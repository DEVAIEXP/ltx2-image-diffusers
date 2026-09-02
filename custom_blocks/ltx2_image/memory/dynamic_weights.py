from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import re
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.hooks.hooks import HookRegistry, ModelHook


_DYNAMIC_WEIGHTS_HOOK = "dynamic_weights"


@dataclass(frozen=True)
class DynamicWeightsConfig:
    """Configuration for the experimental dynamic weight planner/runtime."""

    execution_device: str | torch.device = "cuda:0"
    offload_device: str | torch.device = "cpu"
    target_module_classes: tuple[type[nn.Module], ...] = (nn.Linear,)
    skip_modules_pattern: tuple[str, ...] = ()
    always_resident_modules_pattern: tuple[str, ...] = ()
    small_tensor_threshold_bytes: int = 16 * 1024
    execution_mode: str = "plan"
    pin_cpu_memory: bool = False
    pin_cpu_workers: int = 1
    resident_weight_budget_gb: float = 0.0
    resident_weight_selection: str = "spread"
    resident_module_budget_gb: float = 0.0
    resident_module_patterns: tuple[str, ...] = ()
    resident_module_selection: str = "spread"
    verbose: bool = False


@dataclass(frozen=True)
class DynamicWeightPlanEntry:
    module_name: str
    module_type: str
    tensor_name: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    bytes: int
    placement: str


@dataclass
class DynamicWeightsState:
    entries: list[DynamicWeightPlanEntry] = field(default_factory=list)
    module_count: int = 0
    total_bytes: int = 0
    bytes_by_placement: dict[str, int] = field(default_factory=dict)
    setup_seconds_by_action: dict[str, float] = field(default_factory=dict)
    setup_bytes_by_action: dict[str, int] = field(default_factory=dict)
    copy_seconds_by_name: dict[str, float] = field(default_factory=dict)
    copy_bytes_by_name: dict[str, int] = field(default_factory=dict)
    copy_calls_by_name: dict[str, int] = field(default_factory=dict)
    patched_module_count: int = 0

    def add_setup(self, action: str, seconds: float, byte_count: int = 0) -> None:
        self.setup_seconds_by_action[action] = self.setup_seconds_by_action.get(action, 0.0) + seconds
        self.setup_bytes_by_action[action] = self.setup_bytes_by_action.get(action, 0) + byte_count

    def add_copy(self, name: str, seconds: float, byte_count: int) -> None:
        self.copy_seconds_by_name[name] = self.copy_seconds_by_name.get(name, 0.0) + seconds
        self.copy_bytes_by_name[name] = self.copy_bytes_by_name.get(name, 0) + byte_count
        self.copy_calls_by_name[name] = self.copy_calls_by_name.get(name, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "module_count": self.module_count,
            "patched_module_count": self.patched_module_count,
            "total_gb": round(self.total_bytes / 1024**3, 4),
            "bytes_by_placement": {
                key: round(value / 1024**3, 4) for key, value in sorted(self.bytes_by_placement.items())
            },
            "setup_runtime": {
                key: {
                    "seconds": round(self.setup_seconds_by_action[key], 4),
                    "gb": round(self.setup_bytes_by_action.get(key, 0) / 1024**3, 4),
                }
                for key in sorted(self.setup_seconds_by_action)
            },
            "copy_runtime": {
                key: {
                    "calls": self.copy_calls_by_name.get(key, 0),
                    "seconds": round(self.copy_seconds_by_name[key], 4),
                    "gb": round(self.copy_bytes_by_name.get(key, 0) / 1024**3, 4),
                }
                for key in sorted(self.copy_seconds_by_name)
            },
            "entries": [
                {
                    "module_name": entry.module_name,
                    "module_type": entry.module_type,
                    "tensor_name": entry.tensor_name,
                    "shape": list(entry.shape),
                    "dtype": entry.dtype,
                    "device": entry.device,
                    "gb": round(entry.bytes / 1024**3, 6),
                    "placement": entry.placement,
                }
                for entry in self.entries
            ],
        }


class DynamicWeightsHook(ModelHook):
    def __init__(self, config: DynamicWeightsConfig) -> None:
        super().__init__()
        self.config = config
        self.state = DynamicWeightsState()
        self._patched_modules: list[tuple[nn.Module, object]] = []
        self._replaced_parameters: list[tuple[nn.Module, str, nn.Parameter]] = []
        self._linear_weight_store: dict[int, torch.Tensor] = {}
        self._resident_linear_weight_module_ids: set[int] = set()
        self._resident_module_names: set[str] = set()
        self.execution_device = torch.device(config.execution_device)
        self.offload_device = torch.device(config.offload_device)
        self.execution_mode = config.execution_mode.lower()
        self.pin_cpu_workers = max(1, int(config.pin_cpu_workers))
        self.resident_weight_budget_bytes = int(max(0.0, float(config.resident_weight_budget_gb)) * 1024**3)
        self.resident_module_budget_bytes = int(max(0.0, float(config.resident_module_budget_gb)) * 1024**3)
        self.resident_weight_selection = config.resident_weight_selection.lower()
        if self.resident_weight_selection not in {"first", "spread"}:
            raise ValueError("DynamicWeightsConfig.resident_weight_selection must be 'first' or 'spread'")
        self.resident_module_selection = config.resident_module_selection.lower()
        if self.resident_module_selection not in {"first", "spread"}:
            raise ValueError("DynamicWeightsConfig.resident_module_selection must be 'first' or 'spread'")

    def initialize_hook(self, module: nn.Module) -> nn.Module:
        self.state = build_dynamic_weight_plan(module, self.config)
        if self.execution_mode not in {"plan", "linear_runtime", "linear_store_runtime"}:
            raise ValueError(
                "DynamicWeightsConfig.execution_mode must be 'plan', 'linear_runtime', or 'linear_store_runtime'"
            )
        if self.execution_mode in {"linear_runtime", "linear_store_runtime"}:
            self._prepare_linear_runtime(module, use_store=self.execution_mode == "linear_store_runtime")
        if self.config.verbose:
            summary = self.state.as_dict()
            print(
                "  [dynamic-weights] "
                f"mode={self.execution_mode} modules={summary['module_count']} "
                f"patched={summary['patched_module_count']} total_gb={summary['total_gb']:.4f} "
                f"placements={summary['bytes_by_placement']}",
                flush=True,
            )
            if summary["setup_runtime"]:
                print(f"  [dynamic-weights] setup={summary['setup_runtime']}", flush=True)
        return module

    def detach_hook(self, module: nn.Module) -> nn.Module:
        self._restore_patches()
        return module

    def deinitalize_hook(self, module: nn.Module) -> nn.Module:
        self._restore_patches()
        return module

    def pre_forward(self, module: nn.Module, *args, **kwargs) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if self.execution_mode == "plan":
            return args, kwargs
        return (
            tuple(self._move_value_to_execution_device(value, "forward_input") for value in args),
            {
                key: self._move_value_to_execution_device(value, f"forward_input:{key}")
                for key, value in kwargs.items()
            },
        )

    def _restore_patches(self) -> None:
        while self._patched_modules:
            patched_module, original_forward = self._patched_modules.pop()
            patched_module.forward = original_forward
        while self._replaced_parameters:
            patched_module, parameter_name, original_parameter = self._replaced_parameters.pop()
            patched_module._parameters[parameter_name] = original_parameter
        self._linear_weight_store.clear()
        self._resident_linear_weight_module_ids.clear()
        self._resident_module_names.clear()

    def _move_value_to_execution_device(self, value: Any, name: str) -> Any:
        if isinstance(value, torch.Tensor):
            if value.device == self.execution_device:
                return value
            if value.device.type == "meta":
                return value
            start = time.perf_counter()
            moved = value.to(device=self.execution_device, non_blocking=True)
            self.state.add_copy(name, time.perf_counter() - start, _tensor_size_bytes(moved))
            return moved
        if isinstance(value, tuple):
            return tuple(self._move_value_to_execution_device(item, name) for item in value)
        if isinstance(value, list):
            return [self._move_value_to_execution_device(item, name) for item in value]
        if isinstance(value, dict):
            return {key: self._move_value_to_execution_device(item, f"{name}:{key}") for key, item in value.items()}
        return value

    def _prepare_linear_runtime(self, module: nn.Module, *, use_store: bool) -> None:
        skip_patterns = tuple(re.compile(pattern) for pattern in self.config.skip_modules_pattern)
        resident_patterns = tuple(re.compile(pattern) for pattern in self.config.always_resident_modules_pattern)
        linear_modules_to_pin: list[nn.Linear] = []
        store_weights_to_pin: list[int] = []

        self._move_root_local_tensors_to_device(module)
        if self.resident_module_budget_bytes > 0 and self.config.resident_module_patterns:
            start = time.perf_counter()
            selected_bytes = self._select_resident_modules(module, skip_patterns)
            self.state.add_setup("select_resident_modules", time.perf_counter() - start, selected_bytes)
        if use_store and self.resident_weight_budget_bytes > 0:
            start = time.perf_counter()
            selected_bytes = self._select_resident_linear_weights(module, skip_patterns, resident_patterns)
            self.state.add_setup("select_resident_linear_weights", time.perf_counter() - start, selected_bytes)

        for module_name, submodule in module.named_modules():
            if module_name == "":
                continue
            if skip_patterns and any(pattern.search(module_name) for pattern in skip_patterns):
                continue
            if self._is_descendant_of_resident_module(module_name):
                continue
            if module_name in self._resident_module_names:
                start = time.perf_counter()
                before = _module_size_bytes(submodule)
                submodule.to(self.execution_device)
                self.state.add_setup("resident_budget_modules_to_device", time.perf_counter() - start, before)
                continue

            is_resident_module = bool(
                resident_patterns and any(pattern.search(module_name) for pattern in resident_patterns)
            )
            if is_resident_module:
                start = time.perf_counter()
                before = _module_size_bytes(submodule)
                submodule.to(self.execution_device)
                self.state.add_setup("resident_modules_to_device", time.perf_counter() - start, before)
                continue

            if isinstance(submodule, nn.Linear):
                if use_store and id(submodule) in self._resident_linear_weight_module_ids:
                    self._move_resident_linear_to_device(submodule)
                    continue
                self._move_linear_to_runtime_devices(submodule, linear_modules_to_pin, store_weights_to_pin, use_store)
                self._patch_linear(submodule)
            else:
                self._move_small_local_tensors_to_device(submodule)

        if self.config.pin_cpu_memory and linear_modules_to_pin:
            start = time.perf_counter()
            pinned_bytes = self._pin_linear_weights(linear_modules_to_pin)
            self.state.add_setup("pin_linear_weights", time.perf_counter() - start, pinned_bytes)
        if self.config.pin_cpu_memory and store_weights_to_pin:
            start = time.perf_counter()
            pinned_bytes = self._pin_stored_linear_weights(store_weights_to_pin)
            self.state.add_setup("pin_stored_linear_weights", time.perf_counter() - start, pinned_bytes)

    def _move_linear_to_runtime_devices(
        self,
        linear: nn.Linear,
        linear_modules_to_pin: list[nn.Linear],
        store_weights_to_pin: list[int],
        use_store: bool,
    ) -> None:
        if linear.weight.device != self.offload_device:
            start = time.perf_counter()
            weight_bytes = _tensor_size_bytes(linear.weight.data)
            linear.weight.data = linear.weight.data.to(self.offload_device)
            self.state.add_setup("linear_weights_to_offload", time.perf_counter() - start, weight_bytes)
        if linear.bias is not None and linear.bias.device != self.execution_device:
            start = time.perf_counter()
            bias_bytes = _tensor_size_bytes(linear.bias.data)
            linear.bias.data = linear.bias.data.to(self.execution_device)
            self.state.add_setup("linear_bias_to_device", time.perf_counter() - start, bias_bytes)
        if use_store:
            self._store_linear_weight(linear)
            if self._linear_weight_store[id(linear)].device.type == "cpu" and not self._linear_weight_store[id(linear)].is_pinned():
                store_weights_to_pin.append(id(linear))
            return
        if linear.weight.device.type == "cpu" and not linear.weight.data.is_pinned():
            linear_modules_to_pin.append(linear)

    def _select_resident_modules(
        self,
        module: nn.Module,
        skip_patterns: tuple[re.Pattern[str], ...],
    ) -> int:
        module_patterns = tuple(re.compile(pattern) for pattern in self.config.resident_module_patterns)
        candidates: list[tuple[str, nn.Module, int]] = []
        for module_name, submodule in module.named_modules():
            if module_name == "":
                continue
            if skip_patterns and any(pattern.search(module_name) for pattern in skip_patterns):
                continue
            if not any(pattern.search(module_name) for pattern in module_patterns):
                continue
            if any(_is_module_descendant(module_name, selected_name) for selected_name in self._resident_module_names):
                continue
            candidates.append((module_name, submodule, _module_size_bytes(submodule)))

        ordered_candidates = candidates
        if self.resident_module_selection == "spread":
            ordered_candidates = _spread_order(candidates, self.resident_module_budget_bytes)

        selected_bytes = 0
        for module_name, _, module_bytes in ordered_candidates:
            if selected_bytes + module_bytes > self.resident_module_budget_bytes:
                continue
            self._resident_module_names.add(module_name)
            selected_bytes += module_bytes
        return selected_bytes

    def _is_descendant_of_resident_module(self, module_name: str) -> bool:
        return any(_is_module_descendant(module_name, resident_name) for resident_name in self._resident_module_names)

    def _select_resident_linear_weights(
        self,
        module: nn.Module,
        skip_patterns: tuple[re.Pattern[str], ...],
        resident_patterns: tuple[re.Pattern[str], ...],
    ) -> int:
        candidates: list[tuple[str, nn.Linear, int]] = []
        for module_name, submodule in module.named_modules():
            if module_name == "" or not isinstance(submodule, nn.Linear):
                continue
            if skip_patterns and any(pattern.search(module_name) for pattern in skip_patterns):
                continue
            if self._is_descendant_of_resident_module(module_name):
                continue
            if resident_patterns and any(pattern.search(module_name) for pattern in resident_patterns):
                continue
            candidates.append((module_name, submodule, _tensor_size_bytes(submodule.weight.data)))

        if self.resident_weight_selection == "first":
            ordered_candidates = candidates
        else:
            ordered_candidates = _spread_order(candidates, self.resident_weight_budget_bytes)

        selected_bytes = 0
        for _, linear, weight_bytes in ordered_candidates:
            if selected_bytes + weight_bytes > self.resident_weight_budget_bytes:
                continue
            self._resident_linear_weight_module_ids.add(id(linear))
            selected_bytes += weight_bytes
        return selected_bytes

    def _move_resident_linear_to_device(self, linear: nn.Linear) -> None:
        start = time.perf_counter()
        moved_bytes = 0
        if linear.weight.device != self.execution_device:
            weight_bytes = _tensor_size_bytes(linear.weight.data)
            linear.weight.data = linear.weight.data.to(self.execution_device, non_blocking=True)
            moved_bytes += weight_bytes
        if linear.bias is not None and linear.bias.device != self.execution_device:
            bias_bytes = _tensor_size_bytes(linear.bias.data)
            linear.bias.data = linear.bias.data.to(self.execution_device, non_blocking=True)
            moved_bytes += bias_bytes
        if moved_bytes:
            self.state.add_setup("resident_linear_weights_to_device", time.perf_counter() - start, moved_bytes)

    def _move_root_local_tensors_to_device(self, module: nn.Module) -> None:
        start = time.perf_counter()
        moved_bytes = 0
        for parameter in module.parameters(recurse=False):
            if parameter.device == self.execution_device:
                continue
            tensor_bytes = _tensor_size_bytes(parameter.data)
            parameter.data = parameter.data.to(self.execution_device)
            moved_bytes += tensor_bytes
        for buffer in module.buffers(recurse=False):
            if buffer.device == self.execution_device:
                continue
            tensor_bytes = _tensor_size_bytes(buffer.data)
            buffer.data = buffer.data.to(self.execution_device)
            moved_bytes += tensor_bytes
        if moved_bytes:
            self.state.add_setup("root_tensors_to_device", time.perf_counter() - start, moved_bytes)

    def _store_linear_weight(self, linear: nn.Linear) -> None:
        if id(linear) in self._linear_weight_store:
            return
        original_parameter = linear._parameters["weight"]
        self._linear_weight_store[id(linear)] = original_parameter.detach()
        self._replaced_parameters.append((linear, "weight", original_parameter))
        meta_weight = torch.empty_strided(
            tuple(original_parameter.shape),
            tuple(original_parameter.stride()),
            device="meta",
            dtype=original_parameter.dtype,
        )
        linear._parameters["weight"] = nn.Parameter(meta_weight, requires_grad=original_parameter.requires_grad)

    def _move_small_local_tensors_to_device(self, module: nn.Module) -> int:
        moved_bytes = 0
        for parameter in module.parameters(recurse=False):
            if parameter.device == self.execution_device:
                continue
            tensor_bytes = _tensor_size_bytes(parameter.data)
            if tensor_bytes > self.config.small_tensor_threshold_bytes:
                continue
            start = time.perf_counter()
            parameter.data = parameter.data.to(self.execution_device)
            self.state.add_setup("small_parameters_to_device", time.perf_counter() - start, tensor_bytes)
            moved_bytes += tensor_bytes
        for buffer in module.buffers(recurse=False):
            if buffer.device == self.execution_device:
                continue
            tensor_bytes = _tensor_size_bytes(buffer.data)
            if tensor_bytes > self.config.small_tensor_threshold_bytes:
                continue
            start = time.perf_counter()
            buffer.data = buffer.data.to(self.execution_device)
            self.state.add_setup("small_buffers_to_device", time.perf_counter() - start, tensor_bytes)
            moved_bytes += tensor_bytes
        return moved_bytes

    def _pin_linear_weights(self, linears: list[nn.Linear]) -> int:
        pinned_bytes = 0

        def pin_linear(linear: nn.Linear):
            pinned = linear.weight.data.pin_memory()
            return linear, pinned, _tensor_size_bytes(pinned)

        def assign_pinned(result) -> None:
            nonlocal pinned_bytes
            linear, pinned, tensor_bytes = result
            linear.weight.data = pinned
            pinned_bytes += tensor_bytes

        if self.pin_cpu_workers == 1:
            for linear in linears:
                assign_pinned(pin_linear(linear))
            return pinned_bytes

        linears_iter = iter(linears)
        with ThreadPoolExecutor(max_workers=self.pin_cpu_workers) as executor:
            futures = set()

            def submit_next() -> bool:
                try:
                    linear = next(linears_iter)
                except StopIteration:
                    return False
                futures.add(executor.submit(pin_linear, linear))
                return True

            for _ in range(self.pin_cpu_workers):
                if not submit_next():
                    break

            while futures:
                for future in as_completed(futures):
                    futures.remove(future)
                    assign_pinned(future.result())
                    submit_next()
                    break
        return pinned_bytes

    def _pin_stored_linear_weights(self, store_keys: list[int]) -> int:
        pinned_bytes = 0

        def pin_key(store_key: int):
            source = self._linear_weight_store[store_key]
            pinned = source.pin_memory()
            return store_key, pinned, _tensor_size_bytes(pinned)

        def assign_pinned(result) -> None:
            nonlocal pinned_bytes
            store_key, pinned, tensor_bytes = result
            self._linear_weight_store[store_key] = pinned
            pinned_bytes += tensor_bytes

        if self.pin_cpu_workers == 1:
            for store_key in store_keys:
                assign_pinned(pin_key(store_key))
            return pinned_bytes

        store_keys_iter = iter(store_keys)
        with ThreadPoolExecutor(max_workers=self.pin_cpu_workers) as executor:
            futures = set()

            def submit_next() -> bool:
                try:
                    store_key = next(store_keys_iter)
                except StopIteration:
                    return False
                futures.add(executor.submit(pin_key, store_key))
                return True

            for _ in range(self.pin_cpu_workers):
                if not submit_next():
                    break

            while futures:
                for future in as_completed(futures):
                    futures.remove(future)
                    assign_pinned(future.result())
                    submit_next()
                    break
        return pinned_bytes

    def _patch_linear(self, linear: nn.Linear) -> None:
        self._patched_modules.append((linear, linear.forward))
        self.state.patched_module_count += 1

        def dynamic_linear_forward(patched_linear, input):
            stored_weight = self._linear_weight_store.get(id(patched_linear))
            weight_source = stored_weight if stored_weight is not None else patched_linear.weight
            weight_copy_name = "linear_store_weight" if stored_weight is not None else "linear_weight"
            weight = self._to_input_device(weight_source, input, weight_copy_name)
            bias = self._to_input_device(patched_linear.bias, input, "linear_bias")
            return F.linear(input, weight, bias)

        linear.forward = dynamic_linear_forward.__get__(linear, linear.__class__)

    def _to_input_device(self, tensor: torch.Tensor | None, input: torch.Tensor, name: str) -> torch.Tensor | None:
        if tensor is None:
            return None
        if tensor.device == input.device and tensor.dtype == input.dtype:
            return tensor
        start = time.perf_counter()
        moved = tensor.to(device=input.device, dtype=input.dtype, non_blocking=True)
        self.state.add_copy(name, time.perf_counter() - start, _tensor_size_bytes(moved))
        return moved


def apply_dynamic_weights(module: nn.Module, config: DynamicWeightsConfig | None = None) -> DynamicWeightsHook:
    config = config or DynamicWeightsConfig()
    registry = HookRegistry.check_if_exists_or_initialize(module)
    existing_hook = registry.get_hook(_DYNAMIC_WEIGHTS_HOOK)
    if existing_hook is not None:
        registry.remove_hook(_DYNAMIC_WEIGHTS_HOOK)
    hook = DynamicWeightsHook(config)
    registry.register_hook(hook, _DYNAMIC_WEIGHTS_HOOK)
    return hook


def remove_dynamic_weights(module: nn.Module, recurse: bool = True) -> None:
    if hasattr(module, "_diffusers_hook"):
        module._diffusers_hook.remove_hook(_DYNAMIC_WEIGHTS_HOOK, recurse=recurse)


def get_dynamic_weights_state(module: nn.Module) -> DynamicWeightsState | None:
    if not hasattr(module, "_diffusers_hook"):
        return None
    hook = module._diffusers_hook.get_hook(_DYNAMIC_WEIGHTS_HOOK)
    if hook is None:
        return None
    return hook.state


def build_dynamic_weight_plan(module: nn.Module, config: DynamicWeightsConfig) -> DynamicWeightsState:
    state = DynamicWeightsState()
    skip_patterns = tuple(re.compile(pattern) for pattern in config.skip_modules_pattern)
    resident_patterns = tuple(re.compile(pattern) for pattern in config.always_resident_modules_pattern)

    for module_name, submodule in module.named_modules():
        if module_name == "":
            continue
        if skip_patterns and any(pattern.search(module_name) for pattern in skip_patterns):
            continue
        if not isinstance(submodule, config.target_module_classes):
            continue

        state.module_count += 1
        module_resident = bool(resident_patterns and any(pattern.search(module_name) for pattern in resident_patterns))
        for tensor_name, tensor in _iter_local_tensors(submodule):
            placement = _classify_tensor(tensor, module_resident, config.small_tensor_threshold_bytes)
            entry = DynamicWeightPlanEntry(
                module_name=module_name,
                module_type=submodule.__class__.__name__,
                tensor_name=tensor_name,
                shape=tuple(tensor.shape),
                dtype=str(tensor.dtype),
                device=str(tensor.device),
                bytes=_tensor_size_bytes(tensor),
                placement=placement,
            )
            state.entries.append(entry)
            state.total_bytes += entry.bytes
            state.bytes_by_placement[placement] = state.bytes_by_placement.get(placement, 0) + entry.bytes

    return state


def _iter_local_tensors(module: nn.Module):
    for name, parameter in module.named_parameters(recurse=False):
        yield name, parameter.data
    for name, buffer in module.named_buffers(recurse=False):
        yield name, buffer.data


def _classify_tensor(tensor: torch.Tensor, module_resident: bool, small_tensor_threshold_bytes: int) -> str:
    if module_resident:
        return "resident_module"
    if _tensor_size_bytes(tensor) <= small_tensor_threshold_bytes:
        return "resident_small"
    return "streamed_large"


def _module_size_bytes(module: nn.Module) -> int:
    size = sum(_tensor_size_bytes(parameter.data) for parameter in module.parameters(recurse=True))
    size += sum(_tensor_size_bytes(buffer.data) for buffer in module.buffers(recurse=True))
    return size


def _is_module_descendant(module_name: str, parent_name: str) -> bool:
    return module_name.startswith(f"{parent_name}.")


def _spread_order(items: list[tuple[str, Any, int]], budget_bytes: int) -> list[tuple[str, Any, int]]:
    if len(items) <= 2:
        return items
    average_bytes = max(1, sum(item[2] for item in items) // len(items))
    target_count = max(1, min(len(items), budget_bytes // average_bytes))
    if target_count == 1:
        spread_indices = [len(items) // 2]
    else:
        spread_indices = [
            round(position * (len(items) - 1) / (target_count - 1))
            for position in range(target_count)
        ]
    seen = set()
    ordered: list[tuple[str, nn.Linear, int]] = []
    for index in spread_indices:
        if index not in seen:
            seen.add(index)
            ordered.append(items[index])
    for index, item in enumerate(items):
        if index not in seen:
            ordered.append(item)
    return ordered


def _tensor_size_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()
