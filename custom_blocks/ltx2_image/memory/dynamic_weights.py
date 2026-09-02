from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

import torch
import torch.nn as nn

from diffusers.hooks.hooks import HookRegistry, ModelHook


_DYNAMIC_WEIGHTS_HOOK = "dynamic_weights"


@dataclass(frozen=True)
class DynamicWeightsConfig:
    """Configuration for the experimental dynamic weight planner.

    This first version is intentionally planner-only. It follows the Diffusers
    hook style while we validate model-agnostic load-list behavior before
    adding lazy materialization or caching.
    """

    execution_device: str | torch.device = "cuda:0"
    offload_device: str | torch.device = "cpu"
    target_module_classes: tuple[type[nn.Module], ...] = (nn.Linear,)
    skip_modules_pattern: tuple[str, ...] = ()
    always_resident_modules_pattern: tuple[str, ...] = ()
    small_tensor_threshold_bytes: int = 16 * 1024
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

    def as_dict(self) -> dict[str, Any]:
        return {
            "module_count": self.module_count,
            "total_gb": round(self.total_bytes / 1024**3, 4),
            "bytes_by_placement": {
                key: round(value / 1024**3, 4) for key, value in sorted(self.bytes_by_placement.items())
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

    def initialize_hook(self, module: nn.Module) -> nn.Module:
        self.state = build_dynamic_weight_plan(module, self.config)
        if self.config.verbose:
            summary = self.state.as_dict()
            print(
                "  [dynamic-weights] "
                f"modules={summary['module_count']} total_gb={summary['total_gb']:.4f} "
                f"placements={summary['bytes_by_placement']}",
                flush=True,
            )
        return module


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


def _tensor_size_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()
