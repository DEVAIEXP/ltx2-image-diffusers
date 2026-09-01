"""
Experimental dynamic block manager for the local LTX2 image transformer.

This is intentionally LTX-specific for now. It keeps the transformer shell,
small resident modules, and an optional prefix of transformer blocks on the
execution device. The remaining transformer blocks stay on CPU and are moved to
the execution device only while their forward pass runs. In prefetch modes, the
next streamed block is scheduled on a side CUDA stream while the current block
runs on the default stream.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LTX2DynamicBlockManager:
    VALID_MODES = {
        "sync",
        "prefetch",
        "block_prefetch_v2",
        "block_swap",
        "manual_linear",
        "manual_buffered",
        "manual_cached",
        "manual_selective",
        "manual_linear_static_plan",
        "manual_block_staged",
        "manual_hot_blocks",
    }
    RECOMMENDED_MODE = "manual_linear"

    device: str | torch.device = "cuda:0"
    offload_device: str | torch.device = "cpu"
    enabled: bool = True
    mode: str = "sync"
    pinned_blocks: int = 0
    synchronize: bool = True
    empty_cache_after_offload: bool = False
    verbose: bool = False
    weight_cache_gb: float = 4.0
    pin_cpu_memory: bool = False
    profile: bool = False
    profile_sync_copies: bool = False
    hot_blocks: tuple[int, ...] | list[int] | None = None
    hot_block_budget_gb: float = 0.0
    hot_block_stride: int = 3
    hot_block_offset: int = 0
    streamed_copy_mode: str = "direct"
    keep_streamed_small_tensors_resident: bool = False
    lazy_pin_cpu_memory: bool = False

    def __post_init__(self):
        self.device = torch.device(self.device)
        self.offload_device = torch.device(self.offload_device)
        self.mode = self.mode.lower()
        if self.mode not in self.VALID_MODES:
            valid_modes = ", ".join(sorted(self.VALID_MODES))
            raise ValueError(f"Unsupported LTX2DynamicBlockManager mode {self.mode!r}. Valid modes: {valid_modes}")
        self.pinned_blocks = max(0, int(self.pinned_blocks))
        self.weight_cache_gb = max(0.0, float(self.weight_cache_gb))
        self.hot_blocks = tuple(sorted({int(block) for block in (self.hot_blocks or []) if int(block) >= 0}))
        self.hot_block_budget_gb = max(0.0, float(self.hot_block_budget_gb))
        self.hot_block_stride = max(1, int(self.hot_block_stride))
        self.hot_block_offset = max(0, int(self.hot_block_offset))
        self.streamed_copy_mode = self.streamed_copy_mode.lower()
        if self.streamed_copy_mode not in {"direct", "buffered", "host_buffered"}:
            raise ValueError("streamed_copy_mode must be 'direct', 'buffered', or 'host_buffered'")
        self._active_block: int | None = None
        self._block_count = 0
        self._blocks: nn.ModuleList | None = None
        self._prefetch_stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        self._offload_stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        self._prefetched_index: int | None = None
        self._pending_offload: list[tuple[int, nn.Module]] = []
        self._patched_modules: list[tuple[nn.Module, object]] = []
        self._weight_cache: OrderedDict[tuple[int, str, torch.dtype], torch.Tensor] = OrderedDict()
        self._weight_cache_bytes = 0
        self._weight_cache_limit_bytes = int(self.weight_cache_gb * 1024**3)
        self._buffered_tensors: dict[tuple[str, str, torch.dtype], torch.Tensor] = {}
        self._host_buffered_tensors: dict[tuple[int, torch.dtype], torch.Tensor] = {}
        self._host_buffered_bytes = 0
        self._staged_modules: list[nn.Module] = []
        self._hot_block_indices: set[int] = set()
        self._selective_resident_bytes = 0
        self._streamed_small_resident_bytes = 0
        self._pinned_cpu_bytes = 0
        self._lazy_pinned_cpu_bytes = 0
        self._lazy_pinned_tensor_ids: set[int] = set()
        self._profile_current_block: int | None = None
        self._profile_stats = {
            "setup_runtime": {},
            "block_runtime": {},
            "copy_runtime": {},
        }

    @property
    def prefetch_enabled(self) -> bool:
        return self.mode in {"prefetch", "block_prefetch_v2"} and self._prefetch_stream is not None

    @property
    def block_prefetch_v2_enabled(self) -> bool:
        return self.mode == "block_prefetch_v2" and self._prefetch_stream is not None

    @property
    def block_swap_enabled(self) -> bool:
        return self.mode == "block_swap"

    @property
    def manual_patch_enabled(self) -> bool:
        return self.mode in {"manual_linear", "manual_buffered", "manual_cached", "manual_selective", "manual_linear_static_plan", "manual_block_staged", "manual_hot_blocks"}

    @property
    def manual_cache_enabled(self) -> bool:
        return self.mode == "manual_cached" and self._weight_cache_limit_bytes > 0

    @property
    def manual_buffered_enabled(self) -> bool:
        return self.mode == "manual_buffered"

    @property
    def streamed_buffered_copy_enabled(self) -> bool:
        return self.manual_hot_blocks_enabled and self.streamed_copy_mode == "buffered"

    @property
    def streamed_host_buffered_copy_enabled(self) -> bool:
        return self.manual_hot_blocks_enabled and self.streamed_copy_mode == "host_buffered"

    @property
    def manual_selective_enabled(self) -> bool:
        return self.mode == "manual_selective"

    @property
    def manual_static_plan_enabled(self) -> bool:
        return self.mode == "manual_linear_static_plan"

    @property
    def manual_block_staged_enabled(self) -> bool:
        return self.mode == "manual_block_staged"

    @property
    def manual_hot_blocks_enabled(self) -> bool:
        return self.mode == "manual_hot_blocks"

    def attach(self, transformer: nn.Module) -> None:
        self.prepare_transformer(transformer)
        transformer.set_memory_manager(self)

    def prepare_transformer(self, transformer: nn.Module) -> None:
        if not self.enabled:
            return

        # Keep non-block modules resident. The large repeated body is transformer_blocks.
        setup_start = time.perf_counter()
        resident_names = ["proj_in", "time_embed", "prompt_adaln", "rope", "norm_out", "proj_out"]
        for name in resident_names:
            module = getattr(transformer, name, None)
            if module is not None:
                module.to(self.device)
        self._profile_add("setup_runtime", "resident_modules_to_device", time.perf_counter() - setup_start)

        # Root-level parameters are not covered by moving child modules.
        setup_start = time.perf_counter()
        for _, parameter in transformer.named_parameters(recurse=False):
            parameter.data = parameter.data.to(self.device)
            if parameter._grad is not None:
                parameter._grad.data = parameter._grad.data.to(self.device)

        for _, buffer in transformer.named_buffers(recurse=False):
            buffer.data = buffer.data.to(self.device)
        self._profile_add("setup_runtime", "root_tensors_to_device", time.perf_counter() - setup_start)

        blocks = transformer.transformer_blocks
        self._blocks = blocks
        self._block_count = len(blocks)
        pinned_count = min(self.pinned_blocks, self._block_count)
        setup_start = time.perf_counter()
        self._hot_block_indices = self._select_hot_blocks_by_budget(blocks, pinned_count)
        self._profile_add("setup_runtime", "select_hot_blocks", time.perf_counter() - setup_start)
        setup_start = time.perf_counter()
        for block_index, block in enumerate(blocks):
            keep_resident = block_index < pinned_count or (self.manual_hot_blocks_enabled and block_index in self._hot_block_indices)
            target_device = self.device if keep_resident else self.offload_device
            block.to(target_device)
        self._profile_add("setup_runtime", "blocks_to_target_devices", time.perf_counter() - setup_start)

        if self.keep_streamed_small_tensors_resident:
            setup_start = time.perf_counter()
            self._move_streamed_small_tensors_to_device(blocks, pinned_count=pinned_count)
            self._profile_add("setup_runtime", "streamed_small_tensors_to_device", time.perf_counter() - setup_start)

        if self.pin_cpu_memory and not self.lazy_pin_cpu_memory and self.offload_device.type == "cpu":
            setup_start = time.perf_counter()
            self._pin_cpu_blocks(blocks, pinned_count=pinned_count)
            self._profile_add("setup_runtime", "pin_cpu_blocks", time.perf_counter() - setup_start, self._pinned_cpu_bytes)

        if self.manual_selective_enabled:
            setup_start = time.perf_counter()
            self._pin_selective_tensors(blocks, pinned_count=pinned_count)
            self._profile_add("setup_runtime", "selective_tensors_to_device", time.perf_counter() - setup_start)

        if self.manual_patch_enabled:
            setup_start = time.perf_counter()
            self._patch_manual_modules(blocks)
            self._profile_add("setup_runtime", "patch_manual_modules", time.perf_counter() - setup_start)

        if self.device.type == "cuda":
            setup_start = time.perf_counter()
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            self._profile_add("setup_runtime", "cuda_sync_empty_cache", time.perf_counter() - setup_start)

        if self.verbose:
            streamed = max(0, self._block_count - pinned_count - len(self._hot_block_indices))
            cache_gb = self._weight_cache_limit_bytes / 1024**3
            selective_gb = self._selective_resident_bytes / 1024**3
            streamed_small_gb = self._streamed_small_resident_bytes / 1024**3
            pinned_cpu_gb = self._pinned_cpu_bytes / 1024**3
            print(
                f"  [manager] mode={self.mode} pinned_blocks={pinned_count} "
                f"hot_blocks={sorted(self._hot_block_indices)} hot_block_budget_gb={self.hot_block_budget_gb:g} "
                f"hot_block_stride={self.hot_block_stride} hot_block_offset={self.hot_block_offset} "
                f"streamed_blocks={streamed} streamed_copy_mode={self.streamed_copy_mode} weight_cache_gb={cache_gb:g} "
                f"selective_resident_gb={selective_gb:.3f} streamed_small_resident_gb={streamed_small_gb:.3f} "
                f"pinned_cpu_gb={pinned_cpu_gb:.3f}",
                flush=True,
            )

    def _block_size_bytes(self, block: nn.Module) -> int:
        size = sum(self._tensor_size_bytes(parameter.data) for parameter in block.parameters(recurse=True))
        size += sum(self._tensor_size_bytes(buffer.data) for buffer in block.buffers(recurse=True))
        return size

    def _select_hot_blocks_by_budget(self, blocks: nn.ModuleList, pinned_count: int) -> set[int]:
        explicit_blocks = {block for block in self.hot_blocks if block < len(blocks)}
        if explicit_blocks or not self.manual_hot_blocks_enabled or self.hot_block_budget_gb <= 0:
            return explicit_blocks

        budget_bytes = int(self.hot_block_budget_gb * 1024**3)
        if budget_bytes <= 0:
            return set()

        start_index = pinned_count + self.hot_block_offset
        if start_index >= len(blocks):
            return set()

        candidates = list(range(start_index, len(blocks), self.hot_block_stride))
        selected: list[int] = []
        used_bytes = 0
        for block_index in candidates:
            block_bytes = self._block_size_bytes(blocks[block_index])
            if selected and used_bytes + block_bytes > budget_bytes:
                break
            if block_bytes > budget_bytes:
                continue
            selected.append(block_index)
            used_bytes += block_bytes
        return set(selected)
    def _profile_copy_key(self, tensor_name: str) -> str:
        block = self._profile_current_block
        if block is None:
            return tensor_name
        return f"block:{block}/{tensor_name}"

    def _maybe_sync_profile_copy(self) -> None:
        if self.profile_sync_copies and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _profile_add(self, bucket: str, key: str | int, elapsed: float, byte_count: int = 0) -> None:
        if not self.profile:
            return
        stats = self._profile_stats[bucket].setdefault(str(key), {"calls": 0, "seconds": 0.0, "bytes": 0})
        stats["calls"] += 1
        stats["seconds"] += elapsed
        stats["bytes"] += byte_count

    def _profile_summary_bucket(self, bucket: str) -> dict:
        return {
            key: {
                "calls": value["calls"],
                "seconds": round(value["seconds"], 4),
                "gb": round(value["bytes"] / 1024**3, 4),
            }
            for key, value in sorted(self._profile_stats[bucket].items(), key=lambda item: item[0])
        }

    @property
    def selected_hot_blocks(self) -> tuple[int, ...]:
        return tuple(sorted(self._hot_block_indices))

    def profile_summary(self) -> dict:
        return {
            "mode": self.mode,
            "hot_blocks": list(self.selected_hot_blocks),
            "hot_block_budget_gb": self.hot_block_budget_gb,
            "hot_block_stride": self.hot_block_stride,
            "hot_block_offset": self.hot_block_offset,
            "streamed_copy_mode": self.streamed_copy_mode,
            "keep_streamed_small_tensors_resident": self.keep_streamed_small_tensors_resident,
            "streamed_small_resident_gb": round(self._streamed_small_resident_bytes / 1024**3, 4),
            "host_buffered_cpu_gb": round(self._host_buffered_bytes / 1024**3, 4),
            "lazy_pin_cpu_memory": self.lazy_pin_cpu_memory,
            "lazy_pinned_cpu_gb": round(self._lazy_pinned_cpu_bytes / 1024**3, 4),
            "setup_runtime": self._profile_summary_bucket("setup_runtime"),
            "block_runtime": self._profile_summary_bucket("block_runtime"),
            "copy_runtime": self._profile_summary_bucket("copy_runtime"),
        }

    def _split_profile_copy_key(self, key: str) -> tuple[str | None, str]:
        if key.startswith("block:") and "/" in key:
            block, tensor_name = key.split("/", 1)
            return block.removeprefix("block:"), tensor_name
        return None, key

    def _profile_totals_by_copy_type(self, copy_runtime: dict) -> dict[str, dict[str, float]]:
        totals: dict[str, dict[str, float]] = {}
        for key, value in copy_runtime.items():
            _, tensor_name = self._split_profile_copy_key(key)
            stats = totals.setdefault(tensor_name, {"calls": 0, "seconds": 0.0, "gb": 0.0})
            stats["calls"] += value["calls"]
            stats["seconds"] += value["seconds"]
            stats["gb"] += value["gb"]
        return dict(sorted(totals.items(), key=lambda item: item[1]["seconds"], reverse=True))

    def print_profile_summary(self, *, full: bool = False, top_n: int = 8) -> None:
        if not self.profile:
            return
        summary = self.profile_summary()
        block_runtime = summary["block_runtime"]
        copy_runtime = summary["copy_runtime"]
        setup_runtime = summary["setup_runtime"]
        block_total = sum(value["seconds"] for value in block_runtime.values())
        copy_total = sum(value["seconds"] for value in copy_runtime.values())
        copy_gb = sum(value["gb"] for value in copy_runtime.values())

        print(
            f"  [manager-profile] summary: mode={summary['mode']} "
            f"blocks={len(block_runtime)} block_seconds={block_total:.4f} "
            f"copy_seconds={copy_total:.4f} copy_gb={copy_gb:.4f}",
            flush=True,
        )

        print("  [manager-profile] setup_runtime:", flush=True)
        for key, value in sorted(setup_runtime.items(), key=lambda item: item[1]["seconds"], reverse=True):
            print(
                f"    {key}: calls={value['calls']} seconds={value['seconds']:.4f} gb={value['gb']:.4f}",
                flush=True,
            )

        print("  [manager-profile] copy_runtime_by_type:", flush=True)
        for key, value in self._profile_totals_by_copy_type(copy_runtime).items():
            print(
                f"    {key}: calls={value['calls']} seconds={value['seconds']:.4f} gb={value['gb']:.4f}",
                flush=True,
            )

        print(f"  [manager-profile] slowest_blocks_top_{top_n}:", flush=True)
        slowest_blocks = sorted(block_runtime.items(), key=lambda item: item[1]["seconds"], reverse=True)[:top_n]
        for key, value in slowest_blocks:
            print(
                f"    block {key}: calls={value['calls']} seconds={value['seconds']:.4f}",
                flush=True,
            )

        print(f"  [manager-profile] slowest_copies_top_{top_n}:", flush=True)
        slowest_copies = sorted(copy_runtime.items(), key=lambda item: item[1]["seconds"], reverse=True)[:top_n]
        for key, value in slowest_copies:
            print(
                f"    {key}: calls={value['calls']} seconds={value['seconds']:.4f} gb={value['gb']:.4f}",
                flush=True,
            )

        if not full:
            print(
                "  [manager-profile] full profile saved in metrics JSON; set "
                "LTX_IMAGE_TRANSFORMER_MANAGER_PROFILE_FULL=1 to print it.",
                flush=True,
            )
            return

        print("  [manager-profile] block_runtime_full:", flush=True)
        for key, value in block_runtime.items():
            print(
                f"    block {key}: calls={value['calls']} seconds={value['seconds']:.4f}",
                flush=True,
            )
        print("  [manager-profile] copy_runtime_full:", flush=True)
        for key, value in copy_runtime.items():
            print(
                f"    {key}: calls={value['calls']} seconds={value['seconds']:.4f} gb={value['gb']:.4f}",
                flush=True,
            )
    def _tensor_size_bytes(self, tensor: torch.Tensor) -> int:
        return tensor.numel() * tensor.element_size()

    def _pin_cpu_blocks(self, blocks: nn.ModuleList, pinned_count: int) -> None:
        self._pinned_cpu_bytes = 0
        for block_index, block in enumerate(blocks):
            if block_index < pinned_count:
                continue
            for parameter in block.parameters(recurse=True):
                if parameter.device.type == "cpu" and not parameter.data.is_pinned():
                    parameter.data = parameter.data.pin_memory()
                    self._pinned_cpu_bytes += self._tensor_size_bytes(parameter.data)
            for buffer in block.buffers(recurse=True):
                if buffer.device.type == "cpu" and not buffer.is_pinned():
                    pinned = buffer.pin_memory()
                    buffer.data = pinned
                    self._pinned_cpu_bytes += self._tensor_size_bytes(buffer.data)

    def _maybe_lazy_pin_cpu_tensor(self, tensor: torch.Tensor, tensor_name: str) -> torch.Tensor:
        if not self.lazy_pin_cpu_memory or tensor.device.type != "cpu" or tensor.is_pinned():
            return tensor

        tensor_id = id(tensor)
        start_time = time.perf_counter()
        pinned = tensor.pin_memory()
        elapsed = time.perf_counter() - start_time
        pinned_bytes = self._tensor_size_bytes(pinned)
        if tensor_id not in self._lazy_pinned_tensor_ids:
            self._lazy_pinned_tensor_ids.add(tensor_id)
            self._lazy_pinned_cpu_bytes += pinned_bytes
        self._profile_add("setup_runtime", f"lazy_pin_{tensor_name}", elapsed, pinned_bytes)

        if isinstance(tensor, nn.Parameter):
            tensor.data = pinned
            return tensor
        return pinned

    def _move_parameter_to_device(self, parameter: nn.Parameter) -> int:
        original_bytes = self._tensor_size_bytes(parameter.data)
        parameter.data = parameter.data.to(self.device)
        if parameter._grad is not None:
            parameter._grad.data = parameter._grad.data.to(self.device)
        return original_bytes

    def _move_streamed_small_tensors_to_device(self, blocks: nn.ModuleList, pinned_count: int) -> None:
        self._streamed_small_resident_bytes = 0
        for block_index, block in enumerate(blocks):
            if block_index < pinned_count or block_index in self._hot_block_indices:
                continue

            for module in block.modules():
                if isinstance(module, nn.Linear):
                    if module.bias is not None and module.bias.device != self.device:
                        self._streamed_small_resident_bytes += self._move_parameter_to_device(module.bias)
                elif isinstance(module, (nn.RMSNorm, nn.LayerNorm)):
                    for _, parameter in module.named_parameters(recurse=False):
                        if parameter.device != self.device:
                            self._streamed_small_resident_bytes += self._move_parameter_to_device(parameter)
                    for _, buffer in module.named_buffers(recurse=False):
                        if buffer.device != self.device:
                            self._streamed_small_resident_bytes += self._tensor_size_bytes(buffer.data)
                            buffer.data = buffer.data.to(self.device)

    def _pin_selective_tensors(self, blocks: nn.ModuleList, pinned_count: int) -> None:
        self._selective_resident_bytes = 0
        for block_index, block in enumerate(blocks):
            if block_index < pinned_count:
                continue

            for _, parameter in block.named_parameters(recurse=False):
                self._selective_resident_bytes += self._move_parameter_to_device(parameter)
            for _, buffer in block.named_buffers(recurse=False):
                self._selective_resident_bytes += self._tensor_size_bytes(buffer.data)
                buffer.data = buffer.data.to(self.device)

            for module in block.modules():
                if isinstance(module, nn.Linear):
                    if module.bias is not None:
                        self._selective_resident_bytes += self._move_parameter_to_device(module.bias)
                elif isinstance(module, (nn.RMSNorm, nn.LayerNorm)):
                    for _, parameter in module.named_parameters(recurse=False):
                        self._selective_resident_bytes += self._move_parameter_to_device(parameter)
                    for _, buffer in module.named_buffers(recurse=False):
                        self._selective_resident_bytes += self._tensor_size_bytes(buffer.data)
                        buffer.data = buffer.data.to(self.device)

    def _clear_weight_cache(self) -> None:
        self._weight_cache.clear()
        self._weight_cache_bytes = 0

    def _clear_host_buffers(self) -> None:
        self._host_buffered_tensors.clear()
        self._host_buffered_bytes = 0

    def _evict_weight_cache(self) -> None:
        while self._weight_cache_bytes > self._weight_cache_limit_bytes and self._weight_cache:
            _, cached_tensor = self._weight_cache.popitem(last=False)
            self._weight_cache_bytes -= self._tensor_size_bytes(cached_tensor)
            del cached_tensor

    def _cached_to_input_device(self, tensor: torch.Tensor, input: torch.Tensor, tensor_name: str) -> torch.Tensor:
        key = (id(tensor), str(input.device), input.dtype)
        cached = self._weight_cache.get(key)
        if cached is not None:
            self._weight_cache.move_to_end(key)
            return cached

        profile_key = self._profile_copy_key(tensor_name)
        self._maybe_sync_profile_copy()
        start_time = time.perf_counter()
        cached = tensor.to(device=input.device, dtype=input.dtype, non_blocking=True)
        self._maybe_sync_profile_copy()
        self._profile_add("copy_runtime", profile_key, time.perf_counter() - start_time, self._tensor_size_bytes(cached))
        self._weight_cache[key] = cached
        self._weight_cache_bytes += self._tensor_size_bytes(cached)
        self._evict_weight_cache()
        return self._weight_cache.get(key, cached)

    def _buffered_to_input_device(
        self, tensor: torch.Tensor | None, input: torch.Tensor, buffer_name: str
    ) -> torch.Tensor | None:
        if tensor is None:
            return None
        if tensor.device == input.device and tensor.dtype == input.dtype:
            return tensor

        key = (buffer_name, str(input.device), input.dtype)
        numel = tensor.numel()
        buffer = self._buffered_tensors.get(key)
        if buffer is None or buffer.numel() < numel:
            buffer = torch.empty(numel, device=input.device, dtype=input.dtype)
            self._buffered_tensors[key] = buffer

        view = buffer[:numel].view(tensor.shape)
        profile_key = self._profile_copy_key(buffer_name)
        self._maybe_sync_profile_copy()
        start_time = time.perf_counter()
        view.copy_(tensor, non_blocking=True)
        self._maybe_sync_profile_copy()
        self._profile_add("copy_runtime", profile_key, time.perf_counter() - start_time, self._tensor_size_bytes(view))
        return view

    def _host_buffered_to_input_device(
        self, tensor: torch.Tensor | None, input: torch.Tensor, tensor_name: str
    ) -> torch.Tensor | None:
        if tensor is None:
            return None
        if tensor.device == input.device and tensor.dtype == input.dtype:
            return tensor
        if tensor.device.type != "cpu":
            return self._to_input_device_direct(tensor, input, tensor_name)

        key = (id(tensor), tensor.dtype)
        host_tensor = self._host_buffered_tensors.get(key)
        if host_tensor is None:
            start_time = time.perf_counter()
            host_tensor = torch.empty_like(tensor, device="cpu", pin_memory=True)
            host_tensor.copy_(tensor, non_blocking=False)
            self._profile_add("setup_runtime", f"host_buffer_{tensor_name}", time.perf_counter() - start_time, self._tensor_size_bytes(host_tensor))
            self._host_buffered_tensors[key] = host_tensor
            self._host_buffered_bytes += self._tensor_size_bytes(host_tensor)

        return self._to_input_device_direct(host_tensor, input, tensor_name)

    def _to_input_device_direct(
        self, tensor: torch.Tensor, input: torch.Tensor, tensor_name: str
    ) -> torch.Tensor:
        profile_key = self._profile_copy_key(tensor_name)
        self._maybe_sync_profile_copy()
        start_time = time.perf_counter()
        tensor = self._maybe_lazy_pin_cpu_tensor(tensor, tensor_name)
        moved = tensor.to(device=input.device, dtype=input.dtype, non_blocking=True)
        self._maybe_sync_profile_copy()
        self._profile_add("copy_runtime", profile_key, time.perf_counter() - start_time, self._tensor_size_bytes(moved))
        return moved

    def _to_input_device(self, tensor: torch.Tensor | None, input: torch.Tensor, tensor_name: str = "tensor_to_input") -> torch.Tensor | None:
        if tensor is None:
            return None
        if tensor.device == input.device and tensor.dtype == input.dtype:
            return tensor
        if self.manual_cache_enabled:
            return self._cached_to_input_device(tensor, input, tensor_name)
        if self.streamed_buffered_copy_enabled:
            return self._buffered_to_input_device(tensor, input, tensor_name)
        if self.streamed_host_buffered_copy_enabled:
            return self._host_buffered_to_input_device(tensor, input, tensor_name)
        return self._to_input_device_direct(tensor, input, tensor_name)

    def _stage_block_linear_weights(self, block: nn.Module) -> None:
        self._clear_staged_block()
        for module in block.modules():
            if not isinstance(module, nn.Linear):
                continue

            if module.weight.device == self.device:
                staged_weight = module.weight
            else:
                profile_key = self._profile_copy_key("linear_weight_stage")
                self._maybe_sync_profile_copy()
                start_time = time.perf_counter()
                staged_weight = module.weight.to(device=self.device, dtype=module.weight.dtype, non_blocking=True)
                self._maybe_sync_profile_copy()
                self._profile_add("copy_runtime", profile_key, time.perf_counter() - start_time, self._tensor_size_bytes(staged_weight))

            staged_bias = None
            if module.bias is not None:
                if module.bias.device == self.device:
                    staged_bias = module.bias
                else:
                    profile_key = self._profile_copy_key("linear_bias_stage")
                    self._maybe_sync_profile_copy()
                    start_time = time.perf_counter()
                    staged_bias = module.bias.to(device=self.device, dtype=module.bias.dtype, non_blocking=True)
                    self._maybe_sync_profile_copy()
                    self._profile_add("copy_runtime", profile_key, time.perf_counter() - start_time, self._tensor_size_bytes(staged_bias))

            module._ltx2_staged_weight = staged_weight
            module._ltx2_staged_bias = staged_bias
            self._staged_modules.append(module)

        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()

    def _clear_staged_block(self) -> None:
        while self._staged_modules:
            module = self._staged_modules.pop()
            if hasattr(module, "_ltx2_staged_weight"):
                delattr(module, "_ltx2_staged_weight")
            if hasattr(module, "_ltx2_staged_bias"):
                delattr(module, "_ltx2_staged_bias")

    def _patch_manual_modules(self, blocks: nn.ModuleList) -> None:
        if self._patched_modules:
            return

        def dynamic_linear_forward(linear, input):
            weight = getattr(linear, "_ltx2_staged_weight", None)
            bias = getattr(linear, "_ltx2_staged_bias", None)
            if weight is None:
                weight = self._to_input_device(linear.weight, input, "linear_weight")
            elif weight.device != input.device or weight.dtype != input.dtype:
                weight = self._to_input_device(weight, input, "linear_weight")
            if bias is None and linear.bias is not None:
                bias = self._to_input_device(linear.bias, input, "linear_bias")
            elif bias is not None and (bias.device != input.device or bias.dtype != input.dtype):
                bias = self._to_input_device(bias, input, "linear_bias")
            return F.linear(input, weight, bias)

        def dynamic_rms_norm_forward(norm, input):
            weight = self._to_input_device(norm.weight, input, "rms_norm_weight")
            return F.rms_norm(input, norm.normalized_shape, weight, norm.eps)

        def dynamic_layer_norm_forward(norm, input):
            weight = self._to_input_device(norm.weight, input, "layer_norm_weight")
            bias = self._to_input_device(norm.bias, input, "layer_norm_bias")
            return F.layer_norm(input, norm.normalized_shape, weight, bias, norm.eps)

        def buffered_linear_forward(linear, input):
            weight = self._buffered_to_input_device(linear.weight, input, "linear_weight")
            bias = self._buffered_to_input_device(linear.bias, input, "linear_bias")
            return F.linear(input, weight, bias)

        def buffered_rms_norm_forward(norm, input):
            weight = self._buffered_to_input_device(norm.weight, input, "rms_norm_weight")
            return F.rms_norm(input, norm.normalized_shape, weight, norm.eps)

        def buffered_layer_norm_forward(norm, input):
            weight = self._buffered_to_input_device(norm.weight, input, "layer_norm_weight")
            bias = self._buffered_to_input_device(norm.bias, input, "layer_norm_bias")
            return F.layer_norm(input, norm.normalized_shape, weight, bias, norm.eps)

        def make_static_linear_forward(weight_ref, bias_ref):
            def static_linear_forward(_linear, input):
                weight = self._to_input_device(weight_ref, input, "linear_weight")
                bias = self._to_input_device(bias_ref, input, "linear_bias")
                return F.linear(input, weight, bias)

            return static_linear_forward

        def make_static_rms_norm_forward(weight_ref, normalized_shape, eps):
            def static_rms_norm_forward(_norm, input):
                weight = self._to_input_device(weight_ref, input, "rms_norm_weight")
                return F.rms_norm(input, normalized_shape, weight, eps)

            return static_rms_norm_forward

        def make_static_layer_norm_forward(weight_ref, bias_ref, normalized_shape, eps):
            def static_layer_norm_forward(_norm, input):
                weight = self._to_input_device(weight_ref, input, "linear_weight")
                bias = self._to_input_device(bias_ref, input, "linear_bias")
                return F.layer_norm(input, normalized_shape, weight, bias, eps)

            return static_layer_norm_forward

        for module in blocks.modules():
            if isinstance(module, nn.Linear):
                self._patched_modules.append((module, module.forward))
                if self.manual_buffered_enabled:
                    module.forward = buffered_linear_forward.__get__(module, module.__class__)
                elif self.manual_static_plan_enabled:
                    module.forward = make_static_linear_forward(module.weight, module.bias).__get__(module, module.__class__)
                else:
                    module.forward = dynamic_linear_forward.__get__(module, module.__class__)
            elif isinstance(module, nn.RMSNorm):
                self._patched_modules.append((module, module.forward))
                if self.manual_buffered_enabled:
                    module.forward = buffered_rms_norm_forward.__get__(module, module.__class__)
                elif self.manual_static_plan_enabled:
                    module.forward = make_static_rms_norm_forward(
                        module.weight, module.normalized_shape, module.eps
                    ).__get__(module, module.__class__)
                else:
                    module.forward = dynamic_rms_norm_forward.__get__(module, module.__class__)
            elif isinstance(module, nn.LayerNorm):
                self._patched_modules.append((module, module.forward))
                if self.manual_buffered_enabled:
                    module.forward = buffered_layer_norm_forward.__get__(module, module.__class__)
                elif self.manual_static_plan_enabled:
                    module.forward = make_static_layer_norm_forward(
                        module.weight, module.bias, module.normalized_shape, module.eps
                    ).__get__(module, module.__class__)
                else:
                    module.forward = dynamic_layer_norm_forward.__get__(module, module.__class__)

    def _restore_manual_modules(self) -> None:
        for module, original_forward in self._patched_modules:
            module.forward = original_forward
        self._patched_modules.clear()
        self._clear_weight_cache()
        self._clear_staged_block()
        self._clear_host_buffers()

    def _swap_block_to_device(self, block: nn.Module):
        parameter_handles = []
        buffer_handles = []
        for parameter in block.parameters(recurse=True):
            old_data = parameter.data
            if old_data.device != self.device:
                parameter_handles.append((parameter, old_data))
                start_time = time.perf_counter()
                parameter.data = old_data.to(self.device, non_blocking=True)
                self._profile_add("copy_runtime", "block_swap_parameter_to_device", time.perf_counter() - start_time, self._tensor_size_bytes(parameter.data))
        for module in block.modules():
            for name, buffer in module.named_buffers(recurse=False):
                if buffer.device != self.device:
                    buffer_handles.append((module, name, buffer))
                    start_time = time.perf_counter()
                    new_buffer = buffer.to(self.device, non_blocking=True)
                    self._profile_add("copy_runtime", "block_swap_buffer_to_device", time.perf_counter() - start_time, self._tensor_size_bytes(new_buffer))
                    setattr(module, name, new_buffer)
        return parameter_handles, buffer_handles

    def _restore_block_from_swap(self, handles) -> None:
        parameter_handles, buffer_handles = handles
        for parameter, old_data in reversed(parameter_handles):
            parameter.data = old_data
        for module, name, old_buffer in reversed(buffer_handles):
            setattr(module, name, old_buffer)

    def _is_pinned(self, block_index: int) -> bool:
        return block_index < min(self.pinned_blocks, self._block_count) or block_index in self._hot_block_indices

    def _prefetch_next(self, block_index: int) -> None:
        if not self.prefetch_enabled or self._blocks is None:
            return
        next_index = block_index + 1
        if next_index >= self._block_count or self._is_pinned(next_index):
            return
        if self._prefetched_index == next_index:
            return

        next_block = self._blocks[next_index]
        if self.verbose:
            print(f"  [manager] prefetch block {next_index}", flush=True)
        with torch.cuda.stream(self._prefetch_stream):
            next_block.to(self.device, non_blocking=True)
        self._prefetched_index = next_index

    def _wait_for_prefetch(self, block_index: int) -> bool:
        if not self.prefetch_enabled or self._prefetched_index != block_index:
            return False
        torch.cuda.current_stream(self.device).wait_stream(self._prefetch_stream)
        self._prefetched_index = None
        return True

    def _schedule_deferred_offload(self, block_index: int, block: nn.Module) -> None:
        if self._is_pinned(block_index):
            return
        if self.verbose:
            print(f"  [manager] offload block {block_index}", flush=True)
        block.to(self.offload_device, non_blocking=False)

    def _flush_deferred_offloads(self) -> None:
        while self._pending_offload:
            block_index, block = self._pending_offload.pop(0)
            if not self._is_pinned(block_index):
                block.to(self.offload_device, non_blocking=False)
        if self._offload_stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(self._offload_stream)

    @contextmanager
    def use_block(self, block_index: int, block: nn.Module):
        if not self.enabled:
            yield block
            return

        if self.manual_patch_enabled:
            self._profile_current_block = block_index
            start_time = time.perf_counter()
            try:
                if self.manual_block_staged_enabled:
                    self._stage_block_linear_weights(block)
                yield block
            finally:
                if self.manual_block_staged_enabled:
                    self._clear_staged_block()
                self._profile_add("block_runtime", block_index, time.perf_counter() - start_time)
                self._profile_current_block = None
            return
        if self.block_swap_enabled and not self._is_pinned(block_index):
            if self.verbose:
                print(f"  [manager] swap block {block_index}", flush=True)
            handles = self._swap_block_to_device(block)
            start_time = time.perf_counter()
            try:
                yield block
                if self.synchronize and self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
            finally:
                self._profile_add("block_runtime", block_index, time.perf_counter() - start_time)
                self._restore_block_from_swap(handles)
                if self.empty_cache_after_offload and self.device.type == "cuda":
                    torch.cuda.empty_cache()
            return
        if self._is_pinned(block_index):
            self._prefetch_next(block_index)
            start_time = time.perf_counter()
            try:
                yield block
            finally:
                self._profile_add("block_runtime", block_index, time.perf_counter() - start_time)
            return
        was_prefetched = self._wait_for_prefetch(block_index)
        if not was_prefetched:
            if self.verbose:
                print(f"  [manager] onload streamed block {block_index}", flush=True)
            block.to(self.device, non_blocking=self.block_prefetch_v2_enabled)
            if self.synchronize and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

        self._prefetch_next(block_index)
        self._active_block = block_index
        start_time = time.perf_counter()
        try:
            yield block
            if self.synchronize and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        finally:
            self._profile_add("block_runtime", block_index, time.perf_counter() - start_time)
            if self.block_prefetch_v2_enabled:
                self._schedule_deferred_offload(block_index, block)
            else:
                block.to(self.offload_device, non_blocking=False)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                    if self.empty_cache_after_offload:
                        torch.cuda.empty_cache()
            self._active_block = None

    def detach(self, transformer: nn.Module) -> None:
        transformer.set_memory_manager(None)
        self._restore_manual_modules()
        self._flush_deferred_offloads()
        if self._prefetch_stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(self._prefetch_stream)
        if self._offload_stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(self._offload_stream)
        self._prefetched_index = None
        for block in transformer.transformer_blocks:
            block.to(self.offload_device, non_blocking=False)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
