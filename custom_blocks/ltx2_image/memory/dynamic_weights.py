from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.hooks.hooks import HookRegistry, ModelHook


_DEFAULT_TARGET_MODULE_CLASSES = (nn.Linear, nn.Embedding)
_DYNAMIC_WEIGHTS_HOOK = "dynamic_weights"
_DYNAMIC_WEIGHTS_ENV_PREFIX = "DIFFUSERS_DYNAMIC_WEIGHTS_"
_LEGACY_DYNAMIC_WEIGHTS_ENV_PREFIX = "LTX_IMAGE_DYNAMIC_WEIGHTS_"
_PIN_MEMORY_ERRORS = (RuntimeError, getattr(torch, "AcceleratorError", RuntimeError))
_DEFAULT_ALWAYS_RESIDENT_MODULE_PATTERNS = (
    r"(^|\.)(proj_in|time_embed|prompt_adaln|norm_out|proj_out)(\.|$)",
)
_AUTO_BUDGET_DISABLED = "off"
_AUTO_BUDGET_BALANCED = "balanced"
_DYNAMIC_WEIGHTS_PLAN_CACHE: dict[tuple[Any, ...], "DynamicWeightsState"] = {}
_DYNAMIC_WEIGHTS_PLAN_CACHE_LOCK = threading.Lock()
_DYNAMIC_WEIGHTS_PINNED_TENSOR_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}
_DYNAMIC_WEIGHTS_PINNED_TENSOR_CACHE_LOCK = threading.Lock()
_DYNAMIC_WEIGHTS_RESIDENT_DEVICE_TENSOR_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}
_DYNAMIC_WEIGHTS_RESIDENT_DEVICE_TENSOR_CACHE_LOCK = threading.Lock()


def generic_dynamic_weights_env_name(name: str) -> str:
    if name.startswith(_LEGACY_DYNAMIC_WEIGHTS_ENV_PREFIX):
        return _DYNAMIC_WEIGHTS_ENV_PREFIX + name[len(_LEGACY_DYNAMIC_WEIGHTS_ENV_PREFIX) :]
    return name


def legacy_dynamic_weights_env_name(name: str) -> str:
    if name.startswith(_DYNAMIC_WEIGHTS_ENV_PREFIX):
        return _LEGACY_DYNAMIC_WEIGHTS_ENV_PREFIX + name[len(_DYNAMIC_WEIGHTS_ENV_PREFIX) :]
    return name


def dynamic_weights_env_names(name: str) -> tuple[str, ...]:
    generic_name = generic_dynamic_weights_env_name(name)
    legacy_name = legacy_dynamic_weights_env_name(name)
    if generic_name == legacy_name:
        return (name,)
    return (generic_name, legacy_name)


def dynamic_weights_env_value(name: str, default: str = "", environ: Mapping[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    for env_name in dynamic_weights_env_names(name):
        if env_name in env:
            return env[env_name]
    return env.get(name, default)


def _expand_dynamic_weights_preset_aliases(presets: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    expanded_presets: dict[str, dict[str, str]] = {}
    for preset_name, values in presets.items():
        expanded_values = dict(values)
        for env_name, value in values.items():
            for alias in dynamic_weights_env_names(env_name):
                expanded_values.setdefault(alias, value)
        expanded_presets[preset_name] = expanded_values
    return expanded_presets


def is_wsl_environment() -> bool:
    if os.name == "nt":
        return False
    try:
        release = os.uname().release.lower()
    except AttributeError:
        release = ""
    if "microsoft" in release or "wsl" in release:
        return True
    try:
        version = Path("/proc/version").read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return False
    return "microsoft" in version or "wsl" in version


_DYNAMIC_WEIGHTS_PRESET_VALUES: dict[str, dict[str, str]] = {
    "off": {
        "DIFFUSERS_DYNAMIC_WEIGHTS_EXECUTION_MODE": "plan",
        "DIFFUSERS_DYNAMIC_WEIGHTS_PLAN": "0",
    },
    "windows_fast": {
        "DIFFUSERS_RUNNER_TEXT_ENCODER_GROUP_OFFLOAD": "1",
        "DIFFUSERS_RUNNER_TEXT_ENCODER_OFFLOAD_TYPE": "leaf_level",
        "DIFFUSERS_RUNNER_TEXT_ENCODER_OFFLOAD_STREAM": "1",
        "DIFFUSERS_RUNNER_TRANSFORMER_MEMORY_MANAGER": "off",
        "DIFFUSERS_RUNNER_TRANSFORMER_GROUP_OFFLOAD": "0",
        "DIFFUSERS_RUNNER_ATTENTION_BACKEND": "native",
        "DIFFUSERS_DYNAMIC_WEIGHTS_EXECUTION_MODE": "linear_runtime",
        "DIFFUSERS_DYNAMIC_WEIGHTS_PIN_CPU_MEMORY": "1",
        "DIFFUSERS_DYNAMIC_WEIGHTS_LAZY_PIN_CPU_MEMORY": "0",
        "DIFFUSERS_DYNAMIC_WEIGHTS_ALLOW_PIN_MEMORY_FALLBACK": "1",
        "DIFFUSERS_DYNAMIC_WEIGHTS_PIN_CPU_WORKERS": "4",
        "DIFFUSERS_DYNAMIC_WEIGHTS_AUTO_BUDGET_POLICY": "off",
        "DIFFUSERS_DYNAMIC_WEIGHTS_RESIDENT_MODULE_BUDGET_GB": "6",
        "DIFFUSERS_DYNAMIC_WEIGHTS_RESIDENT_MODULE_PATTERNS": "auto",
        "DIFFUSERS_DYNAMIC_WEIGHTS_RESIDENT_MODULE_SELECTION": "spread",
        "DIFFUSERS_DYNAMIC_WEIGHTS_SMALL_TENSOR_THRESHOLD_KB": "1024",
        "DIFFUSERS_RUNNER_PRE_VAE_CLEANUP_REPEATS": "1",
    },
    "linux_native_fast": {
        "DIFFUSERS_RUNNER_TEXT_ENCODER_GROUP_OFFLOAD": "1",
        "DIFFUSERS_RUNNER_TEXT_ENCODER_OFFLOAD_TYPE": "leaf_level",
        "DIFFUSERS_RUNNER_TEXT_ENCODER_OFFLOAD_STREAM": "1",
        "DIFFUSERS_RUNNER_TRANSFORMER_MEMORY_MANAGER": "off",
        "DIFFUSERS_RUNNER_TRANSFORMER_GROUP_OFFLOAD": "0",
        "DIFFUSERS_RUNNER_ATTENTION_BACKEND": "native",
        "DIFFUSERS_DYNAMIC_WEIGHTS_EXECUTION_MODE": "linear_runtime",
        "DIFFUSERS_DYNAMIC_WEIGHTS_PIN_CPU_MEMORY": "1",
        "DIFFUSERS_DYNAMIC_WEIGHTS_LAZY_PIN_CPU_MEMORY": "0",
        "DIFFUSERS_DYNAMIC_WEIGHTS_ALLOW_PIN_MEMORY_FALLBACK": "1",
        "DIFFUSERS_DYNAMIC_WEIGHTS_PIN_CPU_WORKERS": "4",
        "DIFFUSERS_DYNAMIC_WEIGHTS_AUTO_BUDGET_POLICY": "off",
        "DIFFUSERS_DYNAMIC_WEIGHTS_RESIDENT_MODULE_BUDGET_GB": "6",
        "DIFFUSERS_DYNAMIC_WEIGHTS_RESIDENT_MODULE_PATTERNS": "auto",
        "DIFFUSERS_DYNAMIC_WEIGHTS_RESIDENT_MODULE_SELECTION": "spread",
        "DIFFUSERS_DYNAMIC_WEIGHTS_SMALL_TENSOR_THRESHOLD_KB": "1024",
        "DIFFUSERS_RUNNER_PRE_VAE_CLEANUP_REPEATS": "1",
    },
    "wsl_compat": {
        "DIFFUSERS_RUNNER_TEXT_ENCODER_GROUP_OFFLOAD": "1",
        "DIFFUSERS_RUNNER_TEXT_ENCODER_OFFLOAD_TYPE": "leaf_level",
        "DIFFUSERS_RUNNER_TEXT_ENCODER_OFFLOAD_STREAM": "0",
        "DIFFUSERS_RUNNER_TRANSFORMER_MEMORY_MANAGER": "off",
        "DIFFUSERS_RUNNER_TRANSFORMER_GROUP_OFFLOAD": "0",
        "DIFFUSERS_RUNNER_ATTENTION_BACKEND": "native",
        "DIFFUSERS_DYNAMIC_WEIGHTS_EXECUTION_MODE": "linear_runtime",
        "DIFFUSERS_DYNAMIC_WEIGHTS_PIN_CPU_MEMORY": "0",
        "DIFFUSERS_DYNAMIC_WEIGHTS_LAZY_PIN_CPU_MEMORY": "0",
        "DIFFUSERS_DYNAMIC_WEIGHTS_ALLOW_PIN_MEMORY_FALLBACK": "1",
        "DIFFUSERS_DYNAMIC_WEIGHTS_DISABLE_PIN_ON_WSL": "1",
        "DIFFUSERS_DYNAMIC_WEIGHTS_PIN_CPU_WORKERS": "1",
        "DIFFUSERS_DYNAMIC_WEIGHTS_AUTO_BUDGET_POLICY": "off",
        "DIFFUSERS_DYNAMIC_WEIGHTS_RESIDENT_MODULE_BUDGET_GB": "6",
        "DIFFUSERS_DYNAMIC_WEIGHTS_RESIDENT_MODULE_PATTERNS": "auto",
        "DIFFUSERS_DYNAMIC_WEIGHTS_RESIDENT_MODULE_SELECTION": "spread",
        "DIFFUSERS_DYNAMIC_WEIGHTS_SMALL_TENSOR_THRESHOLD_KB": "1024",
        "DIFFUSERS_RUNNER_PRE_VAE_CLEANUP_REPEATS": "3",
    },
    "one_shot_fast": {
        "DIFFUSERS_RUNNER_TRANSFORMER_MEMORY_MANAGER": "off",
        "DIFFUSERS_RUNNER_TRANSFORMER_GROUP_OFFLOAD": "0",
        "DIFFUSERS_RUNNER_ATTENTION_BACKEND": "native",
        "DIFFUSERS_DYNAMIC_WEIGHTS_EXECUTION_MODE": "linear_runtime",
        "DIFFUSERS_DYNAMIC_WEIGHTS_PIN_CPU_MEMORY": "1",
        "DIFFUSERS_DYNAMIC_WEIGHTS_LAZY_PIN_CPU_MEMORY": "0",
        "DIFFUSERS_DYNAMIC_WEIGHTS_ALLOW_PIN_MEMORY_FALLBACK": "1",
        "DIFFUSERS_DYNAMIC_WEIGHTS_PIN_CPU_WORKERS": "4",
        "DIFFUSERS_DYNAMIC_WEIGHTS_AUTO_BUDGET_POLICY": "off",
        "DIFFUSERS_DYNAMIC_WEIGHTS_RESIDENT_MODULE_BUDGET_GB": "6",
        "DIFFUSERS_DYNAMIC_WEIGHTS_RESIDENT_MODULE_PATTERNS": "auto",
        "DIFFUSERS_DYNAMIC_WEIGHTS_RESIDENT_MODULE_SELECTION": "spread",
        "DIFFUSERS_DYNAMIC_WEIGHTS_SMALL_TENSOR_THRESHOLD_KB": "1024",
    },
    "one_shot_overlap": {
        "DIFFUSERS_RUNNER_TEXT_ENCODER_GROUP_OFFLOAD": "1",
        "DIFFUSERS_RUNNER_TEXT_ENCODER_OFFLOAD_TYPE": "leaf_level",
        "DIFFUSERS_RUNNER_TEXT_ENCODER_OFFLOAD_STREAM": "1",
        "DIFFUSERS_RUNNER_TEXT_ENCODER_DYNAMIC_WEIGHTS": "0",
        "DIFFUSERS_RUNNER_TRANSFORMER_MEMORY_MANAGER": "off",
        "DIFFUSERS_RUNNER_TRANSFORMER_GROUP_OFFLOAD": "0",
        "DIFFUSERS_RUNNER_ATTENTION_BACKEND": "native",
        "DIFFUSERS_DYNAMIC_WEIGHTS_EXECUTION_MODE": "linear_runtime",
        "DIFFUSERS_DYNAMIC_WEIGHTS_PIN_CPU_MEMORY": "1",
        "DIFFUSERS_DYNAMIC_WEIGHTS_LAZY_PIN_CPU_MEMORY": "0",
        "DIFFUSERS_DYNAMIC_WEIGHTS_ALLOW_PIN_MEMORY_FALLBACK": "1",
        "DIFFUSERS_DYNAMIC_WEIGHTS_OVERLAP_PIN_SETUP": "1",
        "DIFFUSERS_DYNAMIC_WEIGHTS_PIN_CPU_WORKERS": "4",
        "DIFFUSERS_DYNAMIC_WEIGHTS_AUTO_BUDGET_POLICY": "balanced",
        "DIFFUSERS_DYNAMIC_WEIGHTS_MAX_RESIDENT_MODULE_BUDGET_GB": "6",
        "DIFFUSERS_DYNAMIC_WEIGHTS_MAX_PIN_WEIGHT_BUDGET_GB": "0",
        "DIFFUSERS_DYNAMIC_WEIGHTS_PIN_WEIGHT_BUDGET_GB": "0",
        "DIFFUSERS_DYNAMIC_WEIGHTS_PIN_WEIGHT_BUDGET_RATIO": "0",
        "DIFFUSERS_DYNAMIC_WEIGHTS_PIN_WEIGHT_SELECTION": "spread",
        "DIFFUSERS_DYNAMIC_WEIGHTS_RESIDENT_MODULE_BUDGET_GB": "0",
        "DIFFUSERS_DYNAMIC_WEIGHTS_RESIDENT_MODULE_PATTERNS": "auto",
        "DIFFUSERS_DYNAMIC_WEIGHTS_RESIDENT_MODULE_SELECTION": "spread",
        "DIFFUSERS_DYNAMIC_WEIGHTS_SMALL_TENSOR_THRESHOLD_KB": "1024",
        "DIFFUSERS_DYNAMIC_WEIGHTS_CACHE_PINNED_WEIGHTS": "0",
        "DIFFUSERS_DYNAMIC_WEIGHTS_CACHE_RESIDENT_DEVICE_TENSORS": "0",
        "DIFFUSERS_RUNNER_TRANSFORMER_PREPARE_REPEATS": "1",
        "DIFFUSERS_RUNNER_GENERATION_REPEATS": "1",
    },
    "planner_balanced": {
        "DIFFUSERS_RUNNER_TEXT_ENCODER_GROUP_OFFLOAD": "1",
        "DIFFUSERS_RUNNER_TEXT_ENCODER_OFFLOAD_TYPE": "leaf_level",
        "DIFFUSERS_RUNNER_TEXT_ENCODER_OFFLOAD_STREAM": "1",
        "DIFFUSERS_RUNNER_TEXT_ENCODER_DYNAMIC_WEIGHTS": "0",
        "DIFFUSERS_RUNNER_TRANSFORMER_MEMORY_MANAGER": "off",
        "DIFFUSERS_RUNNER_TRANSFORMER_GROUP_OFFLOAD": "0",
        "DIFFUSERS_RUNNER_ATTENTION_BACKEND": "native",
        "DIFFUSERS_DYNAMIC_WEIGHTS_EXECUTION_MODE": "linear_runtime",
        "DIFFUSERS_DYNAMIC_WEIGHTS_PIN_CPU_MEMORY": "1",
        "DIFFUSERS_DYNAMIC_WEIGHTS_LAZY_PIN_CPU_MEMORY": "0",
        "DIFFUSERS_DYNAMIC_WEIGHTS_ALLOW_PIN_MEMORY_FALLBACK": "1",
        "DIFFUSERS_DYNAMIC_WEIGHTS_PIN_CPU_WORKERS": "4",
        "DIFFUSERS_DYNAMIC_WEIGHTS_AUTO_BUDGET_POLICY": "balanced",
        "DIFFUSERS_DYNAMIC_WEIGHTS_MAX_RESIDENT_MODULE_BUDGET_GB": "6",
        "DIFFUSERS_DYNAMIC_WEIGHTS_MAX_PIN_WEIGHT_BUDGET_GB": "0",
        "DIFFUSERS_DYNAMIC_WEIGHTS_PIN_WEIGHT_BUDGET_GB": "0",
        "DIFFUSERS_DYNAMIC_WEIGHTS_PIN_WEIGHT_BUDGET_RATIO": "0",
        "DIFFUSERS_DYNAMIC_WEIGHTS_PIN_WEIGHT_SELECTION": "spread",
        "DIFFUSERS_DYNAMIC_WEIGHTS_RESIDENT_MODULE_BUDGET_GB": "0",
        "DIFFUSERS_DYNAMIC_WEIGHTS_RESIDENT_MODULE_PATTERNS": "auto",
        "DIFFUSERS_DYNAMIC_WEIGHTS_RESIDENT_MODULE_SELECTION": "spread",
        "DIFFUSERS_DYNAMIC_WEIGHTS_SMALL_TENSOR_THRESHOLD_KB": "1024",
    },
    "warm_server": {
        "DIFFUSERS_RUNNER_GENERATION_REPEATS": "2",
        "DIFFUSERS_RUNNER_TRANSFORMER_MEMORY_MANAGER": "off",
        "DIFFUSERS_RUNNER_TRANSFORMER_GROUP_OFFLOAD": "0",
        "DIFFUSERS_RUNNER_ATTENTION_BACKEND": "native",
        "DIFFUSERS_DYNAMIC_WEIGHTS_EXECUTION_MODE": "linear_runtime",
        "DIFFUSERS_DYNAMIC_WEIGHTS_PIN_CPU_MEMORY": "1",
        "DIFFUSERS_DYNAMIC_WEIGHTS_LAZY_PIN_CPU_MEMORY": "0",
        "DIFFUSERS_DYNAMIC_WEIGHTS_ALLOW_PIN_MEMORY_FALLBACK": "1",
        "DIFFUSERS_DYNAMIC_WEIGHTS_PIN_CPU_WORKERS": "4",
        "DIFFUSERS_DYNAMIC_WEIGHTS_AUTO_BUDGET_POLICY": "off",
        "DIFFUSERS_DYNAMIC_WEIGHTS_RESIDENT_MODULE_BUDGET_GB": "6",
        "DIFFUSERS_DYNAMIC_WEIGHTS_RESIDENT_MODULE_PATTERNS": "auto",
        "DIFFUSERS_DYNAMIC_WEIGHTS_RESIDENT_MODULE_SELECTION": "spread",
        "DIFFUSERS_DYNAMIC_WEIGHTS_SMALL_TENSOR_THRESHOLD_KB": "1024",
    },
    "low_ram": {
        "DIFFUSERS_RUNNER_TRANSFORMER_MEMORY_MANAGER": "off",
        "DIFFUSERS_RUNNER_TRANSFORMER_GROUP_OFFLOAD": "0",
        "DIFFUSERS_RUNNER_ATTENTION_BACKEND": "native",
        "DIFFUSERS_DYNAMIC_WEIGHTS_EXECUTION_MODE": "linear_runtime",
        "DIFFUSERS_DYNAMIC_WEIGHTS_PIN_CPU_MEMORY": "1",
        "DIFFUSERS_DYNAMIC_WEIGHTS_LAZY_PIN_CPU_MEMORY": "0",
        "DIFFUSERS_DYNAMIC_WEIGHTS_ALLOW_PIN_MEMORY_FALLBACK": "1",
        "DIFFUSERS_DYNAMIC_WEIGHTS_PIN_CPU_WORKERS": "2",
        "DIFFUSERS_DYNAMIC_WEIGHTS_AUTO_BUDGET_POLICY": "off",
        "DIFFUSERS_DYNAMIC_WEIGHTS_RESIDENT_MODULE_BUDGET_GB": "3",
        "DIFFUSERS_DYNAMIC_WEIGHTS_RESIDENT_MODULE_PATTERNS": "auto",
        "DIFFUSERS_DYNAMIC_WEIGHTS_RESIDENT_MODULE_SELECTION": "spread",
        "DIFFUSERS_DYNAMIC_WEIGHTS_SMALL_TENSOR_THRESHOLD_KB": "1024",
    },
    "compat": {
        "DIFFUSERS_RUNNER_TRANSFORMER_MEMORY_MANAGER": "off",
        "DIFFUSERS_RUNNER_TRANSFORMER_GROUP_OFFLOAD": "0",
        "DIFFUSERS_RUNNER_ATTENTION_BACKEND": "native",
        "DIFFUSERS_DYNAMIC_WEIGHTS_EXECUTION_MODE": "linear_runtime",
        "DIFFUSERS_DYNAMIC_WEIGHTS_PIN_CPU_MEMORY": "0",
        "DIFFUSERS_DYNAMIC_WEIGHTS_LAZY_PIN_CPU_MEMORY": "0",
        "DIFFUSERS_DYNAMIC_WEIGHTS_ALLOW_PIN_MEMORY_FALLBACK": "1",
        "DIFFUSERS_DYNAMIC_WEIGHTS_AUTO_BUDGET_POLICY": "off",
        "DIFFUSERS_DYNAMIC_WEIGHTS_RESIDENT_MODULE_BUDGET_GB": "3",
        "DIFFUSERS_DYNAMIC_WEIGHTS_RESIDENT_MODULE_PATTERNS": "auto",
        "DIFFUSERS_DYNAMIC_WEIGHTS_RESIDENT_MODULE_SELECTION": "spread",
        "DIFFUSERS_DYNAMIC_WEIGHTS_SMALL_TENSOR_THRESHOLD_KB": "1024",
    },
    "long_steps": {
        "DIFFUSERS_RUNNER_TRANSFORMER_MEMORY_MANAGER": "off",
        "DIFFUSERS_RUNNER_TRANSFORMER_GROUP_OFFLOAD": "0",
        "DIFFUSERS_RUNNER_ATTENTION_BACKEND": "native",
        "DIFFUSERS_DYNAMIC_WEIGHTS_EXECUTION_MODE": "linear_runtime",
        "DIFFUSERS_DYNAMIC_WEIGHTS_PIN_CPU_MEMORY": "1",
        "DIFFUSERS_DYNAMIC_WEIGHTS_LAZY_PIN_CPU_MEMORY": "1",
        "DIFFUSERS_DYNAMIC_WEIGHTS_ALLOW_PIN_MEMORY_FALLBACK": "1",
        "DIFFUSERS_DYNAMIC_WEIGHTS_PIN_CPU_WORKERS": "4",
        "DIFFUSERS_DYNAMIC_WEIGHTS_AUTO_BUDGET_POLICY": "off",
        "DIFFUSERS_DYNAMIC_WEIGHTS_RESIDENT_MODULE_BUDGET_GB": "6",
        "DIFFUSERS_DYNAMIC_WEIGHTS_RESIDENT_MODULE_PATTERNS": "auto",
        "DIFFUSERS_DYNAMIC_WEIGHTS_RESIDENT_MODULE_SELECTION": "spread",
        "DIFFUSERS_DYNAMIC_WEIGHTS_SMALL_TENSOR_THRESHOLD_KB": "1024",
    },
}


DYNAMIC_WEIGHTS_PRESETS = _expand_dynamic_weights_preset_aliases(_DYNAMIC_WEIGHTS_PRESET_VALUES)


def resolve_dynamic_weights_preset(requested_preset: str, *, running_on_wsl: bool | None = None) -> str:
    requested_preset = requested_preset.strip().lower()
    if requested_preset != "auto":
        if requested_preset and requested_preset not in DYNAMIC_WEIGHTS_PRESETS:
            valid_presets = ", ".join(["auto", *sorted(DYNAMIC_WEIGHTS_PRESETS)])
            raise ValueError(
                f"Invalid DIFFUSERS_DYNAMIC_WEIGHTS_PRESET/LTX_IMAGE_DYNAMIC_WEIGHTS_PRESET={requested_preset!r}. "
                f"Valid values: {valid_presets}"
            )
        return requested_preset

    effective_running_on_wsl = is_wsl_environment() if running_on_wsl is None else running_on_wsl
    if effective_running_on_wsl:
        return "wsl_compat"
    if os.name == "nt":
        return "windows_fast"
    return "linux_native_fast"


def dynamic_weights_preset_env_value(
    name: str,
    default: str = "",
    *,
    requested_preset: str = "",
    effective_preset: str | None = None,
    running_on_wsl: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    env = os.environ if environ is None else environ
    for env_name in dynamic_weights_env_names(name):
        if env_name in env:
            return env[env_name]

    preset_name = effective_preset
    if preset_name is None:
        preset_name = resolve_dynamic_weights_preset(requested_preset, running_on_wsl=running_on_wsl)
    if not preset_name:
        return default

    preset_values = DYNAMIC_WEIGHTS_PRESETS[preset_name]
    for env_name in dynamic_weights_env_names(name):
        if env_name in preset_values:
            return preset_values[env_name]
    return preset_values.get(name, default)


def _parse_bool_value(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_pattern_list_value(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in re.split(r"[;,]", value) if item.strip())


@dataclass(frozen=True)
class DynamicWeightsConfig:
    """Configuration for the experimental dynamic weight planner/runtime."""

    execution_device: str | torch.device = "cuda:0"
    offload_device: str | torch.device = "cpu"
    target_module_classes: tuple[type[nn.Module], ...] = _DEFAULT_TARGET_MODULE_CLASSES
    skip_modules_pattern: tuple[str, ...] = ()
    always_resident_modules_pattern: tuple[str, ...] = ()
    small_tensor_threshold_bytes: int = 16 * 1024
    execution_mode: str = "plan"
    pin_cpu_memory: bool = False
    lazy_pin_cpu_memory: bool = False
    allow_pin_memory_fallback: bool = True
    overlap_pin_setup: bool = False
    pin_cpu_workers: int = 1
    cache_plan: bool = True
    cache_pinned_weights: bool = False
    pinned_weight_cache_namespace: str = ""
    cache_resident_device_tensors: bool = False
    resident_device_cache_namespace: str = ""
    auto_budget_policy: str = _AUTO_BUDGET_DISABLED
    max_resident_module_budget_gb: float = 6.0
    max_pin_weight_budget_gb: float = 0.0
    pin_weight_budget_gb: float = 0.0
    pin_weight_budget_ratio: float = 0.0
    pin_weight_selection: str = "spread"
    resident_weight_budget_gb: float = 0.0
    resident_weight_selection: str = "spread"
    resident_module_budget_gb: float = 0.0
    resident_module_patterns: tuple[str, ...] = ()
    resident_module_selection: str = "spread"
    verbose: bool = False
    show_profile: bool = True


@dataclass(frozen=True)
class DynamicWeightsSettings:
    """Environment-derived dynamic weights settings plus the resolved runtime config."""

    requested_preset: str
    effective_preset: str
    enabled: bool
    plan: bool
    config: DynamicWeightsConfig
    requested_pin_cpu_memory: bool
    effective_pin_cpu_memory: bool
    disable_pin_on_wsl: bool
    running_on_wsl: bool

    @property
    def execution_mode(self) -> str:
        return self.config.execution_mode

    def preset_value(self, name: str, default: str = "", environ: Mapping[str, str] | None = None) -> str:
        return dynamic_weights_preset_env_value(
            name,
            default,
            requested_preset=self.requested_preset,
            effective_preset=self.effective_preset,
            running_on_wsl=self.running_on_wsl,
            environ=environ,
        )

    def preset_bool(self, name: str, default: str = "0", environ: Mapping[str, str] | None = None) -> bool:
        return _parse_bool_value(self.preset_value(name, default, environ))

    def as_metrics(self) -> dict[str, Any]:
        config = self.config
        return {
            "dynamic_weights_requested_preset": self.requested_preset or None,
            "dynamic_weights_effective_preset": self.effective_preset or None,
            "dynamic_weights_execution_mode": config.execution_mode,
            "dynamic_weights_pin_cpu_memory": self.requested_pin_cpu_memory,
            "dynamic_weights_effective_pin_cpu_memory": self.effective_pin_cpu_memory,
            "dynamic_weights_lazy_pin_cpu_memory": config.lazy_pin_cpu_memory,
            "dynamic_weights_allow_pin_memory_fallback": config.allow_pin_memory_fallback,
            "dynamic_weights_overlap_pin_setup": config.overlap_pin_setup,
            "dynamic_weights_disable_pin_on_wsl": self.disable_pin_on_wsl,
            "dynamic_weights_pin_cpu_workers": config.pin_cpu_workers,
            "dynamic_weights_cache_plan": config.cache_plan,
            "dynamic_weights_cache_pinned_weights": config.cache_pinned_weights,
            "dynamic_weights_pinned_weight_cache_namespace": config.pinned_weight_cache_namespace or None,
            "dynamic_weights_cache_resident_device_tensors": config.cache_resident_device_tensors,
            "dynamic_weights_resident_device_cache_namespace": config.resident_device_cache_namespace or None,
            "dynamic_weights_auto_budget_policy": config.auto_budget_policy,
            "dynamic_weights_max_resident_module_budget_gb": config.max_resident_module_budget_gb,
            "dynamic_weights_max_pin_weight_budget_gb": config.max_pin_weight_budget_gb,
            "dynamic_weights_pin_weight_budget_gb": config.pin_weight_budget_gb,
            "dynamic_weights_pin_weight_budget_ratio": config.pin_weight_budget_ratio,
            "dynamic_weights_pin_weight_selection": config.pin_weight_selection,
            "dynamic_weights_small_tensor_threshold_kb": config.small_tensor_threshold_bytes // 1024,
            "dynamic_weights_resident_weight_budget_gb": config.resident_weight_budget_gb,
            "dynamic_weights_resident_weight_selection": config.resident_weight_selection,
            "dynamic_weights_resident_module_budget_gb": config.resident_module_budget_gb,
            "dynamic_weights_resident_module_patterns": config.resident_module_patterns,
            "dynamic_weights_resident_module_selection": config.resident_module_selection,
            "dynamic_weights_show_profile": config.show_profile,
        }

    @classmethod
    def from_env(cls, **kwargs: Any) -> "DynamicWeightsSettings":
        return load_dynamic_weights_settings_from_env(**kwargs)


def build_dynamic_weights_event_payload(
    settings: DynamicWeightsSettings,
    state: "DynamicWeightsState",
    config: DynamicWeightsConfig | None = None,
) -> dict[str, Any]:
    summary = state.as_dict()
    config = config or settings.config
    return {
        "module_count": summary["module_count"],
        "total_gb": summary["total_gb"],
        "bytes_by_placement": summary["bytes_by_placement"],
        "execution_mode": config.execution_mode,
        "pin_cpu_memory": settings.effective_pin_cpu_memory,
        "lazy_pin_cpu_memory": config.lazy_pin_cpu_memory,
        "allow_pin_memory_fallback": config.allow_pin_memory_fallback,
        "cache_plan": config.cache_plan,
        "cache_pinned_weights": config.cache_pinned_weights,
        "pinned_weight_cache_namespace": config.pinned_weight_cache_namespace or None,
        "cache_resident_device_tensors": config.cache_resident_device_tensors,
        "resident_device_cache_namespace": config.resident_device_cache_namespace or None,
        "auto_budget_policy": config.auto_budget_policy,
        "max_resident_module_budget_gb": config.max_resident_module_budget_gb,
        "max_pin_weight_budget_gb": config.max_pin_weight_budget_gb,
        "pin_weight_budget_gb": config.pin_weight_budget_gb,
        "pin_weight_budget_ratio": config.pin_weight_budget_ratio,
        "pin_weight_selection": config.pin_weight_selection,
        "patched_module_count": summary["patched_module_count"],
        "resolved_resident_module_patterns": summary["resolved_resident_module_patterns"],
        "planner_decisions": summary["planner_decisions"],
        "setup_runtime": summary["setup_runtime"],
    }


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
class DynamicWeightsLoadResult:
    module: nn.Module
    hook: "DynamicWeightsHook | None" = None
    should_move_to_execution_device: bool = True


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
    selected_resident_modules: list[str] = field(default_factory=list)
    selected_resident_linear_weights: list[str] = field(default_factory=list)
    selected_pinned_linear_weights: list[str] = field(default_factory=list)
    resolved_resident_module_patterns: list[str] = field(default_factory=list)
    planner_decisions: dict[str, Any] = field(default_factory=dict)

    def clone_for_runtime(self) -> "DynamicWeightsState":
        return DynamicWeightsState(
            entries=list(self.entries),
            module_count=self.module_count,
            total_bytes=self.total_bytes,
            bytes_by_placement=dict(self.bytes_by_placement),
            resolved_resident_module_patterns=list(self.resolved_resident_module_patterns),
            planner_decisions=dict(self.planner_decisions),
        )

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
            "selected_resident_modules": self.selected_resident_modules,
            "selected_resident_linear_weights": self.selected_resident_linear_weights,
            "selected_pinned_linear_weights": self.selected_pinned_linear_weights,
            "resolved_resident_module_patterns": self.resolved_resident_module_patterns,
            "planner_decisions": self.planner_decisions,
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
        self._lazy_pinned_tensor_ids: set[int] = set()
        self._pin_memory_disabled = False
        self._pin_lock = threading.Lock()
        self._resident_module_names: set[str] = set()
        self._pinned_weight_cache_namespace = ""
        self._resident_device_cache_namespace = ""
        self.execution_device = torch.device(config.execution_device)
        self.offload_device = torch.device(config.offload_device)
        self.execution_mode = config.execution_mode.lower()
        self.pin_cpu_workers = max(1, int(config.pin_cpu_workers))
        self.auto_budget_policy = config.auto_budget_policy.lower()
        if self.auto_budget_policy not in {_AUTO_BUDGET_DISABLED, _AUTO_BUDGET_BALANCED}:
            raise ValueError("DynamicWeightsConfig.auto_budget_policy must be 'off' or 'balanced'")
        self.max_resident_module_budget_bytes = int(max(0.0, float(config.max_resident_module_budget_gb)) * 1024**3)
        self.max_pin_weight_budget_bytes = int(max(0.0, float(config.max_pin_weight_budget_gb)) * 1024**3)
        self.pin_weight_budget_bytes = int(max(0.0, float(config.pin_weight_budget_gb)) * 1024**3)
        self.pin_weight_selection = config.pin_weight_selection.lower()
        if self.pin_weight_selection not in {"first", "spread", "largest"}:
            raise ValueError("DynamicWeightsConfig.pin_weight_selection must be 'first', 'spread', or 'largest'")
        self.pin_weight_budget_ratio = max(0.0, min(1.0, float(config.pin_weight_budget_ratio)))
        self.resident_weight_budget_bytes = int(max(0.0, float(config.resident_weight_budget_gb)) * 1024**3)
        self.resident_module_budget_bytes = int(max(0.0, float(config.resident_module_budget_gb)) * 1024**3)
        self.resident_weight_selection = config.resident_weight_selection.lower()
        if self.resident_weight_selection not in {"first", "spread", "largest"}:
            raise ValueError("DynamicWeightsConfig.resident_weight_selection must be 'first', 'spread', or 'largest'")
        self.resident_module_selection = config.resident_module_selection.lower()
        if self.resident_module_selection not in {"first", "spread", "largest"}:
            raise ValueError("DynamicWeightsConfig.resident_module_selection must be 'first', 'spread', or 'largest'")

    def initialize_hook(self, module: nn.Module) -> nn.Module:
        self._pinned_weight_cache_namespace = (
            self.config.pinned_weight_cache_namespace.strip()
            or f"{module.__class__.__module__}.{module.__class__.__qualname__}"
        )
        self._resident_device_cache_namespace = (
            self.config.resident_device_cache_namespace.strip()
            or self._pinned_weight_cache_namespace
        )
        self.state = build_dynamic_weight_plan(module, self.config)
        if self.execution_mode not in {"plan", "linear_runtime"}:
            raise ValueError("DynamicWeightsConfig.execution_mode must be 'plan' or 'linear_runtime'")
        if self.execution_mode == "linear_runtime":
            self._prepare_linear_runtime(module)
        if self.config.verbose:
            summary = self.state.as_dict()
            print(
                "  [dynamic-weights] "
                f"mode={self.execution_mode} modules={summary['module_count']} "
                f"patched={summary['patched_module_count']} total_gb={summary['total_gb']:.4f} "
                f"placements={summary['bytes_by_placement']}",
                flush=True,
            )
            if self.config.pin_cpu_memory and self.pin_weight_budget_bytes > 0:
                print(
                    "  [dynamic-weights] warning: partial pinned-memory budgets can be slower than full pinning; "
                    "clear DIFFUSERS_DYNAMIC_WEIGHTS_PIN_WEIGHT_BUDGET_GB/LTX_IMAGE_DYNAMIC_WEIGHTS_PIN_WEIGHT_BUDGET_GB "
                    "when benchmarking the fast preset.",
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
        self._lazy_pinned_tensor_ids.clear()
        self._pin_memory_disabled = False
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

    def _prepare_linear_runtime(self, module: nn.Module) -> None:
        skip_patterns = tuple(re.compile(pattern) for pattern in self.config.skip_modules_pattern)
        resident_patterns = tuple(re.compile(pattern) for pattern in self.config.always_resident_modules_pattern)
        if self.state.resolved_resident_module_patterns:
            resident_module_patterns = tuple(
                re.compile(pattern) for pattern in self.state.resolved_resident_module_patterns
            )
        else:
            resident_module_patterns = _resolve_resident_module_patterns(module, self.config)
            self.state.resolved_resident_module_patterns = [pattern.pattern for pattern in resident_module_patterns]
        modules_to_pin: list[tuple[str, nn.Module, int]] = []
        eager_pin_cpu_memory = self.config.pin_cpu_memory and not self.config.lazy_pin_cpu_memory
        pin_work = None

        self._apply_auto_budget_policy(module, skip_patterns, resident_patterns, resident_module_patterns)

        self._move_root_local_tensors_to_device(module)
        if self.resident_module_budget_bytes > 0 and resident_module_patterns:
            start = time.perf_counter()
            selected_bytes = self._select_resident_modules(module, skip_patterns, resident_module_patterns)
            self.state.add_setup("select_resident_modules", time.perf_counter() - start, selected_bytes)

        if eager_pin_cpu_memory and self.config.overlap_pin_setup and self.pin_cpu_workers > 1:
            candidates = self._collect_linear_modules_to_pin(module, skip_patterns, resident_patterns)
            selected_linears_to_pin = self._select_linear_weights_to_pin(candidates)
            if selected_linears_to_pin:
                pin_work = self._start_pin_linear_weights(selected_linears_to_pin)

        for module_name, submodule in module.named_modules():
            if module_name == "":
                continue
            if skip_patterns and any(pattern.search(module_name) for pattern in skip_patterns):
                continue
            if self._is_descendant_of_resident_module(module_name):
                continue
            if module_name in self._resident_module_names:
                start = time.perf_counter()
                before = self._move_resident_module_to_execution_device(module_name, submodule)
                self.state.add_setup("resident_budget_modules_to_device", time.perf_counter() - start, before)
                continue

            is_resident_module = bool(
                resident_patterns and any(pattern.search(module_name) for pattern in resident_patterns)
            )
            if is_resident_module:
                start = time.perf_counter()
                before = self._move_resident_module_to_execution_device(module_name, submodule)
                self.state.add_setup("resident_modules_to_device", time.perf_counter() - start, before)
                continue

            if isinstance(submodule, nn.Linear):
                self._move_linear_to_runtime_devices(
                    module_name,
                    submodule,
                    modules_to_pin if pin_work is None else None,
                )
                self._patch_linear(submodule)
            elif isinstance(submodule, nn.Embedding):
                self._move_embedding_to_runtime_devices(
                    module_name,
                    submodule,
                    modules_to_pin if pin_work is None else None,
                )
                self._patch_embedding(submodule)
            else:
                self._move_small_local_tensors_to_device(submodule)

        if pin_work is not None:
            start, finish_pin_work = pin_work
            pinned_bytes = finish_pin_work()
            self.state.add_setup("pin_linear_weights_overlapped", time.perf_counter() - start, pinned_bytes)
        elif eager_pin_cpu_memory and modules_to_pin:
            start = time.perf_counter()
            selected_linears_to_pin = self._select_linear_weights_to_pin(modules_to_pin)
            pinned_bytes = self._pin_linear_weights(selected_linears_to_pin)
            self.state.add_setup("pin_linear_weights", time.perf_counter() - start, pinned_bytes)

    def _collect_linear_modules_to_pin(
        self,
        module: nn.Module,
        skip_patterns: tuple[re.Pattern[str], ...],
        resident_patterns: tuple[re.Pattern[str], ...],
    ) -> list[tuple[str, nn.Module, int]]:
        candidates: list[tuple[str, nn.Module, int]] = []
        for module_name, submodule in module.named_modules():
            if module_name == "" or not isinstance(submodule, (nn.Linear, nn.Embedding)):
                continue
            if skip_patterns and any(pattern.search(module_name) for pattern in skip_patterns):
                continue
            if self._is_descendant_of_resident_module(module_name):
                continue
            if module_name in self._resident_module_names:
                continue
            if resident_patterns and any(pattern.search(module_name) for pattern in resident_patterns):
                continue
            if submodule.weight.device.type == "cpu" and not submodule.weight.data.is_pinned():
                candidates.append((module_name, submodule, _tensor_size_bytes(submodule.weight.data)))
        return candidates

    def _move_linear_to_runtime_devices(
        self,
        module_name: str,
        linear: nn.Linear,
        modules_to_pin: list[tuple[str, nn.Module, int]] | None,
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
        if modules_to_pin is not None and linear.weight.device.type == "cpu" and not linear.weight.data.is_pinned():
            modules_to_pin.append((module_name, linear, _tensor_size_bytes(linear.weight.data)))

    def _move_embedding_to_runtime_devices(
        self,
        module_name: str,
        embedding: nn.Embedding,
        modules_to_pin: list[tuple[str, nn.Module, int]] | None,
    ) -> None:
        if embedding.weight.device != self.offload_device:
            start = time.perf_counter()
            weight_bytes = _tensor_size_bytes(embedding.weight.data)
            embedding.weight.data = embedding.weight.data.to(self.offload_device)
            self.state.add_setup("embedding_weights_to_offload", time.perf_counter() - start, weight_bytes)
        if modules_to_pin is not None and embedding.weight.device.type == "cpu" and not embedding.weight.data.is_pinned():
            modules_to_pin.append((module_name, embedding, _tensor_size_bytes(embedding.weight.data)))

    def _select_linear_weights_to_pin(self, candidates: list[tuple[str, nn.Module, int]]) -> list[tuple[str, nn.Module]]:
        candidate_bytes = sum(weight_bytes for _, _, weight_bytes in candidates)
        pin_weight_budget_bytes = self.pin_weight_budget_bytes
        if pin_weight_budget_bytes <= 0 and self.pin_weight_budget_ratio > 0:
            pin_weight_budget_bytes = int(candidate_bytes * self.pin_weight_budget_ratio)
        if pin_weight_budget_bytes > 0 and self.max_pin_weight_budget_bytes > 0:
            pin_weight_budget_bytes = min(pin_weight_budget_bytes, self.max_pin_weight_budget_bytes)
        if candidate_bytes > 0 and pin_weight_budget_bytes >= int(candidate_bytes * 0.95):
            pin_weight_budget_bytes = 0
            self.state.planner_decisions["pin_weight_budget_snap_to_full"] = True
        self.state.planner_decisions["resolved_pin_weight_budget_gb"] = round(pin_weight_budget_bytes / 1024**3, 4)
        if pin_weight_budget_bytes <= 0:
            self.state.selected_pinned_linear_weights = [module_name for module_name, _, _ in candidates]
            return [(module_name, linear) for module_name, linear, _ in candidates]

        ordered_candidates = candidates
        if self.pin_weight_selection == "spread":
            ordered_candidates = _spread_order(candidates, pin_weight_budget_bytes)
        elif self.pin_weight_selection == "largest":
            ordered_candidates = _largest_first_order(candidates)

        selected_linears: list[tuple[str, nn.Module]] = []
        selected_bytes = 0
        selected_names: list[str] = []
        for module_name, linear, weight_bytes in ordered_candidates:
            if selected_bytes + weight_bytes > pin_weight_budget_bytes:
                continue
            selected_linears.append((module_name, linear))
            selected_names.append(module_name)
            selected_bytes += weight_bytes

        self.state.selected_pinned_linear_weights = selected_names
        return selected_linears

    def _apply_auto_budget_policy(
        self,
        module: nn.Module,
        skip_patterns: tuple[re.Pattern[str], ...],
        resident_patterns: tuple[re.Pattern[str], ...],
        resident_module_patterns: tuple[re.Pattern[str], ...],
    ) -> None:
        if self.auto_budget_policy == _AUTO_BUDGET_DISABLED:
            return

        candidate_weight_bytes = sum(
            weight_bytes
            for _, _, weight_bytes in self._collect_linear_modules_to_pin(module, skip_patterns, resident_patterns)
        )
        candidate_module_bytes = 0
        if resident_module_patterns:
            candidate_module_bytes = sum(
                module_bytes
                for module_name, submodule in module.named_modules()
                if module_name
                and not (skip_patterns and any(pattern.search(module_name) for pattern in skip_patterns))
                and any(pattern.search(module_name) for pattern in resident_module_patterns)
                for module_bytes in (_module_size_bytes(submodule),)
            )

        self.state.planner_decisions.update(
            {
                "auto_budget_policy": self.auto_budget_policy,
                "candidate_weight_gb": round(candidate_weight_bytes / 1024**3, 4),
                "candidate_resident_module_gb": round(candidate_module_bytes / 1024**3, 4),
            }
        )

        if self.resident_module_budget_bytes <= 0 and candidate_module_bytes > 0:
            budget = int(candidate_module_bytes * 0.25)
            if self.max_resident_module_budget_bytes > 0:
                budget = min(budget, self.max_resident_module_budget_bytes)
            self.resident_module_budget_bytes = budget
            self.state.planner_decisions["auto_resident_module_budget_gb"] = round(budget / 1024**3, 4)

        if self.pin_weight_budget_bytes <= 0 and self.pin_weight_budget_ratio <= 0 and candidate_weight_bytes > 0:
            if candidate_weight_bytes <= 8 * 1024**3:
                ratio = 1.0
            elif candidate_weight_bytes <= 16 * 1024**3:
                ratio = 0.85
            else:
                ratio = 0.75
            budget = int(candidate_weight_bytes * ratio)
            if self.max_pin_weight_budget_bytes > 0:
                budget = min(budget, self.max_pin_weight_budget_bytes)
            self.pin_weight_budget_bytes = budget
            self.state.planner_decisions["auto_pin_weight_budget_gb"] = round(budget / 1024**3, 4)
            self.state.planner_decisions["auto_pin_weight_budget_ratio"] = ratio

    def _select_resident_modules(
        self,
        module: nn.Module,
        skip_patterns: tuple[re.Pattern[str], ...],
        module_patterns: tuple[re.Pattern[str], ...],
    ) -> int:
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
        elif self.resident_module_selection == "largest":
            ordered_candidates = _largest_first_order(candidates)

        selected_bytes = 0
        for module_name, _, module_bytes in ordered_candidates:
            if selected_bytes + module_bytes > self.resident_module_budget_bytes:
                continue
            self._resident_module_names.add(module_name)
            self.state.selected_resident_modules.append(module_name)
            selected_bytes += module_bytes
        return selected_bytes

    def _is_descendant_of_resident_module(self, module_name: str) -> bool:
        return any(_is_module_descendant(module_name, resident_name) for resident_name in self._resident_module_names)

    def _move_resident_module_to_execution_device(self, module_name: str, module: nn.Module) -> int:
        if not self.config.cache_resident_device_tensors:
            module_bytes = _module_size_bytes(module)
            module.to(self.execution_device)
            return module_bytes

        moved_bytes = 0
        for tensor_name, parameter in module.named_parameters(recurse=True):
            tensor_bytes = _tensor_size_bytes(parameter.data)
            moved_bytes += tensor_bytes
            cached = self._get_cached_resident_device_tensor(module_name, tensor_name, parameter.data)
            if cached is not None:
                parameter.data = cached
                self.state.add_setup("resident_device_cache_hit", 0.0, tensor_bytes)
                continue
            if parameter.device != self.execution_device:
                parameter.data = parameter.data.to(self.execution_device)
            parameter.data = self._store_cached_resident_device_tensor(module_name, tensor_name, parameter.data)

        for tensor_name, buffer in module.named_buffers(recurse=True):
            tensor_bytes = _tensor_size_bytes(buffer.data)
            moved_bytes += tensor_bytes
            cached = self._get_cached_resident_device_tensor(module_name, tensor_name, buffer.data)
            if cached is not None:
                buffer.data = cached
                self.state.add_setup("resident_device_cache_hit", 0.0, tensor_bytes)
                continue
            if buffer.device != self.execution_device:
                buffer.data = buffer.data.to(self.execution_device)
            buffer.data = self._store_cached_resident_device_tensor(module_name, tensor_name, buffer.data)
        return moved_bytes

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

    def _pin_linear_weights(self, linears: list[tuple[str, nn.Module]]) -> int:
        pinned_bytes = 0

        def pin_linear(module_name: str, linear: nn.Linear):
            cached = self._get_cached_pinned_weight(module_name, linear.weight.data)
            if cached is not None:
                return linear, cached, _tensor_size_bytes(cached), True
            pinned = linear.weight.data.pin_memory()
            pinned = self._store_cached_pinned_weight(module_name, pinned)
            return linear, pinned, _tensor_size_bytes(pinned), False

        def assign_pinned(result) -> None:
            nonlocal pinned_bytes
            linear, pinned, tensor_bytes, cache_hit = result
            linear.weight.data = pinned
            pinned_bytes += tensor_bytes
            if cache_hit:
                self.state.add_setup("pinned_weight_cache_hit", 0.0, tensor_bytes)

        if self.pin_cpu_workers == 1:
            for module_name, linear in linears:
                try:
                    assign_pinned(pin_linear(module_name, linear))
                except _PIN_MEMORY_ERRORS as exc:
                    if not self.config.allow_pin_memory_fallback:
                        raise
                    self._disable_pin_memory("pin_linear_weights_failed", exc)
                    break
            return pinned_bytes

        linears_iter = iter(linears)
        with ThreadPoolExecutor(max_workers=self.pin_cpu_workers) as executor:
            futures = set()

            def submit_next() -> bool:
                try:
                    module_name, linear = next(linears_iter)
                except StopIteration:
                    return False
                futures.add(executor.submit(pin_linear, module_name, linear))
                return True

            for _ in range(self.pin_cpu_workers):
                if not submit_next():
                    break

            while futures:
                for future in as_completed(futures):
                    futures.remove(future)
                    try:
                        assign_pinned(future.result())
                    except _PIN_MEMORY_ERRORS as exc:
                        if not self.config.allow_pin_memory_fallback:
                            raise
                        self._disable_pin_memory("pin_linear_weights_failed", exc)
                        for pending in futures:
                            pending.cancel()
                        return pinned_bytes
                    if not self._pin_memory_disabled:
                        submit_next()
                    break
        return pinned_bytes

    def _start_pin_linear_weights(self, linears: list[tuple[str, nn.Module]]):
        start = time.perf_counter()
        executor = ThreadPoolExecutor(max_workers=self.pin_cpu_workers)
        linears_iter = iter(linears)
        futures = set()
        pinned_bytes = 0

        def pin_and_assign(module_name: str, linear: nn.Linear) -> tuple[int, bool]:
            cached = self._get_cached_pinned_weight(module_name, linear.weight.data)
            if cached is not None:
                with self._pin_lock:
                    linear.weight.data = cached
                return _tensor_size_bytes(cached), True
            pinned = linear.weight.data.pin_memory()
            pinned = self._store_cached_pinned_weight(module_name, pinned)
            tensor_bytes = _tensor_size_bytes(pinned)
            with self._pin_lock:
                linear.weight.data = pinned
            return tensor_bytes, False

        def submit_next() -> bool:
            try:
                module_name, linear = next(linears_iter)
            except StopIteration:
                return False
            futures.add(executor.submit(pin_and_assign, module_name, linear))
            return True

        for _ in range(self.pin_cpu_workers):
            if not submit_next():
                break

        def finish() -> int:
            nonlocal pinned_bytes
            try:
                while futures:
                    for future in as_completed(futures):
                        futures.remove(future)
                        try:
                            tensor_bytes, cache_hit = future.result()
                            pinned_bytes += tensor_bytes
                            if cache_hit:
                                self.state.add_setup("pinned_weight_cache_hit", 0.0, tensor_bytes)
                        except _PIN_MEMORY_ERRORS as exc:
                            if not self.config.allow_pin_memory_fallback:
                                raise
                            self._disable_pin_memory("pin_linear_weights_failed", exc)
                            for pending in futures:
                                pending.cancel()
                            return pinned_bytes
                        if not self._pin_memory_disabled:
                            submit_next()
                        break
                return pinned_bytes
            finally:
                executor.shutdown(wait=True, cancel_futures=True)

        return start, finish

    def _get_cached_pinned_weight(self, module_name: str, tensor: torch.Tensor) -> torch.Tensor | None:
        if not self.config.cache_pinned_weights:
            return None
        cache_key = self._pinned_weight_cache_key(module_name, tensor)
        with _DYNAMIC_WEIGHTS_PINNED_TENSOR_CACHE_LOCK:
            cached = _DYNAMIC_WEIGHTS_PINNED_TENSOR_CACHE.get(cache_key)
        if cached is None:
            return None
        if cached.shape != tensor.shape or cached.dtype != tensor.dtype:
            return None
        return cached

    def _store_cached_pinned_weight(self, module_name: str, tensor: torch.Tensor) -> torch.Tensor:
        if not self.config.cache_pinned_weights or not tensor.is_pinned():
            return tensor
        cache_key = self._pinned_weight_cache_key(module_name, tensor)
        with _DYNAMIC_WEIGHTS_PINNED_TENSOR_CACHE_LOCK:
            cached = _DYNAMIC_WEIGHTS_PINNED_TENSOR_CACHE.setdefault(cache_key, tensor)
        return cached

    def _pinned_weight_cache_key(self, module_name: str, tensor: torch.Tensor) -> tuple[Any, ...]:
        return (
            "v1",
            self._pinned_weight_cache_namespace,
            module_name,
            tuple(tensor.shape),
            str(tensor.dtype),
            tensor.numel(),
            tensor.element_size(),
        )

    def _get_cached_resident_device_tensor(
        self,
        module_name: str,
        tensor_name: str,
        tensor: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self.config.cache_resident_device_tensors:
            return None
        cache_key = self._resident_device_cache_key(module_name, tensor_name, tensor)
        with _DYNAMIC_WEIGHTS_RESIDENT_DEVICE_TENSOR_CACHE_LOCK:
            cached = _DYNAMIC_WEIGHTS_RESIDENT_DEVICE_TENSOR_CACHE.get(cache_key)
        if cached is None:
            return None
        if cached.device != self.execution_device or cached.shape != tensor.shape or cached.dtype != tensor.dtype:
            return None
        return cached

    def _store_cached_resident_device_tensor(
        self,
        module_name: str,
        tensor_name: str,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        if not self.config.cache_resident_device_tensors or tensor.device != self.execution_device:
            return tensor
        cache_key = self._resident_device_cache_key(module_name, tensor_name, tensor)
        with _DYNAMIC_WEIGHTS_RESIDENT_DEVICE_TENSOR_CACHE_LOCK:
            cached = _DYNAMIC_WEIGHTS_RESIDENT_DEVICE_TENSOR_CACHE.setdefault(cache_key, tensor)
        return cached

    def _resident_device_cache_key(self, module_name: str, tensor_name: str, tensor: torch.Tensor) -> tuple[Any, ...]:
        return (
            "v1",
            self._resident_device_cache_namespace,
            module_name,
            tensor_name,
            tuple(tensor.shape),
            str(tensor.dtype),
            tensor.numel(),
            tensor.element_size(),
        )

    def _disable_pin_memory(self, action: str, exc: BaseException) -> None:
        if self._pin_memory_disabled:
            return
        self._pin_memory_disabled = True
        self.state.add_setup(action, 0.0, 0)
        if self.config.verbose:
            print(
                f"  [dynamic-weights] {action}: disabling pinned CPU memory fallback after {type(exc).__name__}: {exc}",
                flush=True,
            )

    def _patch_linear(self, linear: nn.Linear) -> None:
        self._patched_modules.append((linear, linear.forward))
        self.state.patched_module_count += 1

        def dynamic_linear_forward(patched_linear, input):
            weight = self._to_input_device(patched_linear.weight, input, "linear_weight")
            bias = self._to_input_device(patched_linear.bias, input, "linear_bias")
            return F.linear(input, weight, bias)

        linear.forward = dynamic_linear_forward.__get__(linear, linear.__class__)

    def _patch_embedding(self, embedding: nn.Embedding) -> None:
        self._patched_modules.append((embedding, embedding.forward))
        self.state.patched_module_count += 1

        def dynamic_embedding_forward(patched_embedding, input):
            weight = self._to_input_device(patched_embedding.weight, input, "embedding_weight", cast_to_input_dtype=False)
            return F.embedding(
                input,
                weight,
                patched_embedding.padding_idx,
                patched_embedding.max_norm,
                patched_embedding.norm_type,
                patched_embedding.scale_grad_by_freq,
                patched_embedding.sparse,
            )

        embedding.forward = dynamic_embedding_forward.__get__(embedding, embedding.__class__)

    def _to_input_device(
        self,
        tensor: torch.Tensor | None,
        input: torch.Tensor,
        name: str,
        *,
        cast_to_input_dtype: bool = True,
    ) -> torch.Tensor | None:
        if tensor is None:
            return None
        if tensor.device == input.device and (not cast_to_input_dtype or tensor.dtype == input.dtype):
            return tensor
        tensor = self._lazy_pin_tensor(tensor, name)
        start = time.perf_counter()
        dtype = input.dtype if cast_to_input_dtype else tensor.dtype
        moved = tensor.to(device=input.device, dtype=dtype, non_blocking=True)
        self.state.add_copy(name, time.perf_counter() - start, _tensor_size_bytes(moved))
        return moved

    def _lazy_pin_tensor(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        if self._pin_memory_disabled or not self.config.pin_cpu_memory or not self.config.lazy_pin_cpu_memory:
            return tensor
        if tensor.device.type == "meta":
            return tensor
        if tensor.device.type != "cpu" or tensor.is_pinned():
            return tensor

        start = time.perf_counter()
        try:
            pinned = tensor.pin_memory()
        except _PIN_MEMORY_ERRORS as exc:
            if not self.config.allow_pin_memory_fallback:
                raise
            self._disable_pin_memory(f"lazy_pin_{name}_failed", exc)
            return tensor
        seconds = time.perf_counter() - start
        pinned_bytes = _tensor_size_bytes(pinned)

        if isinstance(tensor, nn.Parameter):
            tensor.data = pinned
            pinned_tensor = tensor
            tensor_id = id(tensor)
        else:
            pinned_tensor = pinned
            tensor_id = id(pinned)

        if tensor_id not in self._lazy_pinned_tensor_ids:
            self._lazy_pinned_tensor_ids.add(tensor_id)
            self.state.add_setup(f"lazy_pin_{name}", seconds, pinned_bytes)
        return pinned_tensor

    def print_profile_summary(self, *, top_n: int = 8) -> None:
        summary = self.state.as_dict()
        setup_runtime = summary["setup_runtime"]
        setup_seconds = sum(item["seconds"] for item in setup_runtime.values())
        setup_gb = sum(item["gb"] for item in setup_runtime.values())
        copy_runtime = summary["copy_runtime"]
        copy_seconds = sum(item["seconds"] for item in copy_runtime.values())
        copy_gb = sum(item["gb"] for item in copy_runtime.values())
        print(
            "  [dynamic-weights-profile] "
            f"summary: mode={self.execution_mode} modules={summary['module_count']} "
            f"patched={summary['patched_module_count']} setup_seconds={setup_seconds:.4f} "
            f"setup_gb={setup_gb:.4f} copy_seconds={copy_seconds:.4f} "
            f"copy_gb={copy_gb:.4f}",
            flush=True,
        )

        selected_modules = summary["selected_resident_modules"]
        resolved_patterns = summary["resolved_resident_module_patterns"]
        if resolved_patterns:
            print(
                "  [dynamic-weights-profile] "
                f"resolved_resident_module_patterns={resolved_patterns}",
                flush=True,
            )

        planner_decisions = summary["planner_decisions"]
        if planner_decisions:
            print(
                "  [dynamic-weights-profile] "
                f"planner_decisions={planner_decisions}",
                flush=True,
            )

        if selected_modules:
            print(
                "  [dynamic-weights-profile] "
                f"resident_modules={selected_modules}",
                flush=True,
            )

        selected_linear_weights = summary["selected_resident_linear_weights"]
        if selected_linear_weights:
            print(
                "  [dynamic-weights-profile] "
                f"resident_linear_weights={selected_linear_weights[:top_n]} "
                f"count={len(selected_linear_weights)}",
                flush=True,
            )

        selected_pinned_linear_weights = summary["selected_pinned_linear_weights"]
        if selected_pinned_linear_weights:
            print(
                "  [dynamic-weights-profile] "
                f"pinned_linear_weights={selected_pinned_linear_weights[:top_n]} "
                f"count={len(selected_pinned_linear_weights)}",
                flush=True,
            )

        if setup_runtime:
            print("  [dynamic-weights-profile] setup_runtime_by_type:", flush=True)
            for key, item in sorted(setup_runtime.items(), key=lambda pair: pair[1]["seconds"], reverse=True)[:top_n]:
                print(
                    f"    {key}: seconds={item['seconds']:.4f} gb={item['gb']:.4f}",
                    flush=True,
                )

        if copy_runtime:
            print("  [dynamic-weights-profile] copy_runtime_by_type:", flush=True)
            for name, item in sorted(copy_runtime.items(), key=lambda pair: pair[1]["seconds"], reverse=True)[:top_n]:
                print(
                    f"    {name}: calls={item['calls']} seconds={item['seconds']:.4f} gb={item['gb']:.4f}",
                    flush=True,
                )


def load_dynamic_weights_settings_from_env(
    *,
    execution_device: str | torch.device = "cuda:0",
    offload_device: str | torch.device = "cpu",
    target_module_classes: tuple[type[nn.Module], ...] = _DEFAULT_TARGET_MODULE_CLASSES,
    skip_modules_pattern: tuple[str, ...] = (),
    always_resident_modules_pattern: tuple[str, ...] | None = None,
    running_on_wsl: bool | None = None,
    environ: Mapping[str, str] | None = None,
    default_preset: str = "",
) -> DynamicWeightsSettings:
    env = os.environ if environ is None else environ
    is_wsl = is_wsl_environment() if running_on_wsl is None else running_on_wsl
    requested_preset = dynamic_weights_env_value("DIFFUSERS_DYNAMIC_WEIGHTS_PRESET", default_preset, env).strip().lower()
    effective_preset = resolve_dynamic_weights_preset(requested_preset, running_on_wsl=is_wsl)

    def preset_env(name: str, default: str = "") -> str:
        return dynamic_weights_preset_env_value(
            name,
            default,
            requested_preset=requested_preset,
            effective_preset=effective_preset,
            running_on_wsl=is_wsl,
            environ=env,
        )

    def preset_bool(name: str, default: str = "0") -> bool:
        return _parse_bool_value(preset_env(name, default))

    default_always_resident_patterns = _DEFAULT_ALWAYS_RESIDENT_MODULE_PATTERNS
    if always_resident_modules_pattern is not None:
        default_always_resident_patterns = always_resident_modules_pattern

    plan = preset_bool("DIFFUSERS_DYNAMIC_WEIGHTS_PLAN")
    execution_mode = preset_env("DIFFUSERS_DYNAMIC_WEIGHTS_EXECUTION_MODE", "plan").lower()
    requested_pin_cpu_memory = preset_bool("DIFFUSERS_DYNAMIC_WEIGHTS_PIN_CPU_MEMORY")
    lazy_pin_cpu_memory = preset_bool("DIFFUSERS_DYNAMIC_WEIGHTS_LAZY_PIN_CPU_MEMORY")
    allow_pin_memory_fallback = preset_bool("DIFFUSERS_DYNAMIC_WEIGHTS_ALLOW_PIN_MEMORY_FALLBACK", "1")
    overlap_pin_setup = preset_bool("DIFFUSERS_DYNAMIC_WEIGHTS_OVERLAP_PIN_SETUP", "0")
    disable_pin_on_wsl = preset_bool("DIFFUSERS_DYNAMIC_WEIGHTS_DISABLE_PIN_ON_WSL", "1")
    effective_pin_cpu_memory = requested_pin_cpu_memory and not (is_wsl and disable_pin_on_wsl)

    config = DynamicWeightsConfig(
        execution_device=execution_device,
        offload_device=offload_device,
        target_module_classes=target_module_classes,
        skip_modules_pattern=skip_modules_pattern,
        always_resident_modules_pattern=_parse_pattern_list_value(
            preset_env(
                "DIFFUSERS_DYNAMIC_WEIGHTS_ALWAYS_RESIDENT_MODULE_PATTERNS",
                ",".join(default_always_resident_patterns),
            )
        ),
        small_tensor_threshold_bytes=int(preset_env("DIFFUSERS_DYNAMIC_WEIGHTS_SMALL_TENSOR_THRESHOLD_KB", "1024")) * 1024,
        execution_mode=execution_mode,
        pin_cpu_memory=effective_pin_cpu_memory,
        lazy_pin_cpu_memory=lazy_pin_cpu_memory,
        allow_pin_memory_fallback=allow_pin_memory_fallback,
        overlap_pin_setup=overlap_pin_setup,
        pin_cpu_workers=int(preset_env("DIFFUSERS_DYNAMIC_WEIGHTS_PIN_CPU_WORKERS", "4")),
        cache_plan=preset_bool("DIFFUSERS_DYNAMIC_WEIGHTS_CACHE_PLAN", "1"),
        cache_pinned_weights=preset_bool("DIFFUSERS_DYNAMIC_WEIGHTS_CACHE_PINNED_WEIGHTS", "0"),
        pinned_weight_cache_namespace=preset_env("DIFFUSERS_DYNAMIC_WEIGHTS_PINNED_WEIGHT_CACHE_NAMESPACE", ""),
        cache_resident_device_tensors=preset_bool("DIFFUSERS_DYNAMIC_WEIGHTS_CACHE_RESIDENT_DEVICE_TENSORS", "0"),
        resident_device_cache_namespace=preset_env("DIFFUSERS_DYNAMIC_WEIGHTS_RESIDENT_DEVICE_CACHE_NAMESPACE", ""),
        auto_budget_policy=preset_env("DIFFUSERS_DYNAMIC_WEIGHTS_AUTO_BUDGET_POLICY", "off").lower(),
        max_resident_module_budget_gb=float(
            preset_env("DIFFUSERS_DYNAMIC_WEIGHTS_MAX_RESIDENT_MODULE_BUDGET_GB", "6.0")
        ),
        max_pin_weight_budget_gb=float(preset_env("DIFFUSERS_DYNAMIC_WEIGHTS_MAX_PIN_WEIGHT_BUDGET_GB", "0.0")),
        pin_weight_budget_gb=float(preset_env("DIFFUSERS_DYNAMIC_WEIGHTS_PIN_WEIGHT_BUDGET_GB", "0.0")),
        pin_weight_budget_ratio=float(preset_env("DIFFUSERS_DYNAMIC_WEIGHTS_PIN_WEIGHT_BUDGET_RATIO", "0.0")),
        pin_weight_selection=preset_env("DIFFUSERS_DYNAMIC_WEIGHTS_PIN_WEIGHT_SELECTION", "spread").lower(),
        resident_weight_budget_gb=float(preset_env("DIFFUSERS_DYNAMIC_WEIGHTS_RESIDENT_WEIGHT_BUDGET_GB", "0.0")),
        resident_weight_selection=preset_env("DIFFUSERS_DYNAMIC_WEIGHTS_RESIDENT_WEIGHT_SELECTION", "spread").lower(),
        resident_module_budget_gb=float(preset_env("DIFFUSERS_DYNAMIC_WEIGHTS_RESIDENT_MODULE_BUDGET_GB", "0.0")),
        resident_module_patterns=_parse_pattern_list_value(preset_env("DIFFUSERS_DYNAMIC_WEIGHTS_RESIDENT_MODULE_PATTERNS", "")),
        resident_module_selection=preset_env("DIFFUSERS_DYNAMIC_WEIGHTS_RESIDENT_MODULE_SELECTION", "spread").lower(),
        verbose=preset_bool("DIFFUSERS_DYNAMIC_WEIGHTS_VERBOSE", "0"),
        show_profile=preset_bool("DIFFUSERS_DYNAMIC_WEIGHTS_SHOW_PROFILE", "0"),
    )
    return DynamicWeightsSettings(
        requested_preset=requested_preset,
        effective_preset=effective_preset,
        enabled=plan or execution_mode != "plan",
        plan=plan,
        config=config,
        requested_pin_cpu_memory=requested_pin_cpu_memory,
        effective_pin_cpu_memory=effective_pin_cpu_memory,
        disable_pin_on_wsl=disable_pin_on_wsl,
        running_on_wsl=is_wsl,
    )


def apply_dynamic_weights(module: nn.Module, config: DynamicWeightsConfig | None = None) -> DynamicWeightsHook:
    config = config or DynamicWeightsConfig()
    registry = HookRegistry.check_if_exists_or_initialize(module)
    existing_hook = registry.get_hook(_DYNAMIC_WEIGHTS_HOOK)
    if existing_hook is not None:
        registry.remove_hook(_DYNAMIC_WEIGHTS_HOOK)
    hook = DynamicWeightsHook(config)
    registry.register_hook(hook, _DYNAMIC_WEIGHTS_HOOK)
    return hook


def from_pretrained_with_dynamic_weights(
    pretrained_model_name_or_path: str | os.PathLike[str],
    *,
    model_loader: Any | None = None,
    dynamic_weights_config: DynamicWeightsConfig | None = None,
    apply_dynamic: bool = False,
    **from_pretrained_kwargs: Any,
) -> DynamicWeightsLoadResult:
    if model_loader is None:
        from diffusers import AutoModel

        model_loader = AutoModel
    from_pretrained = getattr(model_loader, "from_pretrained", model_loader)
    module = from_pretrained(pretrained_model_name_or_path, **from_pretrained_kwargs)
    if not apply_dynamic:
        return DynamicWeightsLoadResult(module=module)

    config = dynamic_weights_config or DynamicWeightsConfig()
    hook = apply_dynamic_weights(module, config)
    return DynamicWeightsLoadResult(
        module=module,
        hook=hook,
        should_move_to_execution_device=config.execution_mode.lower() == "plan",
    )


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


def clear_dynamic_weights_plan_cache() -> None:
    with _DYNAMIC_WEIGHTS_PLAN_CACHE_LOCK:
        _DYNAMIC_WEIGHTS_PLAN_CACHE.clear()


def clear_dynamic_weights_pinned_tensor_cache() -> None:
    with _DYNAMIC_WEIGHTS_PINNED_TENSOR_CACHE_LOCK:
        _DYNAMIC_WEIGHTS_PINNED_TENSOR_CACHE.clear()


def clear_dynamic_weights_resident_device_tensor_cache() -> None:
    with _DYNAMIC_WEIGHTS_RESIDENT_DEVICE_TENSOR_CACHE_LOCK:
        _DYNAMIC_WEIGHTS_RESIDENT_DEVICE_TENSOR_CACHE.clear()


def build_dynamic_weight_plan(module: nn.Module, config: DynamicWeightsConfig) -> DynamicWeightsState:
    if not config.cache_plan:
        state = _build_dynamic_weight_plan_uncached(module, config)
        state.planner_decisions["plan_cache"] = "disabled"
        return state

    cache_key = _dynamic_weight_plan_cache_key(module, config)
    with _DYNAMIC_WEIGHTS_PLAN_CACHE_LOCK:
        cached_state = _DYNAMIC_WEIGHTS_PLAN_CACHE.get(cache_key)
    if cached_state is not None:
        state = cached_state.clone_for_runtime()
        state.planner_decisions["plan_cache"] = "hit"
        return state

    state = _build_dynamic_weight_plan_uncached(module, config)
    state.resolved_resident_module_patterns = [
        pattern.pattern for pattern in _resolve_resident_module_patterns(module, config)
    ]
    state.planner_decisions["plan_cache"] = "miss"
    with _DYNAMIC_WEIGHTS_PLAN_CACHE_LOCK:
        _DYNAMIC_WEIGHTS_PLAN_CACHE[cache_key] = state.clone_for_runtime()
    return state


def _build_dynamic_weight_plan_uncached(module: nn.Module, config: DynamicWeightsConfig) -> DynamicWeightsState:
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


def _dynamic_weight_plan_cache_key(module: nn.Module, config: DynamicWeightsConfig) -> tuple[Any, ...]:
    return (
        module.__class__.__module__,
        module.__class__.__qualname__,
        tuple(cls.__module__ + "." + cls.__qualname__ for cls in config.target_module_classes),
        tuple(config.skip_modules_pattern),
        tuple(config.always_resident_modules_pattern),
        tuple(config.resident_module_patterns),
        int(config.small_tensor_threshold_bytes),
        _dynamic_weight_plan_structure_signature(module, config),
    )


def _dynamic_weight_plan_structure_signature(
    module: nn.Module,
    config: DynamicWeightsConfig,
) -> tuple[tuple[Any, ...], ...]:
    skip_patterns = tuple(re.compile(pattern) for pattern in config.skip_modules_pattern)
    signature: list[tuple[Any, ...]] = []
    for module_name, submodule in module.named_modules():
        if module_name == "":
            continue
        if skip_patterns and any(pattern.search(module_name) for pattern in skip_patterns):
            continue
        if not isinstance(submodule, config.target_module_classes):
            continue
        tensor_signature = tuple(
            (
                tensor_name,
                tuple(tensor.shape),
                str(tensor.dtype),
                str(tensor.device),
                tensor.numel(),
                tensor.element_size(),
            )
            for tensor_name, tensor in _iter_local_tensors(submodule)
        )
        signature.append(
            (
                module_name,
                submodule.__class__.__module__,
                submodule.__class__.__qualname__,
                tensor_signature,
            )
        )
    return tuple(signature)


def _iter_local_tensors(module: nn.Module):
    for name, parameter in module.named_parameters(recurse=False):
        yield name, parameter.data
    for name, buffer in module.named_buffers(recurse=False):
        yield name, buffer.data


def _resolve_resident_module_patterns(
    module: nn.Module,
    config: DynamicWeightsConfig,
) -> tuple[re.Pattern[str], ...]:
    raw_patterns = tuple(pattern.strip() for pattern in config.resident_module_patterns if pattern.strip())
    explicit_patterns = tuple(pattern for pattern in raw_patterns if pattern.lower() != "auto")
    compiled_patterns = [re.compile(pattern) for pattern in explicit_patterns]
    if not any(pattern.lower() == "auto" for pattern in raw_patterns):
        return tuple(compiled_patterns)

    inferred_patterns = _infer_repeated_resident_module_patterns(module, config)
    return tuple([*compiled_patterns, *inferred_patterns])


def _infer_repeated_resident_module_patterns(
    module: nn.Module,
    config: DynamicWeightsConfig,
) -> tuple[re.Pattern[str], ...]:
    skip_patterns = tuple(re.compile(pattern) for pattern in config.skip_modules_pattern)
    repeated_module_pattern = re.compile(r"^(.+)\.(\d+)$")
    groups: dict[str, list[tuple[str, int]]] = {}

    for module_name, submodule in module.named_modules():
        match = repeated_module_pattern.match(module_name)
        if match is None:
            continue
        if skip_patterns and any(pattern.search(module_name) for pattern in skip_patterns):
            continue
        target_bytes = _target_module_size_bytes(submodule, config.target_module_classes)
        if target_bytes <= 0:
            continue
        groups.setdefault(match.group(1), []).append((module_name, target_bytes))

    candidates = [
        (prefix, members, sum(member_bytes for _, member_bytes in members))
        for prefix, members in groups.items()
        if len(members) >= 2
    ]
    if not candidates:
        return ()

    prefix, _, _ = max(candidates, key=lambda item: (item[2], len(item[1])))
    return (re.compile(rf"^{re.escape(prefix)}\.\d+$"),)


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


def _target_module_size_bytes(module: nn.Module, target_module_classes: tuple[type[nn.Module], ...]) -> int:
    size = 0
    for submodule in module.modules():
        if not isinstance(submodule, target_module_classes):
            continue
        size += sum(_tensor_size_bytes(parameter.data) for parameter in submodule.parameters(recurse=False))
        size += sum(_tensor_size_bytes(buffer.data) for buffer in submodule.buffers(recurse=False))
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


def _largest_first_order(items: list[tuple[str, Any, int]]) -> list[tuple[str, Any, int]]:
    return sorted(items, key=lambda item: item[2], reverse=True)


def _tensor_size_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()
