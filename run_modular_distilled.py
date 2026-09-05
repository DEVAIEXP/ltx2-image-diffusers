"""
Local low-VRAM parity runner for the experimental LTX 2.3 distilled modular T2I blocks.
"""

import contextlib
from dataclasses import replace
import json
import os
from pathlib import Path
import re
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from diffusers import AutoencoderKLLTX2Video, FlowMatchEulerDiscreteScheduler
from diffusers.hooks import apply_group_offloading
from diffusers.models.attention_dispatch import AttentionBackendName, _AttentionBackendRegistry, attention_backend
from transformers import Gemma3ForConditionalGeneration, GemmaTokenizerFast

from custom_blocks.ltx2_image import LTX2ImageDistilledBlocks, LTX2ImageTextEncoderStep
from custom_blocks.ltx2_image.connectors_ltx2_image import LTX2ImageTextConnectors
from custom_blocks.ltx2_image.modular_blocks_ltx2_image import (
    LTX2ImageConnectorStep,
    LTX2ImageDenoiseStep,
    LTX2ImagePrepareLatentsStep,
)
from diffusers_dynamic_offloader import (
    DynamicOffloadSettings,
    enable_dynamic_offload,
    format_dynamic_offload_presets,
    from_pretrained_with_dynamic_offload,
    is_wsl_environment,
    purge_windows_standby_cache_event,
    remove_dynamic_offload,
)
from inference_utils import RunTracker, flush


def env_value(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def parse_bool_env(name: str, default: str = "0") -> bool:
    return env_value(name, default).strip().lower() in {"1", "true", "yes", "on"}


def parse_metrics_level() -> int:
    value = env_value("DDO_RUNNER_METRICS_LEVEL", "0").strip()
    try:
        level = int(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid DDO_RUNNER_METRICS_LEVEL={value!r}. "
            "Valid values: 0, 1, 2."
        ) from exc
    if level not in {0, 1, 2}:
        raise ValueError(
            f"Invalid DDO_RUNNER_METRICS_LEVEL={value!r}. "
            "Valid values: 0, 1, 2."
        )
    return level


RUNNING_ON_WSL = is_wsl_environment()
DEVICE = env_value("DDO_RUNNER_DEVICE", "cuda:0")
OFFLOAD_DEVICE = "cpu"
DTYPE = torch.bfloat16

MODEL_TAG = "distilled_modular"
MODEL_PATH = env_value("DDO_RUNNER_MODEL_PATH", r"E:\model\ltx2.3-image-distilled-1.1")
TEXT_ENCODER_LOW_CPU_MEM_USAGE = True
MODEL_LOW_CPU_MEM_USAGE = parse_bool_env("DDO_RUNNER_LOW_CPU_MEM_USAGE", "1")
DYNAMIC_OFFLOAD_SETTINGS = DynamicOffloadSettings.from_env(
    execution_device=DEVICE,
    offload_device=OFFLOAD_DEVICE,
    running_on_wsl=RUNNING_ON_WSL,
    default_preset="auto",
)
REQUESTED_DYNAMIC_OFFLOAD_PRESET = DYNAMIC_OFFLOAD_SETTINGS.requested_preset
DYNAMIC_OFFLOAD_PRESET = DYNAMIC_OFFLOAD_SETTINGS.effective_preset


def preset_env(name: str, default: str = "") -> str:
    if name in os.environ:
        return os.environ[name]
    value = DYNAMIC_OFFLOAD_SETTINGS.preset_value(name)
    if value != "":
        return value
    return default


def parse_bool_preset_env(name: str, default: str = "0") -> bool:
    return preset_env(name, default).strip().lower() in {"1", "true", "yes", "on"}


def parse_pattern_list_env(name: str, default: str = "") -> tuple[str, ...]:
    return tuple(item.strip() for item in re.split(r"[;,]", preset_env(name, default)) if item.strip())


def parse_float_preset_env(name: str, default: str = "0.0") -> float:
    return float(preset_env(name, default))


AUTO_CPU_OFFLOAD = parse_bool_env("DDO_RUNNER_AUTO_CPU_OFFLOAD")
TEXT_ENCODER_GROUP_OFFLOAD = parse_bool_preset_env("DDO_RUNNER_TEXT_ENCODER_GROUP_OFFLOAD", "1")
TEXT_ENCODER_DYNAMIC_OFFLOAD = parse_bool_preset_env("DDO_RUNNER_TEXT_ENCODER_DYNAMIC_OFFLOAD")
TEXT_ENCODER_DYNAMIC_OFFLOAD_PIN_CPU_MEMORY = parse_bool_preset_env(
    "DDO_RUNNER_TEXT_ENCODER_DYNAMIC_OFFLOAD_PIN_CPU_MEMORY",
    "0",
)
TEXT_ENCODER_DYNAMIC_OFFLOAD_RESIDENT_MODULE_BUDGET_GB = parse_float_preset_env(
    "DDO_RUNNER_TEXT_ENCODER_DYNAMIC_OFFLOAD_RESIDENT_MODULE_BUDGET_GB",
    "3.0",
)
TEXT_ENCODER_DYNAMIC_OFFLOAD_PIN_WEIGHT_BUDGET_GB = parse_float_preset_env(
    "DDO_RUNNER_TEXT_ENCODER_DYNAMIC_OFFLOAD_PIN_WEIGHT_BUDGET_GB",
    "0.0",
)
TEXT_ENCODER_DYNAMIC_OFFLOAD_PIN_WEIGHT_BUDGET_RATIO = parse_float_preset_env(
    "DDO_RUNNER_TEXT_ENCODER_DYNAMIC_OFFLOAD_PIN_WEIGHT_BUDGET_RATIO",
    "0.0",
)
TEXT_ENCODER_DYNAMIC_OFFLOAD_PIN_WEIGHT_SELECTION = preset_env(
    "DDO_RUNNER_TEXT_ENCODER_DYNAMIC_OFFLOAD_PIN_WEIGHT_SELECTION",
    DYNAMIC_OFFLOAD_SETTINGS.config.pin_weight_selection,
).lower()
TEXT_ENCODER_DYNAMIC_OFFLOAD_AUTO_BUDGET_POLICY = preset_env(
    "DDO_RUNNER_TEXT_ENCODER_DYNAMIC_OFFLOAD_AUTO_BUDGET_POLICY",
    DYNAMIC_OFFLOAD_SETTINGS.config.auto_budget_policy,
).lower()
TEXT_ENCODER_DYNAMIC_OFFLOAD_MAX_RESIDENT_MODULE_BUDGET_GB = parse_float_preset_env(
    "DDO_RUNNER_TEXT_ENCODER_DYNAMIC_OFFLOAD_MAX_RESIDENT_MODULE_BUDGET_GB",
    str(DYNAMIC_OFFLOAD_SETTINGS.config.max_resident_module_budget_gb),
)
TEXT_ENCODER_DYNAMIC_OFFLOAD_MAX_PIN_WEIGHT_BUDGET_GB = parse_float_preset_env(
    "DDO_RUNNER_TEXT_ENCODER_DYNAMIC_OFFLOAD_MAX_PIN_WEIGHT_BUDGET_GB",
    str(DYNAMIC_OFFLOAD_SETTINGS.config.max_pin_weight_budget_gb),
)
TEXT_ENCODER_DYNAMIC_OFFLOAD_SKIP_MODULES = parse_pattern_list_env(
    "DDO_RUNNER_TEXT_ENCODER_DYNAMIC_OFFLOAD_SKIP_MODULE_PATTERNS",
    r"(^|\.)vision_tower(\.|$)",
)
TRANSFORMER_GROUP_OFFLOAD = parse_bool_preset_env("DDO_RUNNER_TRANSFORMER_GROUP_OFFLOAD")
TRANSFORMER_MEMORY_MANAGER = preset_env("DDO_RUNNER_TRANSFORMER_MEMORY_MANAGER", "off").lower()
DYNAMIC_OFFLOAD_CONFIG = DYNAMIC_OFFLOAD_SETTINGS.config
DYNAMIC_OFFLOAD_EXECUTION_MODE = DYNAMIC_OFFLOAD_CONFIG.execution_mode
DYNAMIC_OFFLOAD_PIN_CPU_MEMORY = DYNAMIC_OFFLOAD_SETTINGS.requested_pin_cpu_memory
DYNAMIC_OFFLOAD_SHOW_PROFILE = DYNAMIC_OFFLOAD_CONFIG.show_profile
DYNAMIC_OFFLOAD_EFFECTIVE_PIN_CPU_MEMORY = DYNAMIC_OFFLOAD_SETTINGS.effective_pin_cpu_memory
PRE_VAE_CLEANUP_REPEATS = int(preset_env("DDO_RUNNER_PRE_VAE_CLEANUP_REPEATS", "3" if RUNNING_ON_WSL else "1"))
RESET_DYNAMIC_MEMORY_AFTER_RUN = parse_bool_env("DDO_RUNNER_RESET_DYNAMIC_MEMORY_AFTER_RUN")
PURGE_WINDOWS_STANDBY_BEFORE_RUN = parse_bool_env("DDO_RUNNER_PURGE_WINDOWS_STANDBY_BEFORE_RUN")
PURGE_WINDOWS_STANDBY_AFTER_TEXT_ENCODER = parse_bool_env("DDO_RUNNER_PURGE_WINDOWS_STANDBY_AFTER_TEXT_ENCODER")
PURGE_WINDOWS_STANDBY_BEFORE_TRANSFORMER = parse_bool_env("DDO_RUNNER_PURGE_WINDOWS_STANDBY_BEFORE_TRANSFORMER")
PURGE_WINDOWS_STANDBY_AFTER_RUN = parse_bool_env("DDO_RUNNER_PURGE_WINDOWS_STANDBY_AFTER_RUN")
METRICS_LEVEL = parse_metrics_level()
SHOW_METRICS = METRICS_LEVEL >= 1 or parse_bool_env("DDO_RUNNER_SHOW_METRICS")
SAVE_METRICS = METRICS_LEVEL >= 2 or parse_bool_env("DDO_RUNNER_SAVE_METRICS")
ATTENTION_BACKEND = preset_env("DDO_RUNNER_ATTENTION_BACKEND", "native").lower()
FLASH_COMPATIBLE_ATTENTION_BACKENDS = {"flash", "flash_hub", "_native_flash", "_flash_3", "_flash_3_hub"}
DROP_TRIVIAL_ATTENTION_MASK = (
    parse_bool_env("DDO_RUNNER_DROP_TRIVIAL_ATTENTION_MASK")
    or ATTENTION_BACKEND in FLASH_COMPATIBLE_ATTENTION_BACKENDS
)
GROUP_OFFLOAD_CONFIG = {
    "mode": "components_manager_auto_cpu_offload" if AUTO_CPU_OFFLOAD else "disabled",
    "device": DEVICE,
    "text_encoder_group_offload": TEXT_ENCODER_GROUP_OFFLOAD,
    "text_encoder_offload_type": preset_env("DDO_RUNNER_TEXT_ENCODER_OFFLOAD_TYPE", "leaf_level"),
    "text_encoder_use_stream": parse_bool_preset_env("DDO_RUNNER_TEXT_ENCODER_OFFLOAD_STREAM", "1"),
    "text_encoder_record_stream": parse_bool_preset_env("DDO_RUNNER_TEXT_ENCODER_OFFLOAD_RECORD_STREAM"),
    "text_encoder_num_blocks_per_group": int(preset_env("DDO_RUNNER_TEXT_ENCODER_NUM_BLOCKS_PER_GROUP", "1")),
    "transformer_group_offload": TRANSFORMER_GROUP_OFFLOAD,
    "transformer_offload_type": preset_env("DDO_RUNNER_TRANSFORMER_OFFLOAD_TYPE", "leaf_level"),
    "transformer_use_stream": parse_bool_preset_env("DDO_RUNNER_TRANSFORMER_OFFLOAD_STREAM", "1"),
    "transformer_record_stream": parse_bool_preset_env("DDO_RUNNER_TRANSFORMER_OFFLOAD_RECORD_STREAM"),
    "transformer_low_cpu_mem_usage": parse_bool_preset_env(
        "DDO_RUNNER_TRANSFORMER_OFFLOAD_LOW_CPU_MEM_USAGE",
        "1" if MODEL_LOW_CPU_MEM_USAGE else "0",
    ),
    "transformer_num_blocks_per_group": int(preset_env("DDO_RUNNER_TRANSFORMER_NUM_BLOCKS_PER_GROUP", "1")),
}

WIDTH = int(env_value("DDO_RUNNER_WIDTH", "1280"))
HEIGHT = int(env_value("DDO_RUNNER_HEIGHT", "704"))
SEED = int(env_value("DDO_RUNNER_SEED", "43"))
NUM_INFERENCE_STEPS = int(env_value("DDO_RUNNER_STEPS", "8"))
GUIDANCE_SCALE = float(env_value("DDO_RUNNER_GUIDANCE_SCALE", "1.0"))
GUIDANCE_RESCALE = float(env_value("DDO_RUNNER_GUIDANCE_RESCALE", "0.7"))
DECODE_TIMESTEP = float(env_value("DDO_RUNNER_DECODE_TIMESTEP", "0.0"))
DECODE_NOISE_SCALE_ENV = env_value("DDO_RUNNER_DECODE_NOISE_SCALE")
DECODE_NOISE_SCALE = None if DECODE_NOISE_SCALE_ENV in (None, "") else float(DECODE_NOISE_SCALE_ENV)
PAG_ENABLED = parse_bool_env("DDO_RUNNER_PAG_ENABLED")
PAG_SCALE = float(env_value("DDO_RUNNER_PAG_SCALE", "0.2"))
PAG_APPLIED_LAYERS = [int(x) for x in env_value("DDO_RUNNER_PAG_LAYERS", "28").split(",") if x]
FAKE_PROMPT_EMBEDS = parse_bool_env("DDO_RUNNER_FAKE_PROMPT")
GENERATION_REPEATS = max(1, int(preset_env("DDO_RUNNER_GENERATION_REPEATS", "1")))
TRANSFORMER_PREPARE_REPEATS = max(1, int(preset_env("DDO_RUNNER_TRANSFORMER_PREPARE_REPEATS", "1")))

prompt = env_value(
    "DDO_RUNNER_PROMPT",
    "Fisheye close-up of a calico cat wearing a tiny flower crown, sniffing the camera lens in a sunny park, with bright colors, realistic fur detail, and playful viral-pet energy.",
)
negative_prompt = env_value("DDO_RUNNER_NEGATIVE_PROMPT", "")


def build_run_slug(seed):
    pag_tag = f"pag{PAG_SCALE:g}_layers{'-'.join(map(str, PAG_APPLIED_LAYERS))}" if PAG_ENABLED else "nopag"
    return "_".join(
        [
            "ltx23_image",
            MODEL_TAG,
            "bf16",
            "text_encoder_original",
            pag_tag,
            f"{WIDTH}x{HEIGHT}",
            f"steps{NUM_INFERENCE_STEPS}",
            f"seed{seed}",
        ]
    )


def apply_model_group_offload(model, *, prefix):
    offload_type = GROUP_OFFLOAD_CONFIG[f"{prefix}_offload_type"]
    kwargs = {
        "onload_device": torch.device(DEVICE),
        "offload_device": torch.device(OFFLOAD_DEVICE),
        "offload_type": offload_type,
        "use_stream": GROUP_OFFLOAD_CONFIG[f"{prefix}_use_stream"],
        "record_stream": GROUP_OFFLOAD_CONFIG[f"{prefix}_record_stream"],
        "low_cpu_mem_usage": (
            TEXT_ENCODER_LOW_CPU_MEM_USAGE
            if prefix == "text_encoder"
            else GROUP_OFFLOAD_CONFIG["transformer_low_cpu_mem_usage"]
        ),
    }
    if offload_type == "block_level":
        kwargs["num_blocks_per_group"] = GROUP_OFFLOAD_CONFIG[f"{prefix}_num_blocks_per_group"]
    apply_group_offloading(model, **kwargs)


def get_attention_backend():
    if ATTENTION_BACKEND in ("", "default", "none"):
        return None
    try:
        return AttentionBackendName(ATTENTION_BACKEND)
    except ValueError as exc:
        valid = ", ".join(backend.value for backend in AttentionBackendName)
        raise ValueError(
            f"Invalid DDO_RUNNER_ATTENTION_BACKEND={ATTENTION_BACKEND!r}. "
            f"Valid values: {valid}"
        ) from exc


def _is_trivial_zero_attention_mask(attn_mask):
    if attn_mask is None:
        return False
    with torch.no_grad():
        return bool(torch.all(attn_mask == 0).item())


def install_trivial_mask_flash_wrapper():
    if not DROP_TRIVIAL_ATTENTION_MASK:
        return False

    wrapped_any = False
    for backend in (AttentionBackendName.FLASH, AttentionBackendName._NATIVE_FLASH):
        backend_fn = _AttentionBackendRegistry._backends.get(backend)
        if backend_fn is None or getattr(backend_fn, "_ltx2_trivial_mask_wrapper", False):
            continue

        def wrapped_backend_fn(*args, _backend_fn=backend_fn, _backend=backend, **kwargs):
            attn_mask = kwargs.get("attn_mask")
            if _is_trivial_zero_attention_mask(attn_mask):
                if env_value("DDO_RUNNER_LOG_ATTENTION_MASK", "0") == "1":
                    print(f"  [attention_mask] dropping trivial mask inside backend {_backend.value}", flush=True)
                kwargs["attn_mask"] = None
            return _backend_fn(*args, **kwargs)

        wrapped_backend_fn._ltx2_trivial_mask_wrapper = True
        _AttentionBackendRegistry._backends[backend] = wrapped_backend_fn
        wrapped_any = True
    return wrapped_any

def get_attention_backend_context():
    backend = get_attention_backend()
    if backend is None:
        return contextlib.nullcontext()
    return attention_backend(backend)


def apply_transformer_attention_backend(transformer):
    backend = get_attention_backend()
    transformer.drop_trivial_attention_mask = DROP_TRIVIAL_ATTENTION_MASK
    patched = 0
    for module in transformer.modules():
        processor = getattr(module, "processor", None)
        if processor is not None and hasattr(processor, "_attention_backend"):
            processor._attention_backend = backend
            patched += 1
    return patched, None if backend is None else backend.value, DROP_TRIVIAL_ATTENTION_MASK


def cleanup_runtime_state(record_event, event_name: str, *, repeats: int = 1, collect_cuda_ipc: bool = False) -> None:
    if repeats <= 0:
        return

    before_free_gb = before_total_gb = None
    after_free_gb = after_total_gb = None
    if torch.cuda.is_available():
        before_free, before_total = torch.cuda.mem_get_info(DEVICE)
        before_free_gb = before_free / 1024**3
        before_total_gb = before_total / 1024**3

    event_t0 = time.time()
    for _ in range(repeats):
        flush()
        if collect_cuda_ipc and torch.cuda.is_available() and hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()

    if torch.cuda.is_available():
        after_free, after_total = torch.cuda.mem_get_info(DEVICE)
        after_free_gb = after_free / 1024**3
        after_total_gb = after_total / 1024**3

    record_event(
        event_name,
        time.time() - event_t0,
        repeats=repeats,
        collect_cuda_ipc=collect_cuda_ipc,
        before_free_vram_gb=None if before_free_gb is None else round(before_free_gb, 4),
        before_total_vram_gb=None if before_total_gb is None else round(before_total_gb, 4),
        after_free_vram_gb=None if after_free_gb is None else round(after_free_gb, 4),
        after_total_vram_gb=None if after_total_gb is None else round(after_total_gb, 4),
    )


def cleanup_before_vae_decode(record_event) -> None:
    cleanup_runtime_state(
        record_event,
        "cleanup_before_vae_decode",
        repeats=PRE_VAE_CLEANUP_REPEATS,
        collect_cuda_ipc=True,
    )


def denoise_progress_callback(components, step_index, timestep, callback_kwargs):
    now = time.perf_counter()
    last_time = getattr(denoise_progress_callback, "last_time", now)
    start_time = getattr(denoise_progress_callback, "start_time", last_time)
    step_elapsed = now - last_time
    total_elapsed = now - start_time
    denoise_progress_callback.last_time = now
    denoise_progress_callback.step_times.append(step_elapsed)

    used_gb = torch.cuda.memory_allocated(DEVICE) / 1024**3
    reserved_gb = torch.cuda.memory_reserved(DEVICE) / 1024**3
    total_steps = len(denoise_progress_callback.timesteps)
    avg_elapsed = total_elapsed / (step_index + 1)
    if SHOW_METRICS:
        print(
            f"  [denoise] step {step_index + 1}/{total_steps} timestep={float(timestep):.4f} "
            f"elapsed={step_elapsed:.4f}s avg={avg_elapsed:.4f}s/it "
            f"torch_alloc={used_gb:.2f} GiB torch_reserved={reserved_gb:.2f} GiB",
            flush=True,
        )
    return callback_kwargs


def main():
    if parse_bool_env("DDO_RUNNER_PRINT_DYNAMIC_OFFLOAD_PRESETS"):
        print(format_dynamic_offload_presets(default_preset="auto", running_on_wsl=RUNNING_ON_WSL))
        return

    seed = SEED or torch.randint(0, 2**32, (1,)).item()
    if not SEED:
        print(f"Using random seed: {seed}")
    generator = torch.Generator(device="cpu").manual_seed(seed)

    run_slug = build_run_slug(seed)
    output_dir = Path("outputs/ltx_image_modular")
    metrics_dir = output_dir / "metrics"
    run_metrics = {
        "run_slug": run_slug,
        "model_tag": MODEL_TAG,
        "model_path": MODEL_PATH,
        "width": WIDTH,
        "height": HEIGHT,
        "seed": seed,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "generation_repeats": GENERATION_REPEATS,
        "transformer_prepare_repeats": TRANSFORMER_PREPARE_REPEATS,
        "guidance_scale": GUIDANCE_SCALE,
        "guidance_rescale": GUIDANCE_RESCALE,
        "vae_decode_timestep": DECODE_TIMESTEP,
        "vae_decode_noise_scale": DECODE_NOISE_SCALE,
        "pag_enabled": PAG_ENABLED,
        "pag_scale": PAG_SCALE if PAG_ENABLED else 0.0,
        "pag_applied_layers": PAG_APPLIED_LAYERS if PAG_ENABLED else None,
        "dtype": str(DTYPE),
        "text_encoder_low_cpu_mem_usage": TEXT_ENCODER_LOW_CPU_MEM_USAGE,
        "text_encoder_dynamic_offload": TEXT_ENCODER_DYNAMIC_OFFLOAD,
        "text_encoder_dynamic_offload_pin_cpu_memory": TEXT_ENCODER_DYNAMIC_OFFLOAD_PIN_CPU_MEMORY,
        "text_encoder_dynamic_offload_resident_module_budget_gb": TEXT_ENCODER_DYNAMIC_OFFLOAD_RESIDENT_MODULE_BUDGET_GB,
        "text_encoder_dynamic_offload_pin_weight_budget_gb": TEXT_ENCODER_DYNAMIC_OFFLOAD_PIN_WEIGHT_BUDGET_GB,
        "text_encoder_dynamic_offload_pin_weight_budget_ratio": TEXT_ENCODER_DYNAMIC_OFFLOAD_PIN_WEIGHT_BUDGET_RATIO,
        "text_encoder_dynamic_offload_pin_weight_selection": TEXT_ENCODER_DYNAMIC_OFFLOAD_PIN_WEIGHT_SELECTION,
        "text_encoder_dynamic_offload_auto_budget_policy": TEXT_ENCODER_DYNAMIC_OFFLOAD_AUTO_BUDGET_POLICY,
        "text_encoder_dynamic_offload_max_resident_module_budget_gb": TEXT_ENCODER_DYNAMIC_OFFLOAD_MAX_RESIDENT_MODULE_BUDGET_GB,
        "text_encoder_dynamic_offload_max_pin_weight_budget_gb": TEXT_ENCODER_DYNAMIC_OFFLOAD_MAX_PIN_WEIGHT_BUDGET_GB,
        "text_encoder_dynamic_offload_skip_modules": TEXT_ENCODER_DYNAMIC_OFFLOAD_SKIP_MODULES,
        "model_low_cpu_mem_usage": MODEL_LOW_CPU_MEM_USAGE,
        "running_on_wsl": RUNNING_ON_WSL,
        "reset_dynamic_memory_after_run": RESET_DYNAMIC_MEMORY_AFTER_RUN,
        "purge_windows_standby_before_run": PURGE_WINDOWS_STANDBY_BEFORE_RUN,
        "purge_windows_standby_after_text_encoder": PURGE_WINDOWS_STANDBY_AFTER_TEXT_ENCODER,
        "purge_windows_standby_before_transformer": PURGE_WINDOWS_STANDBY_BEFORE_TRANSFORMER,
        "purge_windows_standby_after_run": PURGE_WINDOWS_STANDBY_AFTER_RUN,
        "metrics_level": METRICS_LEVEL,
        "show_metrics": SHOW_METRICS,
        "save_metrics": SAVE_METRICS,
        "group_offload_config": GROUP_OFFLOAD_CONFIG.copy(),
        "transformer_memory_manager": TRANSFORMER_MEMORY_MANAGER,
        **DYNAMIC_OFFLOAD_SETTINGS.as_metrics(),
        "pre_vae_cleanup_repeats": PRE_VAE_CLEANUP_REPEATS,
        "attention_backend": ATTENTION_BACKEND,
        "drop_trivial_attention_mask": DROP_TRIVIAL_ATTENTION_MASK,
        "events": [],
        "steps": [],
    }

    flash_mask_wrapper_installed = install_trivial_mask_flash_wrapper()
    run_metrics["flash_trivial_mask_wrapper_installed"] = flash_mask_wrapper_installed
    if DYNAMIC_OFFLOAD_PRESET:
        if REQUESTED_DYNAMIC_OFFLOAD_PRESET == "auto":
            print(f"Using DDO preset: auto -> {DYNAMIC_OFFLOAD_PRESET}", flush=True)
        else:
            print(f"Using DDO preset: {DYNAMIC_OFFLOAD_PRESET}", flush=True)
    if DYNAMIC_OFFLOAD_PIN_CPU_MEMORY and not DYNAMIC_OFFLOAD_EFFECTIVE_PIN_CPU_MEMORY:
        print("  [dynamic-offload] disabling pinned CPU memory on WSL; set DDO_DISABLE_PIN_ON_WSL=0 to force it.", flush=True)
    text_encoder_route = (
        "dynamic_offload"
        if TEXT_ENCODER_DYNAMIC_OFFLOAD
        else "group_offload"
        if TEXT_ENCODER_GROUP_OFFLOAD
        else "cuda_to"
    )
    transformer_route = "dynamic_offload" if DYNAMIC_OFFLOAD_SETTINGS.enabled else "standard"
    print(
        "  [runner] "
        f"text_encoder_route={text_encoder_route} "
        f"text_encoder_group_offload={TEXT_ENCODER_GROUP_OFFLOAD} "
        f"text_encoder_dynamic_offload={TEXT_ENCODER_DYNAMIC_OFFLOAD} "
        f"transformer_route={transformer_route} "
        f"preset={DYNAMIC_OFFLOAD_PRESET or 'none'}",
        flush=True,
    )

    tracker = RunTracker(DEVICE, run_metrics, interval=0.1, show_metrics=SHOW_METRICS)
    record_event = tracker.record_event
    step_start = tracker.step_start
    step_end = tracker.step_end

    if PURGE_WINDOWS_STANDBY_BEFORE_RUN:
        purge_windows_standby_cache_event(record_event, "purge_windows_standby_before_run")

    t0 = step_start("Pass 0: Encode prompts")
    if FAKE_PROMPT_EMBEDS:
        prompt_embeds = torch.zeros((1, 1024, 188160), dtype=DTYPE, device=OFFLOAD_DEVICE)
        prompt_attention_mask = torch.ones((1, 1024), dtype=torch.long, device=OFFLOAD_DEVICE)
        record_event("fake_prompt_embeds", 0.0, shape=list(prompt_embeds.shape))
    else:
        event_t0 = time.time()
        text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            subfolder="text_encoder",
            torch_dtype=DTYPE,
            low_cpu_mem_usage=TEXT_ENCODER_LOW_CPU_MEM_USAGE,
        )
        record_event("load_text_encoder", time.time() - event_t0, source=MODEL_PATH)

        event_t0 = time.time()
        text_encoder_dynamic_offload_hook = None
        if TEXT_ENCODER_DYNAMIC_OFFLOAD:
            text_encoder_dynamic_offload_config = replace(
                DYNAMIC_OFFLOAD_CONFIG,
                pin_cpu_memory=TEXT_ENCODER_DYNAMIC_OFFLOAD_PIN_CPU_MEMORY,
                pin_weight_budget_gb=TEXT_ENCODER_DYNAMIC_OFFLOAD_PIN_WEIGHT_BUDGET_GB,
                pin_weight_budget_ratio=TEXT_ENCODER_DYNAMIC_OFFLOAD_PIN_WEIGHT_BUDGET_RATIO,
                pin_weight_selection=TEXT_ENCODER_DYNAMIC_OFFLOAD_PIN_WEIGHT_SELECTION,
                auto_budget_policy=TEXT_ENCODER_DYNAMIC_OFFLOAD_AUTO_BUDGET_POLICY,
                max_resident_module_budget_gb=TEXT_ENCODER_DYNAMIC_OFFLOAD_MAX_RESIDENT_MODULE_BUDGET_GB,
                max_pin_weight_budget_gb=TEXT_ENCODER_DYNAMIC_OFFLOAD_MAX_PIN_WEIGHT_BUDGET_GB,
                resident_module_budget_gb=TEXT_ENCODER_DYNAMIC_OFFLOAD_RESIDENT_MODULE_BUDGET_GB,
                skip_modules_pattern=(
                    *DYNAMIC_OFFLOAD_CONFIG.skip_modules_pattern,
                    *TEXT_ENCODER_DYNAMIC_OFFLOAD_SKIP_MODULES,
                ),
            )
            text_encoder_dynamic_offload = enable_dynamic_offload(
                text_encoder,
                settings=DYNAMIC_OFFLOAD_SETTINGS,
                config=text_encoder_dynamic_offload_config,
                record_event=record_event,
                event_name="setup_text_encoder_dynamic_offload",
            )
            text_encoder_dynamic_offload_hook = text_encoder_dynamic_offload.hook
        elif TEXT_ENCODER_GROUP_OFFLOAD:
            apply_model_group_offload(text_encoder, prefix="text_encoder")
            record_event(
                "setup_text_encoder_group_offload",
                time.time() - event_t0,
                offload_type=GROUP_OFFLOAD_CONFIG["text_encoder_offload_type"],
                use_stream=GROUP_OFFLOAD_CONFIG["text_encoder_use_stream"],
                record_stream=GROUP_OFFLOAD_CONFIG["text_encoder_record_stream"],
                low_cpu_mem_usage=TEXT_ENCODER_LOW_CPU_MEM_USAGE,
            )
        else:
            text_encoder.to(DEVICE)
            record_event("load_text_encoder_to_cuda", time.time() - event_t0)

        event_t0 = time.time()
        tokenizer = GemmaTokenizerFast.from_pretrained(MODEL_PATH, subfolder="tokenizer")
        record_event("load_tokenizer", time.time() - event_t0, source=MODEL_PATH)

        event_t0 = time.time()
        prompt_pipe = LTX2ImageTextEncoderStep().init_pipeline()
        prompt_pipe.update_components(text_encoder=text_encoder, tokenizer=tokenizer)
        record_event("build_prompt_modular_pipeline", time.time() - event_t0, model_path=MODEL_PATH)

        event_t0 = time.time()
        with torch.inference_mode():
            prompt_state = prompt_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=GUIDANCE_SCALE,
                output=["prompt_embeds", "prompt_attention_mask"],
            )
        record_event("encode_prompt_call", time.time() - event_t0, classifier_free_guidance=False)

        prompt_embeds = prompt_state["prompt_embeds"].to(OFFLOAD_DEVICE)
        prompt_attention_mask = prompt_state["prompt_attention_mask"].to(OFFLOAD_DEVICE)
        if text_encoder_dynamic_offload_hook is not None:
            run_metrics["text_encoder_dynamic_offload_runtime_summary"] = text_encoder_dynamic_offload_hook.state.as_dict()
            if DYNAMIC_OFFLOAD_SHOW_PROFILE:
                text_encoder_dynamic_offload_hook.print_profile_summary()
            remove_dynamic_offload(text_encoder)
            text_encoder_dynamic_offload_hook = None
            del text_encoder_dynamic_offload_config
        del prompt_state
        del prompt_pipe, text_encoder, tokenizer
        cleanup_runtime_state(record_event, "cleanup_after_text_encoder")
        if PURGE_WINDOWS_STANDBY_AFTER_TEXT_ENCODER:
            purge_windows_standby_cache_event(record_event, "purge_windows_standby_after_text_encoder")

    if SHOW_METRICS:
        print(f"  prompt_embeds shape: {prompt_embeds.shape}")
    step_end("Pass 0: Encode prompts", t0)

    t0 = step_start(f"Pass 1: Generate at {WIDTH}x{HEIGHT}")

    event_t0 = time.time()
    connectors = LTX2ImageTextConnectors.from_pretrained(
        MODEL_PATH,
        subfolder="connectors",
        torch_dtype=DTYPE,
        low_cpu_mem_usage=MODEL_LOW_CPU_MEM_USAGE,
    ).to(DEVICE)
    record_event("load_connectors_to_cuda", time.time() - event_t0, source=MODEL_PATH)

    event_t0 = time.time()
    connector_pipe = LTX2ImageConnectorStep().init_pipeline()
    connector_pipe.update_components(connectors=connectors)
    record_event("build_connector_modular_pipeline", time.time() - event_t0, model_path=MODEL_PATH)

    event_t0 = time.time()
    connector_state = connector_pipe(
        prompt_embeds=prompt_embeds.to(device=DEVICE, dtype=DTYPE),
        prompt_attention_mask=prompt_attention_mask.to(device=DEVICE),
        negative_prompt_embeds=None,
        negative_prompt_attention_mask=None,
        do_classifier_free_guidance=False,
        pag_scale=PAG_SCALE if PAG_ENABLED else 0.0,
        output=[
            "connector_prompt_embeds",
            "connector_attention_mask",
            "batch_size",
            "transformer_batch_multiplier",
            "do_perturbed_attention_guidance",
        ],
    )
    record_event("connector_modular_pipe_call", time.time() - event_t0)

    connector_prompt_embeds = connector_state["connector_prompt_embeds"].to(OFFLOAD_DEVICE)
    connector_attention_mask = connector_state["connector_attention_mask"].to(OFFLOAD_DEVICE)
    latent_batch_size = connector_state["batch_size"]
    transformer_batch_multiplier = connector_state["transformer_batch_multiplier"]
    do_perturbed_attention_guidance = connector_state["do_perturbed_attention_guidance"]

    del prompt_embeds, prompt_attention_mask, connector_state
    del connector_pipe, connectors
    cleanup_runtime_state(record_event, "cleanup_after_connectors")

    dynamic_offload_enabled = DYNAMIC_OFFLOAD_SETTINGS.enabled
    dynamic_offload_config = DYNAMIC_OFFLOAD_CONFIG if dynamic_offload_enabled else None

    def transformer_prepare_event_name(name: str, repeat_index: int) -> str:
        if TRANSFORMER_PREPARE_REPEATS == 1:
            return name
        return f"{name}_prepare_{repeat_index + 1}"

    def load_and_prepare_transformer(repeat_index: int):
        transformer_load_kwargs = {
            "subfolder": "transformer",
            "torch_dtype": DTYPE,
            "low_cpu_mem_usage": MODEL_LOW_CPU_MEM_USAGE,
        }

        if PURGE_WINDOWS_STANDBY_BEFORE_TRANSFORMER:
            purge_windows_standby_cache_event(
                record_event,
                transformer_prepare_event_name("purge_windows_standby_before_transformer", repeat_index),
            )

        event_t0 = time.time()
        if MODEL_LOW_CPU_MEM_USAGE:
            transformer_load_kwargs["device_map"] = "cpu"
        transformer_load = from_pretrained_with_dynamic_offload(
            MODEL_PATH,
            dynamic_offload_config=dynamic_offload_config,
            apply_dynamic=False,
            **transformer_load_kwargs,
        )
        prepared_transformer = transformer_load.module
        prepared_dynamic_offload_hook = transformer_load.hook
        record_event(
            transformer_prepare_event_name("load_transformer", repeat_index),
            time.time() - event_t0,
            source=MODEL_PATH,
            low_cpu_mem_usage=MODEL_LOW_CPU_MEM_USAGE,
            device_map=transformer_load_kwargs.get("device_map"),
            loader="AutoModel",
            resolved_class=prepared_transformer.__class__.__name__,
        )

        event_t0 = time.time()
        patched_attention_processors, resolved_attention_backend, drop_trivial_attention_mask = (
            apply_transformer_attention_backend(prepared_transformer)
        )
        record_event(
            transformer_prepare_event_name("set_transformer_attention_backend", repeat_index),
            time.time() - event_t0,
            requested_backend=ATTENTION_BACKEND,
            resolved_backend=resolved_attention_backend,
            patched_processors=patched_attention_processors,
            drop_trivial_attention_mask=drop_trivial_attention_mask,
        )

        if dynamic_offload_enabled and DYNAMIC_OFFLOAD_EXECUTION_MODE != "plan" and TRANSFORMER_MEMORY_MANAGER != "off":
            raise ValueError(
                "Dynamic offload execution currently requires "
                "DDO_RUNNER_TRANSFORMER_MEMORY_MANAGER='off'. "
                "Use execution_mode='plan' with the block manager."
            )

        if dynamic_offload_enabled:
            prepared_dynamic_offload = enable_dynamic_offload(
                prepared_transformer,
                settings=DYNAMIC_OFFLOAD_SETTINGS,
                config=dynamic_offload_config,
                record_event=record_event,
                event_name=transformer_prepare_event_name("build_dynamic_offload_plan", repeat_index),
            )
            prepared_dynamic_offload_hook = prepared_dynamic_offload.hook

        event_t0 = time.time()
        if TRANSFORMER_MEMORY_MANAGER != "off":
            raise ValueError(
                "The transformer block manager is no longer used by this runner. "
                "Use DDO_PRESET or transformer group offload."
            )
        if TRANSFORMER_GROUP_OFFLOAD:
            apply_model_group_offload(prepared_transformer, prefix="transformer")
            record_event(
                transformer_prepare_event_name("setup_transformer_group_offload", repeat_index),
                time.time() - event_t0,
                offload_type=GROUP_OFFLOAD_CONFIG["transformer_offload_type"],
                use_stream=GROUP_OFFLOAD_CONFIG["transformer_use_stream"],
                record_stream=GROUP_OFFLOAD_CONFIG["transformer_record_stream"],
                low_cpu_mem_usage=GROUP_OFFLOAD_CONFIG["transformer_low_cpu_mem_usage"],
            )
        elif dynamic_offload_enabled and DYNAMIC_OFFLOAD_EXECUTION_MODE != "plan":
            record_event(
                transformer_prepare_event_name("skip_transformer_to_cuda", repeat_index),
                time.time() - event_t0,
                reason=f"dynamic_offload_{DYNAMIC_OFFLOAD_EXECUTION_MODE}",
            )
        else:
            prepared_transformer.to(DEVICE)
            record_event(transformer_prepare_event_name("load_transformer_to_cuda", repeat_index), time.time() - event_t0)
        return prepared_transformer, prepared_dynamic_offload_hook

    transformer = None
    dynamic_offload_hook = None
    for prepare_repeat_index in range(TRANSFORMER_PREPARE_REPEATS):
        if TRANSFORMER_PREPARE_REPEATS > 1:
            print(
                f"  Transformer prepare repeat {prepare_repeat_index + 1}/{TRANSFORMER_PREPARE_REPEATS}",
                flush=True,
            )
        transformer, dynamic_offload_hook = load_and_prepare_transformer(prepare_repeat_index)
        if prepare_repeat_index + 1 < TRANSFORMER_PREPARE_REPEATS:
            if dynamic_offload_enabled:
                remove_dynamic_offload(transformer)
            del transformer
            dynamic_offload_hook = None
            cleanup_runtime_state(record_event, transformer_prepare_event_name("cleanup_after_transformer_prepare", prepare_repeat_index))

    event_t0 = time.time()
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")
    record_event("load_scheduler", time.time() - event_t0, source=MODEL_PATH)

    event_t0 = time.time()
    prepare_pipe = LTX2ImagePrepareLatentsStep().init_pipeline()
    prepare_pipe.update_components(transformer=transformer, scheduler=scheduler)
    record_event("build_prepare_latents_modular_pipeline", time.time() - event_t0, model_path=MODEL_PATH)

    event_t0 = time.time()
    denoise_pipe = LTX2ImageDenoiseStep().init_pipeline()
    denoise_pipe.update_components(transformer=transformer, scheduler=scheduler)
    record_event("build_denoise_modular_pipeline", time.time() - event_t0, model_path=MODEL_PATH)

    image_latent = None
    latent_height = None
    latent_width = None
    latent_channels = None
    run_metrics["denoise_step_times_by_repeat"] = []

    for repeat_index in range(GENERATION_REPEATS):
        repeat_suffix = "" if GENERATION_REPEATS == 1 else f"_repeat_{repeat_index + 1}"
        if GENERATION_REPEATS > 1:
            print(f"  Warm generation repeat {repeat_index + 1}/{GENERATION_REPEATS}", flush=True)

        event_t0 = time.time()
        prepare_state = prepare_pipe(
            width=WIDTH,
            height=HEIGHT,
            num_inference_steps=NUM_INFERENCE_STEPS,
            batch_size=latent_batch_size,
            transformer_batch_multiplier=transformer_batch_multiplier,
            generator=generator,
            output=["latents", "timesteps", "latent_height", "latent_width", "in_channels", "video_rotary_emb"],
        )
        record_event(f"prepare_latents_modular_pipe_call{repeat_suffix}", time.time() - event_t0)

        denoise_progress_callback.timesteps = prepare_state["timesteps"]
        denoise_progress_callback.start_time = time.perf_counter()
        denoise_progress_callback.last_time = denoise_progress_callback.start_time
        denoise_progress_callback.step_times = []
        print(f"  Starting denoise loop{repeat_suffix} with attention backend: {ATTENTION_BACKEND}", flush=True)
        event_t0 = time.time()
        with get_attention_backend_context():
            denoise_state = denoise_pipe(
                latents=prepare_state["latents"],
                timesteps=prepare_state["timesteps"],
                connector_prompt_embeds=connector_prompt_embeds.to(device=DEVICE, dtype=DTYPE),
                connector_attention_mask=connector_attention_mask.to(device=DEVICE),
                latent_height=prepare_state["latent_height"],
                latent_width=prepare_state["latent_width"],
                video_rotary_emb=prepare_state["video_rotary_emb"],
                batch_size=latent_batch_size,
                transformer_batch_multiplier=transformer_batch_multiplier,
                do_classifier_free_guidance=False,
                do_perturbed_attention_guidance=do_perturbed_attention_guidance,
                guidance_scale=GUIDANCE_SCALE,
                guidance_rescale=GUIDANCE_RESCALE,
                pag_scale=PAG_SCALE if PAG_ENABLED else 0.0,
                pag_applied_layers=PAG_APPLIED_LAYERS if PAG_ENABLED else None,
                callback_on_step_end=denoise_progress_callback,
                callback_on_step_end_tensor_inputs=["latents"],
                output="latents",
            )
        record_event(
            f"denoise_modular_pipe_call{repeat_suffix}",
            time.time() - event_t0,
            attention_backend=ATTENTION_BACKEND,
            repeat_index=repeat_index,
        )
        run_metrics["denoise_step_times_by_repeat"].append(list(denoise_progress_callback.step_times))
        run_metrics["denoise_step_times"] = denoise_progress_callback.step_times

        if image_latent is not None:
            del image_latent
        image_latent = denoise_state.to(OFFLOAD_DEVICE)
        latent_height = prepare_state["latent_height"]
        latent_width = prepare_state["latent_width"]
        latent_channels = prepare_state["in_channels"]
        if SHOW_METRICS:
            print(f"  Image latent: {image_latent.shape}")
        del prepare_state, denoise_state

    if dynamic_offload_enabled and DYNAMIC_OFFLOAD_EXECUTION_MODE != "plan":
        run_metrics["dynamic_offload_runtime_summary"] = dynamic_offload_hook.state.as_dict()
        if DYNAMIC_OFFLOAD_SHOW_PROFILE:
            dynamic_offload_hook.print_profile_summary()
    del connector_prompt_embeds, connector_attention_mask
    if dynamic_offload_enabled:
        if DYNAMIC_OFFLOAD_EXECUTION_MODE != "plan" and "dynamic_offload_runtime_summary" not in run_metrics:
            run_metrics["dynamic_offload_runtime_summary"] = dynamic_offload_hook.state.as_dict()
        remove_dynamic_offload(transformer)
        dynamic_offload_hook = None
    del prepare_pipe, denoise_pipe, transformer, scheduler
    cleanup_before_vae_decode(record_event)
    step_end(f"Pass 1: Generate at {WIDTH}x{HEIGHT}", t0)
    t0 = step_start("Pass 2: Decode VAE")

    event_t0 = time.time()
    vae = AutoencoderKLLTX2Video.from_pretrained(
        MODEL_PATH,
        subfolder="vae",
        torch_dtype=DTYPE,
        low_cpu_mem_usage=MODEL_LOW_CPU_MEM_USAGE,
    ).to(DEVICE)
    record_event("load_vae_to_cuda", time.time() - event_t0, source=MODEL_PATH)

    event_t0 = time.time()
    decode_pipe = LTX2ImageDistilledBlocks().sub_blocks["decode"].init_pipeline()
    decode_pipe.update_components(vae=vae)
    image = decode_pipe(
        latents=image_latent.to(device=DEVICE, dtype=DTYPE),
        batch_size=latent_batch_size,
        latent_height=latent_height,
        latent_width=latent_width,
        in_channels=latent_channels,
        decode_timestep=DECODE_TIMESTEP,
        decode_noise_scale=DECODE_NOISE_SCALE,
        generator=generator,
        output_type="pil",
        output="images",
    )[0]
    record_event("vae_decode_modular_call", time.time() - event_t0)

    del decode_pipe, vae, image_latent
    cleanup_runtime_state(record_event, "cleanup_after_vae_decode")
    step_end("Pass 2: Decode VAE", t0)

    t0 = step_start("Save Image")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run_slug}.png"
    image.save(output_path)
    print(f"  Image saved successfully to: {output_path}")
    step_end("Save Image", t0)

    total_time = tracker.total_elapsed()
    run_metrics["total_elapsed_sec"] = round(total_time, 4)
    run_metrics["global_peak_vram_gb"] = round(tracker.global_peak_vram, 4)
    run_metrics["global_peak_ram_gb"] = round(tracker.global_peak_ram, 4)

    metrics_path = metrics_dir / f"{run_slug}.json"
    if SAVE_METRICS:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(run_metrics, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"  TOTAL: {total_time:.1f}s | Peak VRAM: {tracker.global_peak_vram:.2f} GB | Peak RAM: {tracker.global_peak_ram:.2f} GB")
    print(f"  Output: {output_path}")
    if SAVE_METRICS:
        print(f"  Metrics JSON: {metrics_path}")
    if RESET_DYNAMIC_MEMORY_AFTER_RUN:
        cleanup_runtime_state(record_event, "reset_dynamic_memory_after_run", collect_cuda_ipc=True)
        if SAVE_METRICS:
            metrics_path.write_text(json.dumps(run_metrics, indent=2), encoding="utf-8")
        print("  Dynamic memory state reset after run.")
    if PURGE_WINDOWS_STANDBY_AFTER_RUN:
        purge_windows_standby_cache_event(record_event, "purge_windows_standby_after_run")
        if SAVE_METRICS:
            metrics_path.write_text(json.dumps(run_metrics, indent=2), encoding="utf-8")
    print("=" * 70)


if __name__ == "__main__":
    main()
