"""
Local low-VRAM parity runner for the experimental LTX 2.3 distilled modular T2I blocks.
"""

import contextlib
import json
import os
from pathlib import Path
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
from custom_blocks.ltx2_image.memory import (
    DynamicWeightsSettings,
    apply_dynamic_weights,
    dynamic_weights_env_names,
    dynamic_weights_preset_env_value,
    from_pretrained_with_dynamic_weights,
    is_wsl_environment,
    remove_dynamic_weights,
)
from inference_utils import RunTracker, flush


def env_value(name: str, default: str = "") -> str:
    for env_name in dynamic_weights_env_names(name):
        if env_name in os.environ:
            return os.environ[env_name]
    return os.environ.get(name, default)


def parse_bool_env(name: str, default: str = "0") -> bool:
    return env_value(name, default).strip().lower() in {"1", "true", "yes", "on"}


RUNNING_ON_WSL = is_wsl_environment()
DEVICE = os.environ.get("LTX_IMAGE_DEVICE", "cuda:0")
OFFLOAD_DEVICE = "cpu"
DTYPE = torch.bfloat16

MODEL_TAG = "distilled_modular"
MODEL_PATH = os.environ.get("LTX_IMAGE_MODEL_PATH", r"E:\model\ltx2.3-image-distilled-1.1")
TEXT_ENCODER_LOW_CPU_MEM_USAGE = True
MODEL_LOW_CPU_MEM_USAGE = parse_bool_env("LTX_IMAGE_LOW_CPU_MEM_USAGE", "1")
DYNAMIC_WEIGHTS_SETTINGS = DynamicWeightsSettings.from_env(
    execution_device=DEVICE,
    offload_device=OFFLOAD_DEVICE,
    target_module_classes=(torch.nn.Linear,),
    always_resident_modules_pattern=(
        r"(^|\.)(proj_in|time_embed|prompt_adaln|norm_out|proj_out)(\.|$)",
    ),
    running_on_wsl=RUNNING_ON_WSL,
    default_preset="auto",
)
REQUESTED_DYNAMIC_WEIGHTS_PRESET = DYNAMIC_WEIGHTS_SETTINGS.requested_preset
DYNAMIC_WEIGHTS_PRESET = DYNAMIC_WEIGHTS_SETTINGS.effective_preset


def preset_env(name: str, default: str = "") -> str:
    return dynamic_weights_preset_env_value(
        name,
        default,
        requested_preset=REQUESTED_DYNAMIC_WEIGHTS_PRESET,
        effective_preset=DYNAMIC_WEIGHTS_PRESET,
        running_on_wsl=RUNNING_ON_WSL,
    )


def parse_bool_preset_env(name: str, default: str = "0") -> bool:
    return preset_env(name, default).strip().lower() in {"1", "true", "yes", "on"}

AUTO_CPU_OFFLOAD = parse_bool_env("LTX_IMAGE_AUTO_CPU_OFFLOAD")
TEXT_ENCODER_GROUP_OFFLOAD = parse_bool_preset_env("LTX_IMAGE_TEXT_ENCODER_GROUP_OFFLOAD", "1")
TRANSFORMER_GROUP_OFFLOAD = parse_bool_preset_env("LTX_IMAGE_TRANSFORMER_GROUP_OFFLOAD")
TRANSFORMER_MEMORY_MANAGER = preset_env("LTX_IMAGE_TRANSFORMER_MEMORY_MANAGER", "off").lower()
DYNAMIC_WEIGHTS_CONFIG = DYNAMIC_WEIGHTS_SETTINGS.config
DYNAMIC_WEIGHTS_PLAN = DYNAMIC_WEIGHTS_SETTINGS.plan
DYNAMIC_WEIGHTS_EXECUTION_MODE = DYNAMIC_WEIGHTS_CONFIG.execution_mode
DYNAMIC_WEIGHTS_PIN_CPU_MEMORY = DYNAMIC_WEIGHTS_SETTINGS.requested_pin_cpu_memory
DYNAMIC_WEIGHTS_LAZY_PIN_CPU_MEMORY = DYNAMIC_WEIGHTS_CONFIG.lazy_pin_cpu_memory
DYNAMIC_WEIGHTS_ALLOW_PIN_MEMORY_FALLBACK = DYNAMIC_WEIGHTS_CONFIG.allow_pin_memory_fallback
DYNAMIC_WEIGHTS_DISABLE_PIN_ON_WSL = DYNAMIC_WEIGHTS_SETTINGS.disable_pin_on_wsl
DYNAMIC_WEIGHTS_PIN_CPU_WORKERS = DYNAMIC_WEIGHTS_CONFIG.pin_cpu_workers
DYNAMIC_WEIGHTS_PIN_WEIGHT_BUDGET_GB = DYNAMIC_WEIGHTS_CONFIG.pin_weight_budget_gb
DYNAMIC_WEIGHTS_PIN_WEIGHT_SELECTION = DYNAMIC_WEIGHTS_CONFIG.pin_weight_selection
DYNAMIC_WEIGHTS_SMALL_TENSOR_THRESHOLD_KB = DYNAMIC_WEIGHTS_CONFIG.small_tensor_threshold_bytes // 1024
DYNAMIC_WEIGHTS_RESIDENT_WEIGHT_BUDGET_GB = DYNAMIC_WEIGHTS_CONFIG.resident_weight_budget_gb
DYNAMIC_WEIGHTS_RESIDENT_WEIGHT_SELECTION = DYNAMIC_WEIGHTS_CONFIG.resident_weight_selection
DYNAMIC_WEIGHTS_RESIDENT_MODULE_BUDGET_GB = DYNAMIC_WEIGHTS_CONFIG.resident_module_budget_gb
DYNAMIC_WEIGHTS_RESIDENT_MODULE_PATTERNS = DYNAMIC_WEIGHTS_CONFIG.resident_module_patterns
DYNAMIC_WEIGHTS_RESIDENT_MODULE_SELECTION = DYNAMIC_WEIGHTS_CONFIG.resident_module_selection
DYNAMIC_WEIGHTS_VERBOSE = DYNAMIC_WEIGHTS_CONFIG.verbose
DYNAMIC_WEIGHTS_EFFECTIVE_PIN_CPU_MEMORY = DYNAMIC_WEIGHTS_SETTINGS.effective_pin_cpu_memory
PRE_VAE_CLEANUP_REPEATS = int(preset_env("LTX_IMAGE_PRE_VAE_CLEANUP_REPEATS", "3" if RUNNING_ON_WSL else "1"))
RESET_DYNAMIC_MEMORY_AFTER_RUN = parse_bool_env("LTX_IMAGE_RESET_DYNAMIC_MEMORY_AFTER_RUN")
PURGE_WINDOWS_STANDBY_BEFORE_RUN = parse_bool_env("LTX_IMAGE_PURGE_WINDOWS_STANDBY_BEFORE_RUN")
PURGE_WINDOWS_STANDBY_BEFORE_TRANSFORMER = parse_bool_env("LTX_IMAGE_PURGE_WINDOWS_STANDBY_BEFORE_TRANSFORMER")
PURGE_WINDOWS_STANDBY_AFTER_RUN = parse_bool_env("LTX_IMAGE_PURGE_WINDOWS_STANDBY_AFTER_RUN")
ATTENTION_BACKEND = preset_env("LTX_IMAGE_ATTENTION_BACKEND", "native").lower()
FLASH_COMPATIBLE_ATTENTION_BACKENDS = {"flash", "flash_hub", "_native_flash", "_flash_3", "_flash_3_hub"}
DROP_TRIVIAL_ATTENTION_MASK = (
    parse_bool_env("LTX_IMAGE_DROP_TRIVIAL_ATTENTION_MASK")
    or ATTENTION_BACKEND in FLASH_COMPATIBLE_ATTENTION_BACKENDS
)
GROUP_OFFLOAD_CONFIG = {
    "mode": "components_manager_auto_cpu_offload" if AUTO_CPU_OFFLOAD else "disabled",
    "device": DEVICE,
    "text_encoder_group_offload": TEXT_ENCODER_GROUP_OFFLOAD,
    "text_encoder_offload_type": preset_env("LTX_IMAGE_TEXT_ENCODER_OFFLOAD_TYPE", "leaf_level"),
    "text_encoder_use_stream": parse_bool_preset_env("LTX_IMAGE_TEXT_ENCODER_OFFLOAD_STREAM", "1"),
    "text_encoder_num_blocks_per_group": int(preset_env("LTX_IMAGE_TEXT_ENCODER_NUM_BLOCKS_PER_GROUP", "1")),
    "transformer_group_offload": TRANSFORMER_GROUP_OFFLOAD,
    "transformer_offload_type": os.environ.get("LTX_IMAGE_TRANSFORMER_OFFLOAD_TYPE", "leaf_level"),
    "transformer_use_stream": parse_bool_env("LTX_IMAGE_TRANSFORMER_OFFLOAD_STREAM", "1"),
    "transformer_num_blocks_per_group": int(os.environ.get("LTX_IMAGE_TRANSFORMER_NUM_BLOCKS_PER_GROUP", "1")),
}

WIDTH = int(os.environ.get("LTX_IMAGE_WIDTH", "1280"))
HEIGHT = int(os.environ.get("LTX_IMAGE_HEIGHT", "704"))
SEED = int(os.environ.get("LTX_IMAGE_SEED", "43"))
NUM_INFERENCE_STEPS = int(os.environ.get("LTX_IMAGE_STEPS", "8"))
GUIDANCE_SCALE = float(os.environ.get("LTX_IMAGE_GUIDANCE_SCALE", "1.0"))
GUIDANCE_RESCALE = float(os.environ.get("LTX_IMAGE_GUIDANCE_RESCALE", "0.7"))
DECODE_TIMESTEP = float(os.environ.get("LTX_IMAGE_DECODE_TIMESTEP", "0.0"))
DECODE_NOISE_SCALE_ENV = os.environ.get("LTX_IMAGE_DECODE_NOISE_SCALE")
DECODE_NOISE_SCALE = None if DECODE_NOISE_SCALE_ENV in (None, "") else float(DECODE_NOISE_SCALE_ENV)
PAG_ENABLED = parse_bool_env("LTX_IMAGE_PAG_ENABLED")
PAG_SCALE = float(os.environ.get("LTX_IMAGE_PAG_SCALE", "0.2"))
PAG_APPLIED_LAYERS = [int(x) for x in os.environ.get("LTX_IMAGE_PAG_LAYERS", "28").split(",") if x]
FAKE_PROMPT_EMBEDS = parse_bool_env("LTX_IMAGE_FAKE_PROMPT")
GENERATION_REPEATS = max(1, int(preset_env("LTX_IMAGE_GENERATION_REPEATS", "1")))

prompt = os.environ.get(
    "LTX_IMAGE_PROMPT",
    "Fisheye close-up of a calico cat wearing a tiny flower crown, sniffing the camera lens in a sunny park, with bright colors, realistic fur detail, and playful viral-pet energy.",
)
negative_prompt = os.environ.get("LTX_IMAGE_NEGATIVE_PROMPT", "")


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
        "low_cpu_mem_usage": TEXT_ENCODER_LOW_CPU_MEM_USAGE if prefix == "text_encoder" else MODEL_LOW_CPU_MEM_USAGE,
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
        raise ValueError(f"Invalid LTX_IMAGE_ATTENTION_BACKEND={ATTENTION_BACKEND!r}. Valid values: {valid}") from exc


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
                if os.environ.get("LTX_IMAGE_LOG_ATTENTION_MASK", "0") == "1":
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


def purge_windows_standby_cache() -> dict:
    if os.name != "nt":
        raise RuntimeError("Windows standby cache purge is only available on Windows.")

    import ctypes
    from ctypes import wintypes

    system_memory_list_information = 80
    memory_purge_standby_list = 4
    token_adjust_privileges = 0x0020
    token_query = 0x0008
    se_privilege_enabled = 0x00000002
    error_not_all_assigned = 1300

    class LUID(ctypes.Structure):
        _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]

    class TOKEN_PRIVILEGES(ctypes.Structure):
        _fields_ = [
            ("PrivilegeCount", wintypes.DWORD),
            ("Luid", LUID),
            ("Attributes", wintypes.DWORD),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    ntdll = ctypes.WinDLL("ntdll")

    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    advapi32.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    advapi32.OpenProcessToken.restype = wintypes.BOOL
    advapi32.LookupPrivilegeValueW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, ctypes.POINTER(LUID)]
    advapi32.LookupPrivilegeValueW.restype = wintypes.BOOL
    advapi32.AdjustTokenPrivileges.argtypes = [
        wintypes.HANDLE,
        wintypes.BOOL,
        ctypes.POINTER(TOKEN_PRIVILEGES),
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    advapi32.AdjustTokenPrivileges.restype = wintypes.BOOL

    ntdll.NtSetSystemInformation.argtypes = [wintypes.ULONG, ctypes.c_void_p, wintypes.ULONG]
    ntdll.NtSetSystemInformation.restype = wintypes.LONG

    token = wintypes.HANDLE()
    process = kernel32.GetCurrentProcess()
    if not advapi32.OpenProcessToken(process, token_adjust_privileges | token_query, ctypes.byref(token)):
        raise ctypes.WinError(ctypes.get_last_error())

    try:
        luid = LUID()
        if not advapi32.LookupPrivilegeValueW(None, "SeProfileSingleProcessPrivilege", ctypes.byref(luid)):
            raise ctypes.WinError(ctypes.get_last_error())

        privileges = TOKEN_PRIVILEGES(1, luid, se_privilege_enabled)
        ctypes.set_last_error(0)
        if not advapi32.AdjustTokenPrivileges(token, False, ctypes.byref(privileges), 0, None, None):
            raise ctypes.WinError(ctypes.get_last_error())
        last_error = ctypes.get_last_error()
        if last_error == error_not_all_assigned:
            raise PermissionError("SeProfileSingleProcessPrivilege is not assigned to this process token.")
    finally:
        kernel32.CloseHandle(token)

    command = ctypes.c_int(memory_purge_standby_list)
    status = ntdll.NtSetSystemInformation(
        system_memory_list_information,
        ctypes.byref(command),
        ctypes.sizeof(command),
    )
    if status != 0:
        raise OSError(f"NtSetSystemInformation failed with NTSTATUS 0x{status & 0xFFFFFFFF:08X}")

    return {
        "system_information_class": system_memory_list_information,
        "command": memory_purge_standby_list,
        "privilege": "SeProfileSingleProcessPrivilege",
    }


def purge_windows_standby_cache_event(record_event, event_name: str) -> None:
    event_t0 = time.time()
    result = purge_windows_standby_cache()
    record_event(event_name, time.time() - event_t0, **result)
    print(f"  [windows-memory] {event_name}: standby cache purge requested", flush=True)


def cleanup_before_vae_decode(record_event) -> None:
    if PRE_VAE_CLEANUP_REPEATS <= 0:
        return

    before_free_gb = before_total_gb = None
    after_free_gb = after_total_gb = None
    if torch.cuda.is_available():
        before_free, before_total = torch.cuda.mem_get_info(DEVICE)
        before_free_gb = before_free / 1024**3
        before_total_gb = before_total / 1024**3

    event_t0 = time.time()
    for _ in range(PRE_VAE_CLEANUP_REPEATS):
        flush()
        if torch.cuda.is_available() and hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()

    if torch.cuda.is_available():
        after_free, after_total = torch.cuda.mem_get_info(DEVICE)
        after_free_gb = after_free / 1024**3
        after_total_gb = after_total / 1024**3

    record_event(
        "cleanup_before_vae_decode",
        time.time() - event_t0,
        repeats=PRE_VAE_CLEANUP_REPEATS,
        before_free_vram_gb=None if before_free_gb is None else round(before_free_gb, 4),
        before_total_vram_gb=None if before_total_gb is None else round(before_total_gb, 4),
        after_free_vram_gb=None if after_free_gb is None else round(after_free_gb, 4),
        after_total_vram_gb=None if after_total_gb is None else round(after_total_gb, 4),
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
    print(
        f"  [denoise] step {step_index + 1}/{total_steps} timestep={float(timestep):.4f} "
        f"elapsed={step_elapsed:.4f}s avg={avg_elapsed:.4f}s/it "
        f"torch_alloc={used_gb:.2f} GiB torch_reserved={reserved_gb:.2f} GiB",
        flush=True,
    )
    return callback_kwargs


def main():
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
        "dynamic_weights_requested_preset": REQUESTED_DYNAMIC_WEIGHTS_PRESET or None,
        "dynamic_weights_effective_preset": DYNAMIC_WEIGHTS_PRESET or None,
        "guidance_scale": GUIDANCE_SCALE,
        "guidance_rescale": GUIDANCE_RESCALE,
        "vae_decode_timestep": DECODE_TIMESTEP,
        "vae_decode_noise_scale": DECODE_NOISE_SCALE,
        "pag_enabled": PAG_ENABLED,
        "pag_scale": PAG_SCALE if PAG_ENABLED else 0.0,
        "pag_applied_layers": PAG_APPLIED_LAYERS if PAG_ENABLED else None,
        "dtype": str(DTYPE),
        "text_encoder_low_cpu_mem_usage": TEXT_ENCODER_LOW_CPU_MEM_USAGE,
        "model_low_cpu_mem_usage": MODEL_LOW_CPU_MEM_USAGE,
        "running_on_wsl": RUNNING_ON_WSL,
        "reset_dynamic_memory_after_run": RESET_DYNAMIC_MEMORY_AFTER_RUN,
        "purge_windows_standby_before_run": PURGE_WINDOWS_STANDBY_BEFORE_RUN,
        "purge_windows_standby_before_transformer": PURGE_WINDOWS_STANDBY_BEFORE_TRANSFORMER,
        "purge_windows_standby_after_run": PURGE_WINDOWS_STANDBY_AFTER_RUN,
        "group_offload_config": GROUP_OFFLOAD_CONFIG.copy(),
        "transformer_memory_manager": TRANSFORMER_MEMORY_MANAGER,
        "dynamic_weights_execution_mode": DYNAMIC_WEIGHTS_EXECUTION_MODE,
        "dynamic_weights_pin_cpu_memory": DYNAMIC_WEIGHTS_PIN_CPU_MEMORY,
        "dynamic_weights_effective_pin_cpu_memory": DYNAMIC_WEIGHTS_EFFECTIVE_PIN_CPU_MEMORY,
        "dynamic_weights_lazy_pin_cpu_memory": DYNAMIC_WEIGHTS_LAZY_PIN_CPU_MEMORY,
        "dynamic_weights_allow_pin_memory_fallback": DYNAMIC_WEIGHTS_ALLOW_PIN_MEMORY_FALLBACK,
        "dynamic_weights_disable_pin_on_wsl": DYNAMIC_WEIGHTS_DISABLE_PIN_ON_WSL,
        "dynamic_weights_pin_cpu_workers": DYNAMIC_WEIGHTS_PIN_CPU_WORKERS,
        "dynamic_weights_pin_weight_budget_gb": DYNAMIC_WEIGHTS_PIN_WEIGHT_BUDGET_GB,
        "dynamic_weights_pin_weight_selection": DYNAMIC_WEIGHTS_PIN_WEIGHT_SELECTION,
        "dynamic_weights_small_tensor_threshold_kb": DYNAMIC_WEIGHTS_SMALL_TENSOR_THRESHOLD_KB,
        "dynamic_weights_resident_weight_budget_gb": DYNAMIC_WEIGHTS_RESIDENT_WEIGHT_BUDGET_GB,
        "dynamic_weights_resident_weight_selection": DYNAMIC_WEIGHTS_RESIDENT_WEIGHT_SELECTION,
        "dynamic_weights_resident_module_budget_gb": DYNAMIC_WEIGHTS_RESIDENT_MODULE_BUDGET_GB,
        "dynamic_weights_resident_module_patterns": DYNAMIC_WEIGHTS_RESIDENT_MODULE_PATTERNS,
        "dynamic_weights_resident_module_selection": DYNAMIC_WEIGHTS_RESIDENT_MODULE_SELECTION,
        "pre_vae_cleanup_repeats": PRE_VAE_CLEANUP_REPEATS,
        "attention_backend": ATTENTION_BACKEND,
        "drop_trivial_attention_mask": DROP_TRIVIAL_ATTENTION_MASK,
        "events": [],
        "steps": [],
    }

    flash_mask_wrapper_installed = install_trivial_mask_flash_wrapper()
    run_metrics["flash_trivial_mask_wrapper_installed"] = flash_mask_wrapper_installed
    if DYNAMIC_WEIGHTS_PRESET:
        if REQUESTED_DYNAMIC_WEIGHTS_PRESET == "auto":
            print(f"Using dynamic weights preset: auto -> {DYNAMIC_WEIGHTS_PRESET}", flush=True)
        else:
            print(f"Using dynamic weights preset: {DYNAMIC_WEIGHTS_PRESET}", flush=True)
    if DYNAMIC_WEIGHTS_PIN_CPU_MEMORY and not DYNAMIC_WEIGHTS_EFFECTIVE_PIN_CPU_MEMORY:
        print("  [dynamic-weights] disabling pinned CPU memory on WSL; set LTX_IMAGE_DYNAMIC_WEIGHTS_DISABLE_PIN_ON_WSL=0 to force it.", flush=True)

    tracker = RunTracker(DEVICE, run_metrics, interval=0.1)
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
        if TEXT_ENCODER_GROUP_OFFLOAD:
            apply_model_group_offload(text_encoder, prefix="text_encoder")
            record_event(
                "setup_text_encoder_group_offload",
                time.time() - event_t0,
                offload_type=GROUP_OFFLOAD_CONFIG["text_encoder_offload_type"],
                use_stream=GROUP_OFFLOAD_CONFIG["text_encoder_use_stream"],
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
        del prompt_pipe, text_encoder, tokenizer
        flush()

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
    flush()
    record_event("offload_connector_outputs", 0.0)

    transformer_load_kwargs = {
        "subfolder": "transformer",
        "torch_dtype": DTYPE,
        "low_cpu_mem_usage": MODEL_LOW_CPU_MEM_USAGE,
    }

    dynamic_weights_enabled = DYNAMIC_WEIGHTS_SETTINGS.enabled
    dynamic_weights_config = DYNAMIC_WEIGHTS_CONFIG if dynamic_weights_enabled else None
    if PURGE_WINDOWS_STANDBY_BEFORE_TRANSFORMER:
        purge_windows_standby_cache_event(record_event, "purge_windows_standby_before_transformer")

    event_t0 = time.time()
    if MODEL_LOW_CPU_MEM_USAGE:
        transformer_load_kwargs["device_map"] = "cpu"
    transformer_load = from_pretrained_with_dynamic_weights(
        MODEL_PATH,
        dynamic_weights_config=dynamic_weights_config,
        apply_dynamic=False,
        **transformer_load_kwargs,
    )
    transformer = transformer_load.module
    dynamic_weights_hook = transformer_load.hook
    record_event(
        "load_transformer",
        time.time() - event_t0,
        source=MODEL_PATH,
        low_cpu_mem_usage=MODEL_LOW_CPU_MEM_USAGE,
        device_map=transformer_load_kwargs.get("device_map"),
        loader="AutoModel",
        resolved_class=transformer.__class__.__name__,
    )

    event_t0 = time.time()
    patched_attention_processors, resolved_attention_backend, drop_trivial_attention_mask = apply_transformer_attention_backend(transformer)
    record_event(
        "set_transformer_attention_backend",
        time.time() - event_t0,
        requested_backend=ATTENTION_BACKEND,
        resolved_backend=resolved_attention_backend,
        patched_processors=patched_attention_processors,
        drop_trivial_attention_mask=drop_trivial_attention_mask,
    )

    if dynamic_weights_enabled and DYNAMIC_WEIGHTS_EXECUTION_MODE != "plan" and TRANSFORMER_MEMORY_MANAGER != "off":
        raise ValueError("Dynamic weights execution currently requires LTX_IMAGE_TRANSFORMER_MEMORY_MANAGER='off'. Use execution_mode='plan' with the block manager.")

    if dynamic_weights_enabled:
        event_t0 = time.time()
        dynamic_weights_hook = apply_dynamic_weights(transformer, dynamic_weights_config)
        dynamic_weights_summary = dynamic_weights_hook.state.as_dict()
        record_event(
            "build_dynamic_weights_plan",
            time.time() - event_t0,
            module_count=dynamic_weights_summary["module_count"],
            total_gb=dynamic_weights_summary["total_gb"],
            bytes_by_placement=dynamic_weights_summary["bytes_by_placement"],
            execution_mode=DYNAMIC_WEIGHTS_EXECUTION_MODE,
            pin_cpu_memory=DYNAMIC_WEIGHTS_EFFECTIVE_PIN_CPU_MEMORY,
            lazy_pin_cpu_memory=DYNAMIC_WEIGHTS_LAZY_PIN_CPU_MEMORY,
            allow_pin_memory_fallback=DYNAMIC_WEIGHTS_ALLOW_PIN_MEMORY_FALLBACK,
            pin_weight_budget_gb=DYNAMIC_WEIGHTS_PIN_WEIGHT_BUDGET_GB,
            pin_weight_selection=DYNAMIC_WEIGHTS_PIN_WEIGHT_SELECTION,
            patched_module_count=dynamic_weights_summary["patched_module_count"],
            resolved_resident_module_patterns=dynamic_weights_summary["resolved_resident_module_patterns"],
            setup_runtime=dynamic_weights_summary["setup_runtime"],
        )

    event_t0 = time.time()
    if TRANSFORMER_MEMORY_MANAGER != "off":
        raise ValueError(
            "The legacy transformer block manager is no longer used by this runner. "
            "Use DIFFUSERS_DYNAMIC_WEIGHTS_PRESET/LTX_IMAGE_DYNAMIC_WEIGHTS_PRESET or transformer group offload."
        )
    if TRANSFORMER_GROUP_OFFLOAD:
        apply_model_group_offload(transformer, prefix="transformer")
        record_event(
            "setup_transformer_group_offload",
            time.time() - event_t0,
            offload_type=GROUP_OFFLOAD_CONFIG["transformer_offload_type"],
            use_stream=GROUP_OFFLOAD_CONFIG["transformer_use_stream"],
            low_cpu_mem_usage=MODEL_LOW_CPU_MEM_USAGE,
        )
    elif dynamic_weights_enabled and DYNAMIC_WEIGHTS_EXECUTION_MODE != "plan":
        record_event(
            "skip_transformer_to_cuda",
            time.time() - event_t0,
            reason=f"dynamic_weights_{DYNAMIC_WEIGHTS_EXECUTION_MODE}",
        )
    else:
        transformer.to(DEVICE)
        record_event("load_transformer_to_cuda", time.time() - event_t0)

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
        print(f"  Image latent: {image_latent.shape}")
        del prepare_state, denoise_state

    if dynamic_weights_enabled and DYNAMIC_WEIGHTS_EXECUTION_MODE != "plan":
        run_metrics["dynamic_weights_runtime_summary"] = dynamic_weights_hook.state.as_dict()
        dynamic_weights_hook.print_profile_summary()
    del connector_prompt_embeds, connector_attention_mask
    if dynamic_weights_enabled:
        if DYNAMIC_WEIGHTS_EXECUTION_MODE != "plan" and "dynamic_weights_runtime_summary" not in run_metrics:
            run_metrics["dynamic_weights_runtime_summary"] = dynamic_weights_hook.state.as_dict()
        remove_dynamic_weights(transformer)
        dynamic_weights_hook = None
    del prepare_pipe, denoise_pipe, transformer, scheduler
    cleanup_before_vae_decode(record_event)
    flush()
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
    flush()
    step_end("Pass 2: Decode VAE", t0)

    t0 = step_start("Save Image")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run_slug}.png"
    image.save(output_path)
    print(f"  Image saved successfully to: {output_path}")
    step_end("Save Image", t0)

    total_time = tracker.total_elapsed()
    run_metrics["total_elapsed_sec"] = round(total_time, 4)
    run_metrics["global_peak_vram_gb"] = round(tracker.global_peak_vram, 4)
    run_metrics["global_peak_ram_gb"] = round(tracker.global_peak_ram, 4)

    metrics_path = metrics_dir / f"{run_slug}.json"
    metrics_path.write_text(json.dumps(run_metrics, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"  TOTAL: {total_time:.1f}s | Peak VRAM: {tracker.global_peak_vram:.2f} GB | Peak RAM: {tracker.global_peak_ram:.2f} GB")
    print(f"  Output: {output_path}")
    print(f"  Metrics JSON: {metrics_path}")
    if RESET_DYNAMIC_MEMORY_AFTER_RUN:
        event_t0 = time.time()
        flush()
        if torch.cuda.is_available() and hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
        record_event("reset_dynamic_memory_after_run", time.time() - event_t0)
        metrics_path.write_text(json.dumps(run_metrics, indent=2), encoding="utf-8")
        print("  Dynamic memory state reset after run.")
    if PURGE_WINDOWS_STANDBY_AFTER_RUN:
        purge_windows_standby_cache_event(record_event, "purge_windows_standby_after_run")
        metrics_path.write_text(json.dumps(run_metrics, indent=2), encoding="utf-8")
    print("=" * 70)


if __name__ == "__main__":
    main()
