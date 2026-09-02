"""
Local low-VRAM parity runner for the experimental LTX 2.3 distilled modular T2I blocks.
"""

import contextlib
import json
import os
from pathlib import Path
import time

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
from custom_blocks.ltx2_image.memory import DynamicWeightsConfig, apply_dynamic_weights, remove_dynamic_weights
from custom_blocks.ltx2_image.memory_manager import LTX2DynamicBlockManager
from custom_blocks.ltx2_image.transformer_ltx2_image import LTX2ImageTransformer2DModel
from inference_utils import RunTracker, flush


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def parse_int_list_env(name: str) -> tuple[int, ...]:
    value = os.environ.get(name, "").strip()
    if not value:
        return ()
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_bool_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


DEVICE = os.environ.get("LTX_IMAGE_DEVICE", "cuda:0")
OFFLOAD_DEVICE = "cpu"
DTYPE = torch.bfloat16

MODEL_TAG = "distilled_modular"
MODEL_PATH = os.environ.get("LTX_IMAGE_MODEL_PATH", r"E:\model\ltx2.3-image-distilled-1.1")
TEXT_ENCODER_LOW_CPU_MEM_USAGE = True
MODEL_LOW_CPU_MEM_USAGE = parse_bool_env("LTX_IMAGE_LOW_CPU_MEM_USAGE", "1")
AUTO_CPU_OFFLOAD = parse_bool_env("LTX_IMAGE_AUTO_CPU_OFFLOAD")
TEXT_ENCODER_GROUP_OFFLOAD = parse_bool_env("LTX_IMAGE_TEXT_ENCODER_GROUP_OFFLOAD", "1")
TRANSFORMER_GROUP_OFFLOAD = parse_bool_env("LTX_IMAGE_TRANSFORMER_GROUP_OFFLOAD")
TRANSFORMER_MEMORY_MANAGER = os.environ.get("LTX_IMAGE_TRANSFORMER_MEMORY_MANAGER", "manual_linear").lower()
TRANSFORMER_MANAGER_PINNED_BLOCKS = int(os.environ.get("LTX_IMAGE_TRANSFORMER_MANAGER_PINNED_BLOCKS", "0"))
TRANSFORMER_MANAGER_HOT_BLOCKS = parse_int_list_env("LTX_IMAGE_TRANSFORMER_HOT_BLOCKS")
TRANSFORMER_MANAGER_HOT_BLOCK_CANDIDATES = parse_int_list_env("LTX_IMAGE_TRANSFORMER_HOT_BLOCK_CANDIDATES")
TRANSFORMER_MANAGER_HOT_BLOCK_BUDGET_GB = float(os.environ.get("LTX_IMAGE_TRANSFORMER_HOT_BLOCK_BUDGET_GB", "0.0"))
TRANSFORMER_MANAGER_HOT_BLOCK_STRIDE = int(os.environ.get("LTX_IMAGE_TRANSFORMER_HOT_BLOCK_STRIDE", "3"))
TRANSFORMER_MANAGER_HOT_BLOCK_OFFSET = int(os.environ.get("LTX_IMAGE_TRANSFORMER_HOT_BLOCK_OFFSET", "0"))
TRANSFORMER_MANAGER_HOT_BLOCK_SELECTION = os.environ.get("LTX_IMAGE_TRANSFORMER_HOT_BLOCK_SELECTION", "stride").lower()
TRANSFORMER_MANAGER_HOT_LINEAR_WEIGHT_BUDGET_GB = float(os.environ.get("LTX_IMAGE_TRANSFORMER_HOT_LINEAR_WEIGHT_BUDGET_GB", "0.0"))
TRANSFORMER_MANAGER_HOT_LINEAR_WEIGHT_STRIDE = int(os.environ.get("LTX_IMAGE_TRANSFORMER_HOT_LINEAR_WEIGHT_STRIDE", "3"))
TRANSFORMER_MANAGER_HOT_LINEAR_WEIGHT_OFFSET = int(os.environ.get("LTX_IMAGE_TRANSFORMER_HOT_LINEAR_WEIGHT_OFFSET", "1"))
TRANSFORMER_MANAGER_STREAMED_COPY_MODE = os.environ.get("LTX_IMAGE_TRANSFORMER_STREAMED_COPY_MODE", "direct").lower()
TRANSFORMER_MANAGER_KEEP_STREAMED_SMALL_TENSORS_RESIDENT = (
    os.environ.get("LTX_IMAGE_TRANSFORMER_KEEP_STREAMED_SMALL_TENSORS_RESIDENT", "0") == "1"
)
TRANSFORMER_MANAGER_SYNCHRONIZE = parse_bool_env("LTX_IMAGE_TRANSFORMER_MANAGER_SYNCHRONIZE")
TRANSFORMER_MANAGER_EMPTY_CACHE = parse_bool_env("LTX_IMAGE_TRANSFORMER_MANAGER_EMPTY_CACHE")
TRANSFORMER_MANAGER_VERBOSE = parse_bool_env("LTX_IMAGE_TRANSFORMER_MANAGER_VERBOSE")
TRANSFORMER_MANAGER_WEIGHT_CACHE_GB = float(os.environ.get("LTX_IMAGE_TRANSFORMER_WEIGHT_CACHE_GB", "0.0"))
TRANSFORMER_MANAGER_PIN_CPU_MEMORY = parse_bool_env("LTX_IMAGE_TRANSFORMER_PIN_CPU_MEMORY")
TRANSFORMER_MANAGER_LAZY_PIN_CPU_MEMORY = parse_bool_env("LTX_IMAGE_TRANSFORMER_LAZY_PIN_CPU_MEMORY")
TRANSFORMER_MANAGER_PIN_CPU_WORKERS = int(os.environ.get("LTX_IMAGE_TRANSFORMER_PIN_CPU_WORKERS", "1"))
TRANSFORMER_MANAGER_SLIDING_WINDOW_SIZE = int(os.environ.get("LTX_IMAGE_TRANSFORMER_SLIDING_WINDOW_SIZE", "0"))
TRANSFORMER_MANAGER_PROFILE = parse_bool_env("LTX_IMAGE_TRANSFORMER_MANAGER_PROFILE")
TRANSFORMER_MANAGER_PROFILE_SYNC_COPIES = parse_bool_env("LTX_IMAGE_TRANSFORMER_MANAGER_PROFILE_SYNC_COPIES")
TRANSFORMER_MANAGER_PROFILE_LAYERS = parse_bool_env("LTX_IMAGE_TRANSFORMER_PROFILE_LAYERS")
TRANSFORMER_MANAGER_PROFILE_SYNC_LAYERS = parse_bool_env("LTX_IMAGE_TRANSFORMER_PROFILE_SYNC_LAYERS")
TRANSFORMER_MANAGER_PROFILE_FULL = parse_bool_env("LTX_IMAGE_TRANSFORMER_MANAGER_PROFILE_FULL")
DYNAMIC_WEIGHTS_PLAN = parse_bool_env("LTX_IMAGE_DYNAMIC_WEIGHTS_PLAN")
DYNAMIC_WEIGHTS_EXECUTION_MODE = os.environ.get("LTX_IMAGE_DYNAMIC_WEIGHTS_EXECUTION_MODE", "plan").lower()
DYNAMIC_WEIGHTS_PIN_CPU_MEMORY = parse_bool_env("LTX_IMAGE_DYNAMIC_WEIGHTS_PIN_CPU_MEMORY")
DYNAMIC_WEIGHTS_PIN_CPU_WORKERS = int(os.environ.get("LTX_IMAGE_DYNAMIC_WEIGHTS_PIN_CPU_WORKERS", "4"))
DYNAMIC_WEIGHTS_VERBOSE = parse_bool_env("LTX_IMAGE_DYNAMIC_WEIGHTS_VERBOSE", "1")
RESET_DYNAMIC_MEMORY_AFTER_RUN = parse_bool_env("LTX_IMAGE_RESET_DYNAMIC_MEMORY_AFTER_RUN")
PURGE_WINDOWS_STANDBY_BEFORE_RUN = parse_bool_env("LTX_IMAGE_PURGE_WINDOWS_STANDBY_BEFORE_RUN")
PURGE_WINDOWS_STANDBY_BEFORE_TRANSFORMER = parse_bool_env("LTX_IMAGE_PURGE_WINDOWS_STANDBY_BEFORE_TRANSFORMER")
PURGE_WINDOWS_STANDBY_AFTER_RUN = parse_bool_env("LTX_IMAGE_PURGE_WINDOWS_STANDBY_AFTER_RUN")
ATTENTION_BACKEND = os.environ.get("LTX_IMAGE_ATTENTION_BACKEND", "native").lower()
FLASH_COMPATIBLE_ATTENTION_BACKENDS = {"flash", "flash_hub", "_native_flash", "_flash_3", "_flash_3_hub"}
DROP_TRIVIAL_ATTENTION_MASK = (
    parse_bool_env("LTX_IMAGE_DROP_TRIVIAL_ATTENTION_MASK")
    or ATTENTION_BACKEND in FLASH_COMPATIBLE_ATTENTION_BACKENDS
)
GROUP_OFFLOAD_CONFIG = {
    "mode": "components_manager_auto_cpu_offload" if AUTO_CPU_OFFLOAD else "disabled",
    "device": DEVICE,
    "text_encoder_group_offload": TEXT_ENCODER_GROUP_OFFLOAD,
    "text_encoder_offload_type": os.environ.get("LTX_IMAGE_TEXT_ENCODER_OFFLOAD_TYPE", "leaf_level"),
    "text_encoder_use_stream": parse_bool_env("LTX_IMAGE_TEXT_ENCODER_OFFLOAD_STREAM", "1"),
    "text_encoder_num_blocks_per_group": int(os.environ.get("LTX_IMAGE_TEXT_ENCODER_NUM_BLOCKS_PER_GROUP", "1")),
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
        "reset_dynamic_memory_after_run": RESET_DYNAMIC_MEMORY_AFTER_RUN,
        "purge_windows_standby_before_run": PURGE_WINDOWS_STANDBY_BEFORE_RUN,
        "purge_windows_standby_before_transformer": PURGE_WINDOWS_STANDBY_BEFORE_TRANSFORMER,
        "purge_windows_standby_after_run": PURGE_WINDOWS_STANDBY_AFTER_RUN,
        "group_offload_config": GROUP_OFFLOAD_CONFIG.copy(),
        "transformer_memory_manager": TRANSFORMER_MEMORY_MANAGER,
        "transformer_manager_pinned_blocks": TRANSFORMER_MANAGER_PINNED_BLOCKS,
        "transformer_manager_weight_cache_gb": TRANSFORMER_MANAGER_WEIGHT_CACHE_GB,
        "transformer_manager_streamed_copy_mode": TRANSFORMER_MANAGER_STREAMED_COPY_MODE,
        "transformer_manager_keep_streamed_small_tensors_resident": TRANSFORMER_MANAGER_KEEP_STREAMED_SMALL_TENSORS_RESIDENT,
        "transformer_manager_pin_cpu_memory": TRANSFORMER_MANAGER_PIN_CPU_MEMORY,
        "transformer_manager_pin_cpu_workers": TRANSFORMER_MANAGER_PIN_CPU_WORKERS,
        "transformer_manager_sliding_window_size": TRANSFORMER_MANAGER_SLIDING_WINDOW_SIZE,
        "transformer_manager_profile_enabled": TRANSFORMER_MANAGER_PROFILE,
        "transformer_manager_profile_sync_copies": TRANSFORMER_MANAGER_PROFILE_SYNC_COPIES,
        "transformer_manager_profile_layers": TRANSFORMER_MANAGER_PROFILE_LAYERS,
        "transformer_manager_profile_sync_layers": TRANSFORMER_MANAGER_PROFILE_SYNC_LAYERS,
        "transformer_manager_profile_full": TRANSFORMER_MANAGER_PROFILE_FULL,
        "attention_backend": ATTENTION_BACKEND,
        "drop_trivial_attention_mask": DROP_TRIVIAL_ATTENTION_MASK,
        "events": [],
        "steps": [],
    }

    flash_mask_wrapper_installed = install_trivial_mask_flash_wrapper()
    run_metrics["flash_trivial_mask_wrapper_installed"] = flash_mask_wrapper_installed

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

    if PURGE_WINDOWS_STANDBY_BEFORE_TRANSFORMER:
        purge_windows_standby_cache_event(record_event, "purge_windows_standby_before_transformer")

    event_t0 = time.time()
    if MODEL_LOW_CPU_MEM_USAGE:
        transformer_load_kwargs["device_map"] = "cpu"
    transformer = LTX2ImageTransformer2DModel.from_pretrained(MODEL_PATH, **transformer_load_kwargs)
    record_event(
        "load_transformer",
        time.time() - event_t0,
        source=MODEL_PATH,
        low_cpu_mem_usage=MODEL_LOW_CPU_MEM_USAGE,
        device_map=transformer_load_kwargs.get("device_map"),
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

    dynamic_weights_enabled = DYNAMIC_WEIGHTS_PLAN or DYNAMIC_WEIGHTS_EXECUTION_MODE != "plan"
    if dynamic_weights_enabled and DYNAMIC_WEIGHTS_EXECUTION_MODE != "plan" and TRANSFORMER_MEMORY_MANAGER != "off":
        raise ValueError("Dynamic weights execution currently requires LTX_IMAGE_TRANSFORMER_MEMORY_MANAGER='off'. Use execution_mode='plan' with the block manager.")

    if dynamic_weights_enabled:
        event_t0 = time.time()
        dynamic_weights_hook = apply_dynamic_weights(
            transformer,
            DynamicWeightsConfig(
                execution_device=DEVICE,
                offload_device=OFFLOAD_DEVICE,
                target_module_classes=(torch.nn.Linear,),
                always_resident_modules_pattern=(
                    r"(^|\.)(proj_in|time_embed|prompt_adaln|norm_out|proj_out)(\.|$)",
                ),
                execution_mode=DYNAMIC_WEIGHTS_EXECUTION_MODE,
                pin_cpu_memory=DYNAMIC_WEIGHTS_PIN_CPU_MEMORY,
                pin_cpu_workers=DYNAMIC_WEIGHTS_PIN_CPU_WORKERS,
                verbose=DYNAMIC_WEIGHTS_VERBOSE,
            ),
        )
        dynamic_weights_summary = dynamic_weights_hook.state.as_dict()
        record_event(
            "build_dynamic_weights_plan",
            time.time() - event_t0,
            module_count=dynamic_weights_summary["module_count"],
            total_gb=dynamic_weights_summary["total_gb"],
            bytes_by_placement=dynamic_weights_summary["bytes_by_placement"],
            execution_mode=DYNAMIC_WEIGHTS_EXECUTION_MODE,
            patched_module_count=dynamic_weights_summary["patched_module_count"],
            setup_runtime=dynamic_weights_summary["setup_runtime"],
        )

    event_t0 = time.time()
    transformer_manager = None
    if TRANSFORMER_MEMORY_MANAGER != "off":
        if TRANSFORMER_GROUP_OFFLOAD:
            print("  Disabling transformer group offload because transformer memory manager is enabled.", flush=True)
        transformer_manager = LTX2DynamicBlockManager(
            device=DEVICE,
            offload_device=OFFLOAD_DEVICE,
            enabled=True,
            mode=TRANSFORMER_MEMORY_MANAGER,
            pinned_blocks=TRANSFORMER_MANAGER_PINNED_BLOCKS,
            hot_blocks=TRANSFORMER_MANAGER_HOT_BLOCKS,
            hot_block_candidates=TRANSFORMER_MANAGER_HOT_BLOCK_CANDIDATES,
            hot_block_budget_gb=TRANSFORMER_MANAGER_HOT_BLOCK_BUDGET_GB,
            hot_block_stride=TRANSFORMER_MANAGER_HOT_BLOCK_STRIDE,
            hot_block_offset=TRANSFORMER_MANAGER_HOT_BLOCK_OFFSET,
            hot_block_selection=TRANSFORMER_MANAGER_HOT_BLOCK_SELECTION,
            hot_linear_weight_budget_gb=TRANSFORMER_MANAGER_HOT_LINEAR_WEIGHT_BUDGET_GB,
            hot_linear_weight_stride=TRANSFORMER_MANAGER_HOT_LINEAR_WEIGHT_STRIDE,
            hot_linear_weight_offset=TRANSFORMER_MANAGER_HOT_LINEAR_WEIGHT_OFFSET,
            streamed_copy_mode=TRANSFORMER_MANAGER_STREAMED_COPY_MODE,
            keep_streamed_small_tensors_resident=TRANSFORMER_MANAGER_KEEP_STREAMED_SMALL_TENSORS_RESIDENT,
            synchronize=TRANSFORMER_MANAGER_SYNCHRONIZE,
            empty_cache_after_offload=TRANSFORMER_MANAGER_EMPTY_CACHE,
            verbose=TRANSFORMER_MANAGER_VERBOSE,
            weight_cache_gb=TRANSFORMER_MANAGER_WEIGHT_CACHE_GB,
            pin_cpu_memory=TRANSFORMER_MANAGER_PIN_CPU_MEMORY,
            lazy_pin_cpu_memory=TRANSFORMER_MANAGER_LAZY_PIN_CPU_MEMORY,
            pin_cpu_workers=TRANSFORMER_MANAGER_PIN_CPU_WORKERS,
            sliding_window_size=TRANSFORMER_MANAGER_SLIDING_WINDOW_SIZE,
            profile=TRANSFORMER_MANAGER_PROFILE,
            profile_sync_copies=TRANSFORMER_MANAGER_PROFILE_SYNC_COPIES,
            profile_layer_runtime=TRANSFORMER_MANAGER_PROFILE_LAYERS,
            profile_sync_layers=TRANSFORMER_MANAGER_PROFILE_SYNC_LAYERS,
        )
        transformer_manager.attach(transformer)
        record_event(
            "setup_transformer_memory_manager",
            time.time() - event_t0,
            mode=TRANSFORMER_MEMORY_MANAGER,
            pinned_blocks=TRANSFORMER_MANAGER_PINNED_BLOCKS,
            hot_blocks=TRANSFORMER_MANAGER_HOT_BLOCKS,
            hot_block_candidates=TRANSFORMER_MANAGER_HOT_BLOCK_CANDIDATES,
            hot_block_budget_gb=TRANSFORMER_MANAGER_HOT_BLOCK_BUDGET_GB,
            hot_block_stride=TRANSFORMER_MANAGER_HOT_BLOCK_STRIDE,
            hot_block_offset=TRANSFORMER_MANAGER_HOT_BLOCK_OFFSET,
            hot_block_selection=TRANSFORMER_MANAGER_HOT_BLOCK_SELECTION,
            hot_linear_weight_budget_gb=TRANSFORMER_MANAGER_HOT_LINEAR_WEIGHT_BUDGET_GB,
            hot_linear_weight_stride=TRANSFORMER_MANAGER_HOT_LINEAR_WEIGHT_STRIDE,
            hot_linear_weight_offset=TRANSFORMER_MANAGER_HOT_LINEAR_WEIGHT_OFFSET,
            streamed_copy_mode=TRANSFORMER_MANAGER_STREAMED_COPY_MODE,
            keep_streamed_small_tensors_resident=TRANSFORMER_MANAGER_KEEP_STREAMED_SMALL_TENSORS_RESIDENT,
            selected_hot_blocks=transformer_manager.selected_hot_blocks,
            synchronize=TRANSFORMER_MANAGER_SYNCHRONIZE,
            empty_cache_after_offload=TRANSFORMER_MANAGER_EMPTY_CACHE,
            weight_cache_gb=TRANSFORMER_MANAGER_WEIGHT_CACHE_GB,
            pin_cpu_memory=TRANSFORMER_MANAGER_PIN_CPU_MEMORY,
            lazy_pin_cpu_memory=TRANSFORMER_MANAGER_LAZY_PIN_CPU_MEMORY,
            pin_cpu_workers=TRANSFORMER_MANAGER_PIN_CPU_WORKERS,
            sliding_window_size=TRANSFORMER_MANAGER_SLIDING_WINDOW_SIZE,
            profile=TRANSFORMER_MANAGER_PROFILE,
        )
    elif TRANSFORMER_GROUP_OFFLOAD:
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
    prepare_state = prepare_pipe(
        width=WIDTH,
        height=HEIGHT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        batch_size=latent_batch_size,
        transformer_batch_multiplier=transformer_batch_multiplier,
        generator=generator,
        output=["latents", "timesteps", "latent_height", "latent_width", "in_channels", "video_rotary_emb"],
    )
    record_event("prepare_latents_modular_pipe_call", time.time() - event_t0)

    event_t0 = time.time()
    denoise_pipe = LTX2ImageDenoiseStep().init_pipeline()
    denoise_pipe.update_components(transformer=transformer, scheduler=scheduler)
    record_event("build_denoise_modular_pipeline", time.time() - event_t0, model_path=MODEL_PATH)

    denoise_progress_callback.timesteps = prepare_state["timesteps"]
    denoise_progress_callback.start_time = time.perf_counter()
    denoise_progress_callback.last_time = denoise_progress_callback.start_time
    denoise_progress_callback.step_times = []
    print(f"  Starting denoise loop with attention backend: {ATTENTION_BACKEND}", flush=True)
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
    record_event("denoise_modular_pipe_call", time.time() - event_t0, attention_backend=ATTENTION_BACKEND)
    run_metrics["denoise_step_times"] = denoise_progress_callback.step_times
    if transformer_manager is not None and TRANSFORMER_MANAGER_PROFILE:
        run_metrics["transformer_manager_profile_summary"] = transformer_manager.profile_summary()
        transformer_manager.print_profile_summary(full=TRANSFORMER_MANAGER_PROFILE_FULL)


    image_latent = denoise_state.to(OFFLOAD_DEVICE)
    latent_height = prepare_state["latent_height"]
    latent_width = prepare_state["latent_width"]
    latent_channels = prepare_state["in_channels"]
    print(f"  Image latent: {image_latent.shape}")

    del connector_prompt_embeds, connector_attention_mask
    if transformer_manager is not None:
        transformer_manager.detach(transformer)
    if dynamic_weights_enabled:
        if DYNAMIC_WEIGHTS_EXECUTION_MODE != "plan":
            run_metrics["dynamic_weights_runtime_summary"] = dynamic_weights_hook.state.as_dict()
        remove_dynamic_weights(transformer)
    del prepare_pipe, denoise_pipe, transformer, scheduler
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
