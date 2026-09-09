"""
Modular LTX 2.3 distilled image runner using official Diffusers group offloading.
"""

import argparse
import json
import os
from pathlib import Path
import re
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_MODULES_CACHE", str((Path(__file__).parent / ".hf_modules").resolve()))

import torch
from diffusers import AutoencoderKLLTX2Video, FlowMatchEulerDiscreteScheduler
from diffusers.hooks import apply_group_offloading
from diffusers.loaders.lora_pipeline import LTX2LoraLoaderMixin
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


DEVICE = "cuda:0"
OFFLOAD_DEVICE = "cpu"
DTYPE = torch.bfloat16

MODEL_TAG = "distilled_modular_diffusers_group_offload"
MODEL_PATH = r"elismasilva/ltx2.3-image-distilled-1.1"
LOW_CPU_MEM_USAGE = True

GROUP_OFFLOAD_CONFIG = {
    "text_encoder_offload_type": "leaf_level",
    "text_encoder_use_stream": True,
    "text_encoder_record_stream": False,
    "text_encoder_low_cpu_mem_usage": True,
    "text_encoder_num_blocks_per_group": 1,
    "transformer_offload_type": "leaf_level",
    "transformer_use_stream": True,
    "transformer_record_stream": False,
    "transformer_low_cpu_mem_usage": True,
    "transformer_num_blocks_per_group": 1,
}

WIDTH = 1280
HEIGHT = 704
SEED = 43
NUM_INFERENCE_STEPS = 8
GUIDANCE_SCALE = 1.0
GUIDANCE_RESCALE = 0.7
DECODE_TIMESTEP = 0.0
DECODE_NOISE_SCALE = None
PAG_ENABLED = False
PAG_SCALE = 0.2
PAG_APPLIED_LAYERS = [28]
GENERATION_REPEATS = 1

SHOW_METRICS = True
SAVE_METRICS = True
SHOW_DENOISE_STEPS = True

CRISP_LORA_ENABLED = False
CRISP_LORA_PATH = "vrgamedevgirl84/LTX_2.3_Crisp_Enhance_Style_LoRa"
CRISP_LORA_WEIGHT_NAME = "LTX2.3_Crisp_Enhance.safetensors"
CRISP_LORA_ADAPTER_NAME = "crisp"
CRISP_LORA_SCALE = 0.3
SOFT_LORA_ENABLED = True
SOFT_LORA_PATH = "vrgamedevgirl84/LTX_2.3_Soft_Enhance_Style_LoRa"
SOFT_LORA_WEIGHT_NAME = "LTX2.3_Soft_Enhance.safetensors"
SOFT_LORA_ADAPTER_NAME = "soft"
SOFT_LORA_SCALE = 0.8

prompt = "Fisheye close-up of a calico cat wearing a tiny flower crown, sniffing the camera lens in a sunny park, with bright colors, realistic fur detail, and playful viral-pet energy."
negative_prompt = ""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=prompt)
    parser.add_argument("--negative-prompt", default=negative_prompt)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--steps", type=int, default=NUM_INFERENCE_STEPS)
    parser.add_argument("--guidance-scale", type=float, default=GUIDANCE_SCALE)
    parser.add_argument("--guidance-rescale", type=float, default=GUIDANCE_RESCALE)
    parser.add_argument("--decode-timestep", type=float, default=DECODE_TIMESTEP)
    parser.add_argument("--decode-noise-scale", type=float, default=DECODE_NOISE_SCALE)
    parser.add_argument("--pag", action="store_true", default=PAG_ENABLED)
    parser.add_argument("--pag-scale", type=float, default=PAG_SCALE)
    parser.add_argument("--pag-layers", default=",".join(map(str, PAG_APPLIED_LAYERS)))
    parser.add_argument("--generation-repeats", type=int, default=GENERATION_REPEATS)
    parser.add_argument("--output-dir", default="outputs/ltx_image_modular")
    parser.add_argument("--show-metrics", action=argparse.BooleanOptionalAction, default=SHOW_METRICS)
    parser.add_argument("--save-metrics", action=argparse.BooleanOptionalAction, default=SAVE_METRICS)
    parser.add_argument("--show-denoise-steps", action=argparse.BooleanOptionalAction, default=SHOW_DENOISE_STEPS)
    parser.add_argument("--crisp-lora", action=argparse.BooleanOptionalAction, default=CRISP_LORA_ENABLED)
    parser.add_argument("--crisp-lora-path", default=CRISP_LORA_PATH)
    parser.add_argument("--crisp-lora-weight-name", default=CRISP_LORA_WEIGHT_NAME)
    parser.add_argument("--crisp-lora-adapter-name", default=CRISP_LORA_ADAPTER_NAME)
    parser.add_argument("--crisp-lora-scale", type=float, default=CRISP_LORA_SCALE)
    parser.add_argument("--soft-lora", action=argparse.BooleanOptionalAction, default=SOFT_LORA_ENABLED)
    parser.add_argument("--soft-lora-path", default=SOFT_LORA_PATH)
    parser.add_argument("--soft-lora-weight-name", default=SOFT_LORA_WEIGHT_NAME)
    parser.add_argument("--soft-lora-adapter-name", default=SOFT_LORA_ADAPTER_NAME)
    parser.add_argument("--soft-lora-scale", type=float, default=SOFT_LORA_SCALE)
    return parser.parse_args()


def build_run_slug(seed):
    pag_tag = f"pag{PAG_SCALE:g}_layers{'-'.join(map(str, PAG_APPLIED_LAYERS))}" if PAG_ENABLED else "nopag"
    lora_tags = []
    if SOFT_LORA_ENABLED:
        lora_tags.append(f"{SOFT_LORA_ADAPTER_NAME}{SOFT_LORA_SCALE:g}")
    if CRISP_LORA_ENABLED:
        lora_tags.append(f"{CRISP_LORA_ADAPTER_NAME}{CRISP_LORA_SCALE:g}")
    lora_tag = "lora_" + "-".join(lora_tags) if lora_tags else "nolora"
    return "_".join(
        [
            "ltx23_image",
            MODEL_TAG,
            "bf16",
            "text_encoder_original",
            pag_tag,
            lora_tag,
            f"{WIDTH}x{HEIGHT}",
            f"steps{NUM_INFERENCE_STEPS}",
            f"seed{seed}",
        ]
    )


def apply_model_group_offload(model, *, prefix: str):
    offload_type = GROUP_OFFLOAD_CONFIG[f"{prefix}_offload_type"]
    kwargs = {
        "onload_device": torch.device(DEVICE),
        "offload_device": torch.device(OFFLOAD_DEVICE),
        "offload_type": offload_type,
        "use_stream": GROUP_OFFLOAD_CONFIG[f"{prefix}_use_stream"],
        "record_stream": GROUP_OFFLOAD_CONFIG[f"{prefix}_record_stream"],
        "low_cpu_mem_usage": GROUP_OFFLOAD_CONFIG[f"{prefix}_low_cpu_mem_usage"],
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


def cleanup_runtime_state(record_event, event_name: str):
    event_t0 = time.time()
    flush()
    record_event(event_name, time.time() - event_t0)


def load_lora_adapter(model, source: str, weight_name: str, adapter_name: str) -> str:
    state_dict, metadata = LTX2LoraLoaderMixin.lora_state_dict(
        source,
        weight_name=weight_name,
        return_lora_metadata=True,
    )
    model.load_lora_adapter(
        state_dict,
        prefix="transformer",
        adapter_name=adapter_name,
        metadata=metadata,
        low_cpu_mem_usage=LOW_CPU_MEM_USAGE,
    )
    return adapter_name


def maybe_load_lora(model, run_metrics, enabled, kind, path, weight_name, adapter_name, scale, record_event):
    if not enabled:
        return None
    event_t0 = time.time()
    actual_adapter_name = load_lora_adapter(model, path, weight_name, adapter_name)
    run_metrics["lora_adapters"].append(
        {
            "kind": kind,
            "path": path,
            "weight_name": weight_name,
            "adapter_name": actual_adapter_name,
            "scale": scale,
        }
    )
    record_event(
        f"load_{kind}_lora",
        time.time() - event_t0,
        path=path,
        weight_name=weight_name,
        adapter_name=actual_adapter_name,
        scale=scale,
    )
    return actual_adapter_name, scale


def activate_loras(model, adapters, record_event):
    if not adapters:
        return
    names, scales = zip(*adapters)
    try:
        model.set_adapters(list(names), weights=list(scales))
    except TypeError:
        model.set_adapters(list(names), adapter_weights=list(scales))
    record_event("activate_loras", 0.0, adapters=dict(adapters))


def make_denoise_progress_callback(tracker):
    def callback(pipe, step_index, timestep, callback_kwargs):
        now = time.perf_counter()
        elapsed = now - callback.last_time
        callback.last_time = now
        callback.step_times.append(elapsed)
        if SHOW_DENOISE_STEPS:
            avg = (now - callback.start_time) / (step_index + 1)
            allocated = torch.cuda.memory_allocated(DEVICE) / (1024**3)
            reserved = torch.cuda.memory_reserved(DEVICE) / (1024**3)
            total_steps = len(callback.timesteps) if callback.timesteps is not None else NUM_INFERENCE_STEPS
            print(
                f"  [denoise] step {step_index + 1}/{total_steps} timestep={float(timestep):.4f} "
                f"elapsed={elapsed:.4f}s avg={avg:.4f}s/it "
                f"torch_alloc={allocated:.2f} GiB torch_reserved={reserved:.2f} GiB",
                flush=True,
            )
        return callback_kwargs

    callback.start_time = 0.0
    callback.last_time = 0.0
    callback.step_times = []
    callback.timesteps = None
    return callback


def main():
    global WIDTH, HEIGHT, SEED, NUM_INFERENCE_STEPS, GUIDANCE_SCALE, GUIDANCE_RESCALE
    global DECODE_TIMESTEP, DECODE_NOISE_SCALE, PAG_ENABLED, PAG_SCALE, PAG_APPLIED_LAYERS
    global GENERATION_REPEATS, SHOW_METRICS, SAVE_METRICS, SHOW_DENOISE_STEPS
    global CRISP_LORA_ENABLED, CRISP_LORA_PATH, CRISP_LORA_WEIGHT_NAME, CRISP_LORA_ADAPTER_NAME, CRISP_LORA_SCALE
    global SOFT_LORA_ENABLED, SOFT_LORA_PATH, SOFT_LORA_WEIGHT_NAME, SOFT_LORA_ADAPTER_NAME, SOFT_LORA_SCALE
    global prompt, negative_prompt

    args = parse_args()
    WIDTH = args.width
    HEIGHT = args.height
    SEED = args.seed
    NUM_INFERENCE_STEPS = args.steps
    GUIDANCE_SCALE = args.guidance_scale
    GUIDANCE_RESCALE = args.guidance_rescale
    DECODE_TIMESTEP = args.decode_timestep
    DECODE_NOISE_SCALE = args.decode_noise_scale
    PAG_ENABLED = args.pag
    PAG_SCALE = args.pag_scale
    PAG_APPLIED_LAYERS = [int(item.strip()) for item in args.pag_layers.split(",") if item.strip()]
    GENERATION_REPEATS = args.generation_repeats
    SHOW_METRICS = args.show_metrics
    SAVE_METRICS = args.save_metrics
    SHOW_DENOISE_STEPS = args.show_denoise_steps
    CRISP_LORA_ENABLED = args.crisp_lora
    CRISP_LORA_PATH = args.crisp_lora_path
    CRISP_LORA_WEIGHT_NAME = args.crisp_lora_weight_name
    CRISP_LORA_ADAPTER_NAME = args.crisp_lora_adapter_name
    CRISP_LORA_SCALE = args.crisp_lora_scale
    SOFT_LORA_ENABLED = args.soft_lora
    SOFT_LORA_PATH = args.soft_lora_path
    SOFT_LORA_WEIGHT_NAME = args.soft_lora_weight_name
    SOFT_LORA_ADAPTER_NAME = args.soft_lora_adapter_name
    SOFT_LORA_SCALE = args.soft_lora_scale
    prompt = args.prompt
    negative_prompt = args.negative_prompt

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this runner.")

    seed = SEED or torch.randint(0, 2**32, (1,)).item()
    if not SEED:
        print(f"  Using random seed: {seed}")
    generator = torch.Generator(device="cpu").manual_seed(seed)

    run_slug = build_run_slug(seed)
    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / "metrics"
    run_metrics = {
        "run_slug": run_slug,
        "model_tag": MODEL_TAG,
        "model_path": MODEL_PATH,
        "text_encoder_kind": "original",
        "transformer_kind": "custom_modular",
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
        "lora_enabled": SOFT_LORA_ENABLED or CRISP_LORA_ENABLED,
        "lora_adapters": [],
        "dtype": str(DTYPE),
        "offload_backend": "diffusers_group_offloading",
        "group_offload_config": GROUP_OFFLOAD_CONFIG.copy(),
        "transformer_memory_manager": TRANSFORMER_MEMORY_MANAGER,
        **DYNAMIC_OFFLOAD_SETTINGS.as_metrics(),
        "pre_vae_cleanup_repeats": PRE_VAE_CLEANUP_REPEATS,
        "attention_backend": ATTENTION_BACKEND,
        "drop_trivial_attention_mask": DROP_TRIVIAL_ATTENTION_MASK,
        "events": [],
        "steps": [],
    }

    tracker = RunTracker(DEVICE, run_metrics, interval=0.1, show_metrics=SHOW_METRICS)
    record_event = tracker.record_event
    step_start = tracker.step_start
    step_end = tracker.step_end
     denoise_progress_callback = make_denoise_progress_callback(tracker)

    print("Using modular Diffusers group offload baseline", flush=True)
    print(
        f"  text_encoder=leaf group offload transformer=leaf group offload "
        f"model_path={MODEL_PATH}",
        flush=True,
    )
    print(f"  VRAM baseline: {torch.cuda.memory_allocated(DEVICE) / (1024**3):.2f} GB", flush=True)

    if PURGE_WINDOWS_STANDBY_BEFORE_RUN:
        purge_windows_standby_cache_event(record_event, "purge_windows_standby_before_run")

    t0 = step_start("Pass 0: Encode prompts")

    event_t0 = time.time()
    text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        subfolder="text_encoder",
        torch_dtype=DTYPE,
        low_cpu_mem_usage=LOW_CPU_MEM_USAGE,
    )
    record_event("load_text_encoder", time.time() - event_t0, source=MODEL_PATH)

    event_t0 = time.time()
    apply_model_group_offload(text_encoder, prefix="text_encoder")
    record_event(
        "setup_text_encoder_group_offload",
        time.time() - event_t0,
        offload_type=GROUP_OFFLOAD_CONFIG["text_encoder_offload_type"],
        use_stream=GROUP_OFFLOAD_CONFIG["text_encoder_use_stream"],
        record_stream=GROUP_OFFLOAD_CONFIG["text_encoder_record_stream"],
    )

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
            output=[
                "prompt_embeds",
                "prompt_attention_mask",
                "negative_prompt_embeds",
                "negative_prompt_attention_mask",
                "batch_size",
                "dtype",
                "do_classifier_free_guidance",
            ],
        )
    record_event("encode_prompt_call", time.time() - event_t0, classifier_free_guidance=False)

    prompt_embeds = prompt_state["prompt_embeds"].to(OFFLOAD_DEVICE)
    prompt_attention_mask = prompt_state["prompt_attention_mask"].to(OFFLOAD_DEVICE)
    latent_batch_size = prompt_state["batch_size"]
    prompt_dtype = prompt_state["dtype"]
    do_classifier_free_guidance = prompt_state["do_classifier_free_guidance"]
    if SHOW_METRICS:
        print(f"  prompt_embeds shape: {prompt_embeds.shape}")

    del prompt_state, prompt_pipe, text_encoder, tokenizer
    cleanup_runtime_state(record_event, "cleanup_after_text_encoder")
    step_end("Pass 0: Encode prompts", t0)

    t0 = step_start(f"Pass 1: Generate at {WIDTH}x{HEIGHT}")

    event_t0 = time.time()
    connectors = LTX2ImageTextConnectors.from_pretrained(
        MODEL_PATH,
        subfolder="connectors",
        torch_dtype=DTYPE,
        low_cpu_mem_usage=LOW_CPU_MEM_USAGE,
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
        do_classifier_free_guidance=do_classifier_free_guidance,
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

    del connector_state, connector_pipe, connectors, prompt_embeds, prompt_attention_mask
    cleanup_runtime_state(record_event, "cleanup_after_connectors")

    transformer_load_kwargs = {
        "subfolder": "transformer",
        "torch_dtype": prompt_dtype,
        "low_cpu_mem_usage": LOW_CPU_MEM_USAGE,
    }
    if LOW_CPU_MEM_USAGE:
        transformer_load_kwargs["device_map"] = "cpu"

    event_t0 = time.time()
    transformer = LTX2ImageTransformer2DModel.from_pretrained(MODEL_PATH, **transformer_load_kwargs)
    record_event(
        "load_transformer",
        time.time() - event_t0,
        source=MODEL_PATH,
        low_cpu_mem_usage=LOW_CPU_MEM_USAGE,
        device_map=transformer_load_kwargs.get("device_map"),
    )

    active_adapters = []
    for item in [
        (
            SOFT_LORA_ENABLED,
            "soft",
            SOFT_LORA_PATH,
            SOFT_LORA_WEIGHT_NAME,
            SOFT_LORA_ADAPTER_NAME,
            SOFT_LORA_SCALE,
        ),
        (
            CRISP_LORA_ENABLED,
            "crisp",
            CRISP_LORA_PATH,
            CRISP_LORA_WEIGHT_NAME,
            CRISP_LORA_ADAPTER_NAME,
            CRISP_LORA_SCALE,
        ),
    ]:
        adapter = maybe_load_lora(transformer, run_metrics, *item, record_event=record_event)
        if adapter is not None:
            active_adapters.append(adapter)
    activate_loras(transformer, active_adapters, record_event)

    event_t0 = time.time()
    apply_model_group_offload(transformer, prefix="transformer")
    record_event(
        "setup_transformer_group_offload",
        time.time() - event_t0,
        offload_type=GROUP_OFFLOAD_CONFIG["transformer_offload_type"],
        use_stream=GROUP_OFFLOAD_CONFIG["transformer_use_stream"],
        record_stream=GROUP_OFFLOAD_CONFIG["transformer_record_stream"],
        low_cpu_mem_usage=GROUP_OFFLOAD_CONFIG["transformer_low_cpu_mem_usage"],
    )

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
        print(f"  Starting denoise loop{repeat_suffix}", flush=True)
        event_t0 = time.time()
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
    cleanup_runtime_state(record_event, "cleanup_before_vae_decode")
    step_end(f"Pass 1: Generate at {WIDTH}x{HEIGHT}", t0)

    t0 = step_start("Pass 2: Decode VAE")

    event_t0 = time.time()
    vae = AutoencoderKLLTX2Video.from_pretrained(
        MODEL_PATH,
        subfolder="vae",
        torch_dtype=DTYPE,
        low_cpu_mem_usage=LOW_CPU_MEM_USAGE,
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
    print("=" * 70)


if __name__ == "__main__":
    main()
