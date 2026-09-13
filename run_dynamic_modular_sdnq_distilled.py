"""
Memory-scoped LTX 2.3 distilled modular SDNQ text-to-image DDO runner.

The transformer is loaded from an SDNQ repository, while the prompt encoder,
connectors, scheduler, and VAE are loaded only for their own phases. Quantized
modules keep their native forward path; the transformer uses DDO's dedicated
``sdnq_runtime`` adapter while the text encoder retains Diffusers group offload.
"""

import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_MODULES_CACHE", str((Path(__file__).parent / ".hf_modules").resolve()))

import torch
from diffusers import AutoencoderKLLTX2Video, FlowMatchEulerDiscreteScheduler
from diffusers.hooks import apply_group_offloading
from diffusers_dynamic_offloader import (
    DynamicOffloadSettings,
    enable_offload,
    format_dynamic_offload_presets,
    is_wsl_environment,
    remove_dynamic_offload,
)
from diffusers.loaders.lora_pipeline import LTX2LoraLoaderMixin
from transformers import Gemma3ForConditionalGeneration, GemmaTokenizerFast

from custom_blocks.ltx2_image import LTX2ImageDistilledBlocks, LTX2ImageTextEncoderStep
from custom_blocks.ltx2_image.connectors_ltx2_image import LTX2ImageTextConnectors
from custom_blocks.ltx2_image.modular_blocks_ltx2_image import (
    LTX2ImageConnectorStep,
    LTX2ImageDenoiseStep,
    LTX2ImagePrepareLatentsStep,
)
from custom_blocks.ltx2_image.transformer_ltx2_image import LTX2ImageTransformer2DModel
from inference_utils import RunTracker, flush, get_sdnq_version

DEVICE = "cuda:0"
OFFLOAD_DEVICE = "cpu"
DTYPE = torch.bfloat16
LOW_CPU_MEM_USAGE = True
TEXT_ENCODER_LOW_CPU_MEM_USAGE = True

MODEL_PATH = os.getenv("MODEL_PATH", r"elismasilva/ltx2.3-image-distilled-1.1")
MODEL_TAG = "distilled_modular_sdnq_ddo_runtime"
SDNQ_MODEL_PATHS = {
    4: "elismasilva/ltx2.3-image-distilled-1.1-sdnq-int4",
    8: "elismasilva/ltx2.3-image-distilled-1.1-sdnq-int8",
}

GROUP_OFFLOAD_CONFIG = {
    "offload_type": "leaf_level",
    "use_stream": True,
    "record_stream": False,
    "low_cpu_mem_usage": True,
}

SHOW_METRICS = True
SAVE_METRICS = True
SHOW_DENOISE_STEPS = True

CRISP_LORA_PATH = "vrgamedevgirl84/LTX_2.3_Crisp_Enhance_Style_LoRa"
CRISP_LORA_WEIGHT_NAME = "LTX2.3_Crisp_Enhance.safetensors"
CRISP_LORA_ADAPTER_NAME = "crisp"
CRISP_LORA_SCALE = 0.3
SOFT_LORA_PATH = "vrgamedevgirl84/LTX_2.3_Soft_Enhance_Style_LoRa"
SOFT_LORA_WEIGHT_NAME = "LTX2.3_Soft_Enhance.safetensors"
SOFT_LORA_ADAPTER_NAME = "soft"
SOFT_LORA_SCALE = 0.15

DEFAULT_PROMPT = (
    "Fisheye close-up of a calico cat wearing a tiny flower crown, sniffing the camera lens "
    "in a sunny park, with bright colors, realistic fur detail, and playful viral-pet energy."
)
DEFAULT_NEGATIVE_PROMPT = (
    ""
)


def parse_args():
    parser = argparse.ArgumentParser(description="LTX 2.3 distilled modular SDNQ text-to-image DDO runner.")
    parser.add_argument("--bits", type=int, choices=(4, 8), default=4, help="SDNQ transformer bit depth.")
    parser.add_argument(
        "--text-encoder-bits",
        type=int,
        choices=(8,),
        default=None,
        help="Use the SDNQ text encoder. Leave unset to use the original text encoder.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--guidance-rescale", type=float, default=0.7)
    parser.add_argument("--decode-timestep", type=float, default=0.0)
    parser.add_argument("--decode-noise-scale", type=float, default=None)
    parser.add_argument("--pag", action="store_true")
    parser.add_argument("--pag-scale", type=float, default=0.2)
    parser.add_argument("--pag-layers", default="28")
    parser.add_argument("--soft-lora", action="store_true", help="Load the Soft Enhance LoRA.")
    parser.add_argument("--soft-lora-path", default=SOFT_LORA_PATH)
    parser.add_argument("--soft-lora-weight-name", default=SOFT_LORA_WEIGHT_NAME)
    parser.add_argument("--soft-lora-adapter-name", default=SOFT_LORA_ADAPTER_NAME)
    parser.add_argument("--soft-lora-scale", type=float, default=SOFT_LORA_SCALE)
    parser.add_argument("--crisp-lora", action="store_true", help="Load the Crisp Enhance LoRA.")
    parser.add_argument("--crisp-lora-path", default=CRISP_LORA_PATH)
    parser.add_argument("--crisp-lora-weight-name", default=CRISP_LORA_WEIGHT_NAME)
    parser.add_argument("--crisp-lora-adapter-name", default=CRISP_LORA_ADAPTER_NAME)
    parser.add_argument("--crisp-lora-scale", type=float, default=CRISP_LORA_SCALE)
    parser.add_argument("--output-dir", default="outputs/ltx_image_modular_sdnq_distilled")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--preset", default="auto", help="Preset do DDO para o transformer SDNQ.")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--ddo-profile", default=None, help="Named DDO denoise workload profile.")
    parser.add_argument("--build-ddo-profile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ddo-profile-steps", type=int, default=2)
    parser.add_argument("--pre-vae-cleanup-repeats", type=int, default=1)
    parser.add_argument("--metrics-level", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--resident-module-budget-gb", type=float, default=None)
    parser.add_argument("--max-resident-module-budget-gb", type=float, default=None)
    parser.add_argument("--pin-weight-budget-gb", type=float, default=None)
    parser.add_argument("--reset-dynamic-memory-after-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--low-cpu-mem-usage", action=argparse.BooleanOptionalAction, default=LOW_CPU_MEM_USAGE)
    parser.add_argument(
        "--text-encoder-low-cpu-mem-usage",
        action=argparse.BooleanOptionalAction,
        default=TEXT_ENCODER_LOW_CPU_MEM_USAGE,
    )
    parser.add_argument("--text-encoder-group-offload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--text-encoder-dynamic-offload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reserved: the experimental SDNQ DDO route currently applies only to the transformer.",
    )
    parser.add_argument("--print-presets", action="store_true")
    parser.add_argument("--show-metrics", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-metrics", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--show-denoise-steps", action=argparse.BooleanOptionalAction, default=SHOW_DENOISE_STEPS)
    return parser.parse_args()


def apply_leaf_group_offload(model) -> None:
    apply_group_offloading(
        model,
        onload_device=torch.device(DEVICE),
        offload_device=torch.device(OFFLOAD_DEVICE),
        **GROUP_OFFLOAD_CONFIG,
    )


def cleanup_runtime_state(record_event, event_name: str, *, repeats: int = 1) -> None:
    event_t0 = time.time()
    for _ in range(max(1, repeats)):
        flush()
    record_event(event_name, time.time() - event_t0, repeats=max(1, repeats))


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


def activate_loras(model, adapters, record_event) -> None:
    if not adapters:
        return
    names, scales = zip(*adapters, strict=False)
    try:
        model.set_adapters(list(names), weights=list(scales))
    except TypeError:
        model.set_adapters(list(names), adapter_weights=list(scales))
    record_event("activate_loras", 0.0, adapters=dict(adapters))


def build_run_slug(args, *, text_encoder_kind: str) -> str:
    pag_tag = f"pag{args.pag_scale:g}_layers{args.pag_layers.replace(',', '-')}" if args.pag else "nopag"
    lora_tags = []
    if args.soft_lora:
        lora_tags.append(f"{args.soft_lora_adapter_name}{args.soft_lora_scale:g}")
    if args.crisp_lora:
        lora_tags.append(f"{args.crisp_lora_adapter_name}{args.crisp_lora_scale:g}")
    lora_tag = "lora_" + "-".join(lora_tags) if lora_tags else "nolora"
    return "_".join(
        [
            "ltx23_image",
            MODEL_TAG,
            f"sdnq{args.bits}",
            f"text_encoder_{text_encoder_kind}",
            pag_tag,
            lora_tag,
            f"{args.width}x{args.height}",
            f"steps{args.steps}",
            f"seed{args.seed}",
        ]
    )


def make_denoise_progress_callback(args, run_metrics):
    def callback(pipe, step_index, timestep, callback_kwargs):
        now = time.perf_counter()
        elapsed = now - callback.last_time
        callback.last_time = now
        callback.step_times.append(elapsed)
        if SHOW_DENOISE_STEPS:
            avg = (now - callback.start_time) / (step_index + 1)
            allocated = torch.cuda.memory_allocated(DEVICE) / (1024**3)
            reserved = torch.cuda.memory_reserved(DEVICE) / (1024**3)
            total_steps = len(callback.timesteps) if callback.timesteps is not None else args.steps
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
    run_metrics["denoise_step_times"] = callback.step_times
    return callback


def main():
    global DEVICE, LOW_CPU_MEM_USAGE, MODEL_PATH, SHOW_METRICS, SAVE_METRICS, SHOW_DENOISE_STEPS
    global TEXT_ENCODER_LOW_CPU_MEM_USAGE

    args = parse_args()
    if args.print_presets:
        print(format_dynamic_offload_presets(default_preset=args.preset, running_on_wsl=is_wsl_environment()))
        return
    if args.build_ddo_profile and not args.ddo_profile:
        raise SystemExit("--build-ddo-profile requires --ddo-profile NAME")
    if args.text_encoder_dynamic_offload:
        raise SystemExit("--text-encoder-dynamic-offload is not implemented for the SDNQ experimental runner.")
    DEVICE = args.device
    MODEL_PATH = args.model_path
    LOW_CPU_MEM_USAGE = args.low_cpu_mem_usage
    TEXT_ENCODER_LOW_CPU_MEM_USAGE = args.text_encoder_low_cpu_mem_usage
    active_inference_steps = max(1, min(args.steps, args.ddo_profile_steps)) if args.build_ddo_profile else args.steps
    SHOW_METRICS = args.show_metrics if args.show_metrics is not None else args.metrics_level >= 1
    SAVE_METRICS = args.save_metrics if args.save_metrics is not None else args.metrics_level >= 2
    SHOW_DENOISE_STEPS = args.show_denoise_steps
    import sdnq  # noqa: F401 - registers SDNQ model classes before loading components.

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this runner.")
    if args.width % 32 != 0 or args.height % 32 != 0:
        raise ValueError("Width and height must be divisible by 32.")
    settings_environ = os.environ.copy()
    if args.resident_module_budget_gb is not None:
        settings_environ["DDO_RESIDENT_MODULE_BUDGET_GB"] = str(args.resident_module_budget_gb)
    if args.max_resident_module_budget_gb is not None:
        settings_environ["DDO_MAX_RESIDENT_MODULE_BUDGET_GB"] = str(args.max_resident_module_budget_gb)
    if args.pin_weight_budget_gb is not None:
        settings_environ["DDO_PIN_WEIGHT_BUDGET_GB"] = str(args.pin_weight_budget_gb)
    ddo_settings = DynamicOffloadSettings.from_env(
        execution_device=DEVICE,
        offload_device=OFFLOAD_DEVICE,
        running_on_wsl=is_wsl_environment(),
        default_preset=args.preset,
        environ=settings_environ,
    )
    ddo_config = replace(ddo_settings.config, execution_mode="sdnq_runtime")
    ddo_settings = replace(ddo_settings, config=ddo_config, enabled=True)

    sdnq_model_path = SDNQ_MODEL_PATHS[args.bits]
    text_encoder_src = sdnq_model_path if args.text_encoder_bits is not None else MODEL_PATH
    text_encoder_kind = f"sdnq{args.text_encoder_bits}" if args.text_encoder_bits is not None else "original"
    transformer_kind = f"sdnq{args.bits}"
    seed = args.seed if args.seed is not None else torch.randint(0, 2**32, (1,)).item()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pag_layers = [int(item.strip()) for item in args.pag_layers.split(",") if item.strip()]
    run_slug = build_run_slug(args, text_encoder_kind=text_encoder_kind)
    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / "metrics"
    output_path = output_dir / f"{run_slug}.png"
    metrics_path = metrics_dir / f"{run_slug}.json"

    run_metrics = {
        "run_slug": run_slug,
        "model_tag": MODEL_TAG,
        "model_path": MODEL_PATH,
        "workflow": "text2image",
        "sdnq_enabled": True,
        "sdnq_bits": args.bits,
        "sdnq_quantized_version": "0.1.6",
        "sdnq_version": get_sdnq_version(),
        "sdnq_model_path": sdnq_model_path,
        "sdnq_text_encoder_bits": args.text_encoder_bits,
        "text_encoder_source": text_encoder_src,
        "text_encoder_kind": text_encoder_kind,
        "transformer_source": sdnq_model_path,
        "transformer_kind": transformer_kind,
        "width": args.width,
        "height": args.height,
        "seed": seed,
        "num_inference_steps": args.steps,
        "active_inference_steps": active_inference_steps,
        "ddo_profile_name": args.ddo_profile,
        "build_ddo_profile": args.build_ddo_profile,
        "ddo_preset": args.preset,
        "metrics_level": args.metrics_level,
        "pre_vae_cleanup_repeats": max(1, args.pre_vae_cleanup_repeats),
        "reset_dynamic_memory_after_run": args.reset_dynamic_memory_after_run,
        "guidance_scale": args.guidance_scale,
        "guidance_rescale": args.guidance_rescale,
        "vae_decode_timestep": args.decode_timestep,
        "vae_decode_noise_scale": args.decode_noise_scale,
        "pag_enabled": args.pag,
        "pag_scale": args.pag_scale if args.pag else 0.0,
        "pag_applied_layers": pag_layers if args.pag else None,
        "lora_enabled": args.soft_lora or args.crisp_lora,
        "lora_adapters": [],
        "dtype": str(DTYPE),
        "offload_backend": "ddo_sdnq_runtime",
        "group_offload_config": GROUP_OFFLOAD_CONFIG.copy(),
        "events": [],
        "steps": [],
    }
    tracker = RunTracker(DEVICE, run_metrics, interval=0.1, show_metrics=SHOW_METRICS)
    record_event = tracker.record_event
    denoise_progress_callback = make_denoise_progress_callback(args, run_metrics)

    print(
        f"Using modular SDNQ: transformer={transformer_kind} text_encoder={text_encoder_kind} "
        "offload=ddo_sdnq_runtime",
        flush=True,
    )
    print(f"  model_path={MODEL_PATH}", flush=True)
    print(f"  sdnq_model_path={sdnq_model_path}", flush=True)

    t0 = tracker.step_start("Pass 0: Encode prompts")
    event_t0 = time.time()
    text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
        text_encoder_src,
        subfolder="text_encoder",
        torch_dtype=DTYPE,
        low_cpu_mem_usage=TEXT_ENCODER_LOW_CPU_MEM_USAGE,
    )
    record_event("load_text_encoder", time.time() - event_t0, source=text_encoder_src, kind=text_encoder_kind)

    event_t0 = time.time()
    if args.text_encoder_group_offload:
        apply_leaf_group_offload(text_encoder)
        record_event("setup_text_encoder_group_offload", time.time() - event_t0)
    else:
        text_encoder.to(DEVICE)
        record_event("setup_text_encoder_onload", time.time() - event_t0)

    event_t0 = time.time()
    tokenizer = GemmaTokenizerFast.from_pretrained(MODEL_PATH, subfolder="tokenizer")
    record_event("load_tokenizer", time.time() - event_t0, source=MODEL_PATH)

    event_t0 = time.time()
    prompt_pipe = LTX2ImageTextEncoderStep().init_pipeline()
    prompt_pipe.update_components(text_encoder=text_encoder, tokenizer=tokenizer)
    record_event("build_prompt_modular_pipeline", time.time() - event_t0)

    event_t0 = time.time()
    with torch.inference_mode():
        prompt_state = prompt_pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            guidance_scale=args.guidance_scale,
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
    record_event("encode_prompt_call", time.time() - event_t0)

    prompt_embeds = prompt_state["prompt_embeds"].to(OFFLOAD_DEVICE)
    prompt_attention_mask = prompt_state["prompt_attention_mask"].to(OFFLOAD_DEVICE)
    negative_prompt_embeds = prompt_state["negative_prompt_embeds"]
    negative_prompt_attention_mask = prompt_state["negative_prompt_attention_mask"]
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(OFFLOAD_DEVICE)
    if negative_prompt_attention_mask is not None:
        negative_prompt_attention_mask = negative_prompt_attention_mask.to(OFFLOAD_DEVICE)
    latent_batch_size = prompt_state["batch_size"]
    prompt_dtype = prompt_state["dtype"]
    do_classifier_free_guidance = prompt_state["do_classifier_free_guidance"]
    if SHOW_METRICS:
        print(f"  prompt_embeds shape: {prompt_embeds.shape}", flush=True)

    del prompt_state, prompt_pipe, text_encoder, tokenizer
    cleanup_runtime_state(record_event, "cleanup_after_text_encoder")
    tracker.step_end("Pass 0: Encode prompts", t0)

    t0 = tracker.step_start(f"Pass 1: Generate at {args.width}x{args.height}")
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
    record_event("build_connector_modular_pipeline", time.time() - event_t0)

    event_t0 = time.time()
    connector_state = connector_pipe(
        prompt_embeds=prompt_embeds.to(device=DEVICE, dtype=DTYPE),
        prompt_attention_mask=prompt_attention_mask.to(device=DEVICE),
        negative_prompt_embeds=negative_prompt_embeds.to(device=DEVICE, dtype=DTYPE)
        if negative_prompt_embeds is not None
        else None,
        negative_prompt_attention_mask=negative_prompt_attention_mask.to(device=DEVICE)
        if negative_prompt_attention_mask is not None
        else None,
        do_classifier_free_guidance=do_classifier_free_guidance,
        pag_scale=args.pag_scale if args.pag else 0.0,
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

    del connector_state, connector_pipe, connectors
    del prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask
    cleanup_runtime_state(record_event, "cleanup_after_connectors")

    transformer_load_kwargs = {
        "subfolder": "transformer",
        "torch_dtype": prompt_dtype,
        "low_cpu_mem_usage": LOW_CPU_MEM_USAGE,
    }
    if LOW_CPU_MEM_USAGE:
        transformer_load_kwargs["device_map"] = "cpu"

    event_t0 = time.time()
    transformer = LTX2ImageTransformer2DModel.from_pretrained(sdnq_model_path, **transformer_load_kwargs)
    record_event("load_sdnq_transformer", time.time() - event_t0, source=sdnq_model_path, kind=transformer_kind)

    active_adapters = []
    for item in [
        (args.soft_lora, "soft", args.soft_lora_path, args.soft_lora_weight_name, args.soft_lora_adapter_name, args.soft_lora_scale),
        (args.crisp_lora, "crisp", args.crisp_lora_path, args.crisp_lora_weight_name, args.crisp_lora_adapter_name, args.crisp_lora_scale),
    ]:
        adapter = maybe_load_lora(transformer, run_metrics, *item, record_event=record_event)
        if adapter is not None:
            active_adapters.append(adapter)
    activate_loras(transformer, active_adapters, record_event)

    event_t0 = time.time()
    transformer_offload = enable_offload(
        transformer,
        settings=ddo_settings,
        config=ddo_config,
        component="transformer",
        route="dynamic_offload",
        execution_device=DEVICE,
        offload_device=OFFLOAD_DEVICE,
        low_cpu_mem_usage=LOW_CPU_MEM_USAGE,
        record_event=record_event,
        dynamic_event_name="setup_transformer_sdnq_dynamic_offload",
        profile_name=args.ddo_profile or "",
        build_profile=args.build_ddo_profile,
    )
    pinning = transformer_offload.event_payload.get("sdnq_parameter_pinning", {})
    if SHOW_METRICS and pinning:
        print(
            "  [sdnq] CPU pinned parameters: "
            f"{pinning.get('pinned_gb', 0.0):.2f} GiB "
            f"({pinning.get('pinned_count', 0)} tensors); "
            f"unpinned={pinning.get('unpinned_gb', 0.0):.2f} GiB "
            f"({pinning.get('unpinned_count', 0)} tensors)",
            flush=True,
        )

    event_t0 = time.time()
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")
    record_event("load_scheduler", time.time() - event_t0, source=MODEL_PATH)

    event_t0 = time.time()
    prepare_pipe = LTX2ImagePrepareLatentsStep().init_pipeline()
    prepare_pipe.update_components(transformer=transformer, scheduler=scheduler)
    record_event("build_prepare_latents_modular_pipeline", time.time() - event_t0)

    event_t0 = time.time()
    denoise_pipe = LTX2ImageDenoiseStep().init_pipeline()
    denoise_pipe.update_components(transformer=transformer, scheduler=scheduler)
    record_event("build_denoise_modular_pipeline", time.time() - event_t0)

    event_t0 = time.time()
    prepare_state = prepare_pipe(
        width=args.width,
        height=args.height,
        num_inference_steps=active_inference_steps,
        batch_size=latent_batch_size,
        transformer_batch_multiplier=transformer_batch_multiplier,
        generator=generator,
        output=["latents", "timesteps", "latent_height", "latent_width", "in_channels", "video_rotary_emb"],
    )
    record_event("prepare_latents_modular_pipe_call", time.time() - event_t0)

    denoise_progress_callback.timesteps = prepare_state["timesteps"]
    denoise_progress_callback.start_time = time.perf_counter()
    denoise_progress_callback.last_time = denoise_progress_callback.start_time
    denoise_progress_callback.step_times = []
    run_metrics["denoise_step_times"] = denoise_progress_callback.step_times
    print("  Starting denoise loop", flush=True)
    event_t0 = time.time()
    with transformer_offload.profile_run() as ddo_profile_report:
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
        do_classifier_free_guidance=do_classifier_free_guidance,
        do_perturbed_attention_guidance=do_perturbed_attention_guidance,
        guidance_scale=args.guidance_scale,
        guidance_rescale=args.guidance_rescale,
        pag_scale=args.pag_scale if args.pag else 0.0,
        pag_applied_layers=pag_layers if args.pag else None,
        callback_on_step_end=denoise_progress_callback,
        callback_on_step_end_tensor_inputs=["latents"],
        output="latents",
        )
    record_event("denoise_modular_pipe_call", time.time() - event_t0)

    image_latent = denoise_state.to(OFFLOAD_DEVICE)
    latent_height = prepare_state["latent_height"]
    latent_width = prepare_state["latent_width"]
    latent_channels = prepare_state["in_channels"]
    if SHOW_METRICS:
        print(f"  Image latent: {image_latent.shape}", flush=True)

    run_metrics["ddo_profile"] = ddo_profile_report
    remove_dynamic_offload(transformer)
    del prepare_state, denoise_state, connector_prompt_embeds, connector_attention_mask
    del prepare_pipe, denoise_pipe, transformer, scheduler
    cleanup_runtime_state(
        record_event,
        "cleanup_before_vae_decode",
        repeats=args.pre_vae_cleanup_repeats,
    )
    tracker.step_end(f"Pass 1: Generate at {args.width}x{args.height}", t0)

    t0 = tracker.step_start("Pass 2: Decode VAE")
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
        decode_timestep=args.decode_timestep,
        decode_noise_scale=args.decode_noise_scale,
        generator=generator,
        output_type="pil",
        output="images",
    )[0]
    record_event("vae_decode_modular_call", time.time() - event_t0)

    del decode_pipe, vae, image_latent
    cleanup_runtime_state(record_event, "cleanup_after_vae_decode")
    tracker.step_end("Pass 2: Decode VAE", t0)

    t0 = tracker.step_start("Save Image")
    output_dir.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"  Image saved successfully to: {output_path}", flush=True)
    tracker.step_end("Save Image", t0)

    total_time = tracker.total_elapsed()
    run_metrics["total_elapsed_sec"] = round(total_time, 4)
    run_metrics["global_peak_vram_gb"] = round(tracker.global_peak_vram, 4)
    run_metrics["global_peak_ram_gb"] = round(tracker.global_peak_ram, 4)
    run_metrics["output_path"] = str(output_path)
    if args.reset_dynamic_memory_after_run:
        cleanup_runtime_state(record_event, "reset_dynamic_memory_after_run")
    if SAVE_METRICS:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(run_metrics, indent=2), encoding="utf-8")
        print(f"  Metrics JSON: {metrics_path}", flush=True)

    print("\n" + "=" * 70)
    print(
        f"  TOTAL: {total_time:.1f}s | Peak VRAM: {tracker.global_peak_vram:.2f} GB | "
        f"Peak RAM: {tracker.global_peak_ram:.2f} GB",
        flush=True,
    )
    print(f"  Output: {output_path}", flush=True)
    print("=" * 70)


if __name__ == "__main__":
    main()
