"""
Memory-scoped LTX 2 image modular image-to-image runner.

This runner uses the custom modular blocks step by step, so VAE encode, text
encoding, connectors, transformer denoise, and VAE decode can be cleaned up
between phases.
"""

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_MODULES_CACHE", str((Path(__file__).parent / ".hf_modules").resolve()))

import torch
from diffusers import AutoencoderKLLTX2Video, FlowMatchEulerDiscreteScheduler
from diffusers.hooks import apply_group_offloading
from diffusers.loaders.lora_pipeline import LTX2LoraLoaderMixin
from transformers import Gemma3ForConditionalGeneration, GemmaTokenizerFast

from custom_blocks.ltx2_image import LTX2ImageDistilledBlocks, LTX2ImageTextEncoderStep, LTX2ImageVaeEncoderStep
from custom_blocks.ltx2_image.connectors_ltx2_image import LTX2ImageTextConnectors
from custom_blocks.ltx2_image.modular_blocks_ltx2_image import (
    LTX2ImageConnectorStep,
    LTX2ImageDenoiseStep,
    LTX2ImagePrepareLatentsStep,
    get_strength_sigmas,
)
from custom_blocks.ltx2_image.transformer_ltx2_image import LTX2ImageTransformer2DModel
from inference_utils import RunTracker, flush


DEVICE = "cuda:0"
OFFLOAD_DEVICE = "cpu"
DTYPE = torch.bfloat16
LOW_CPU_MEM_USAGE = True

MODEL_PATH = "elismasilva/ltx2.3-image-base"
MODEL_TAG = "base_modular_img2img_diffusers_group_offload"

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
CRISP_LORA_SCALE = 0.5
SOFT_LORA_PATH = "vrgamedevgirl84/LTX_2.3_Soft_Enhance_Style_LoRa"
SOFT_LORA_WEIGHT_NAME = "LTX2.3_Soft_Enhance.safetensors"
SOFT_LORA_ADAPTER_NAME = "soft"
SOFT_LORA_SCALE = 0.5

DEFAULT_PROMPT = (
    "Fisheye close-up of a calico cat wearing a tiny flower crown, sniffing the camera lens in a sunny park, "
    "with bright colors, realistic fur detail, and playful viral-pet energy. Preserve the exact same composition, "
    "camera framing, subject identity, pose, lighting, background layout, and geometry. Restore crisp fine detail, "
    "natural micro texture, clean edges, realistic material detail, and high-resolution sharpness. Do not change the scene."
)
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, distorted proportions, unnatural skin tones, deformed features, artifacts, "
    "cartoonish rendering, 3D CGI look, unrealistic materials"
)


def parse_args():
    parser = argparse.ArgumentParser(description="LTX 2 image modular img2img runner.")
    parser.add_argument("--input-image", required=True, help="Image to refine.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--strength", type=float, default=0.20)
    parser.add_argument("--input-noise-sigma", type=float, default=5.0)
    parser.add_argument("--input-sharpen", type=float, default=1.3)
    parser.add_argument("--phase-cutoff", type=float, default=9.0)
    parser.add_argument("--phase-transition-width", type=float, default=2.0)
    parser.add_argument("--phase-pad-factor", type=float, default=1.5)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--guidance-rescale", type=float, default=0.7)
    parser.add_argument("--decode-timestep", type=float, default=0.0)
    parser.add_argument("--decode-noise-scale", type=float, default=None)
    parser.add_argument("--pag", action="store_true")
    parser.add_argument("--pag-scale", type=float, default=0.2)
    parser.add_argument("--pag-layers", default="28")
    parser.add_argument("--no-input-preprocess", action="store_true")
    parser.add_argument("--soft-lora", action="store_true", help="Load the Enhance LoRA for the I2I pass.")
    parser.add_argument("--soft-lora-path", default=SOFT_LORA_PATH)
    parser.add_argument("--soft-lora-weight-name", default=SOFT_LORA_WEIGHT_NAME)
    parser.add_argument("--soft-lora-adapter-name", default=SOFT_LORA_ADAPTER_NAME)
    parser.add_argument("--soft-lora-scale", type=float, default=SOFT_LORA_SCALE)
    parser.add_argument("--crisp-lora", action="store_true", help="Load the Crisp LoRA for the I2I pass.")
    parser.add_argument("--crisp-lora-path", default=CRISP_LORA_PATH)
    parser.add_argument("--crisp-lora-weight-name", default=CRISP_LORA_WEIGHT_NAME)
    parser.add_argument("--crisp-lora-adapter-name", default=CRISP_LORA_ADAPTER_NAME)
    parser.add_argument("--crisp-lora-scale", type=float, default=CRISP_LORA_SCALE)
    parser.add_argument("--output-dir", default="outputs/ltx_image_modular_img2img")
    parser.add_argument("--save-metrics", action=argparse.BooleanOptionalAction, default=SAVE_METRICS)
    return parser.parse_args()


def apply_leaf_group_offload(model) -> None:
    apply_group_offloading(
        model,
        onload_device=torch.device(DEVICE),
        offload_device=torch.device(OFFLOAD_DEVICE),
        **GROUP_OFFLOAD_CONFIG,
    )


def cleanup_runtime_state(record_event, event_name: str) -> None:
    event_t0 = time.time()
    flush()
    record_event(event_name, time.time() - event_t0)


def load_lora_adapter(model, source: str, weight_name: str, adapter_name: str) -> str:
    if hasattr(model, "load_lora_adapter"):
        state_dict, metadata = LTX2LoraLoaderMixin.lora_state_dict(
            source,
            weight_name=weight_name,
            return_lora_metadata=True,
        )
        kwargs = {
            "adapter_name": adapter_name,
            "metadata": metadata,
            "low_cpu_mem_usage": LOW_CPU_MEM_USAGE,
        }
        model.load_lora_adapter(state_dict, prefix="transformer", **kwargs)
        return adapter_name
    if hasattr(model, "load_lora_weights"):
        kwargs = {"adapter_name": adapter_name}
        if weight_name:
            kwargs["weight_name"] = weight_name
        model.load_lora_weights(source, **kwargs)
        return adapter_name
    raise RuntimeError(f"{type(model).__name__} does not support LoRA loading.")


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
    names, scales = zip(*adapters)
    if not hasattr(model, "set_adapters"):
        raise RuntimeError(f"{type(model).__name__} does not support LoRA adapter activation.")
    try:
        model.set_adapters(list(names), weights=list(scales))
    except TypeError:
        model.set_adapters(list(names), adapter_weights=list(scales))
    record_event("activate_loras", 0.0, adapters=dict(adapters))


def build_run_slug(args, *, input_noise_sigma: float, input_sharpen: float, phase_cutoff: float | None) -> str:
    pag_tag = f"pag{args.pag_scale:g}_layers{args.pag_layers.replace(',', '-')}" if args.pag else "nopag"
    input_tag = Path(args.input_image).stem.replace(" ", "_")
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
            "bf16",
            "text_encoder_original",
            pag_tag,
            f"strength{args.strength:g}",
            f"noise{input_noise_sigma:g}",
            f"sharp{input_sharpen:g}",
            f"phase{phase_cutoff:g}" if phase_cutoff is not None else "phaseoff",
            f"{args.width}x{args.height}",
            f"steps{args.steps}",
            f"seed{args.seed}",
            lora_tag,
            input_tag,
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
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this runner.")
    if args.width % 32 != 0 or args.height % 32 != 0:
        raise ValueError("Width and height must be divisible by 32.")

    input_noise_sigma = 0.0 if args.no_input_preprocess else args.input_noise_sigma
    input_sharpen = 1.0 if args.no_input_preprocess else args.input_sharpen
    phase_cutoff = args.phase_cutoff if args.phase_cutoff and args.phase_cutoff > 0 else None
    pag_layers = [int(item.strip()) for item in args.pag_layers.split(",") if item.strip()]
    selected_sigmas, t_start = get_strength_sigmas(args.steps, args.strength)

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    pixel_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    run_slug = build_run_slug(
        args,
        input_noise_sigma=input_noise_sigma,
        input_sharpen=input_sharpen,
        phase_cutoff=phase_cutoff,
    )
    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / "metrics"
    output_path = output_dir / f"{run_slug}.png"
    metrics_path = metrics_dir / f"{run_slug}.json"

    run_metrics = {
        "run_slug": run_slug,
        "model_tag": MODEL_TAG,
        "model_path": MODEL_PATH,
        "workflow": "image2image",
        "input_image": args.input_image,
        "width": args.width,
        "height": args.height,
        "seed": args.seed,
        "num_inference_steps": args.steps,
        "strength": args.strength,
        "effective_denoising_steps": len(selected_sigmas),
        "t_start": t_start,
        "input_noise_sigma": input_noise_sigma,
        "input_sharpen": input_sharpen,
        "phase_cutoff": phase_cutoff,
        "phase_transition_width": args.phase_transition_width,
        "phase_pad_factor": args.phase_pad_factor,
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
        "offload_backend": "diffusers_group_offloading",
        "group_offload_config": GROUP_OFFLOAD_CONFIG.copy(),
        "events": [],
        "steps": [],
    }
    tracker = RunTracker(DEVICE, run_metrics, interval=0.1, show_metrics=SHOW_METRICS)
    record_event = tracker.record_event
    denoise_progress_callback = make_denoise_progress_callback(args, run_metrics)

    print("Using modular img2img Diffusers group offload", flush=True)
    print(f"  model_path={MODEL_PATH}", flush=True)

    t0 = tracker.step_start("Pass 0: Encode prompts")
    event_t0 = time.time()
    text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        subfolder="text_encoder",
        torch_dtype=DTYPE,
        low_cpu_mem_usage=LOW_CPU_MEM_USAGE,
    )
    record_event("load_text_encoder", time.time() - event_t0, source=MODEL_PATH)

    event_t0 = time.time()
    apply_leaf_group_offload(text_encoder)
    record_event("setup_text_encoder_group_offload", time.time() - event_t0)

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

    t0 = tracker.step_start("Pass 1: Encode input image")
    event_t0 = time.time()
    vae = AutoencoderKLLTX2Video.from_pretrained(
        MODEL_PATH,
        subfolder="vae",
        torch_dtype=DTYPE,
        low_cpu_mem_usage=LOW_CPU_MEM_USAGE,
    ).to(DEVICE)
    record_event("load_vae_to_cuda_for_encode", time.time() - event_t0, source=MODEL_PATH)

    event_t0 = time.time()
    vae_encode_pipe = LTX2ImageVaeEncoderStep().init_pipeline()
    vae_encode_pipe.update_components(vae=vae)
    record_event("build_vae_encode_modular_pipeline", time.time() - event_t0)

    event_t0 = time.time()
    image_latent_state = vae_encode_pipe(
        image=args.input_image,
        width=args.width,
        height=args.height,
        generator=generator,
        pixel_generator=pixel_generator,
        input_noise_sigma=input_noise_sigma,
        input_sharpen=input_sharpen,
        output="image_latents",
    )
    record_event("vae_encode_modular_call", time.time() - event_t0)
    image_latents = image_latent_state.to(OFFLOAD_DEVICE)
    if SHOW_METRICS:
        print(f"  image_latents shape: {image_latents.shape}", flush=True)

    del image_latent_state, vae_encode_pipe, vae
    cleanup_runtime_state(record_event, "cleanup_after_vae_encode")
    tracker.step_end("Pass 1: Encode input image", t0)

    t0 = tracker.step_start(f"Pass 2: Generate at {args.width}x{args.height}")
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
    transformer = LTX2ImageTransformer2DModel.from_pretrained(MODEL_PATH, **transformer_load_kwargs)
    record_event("load_transformer", time.time() - event_t0, source=MODEL_PATH)

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
    apply_leaf_group_offload(transformer)
    record_event("setup_transformer_group_offload", time.time() - event_t0)

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
        num_inference_steps=args.steps,
        image_latents=image_latents.to(device=DEVICE, dtype=torch.float32),
        strength=args.strength,
        phase_cutoff=phase_cutoff,
        phase_transition_width=args.phase_transition_width,
        phase_pad_factor=args.phase_pad_factor,
        batch_size=latent_batch_size,
        transformer_batch_multiplier=transformer_batch_multiplier,
        generator=generator,
        output=["latents", "timesteps", "latent_height", "latent_width", "in_channels", "video_rotary_emb"],
    )
    record_event("prepare_latents_modular_pipe_call", time.time() - event_t0)
    del image_latents

    denoise_progress_callback.timesteps = prepare_state["timesteps"]
    denoise_progress_callback.start_time = time.perf_counter()
    denoise_progress_callback.last_time = denoise_progress_callback.start_time
    denoise_progress_callback.step_times = []
    print("  Starting denoise loop", flush=True)
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

    output_latents = denoise_state.to(OFFLOAD_DEVICE)
    latent_height = prepare_state["latent_height"]
    latent_width = prepare_state["latent_width"]
    latent_channels = prepare_state["in_channels"]
    if SHOW_METRICS:
        print(f"  Image latent: {output_latents.shape}", flush=True)

    del prepare_state, denoise_state, connector_prompt_embeds, connector_attention_mask
    del prepare_pipe, denoise_pipe, transformer, scheduler
    cleanup_runtime_state(record_event, "cleanup_before_vae_decode")
    tracker.step_end(f"Pass 2: Generate at {args.width}x{args.height}", t0)

    t0 = tracker.step_start("Pass 3: Decode VAE")
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
        latents=output_latents.to(device=DEVICE, dtype=DTYPE),
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

    del decode_pipe, vae, output_latents
    cleanup_runtime_state(record_event, "cleanup_after_vae_decode")
    tracker.step_end("Pass 3: Decode VAE", t0)

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
    if args.save_metrics:
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
