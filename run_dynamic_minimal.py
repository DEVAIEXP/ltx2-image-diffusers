"""
Minimal LTX 2.3 modular image runner using diffusers-dynamic-offloader.

This is the small integration shape a host app should need:
- let DDO handle preset/lifecycle decisions
- call enable_offload(...) for each large component
"""

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from diffusers import AutoencoderKLLTX2Video, FlowMatchEulerDiscreteScheduler
from diffusers.models.attention_dispatch import AttentionBackendName, attention_backend
from diffusers_dynamic_offloader import (
    enable_offload,
    from_pretrained_with_dynamic_offload,
    maybe_purge_windows_standby_cache,
    remove_dynamic_offload,
)
from transformers import Gemma3ForConditionalGeneration, GemmaTokenizerFast

from custom_blocks.ltx2_image import LTX2ImageDistilledBlocks, LTX2ImageTextEncoderStep
from custom_blocks.ltx2_image.connectors_ltx2_image import LTX2ImageTextConnectors
from custom_blocks.ltx2_image.modular_blocks_ltx2_image import (
    LTX2ImageConnectorStep,
    LTX2ImageDenoiseStep,
    LTX2ImagePrepareLatentsStep,
)
from inference_utils import RunTracker, flush

MODEL_PATH = os.getenv("MODEL_PATH", r"elismasilva/ltx2.3-image-distilled-1.1")
OUTPUT_DIR = Path("outputs/ltx_image_modular")

DEVICE = "cuda:0"
OFFLOAD_DEVICE = "cpu"
DTYPE = torch.bfloat16
PRESET = "auto"
SHOW_METRICS = True
SAVE_METRICS = False
SHOW_DENOISE_STEPS = True

WIDTH = 1280
HEIGHT = 704
STEPS = 8
SEED = 43
GUIDANCE_SCALE = 1.0
GUIDANCE_RESCALE = 0.7

PROMPT = (
    "Fisheye close-up of a calico cat wearing a tiny flower crown, sniffing the camera lens "
    "in a sunny park, with bright colors, realistic fur detail, and playful viral-pet energy."
)
NEGATIVE_PROMPT = ""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--preset", default=PRESET)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--negative-prompt", default=NEGATIVE_PROMPT)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--guidance-scale", type=float, default=GUIDANCE_SCALE)
    parser.add_argument("--guidance-rescale", type=float, default=GUIDANCE_RESCALE)
    parser.add_argument("--ddo-profile", default=None, help="Named DDO denoise workload profile.")
    parser.add_argument("--build-ddo-profile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ddo-profile-steps", type=int, default=2)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--show-metrics", action=argparse.BooleanOptionalAction, default=SHOW_METRICS)
    parser.add_argument("--save-metrics", action=argparse.BooleanOptionalAction, default=SAVE_METRICS)
    parser.add_argument("--show-denoise-steps", action=argparse.BooleanOptionalAction, default=SHOW_DENOISE_STEPS)
    return parser.parse_args()


def cleanup(record_event, name: str, *, repeats: int = 1) -> None:
    event_t0 = time.time()
    for _ in range(repeats):
        flush()
        if torch.cuda.is_available() and hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    record_event(name, time.time() - event_t0, repeats=repeats)


def denoise_callback(components, step_index, timestep, callback_kwargs):
    now = time.perf_counter()
    last_time = getattr(denoise_callback, "last_time", now)
    start_time = getattr(denoise_callback, "start_time", last_time)
    elapsed = now - last_time
    denoise_callback.last_time = now
    avg = (now - start_time) / (step_index + 1)
    alloc = torch.cuda.memory_allocated(DEVICE) / 1024**3
    reserved = torch.cuda.memory_reserved(DEVICE) / 1024**3
    if not SHOW_DENOISE_STEPS:
        return callback_kwargs
    print(
        f"  [denoise] step {step_index + 1}/{STEPS} elapsed={elapsed:.4f}s "
        f"avg={avg:.4f}s/it torch_alloc={alloc:.2f} GiB torch_reserved={reserved:.2f} GiB",
        flush=True,
    )
    return callback_kwargs


def apply_native_attention_backend(transformer):
    patched = 0
    transformer.drop_trivial_attention_mask = False
    for module in transformer.modules():
        processor = getattr(module, "processor", None)
        if processor is not None and hasattr(processor, "_attention_backend"):
            processor._attention_backend = AttentionBackendName.NATIVE
            patched += 1
    return patched


def print_offload_route(component: str, result) -> None:
    settings = result.settings
    preset = "none" if settings is None else settings.effective_preset
    print(
        f"  [ddo] {component}: route={result.route} preset={preset}",
        flush=True,
    )


def main():
    global MODEL_PATH, OUTPUT_DIR, DEVICE, PRESET, WIDTH, HEIGHT, STEPS, SEED
    global GUIDANCE_SCALE, GUIDANCE_RESCALE, PROMPT, NEGATIVE_PROMPT
    global SHOW_METRICS, SAVE_METRICS, SHOW_DENOISE_STEPS

    args = parse_args()
    if args.build_ddo_profile and not args.ddo_profile:
        raise SystemExit("--build-ddo-profile requires --ddo-profile NAME")
    MODEL_PATH = args.model_path
    OUTPUT_DIR = Path(args.output_dir)
    DEVICE = args.device
    PRESET = args.preset
    WIDTH = args.width
    HEIGHT = args.height
    STEPS = args.steps
    active_steps = max(1, min(STEPS, args.ddo_profile_steps)) if args.build_ddo_profile else STEPS
    SEED = args.seed
    GUIDANCE_SCALE = args.guidance_scale
    GUIDANCE_RESCALE = args.guidance_rescale
    PROMPT = args.prompt
    NEGATIVE_PROMPT = args.negative_prompt
    SHOW_METRICS = args.show_metrics
    SAVE_METRICS = args.save_metrics
    SHOW_DENOISE_STEPS = args.show_denoise_steps

    run_metrics = {"events": [], "steps": []}
    tracker = RunTracker(DEVICE, run_metrics, interval=0.1, show_metrics=SHOW_METRICS)
    record_event = tracker.record_event

    maybe_purge_windows_standby_cache("before_run", preset=PRESET, record_event=record_event)

    generator = torch.Generator(device="cpu").manual_seed(SEED)

    t0 = tracker.step_start("Pass 0: Encode prompts")
    event_t0 = time.time()
    text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        subfolder="text_encoder",
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    )
    record_event("load_text_encoder", time.time() - event_t0)

    text_encoder_result = enable_offload(
        text_encoder,
        preset=PRESET,
        component="text_encoder",
        execution_device=DEVICE,
        offload_device=OFFLOAD_DEVICE,
        low_cpu_mem_usage=True,
        record_event=record_event,
    )
    text_encoder = text_encoder_result.module
    print_offload_route("text_encoder", text_encoder_result)

    event_t0 = time.time()
    tokenizer = GemmaTokenizerFast.from_pretrained(MODEL_PATH, subfolder="tokenizer")
    record_event("load_tokenizer", time.time() - event_t0)

    prompt_pipe = LTX2ImageTextEncoderStep().init_pipeline()
    prompt_pipe.update_components(text_encoder=text_encoder, tokenizer=tokenizer)

    event_t0 = time.time()
    with torch.inference_mode():
        prompt_state = prompt_pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            guidance_scale=GUIDANCE_SCALE,
            output=["prompt_embeds", "prompt_attention_mask"],
        )
    record_event("encode_prompt_call", time.time() - event_t0)

    prompt_embeds = prompt_state["prompt_embeds"].to(OFFLOAD_DEVICE)
    prompt_attention_mask = prompt_state["prompt_attention_mask"].to(OFFLOAD_DEVICE)

    if text_encoder_result.route == "dynamic_offload":
        remove_dynamic_offload(text_encoder)
    del text_encoder_result
    del prompt_state, prompt_pipe, text_encoder, tokenizer
    cleanup(record_event, "cleanup_after_text_encoder")
    maybe_purge_windows_standby_cache("after_text_encoder", preset=PRESET, record_event=record_event)
    tracker.step_end("Pass 0: Encode prompts", t0)

    t0 = tracker.step_start(f"Pass 1: Generate at {WIDTH}x{HEIGHT}")
    event_t0 = time.time()
    connectors = LTX2ImageTextConnectors.from_pretrained(
        MODEL_PATH,
        subfolder="connectors",
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    record_event("load_connectors_to_cuda", time.time() - event_t0)

    connector_pipe = LTX2ImageConnectorStep().init_pipeline()
    connector_pipe.update_components(connectors=connectors)
    event_t0 = time.time()
    connector_state = connector_pipe(
        prompt_embeds=prompt_embeds.to(device=DEVICE, dtype=DTYPE),
        prompt_attention_mask=prompt_attention_mask.to(device=DEVICE),
        negative_prompt_embeds=None,
        negative_prompt_attention_mask=None,
        do_classifier_free_guidance=False,
        pag_scale=0.0,
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
    del prompt_embeds, prompt_attention_mask, connector_state, connector_pipe, connectors
    cleanup(record_event, "cleanup_after_connectors")

    maybe_purge_windows_standby_cache("before_transformer", preset=PRESET, record_event=record_event)

    event_t0 = time.time()
    transformer_load = from_pretrained_with_dynamic_offload(
        MODEL_PATH,
        subfolder="transformer",
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
        device_map="cpu",
        apply_dynamic=False,
    )
    transformer = transformer_load.module
    del transformer_load
    record_event("load_transformer", time.time() - event_t0)

    event_t0 = time.time()
    patched_attention_processors = apply_native_attention_backend(transformer)
    record_event("set_transformer_attention_backend", time.time() - event_t0, patched_processors=patched_attention_processors)

    event_t0 = time.time()
    transformer_result = enable_offload(
        transformer,
        preset=PRESET,
        component="transformer",
        execution_device=DEVICE,
        offload_device=OFFLOAD_DEVICE,
        low_cpu_mem_usage=True,
        record_event=record_event,
        profile_name=args.ddo_profile or "",
        build_profile=args.build_ddo_profile,
    )
    transformer = transformer_result.module
    print_offload_route("transformer", transformer_result)
    transformer_hook = transformer_result.hook
    transformer_route = transformer_result.route

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")
    prepare_pipe = LTX2ImagePrepareLatentsStep().init_pipeline()
    denoise_pipe = LTX2ImageDenoiseStep().init_pipeline()
    prepare_pipe.update_components(transformer=transformer, scheduler=scheduler)
    denoise_pipe.update_components(transformer=transformer, scheduler=scheduler)

    prepare_state = prepare_pipe(
        width=WIDTH,
        height=HEIGHT,
        num_inference_steps=active_steps,
        batch_size=latent_batch_size,
        transformer_batch_multiplier=transformer_batch_multiplier,
        generator=generator,
        output=["latents", "timesteps", "latent_height", "latent_width", "in_channels", "video_rotary_emb"],
    )

    print("  Starting denoise loop with attention backend: native", flush=True)
    denoise_callback.start_time = time.perf_counter()
    denoise_callback.last_time = denoise_callback.start_time
    event_t0 = time.time()
    with transformer_result.profile_run() as ddo_profile_report, attention_backend(AttentionBackendName.NATIVE):
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
            pag_scale=0.0,
            pag_applied_layers=None,
            callback_on_step_end=denoise_callback,
            callback_on_step_end_tensor_inputs=["latents"],
            output="latents",
    )
    record_event("denoise_modular_pipe_call", time.time() - event_t0)
    run_metrics["ddo_profile"] = ddo_profile_report
    if transformer_hook is not None:
        transformer_hook.print_profile_summary()

    image_latent = denoise_state.to(OFFLOAD_DEVICE)
    latent_height = prepare_state["latent_height"]
    latent_width = prepare_state["latent_width"]
    latent_channels = prepare_state["in_channels"]

    if transformer_route == "dynamic_offload":
        remove_dynamic_offload(transformer)
    transformer_hook = None
    del prepare_state, denoise_state, prepare_pipe, denoise_pipe, transformer, scheduler
    del transformer_result
    del connector_prompt_embeds, connector_attention_mask
    cleanup(record_event, "cleanup_before_vae_decode")
    tracker.step_end(f"Pass 1: Generate at {WIDTH}x{HEIGHT}", t0)

    if args.build_ddo_profile:
        print(f"  DDO profile calibration complete: {ddo_profile_report}")
        return

    t0 = tracker.step_start("Pass 2: Decode VAE")
    event_t0 = time.time()
    vae = AutoencoderKLLTX2Video.from_pretrained(
        MODEL_PATH,
        subfolder="vae",
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    record_event("load_vae_to_cuda", time.time() - event_t0)

    decode_pipe = LTX2ImageDistilledBlocks().sub_blocks["decode"].init_pipeline()
    decode_pipe.update_components(vae=vae)
    event_t0 = time.time()
    image = decode_pipe(
        latents=image_latent.to(device=DEVICE, dtype=DTYPE),
        batch_size=latent_batch_size,
        latent_height=latent_height,
        latent_width=latent_width,
        in_channels=latent_channels,
        decode_timestep=0.0,
        decode_noise_scale=None,
        generator=generator,
        output_type="pil",
        output="images",
    )[0]
    record_event("vae_decode_modular_call", time.time() - event_t0)
    del decode_pipe, vae, image_latent
    cleanup(record_event, "cleanup_after_vae_decode")
    tracker.step_end("Pass 2: Decode VAE", t0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"ddo_minimal_{WIDTH}x{HEIGHT}_steps{STEPS}_seed{SEED}.png"
    image.save(output_path)

    print("\n" + "=" * 70)
    print(f"  TOTAL: {tracker.total_elapsed():.1f}s")
    print(f"  Output: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
