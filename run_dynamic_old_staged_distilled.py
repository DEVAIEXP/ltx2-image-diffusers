"""Staged traditional LTX 2.3 distilled runner using diffusers-dynamic-offloader.

This keeps the regular Diffusers LTX2ImagePipeline components, but orchestrates
encode, denoise, and VAE decode as separate stages so DDO and memory cleanup can
be applied between components.
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_MODULES_CACHE", str((Path(__file__).parent / ".hf_modules").resolve()))

import torch
from diffusers import AutoencoderKLLTX2Video
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.transformers import LTX2ImageTransformer2DModel
from diffusers.pipelines.ltx2.pipeline_ltx2_image import LTX2ImagePipeline
from diffusers_dynamic_offloader import (
    DynamicOffloadSettings,
    enable_offload,
    enable_pipeline_offload,
    format_dynamic_offload_presets,
    is_wsl_environment,
    maybe_purge_windows_standby_cache,
    remove_dynamic_offload,
)
from transformers import Gemma3ForConditionalGeneration

from inference_utils import RunTracker, flush

DEVICE = "cuda:0"
OFFLOAD_DEVICE = "cpu"
DTYPE = torch.bfloat16

MODEL_TAG = "distilled_dynamic_old_staged"
MODEL_PATH = os.getenv("MODEL_PATH", r"elismasilva/ltx2.3-image-distilled-1.1")
OUTPUT_DIR = Path("outputs/ltx_image_dynamic_old_staged")

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

PIPELINE_COMPONENT_POLICIES = {
    "text_encoder": {"route": "diffusers_group_offload", "offload_type": "leaf_level", "offload_stream": "1"},
    "connectors": {"route": "diffusers_group_offload", "offload_type": "leaf_level", "offload_stream": "1"},
    "transformer": {"route": "auto"},
}

DEFAULT_PROMPT = (
    "Fisheye close-up of a calico cat wearing a tiny flower crown, sniffing the camera lens "
    "in a sunny park, with bright colors, realistic fur detail, and playful viral-pet energy."
)
DEFAULT_NEGATIVE_PROMPT = ""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.environ.get("DDO_RUNNER_MODEL_PATH", MODEL_PATH))
    parser.add_argument("--preset", default=os.environ.get("DDO_PRESET", "auto"))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
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
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--metrics-level", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--show-metrics", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-metrics", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--show-denoise-steps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-presets", action="store_true")
    return parser.parse_args()


def build_run_slug(args, seed: int) -> str:
    pag_tag = f"pag{args.pag_scale:g}_layers{args.pag_layers.replace(',', '-')}" if args.pag else "nopag"
    preset_tag = args.preset.replace("/", "-").replace("\\", "-")
    return "_".join(
        [
            "ltx23_image",
            MODEL_TAG,
            preset_tag,
            "bf16",
            pag_tag,
            f"{args.width}x{args.height}",
            f"steps{args.steps}",
            f"seed{seed}",
        ]
    )


def make_denoise_callback(device: str, total_steps: int, enabled: bool):
    def callback(*callback_args):
        if len(callback_args) == 4:
            _, step_index, timestep, callback_kwargs = callback_args
        else:
            step_index, timestep, callback_kwargs = callback_args

        now = time.perf_counter()
        elapsed = now - callback.last_time
        callback.last_time = now
        callback.step_times.append(elapsed)
        if enabled:
            avg = (now - callback.start_time) / (step_index + 1)
            allocated = torch.cuda.memory_allocated(device) / (1024**3)
            reserved = torch.cuda.memory_reserved(device) / (1024**3)
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
    return callback


def cleanup_runtime_state(record_event, event_name: str, *, repeats: int = 1) -> None:
    event_t0 = time.time()
    for _ in range(max(1, repeats)):
        flush()
        if torch.cuda.is_available() and hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    record_event(event_name, time.time() - event_t0, repeats=repeats)


def print_routes(results: dict) -> None:
    for name, result in results.items():
        settings = result.settings
        preset = "none" if settings is None else settings.effective_preset
        print(f"  [ddo] {name}: route={result.route} preset={preset}", flush=True)


def main():
    args = parse_args()
    show_metrics = args.show_metrics if args.show_metrics is not None else args.metrics_level >= 1
    save_metrics = args.save_metrics if args.save_metrics is not None else args.metrics_level >= 2
    running_on_wsl = is_wsl_environment()
    if args.print_presets:
        print(format_dynamic_offload_presets(default_preset=args.preset, running_on_wsl=running_on_wsl))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this runner.")

    seed = args.seed or torch.randint(0, 2**32, (1,)).item()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / "metrics"
    run_slug = build_run_slug(args, seed)
    settings = DynamicOffloadSettings.from_env(
        execution_device=DEVICE,
        offload_device=OFFLOAD_DEVICE,
        running_on_wsl=running_on_wsl,
        default_preset=args.preset,
        component_policies=PIPELINE_COMPONENT_POLICIES,
    )

    run_metrics = {
        "run_slug": run_slug,
        "model_tag": MODEL_TAG,
        "model_path": args.model_path,
        "pipeline": "LTX2ImagePipeline staged traditional",
        "width": args.width,
        "height": args.height,
        "seed": seed,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "guidance_rescale": args.guidance_rescale,
        "vae_decode_timestep": args.decode_timestep,
        "vae_decode_noise_scale": args.decode_noise_scale,
        "pag_enabled": args.pag,
        "pag_scale": args.pag_scale if args.pag else 0.0,
        "pag_applied_layers": args.pag_layers if args.pag else None,
        "dtype": str(DTYPE),
        "running_on_wsl": running_on_wsl,
        **settings.as_metrics(),
        "events": [],
        "steps": [],
    }
    tracker = RunTracker(DEVICE, run_metrics, interval=0.1, show_metrics=show_metrics)
    record_event = tracker.record_event

    print(f"Using DDO old-staged preset: {settings.effective_preset}", flush=True)
    print(f"  model_path={args.model_path}", flush=True)
    maybe_purge_windows_standby_cache(settings, "before_run", record_event=record_event)

    t0 = tracker.step_start("Pass 0: Encode prompts")
    event_t0 = time.time()
    text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
        args.model_path,
        subfolder="text_encoder",
        torch_dtype=DTYPE,
    )
    record_event("load_text_encoder", time.time() - event_t0, source=args.model_path)

    event_t0 = time.time()
    text_result = enable_offload(
        text_encoder,
        settings=settings,
        component="text_encoder",
        execution_device=DEVICE,
        offload_device=OFFLOAD_DEVICE,
        low_cpu_mem_usage=True,
        record_event=record_event,
    )
    print_routes({"text_encoder": text_result})

    event_t0 = time.time()
    embeds_pipe = LTX2ImagePipeline.from_pretrained(
        args.model_path,
        text_encoder=text_encoder,
        transformer=None,
        connectors=None,
        vae=None,
        scheduler=None,
        torch_dtype=DTYPE,
    )
    record_event("build_prompt_pipeline", time.time() - event_t0, model_path=args.model_path)

    event_t0 = time.time()
    with torch.inference_mode():
        prompt_embeds, prompt_attention_mask, _, _ = embeds_pipe.encode_prompt(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            do_classifier_free_guidance=False,
        )
    record_event("encode_prompt_call", time.time() - event_t0, classifier_free_guidance=False)
    prompt_embeds = prompt_embeds.to(OFFLOAD_DEVICE)
    prompt_attention_mask = prompt_attention_mask.to(OFFLOAD_DEVICE)
    print(f"  prompt_embeds shape: {prompt_embeds.shape}", flush=True)

    del text_result
    del embeds_pipe, text_encoder
    cleanup_runtime_state(record_event, "cleanup_after_text_encoder")
    maybe_purge_windows_standby_cache(settings, "after_text_encoder", record_event=record_event)
    tracker.step_end("Pass 0: Encode prompts", t0)

    t0 = tracker.step_start(f"Pass 1: Generate at {args.width}x{args.height}")
    event_t0 = time.time()
    transformer = LTX2ImageTransformer2DModel.from_pretrained(
        args.model_path,
        subfolder="transformer",
        torch_dtype=DTYPE,
        device_map="cpu",
    )
    record_event("load_transformer", time.time() - event_t0, source=args.model_path)

    event_t0 = time.time()
    denoise_pipe = LTX2ImagePipeline.from_pretrained(
        args.model_path,
        transformer=transformer,
        text_encoder=None,
        tokenizer=None,
        vae=None,
        torch_dtype=DTYPE,
    )
    record_event("build_denoise_pipeline", time.time() - event_t0, model_path=args.model_path)

    event_t0 = time.time()
    denoise_results = enable_pipeline_offload(
        denoise_pipe,
        settings=settings,
        components=("connectors", "transformer"),
        execution_device=DEVICE,
        offload_device=OFFLOAD_DEVICE,
        low_cpu_mem_usage=True,
        record_event=record_event,
    )
    print_routes(denoise_results)
    record_event("setup_denoise_dynamic_offload", time.time() - event_t0)
    maybe_purge_windows_standby_cache(settings, "before_transformer", record_event=record_event)

    callback = make_denoise_callback(DEVICE, args.steps, args.show_denoise_steps)
    callback.start_time = time.perf_counter()
    callback.last_time = callback.start_time

    event_t0 = time.time()
    image_latent = denoise_pipe(
        prompt_embeds=prompt_embeds.to(device=DEVICE, dtype=DTYPE),
        prompt_attention_mask=prompt_attention_mask.to(device=DEVICE),
        negative_prompt_embeds=None,
        negative_prompt_attention_mask=None,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        guidance_rescale=args.guidance_rescale,
        decode_timestep=args.decode_timestep,
        decode_noise_scale=args.decode_noise_scale,
        pag_scale=args.pag_scale if args.pag else 0.0,
        pag_applied_layers=[int(item.strip()) for item in args.pag_layers.split(",") if item.strip()] if args.pag else None,
        generator=generator,
        output_type="latent",
        return_dict=False,
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=["latents"],
    )[0]
    record_event("denoise_pipe_call", time.time() - event_t0)
    run_metrics["denoise_step_times"] = callback.step_times
    print(f"  Image latent: {image_latent.shape}", flush=True)
    image_latent = image_latent.to(OFFLOAD_DEVICE)

    for result in denoise_results.values():
        if result.route == "dynamic_offload":
            remove_dynamic_offload(result.module)
    del denoise_results
    del denoise_pipe, transformer, prompt_embeds, prompt_attention_mask
    cleanup_runtime_state(record_event, "cleanup_before_vae_decode")
    tracker.step_end(f"Pass 1: Generate at {args.width}x{args.height}", t0)

    t0 = tracker.step_start("Pass 2: Decode VAE")
    event_t0 = time.time()
    vae = AutoencoderKLLTX2Video.from_pretrained(
        args.model_path,
        subfolder="vae",
        torch_dtype=DTYPE,
    ).to(DEVICE)
    record_event("load_vae_to_cuda", time.time() - event_t0, source=args.model_path)

    event_t0 = time.time()
    with torch.no_grad():
        latents_gpu = image_latent.to(device=DEVICE, dtype=DTYPE)
        if not vae.config.timestep_conditioning:
            vae_decode_timestep = None
        else:
            vae_decode_timestep = torch.tensor(
                [args.decode_timestep] * latents_gpu.shape[0],
                device=DEVICE,
                dtype=latents_gpu.dtype,
            )
            effective_noise_scale = args.decode_timestep if args.decode_noise_scale is None else args.decode_noise_scale
            if effective_noise_scale != 0.0:
                noise = torch.randn(latents_gpu.shape, generator=generator, device=DEVICE, dtype=latents_gpu.dtype)
                latents_gpu = (1 - effective_noise_scale) * latents_gpu + effective_noise_scale * noise

        latents_mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(DEVICE)
        latents_std = vae.latents_std.view(1, -1, 1, 1, 1).to(DEVICE)
        latents_gpu = (latents_gpu * latents_std) / vae.config.scaling_factor + latents_mean
        decoded = vae.decode(latents_gpu.to(vae.dtype), vae_decode_timestep, return_dict=False)[0]
    record_event("vae_decode_call", time.time() - event_t0)

    image_processor = VaeImageProcessor(vae_scale_factor=vae.spatial_compression_ratio)
    image = image_processor.postprocess(decoded[:, :, 0, :, :].cpu(), output_type="pil")[0]

    del vae, image_latent, latents_gpu, decoded
    cleanup_runtime_state(record_event, "cleanup_after_vae_decode")
    tracker.step_end("Pass 2: Decode VAE", t0)

    t0 = tracker.step_start("Save Image")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run_slug}.png"
    image.save(output_path)
    record_event("save_image", 0.0, output_path=str(output_path))
    tracker.step_end("Save Image", t0)

    maybe_purge_windows_standby_cache(settings, "after_run", record_event=record_event)
    total_time = tracker.total_elapsed()
    run_metrics["total_elapsed_sec"] = round(total_time, 4)
    run_metrics["global_peak_vram_gb"] = round(tracker.global_peak_vram, 4)
    run_metrics["global_peak_ram_gb"] = round(tracker.global_peak_ram, 4)
    run_metrics["output_path"] = str(output_path)

    metrics_path = None
    if save_metrics:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = metrics_dir / f"{run_slug}.json"
        metrics_path.write_text(json.dumps(run_metrics, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"  TOTAL: {total_time:.1f}s | Peak VRAM: {tracker.global_peak_vram:.2f} GB | Peak RAM: {tracker.global_peak_ram:.2f} GB")
    print(f"  Output: {output_path}")
    if metrics_path is not None:
        print(f"  Metrics JSON: {metrics_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
