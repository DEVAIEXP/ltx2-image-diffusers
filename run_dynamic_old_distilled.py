"""Traditional LTX 2.3 distilled pipeline runner using diffusers-dynamic-offloader."""

import argparse
from contextlib import nullcontext
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_MODULES_CACHE", str((Path(__file__).parent / ".hf_modules").resolve()))

import torch
from diffusers.models.transformers import LTX2ImageTransformer2DModel
from diffusers.pipelines.ltx2.pipeline_ltx2_image import LTX2ImagePipeline
from diffusers_dynamic_offloader import (
    DynamicOffloadSettings,
    enable_offload,
    enable_pipeline_offload,
    format_dynamic_offload_presets,
    from_pretrained_with_dynamic_offload,
    is_wsl_environment,
    maybe_purge_windows_standby_cache,
    remove_dynamic_offload,
)

from inference_utils import RunTracker, flush

DEVICE = "cuda:0"
OFFLOAD_DEVICE = "cpu"
DTYPE = torch.bfloat16

MODEL_TAG = "distilled_dynamic_old_pipeline"
MODEL_PATH = os.getenv("MODEL_PATH", r"elismasilva/ltx2.3-image-distilled-1.1")
DEFAULT_COMPONENTS = "auto"
PIPELINE_COMPONENT_POLICIES = {
    "text_encoder": {"route": "diffusers_group_offload", "offload_type": "leaf_level", "offload_stream": "1"},
    "text_encoder_2": {"route": "diffusers_group_offload", "offload_type": "leaf_level", "offload_stream": "1"},
    "connectors": {"route": "diffusers_group_offload", "offload_type": "leaf_level", "offload_stream": "1"},
    "transformer": {"route": "auto"},
    "unet": {"route": "auto"},
    "vae": {"route": "diffusers_group_offload", "offload_type": "leaf_level", "offload_stream": "1"},
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

OUTPUT_DIR = Path("outputs/ltx_image_dynamic_old")
DEFAULT_PROMPT = (
    "Fisheye close-up of a calico cat wearing a tiny flower crown, sniffing the camera lens "
    "in a sunny park, with bright colors, realistic fur detail, and playful viral-pet energy."
)
DEFAULT_NEGATIVE_PROMPT = ""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.environ.get("DDO_RUNNER_MODEL_PATH", MODEL_PATH))
    parser.add_argument("--preset", default=os.environ.get("DDO_PRESET", "auto"))
    parser.add_argument("--components", default=DEFAULT_COMPONENTS)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--steps", type=int, default=NUM_INFERENCE_STEPS)
    parser.add_argument("--guidance-scale", type=float, default=GUIDANCE_SCALE)
    parser.add_argument("--guidance-rescale", type=float, default=GUIDANCE_RESCALE)
    parser.add_argument("--ddo-profile", default=None, help="Named DDO denoise workload profile.")
    parser.add_argument("--build-ddo-profile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ddo-profile-steps", type=int, default=2)
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


def parse_components(value: str) -> tuple[str, ...] | None:
    normalized = value.strip().lower()
    if normalized in {"", "auto"}:
        return None
    if normalized == "none":
        return []
    return tuple(item.strip() for item in value.split(",") if item.strip())


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


def print_routes(results: dict) -> None:
    for name, result in results.items():
        settings = result.settings
        preset = "none" if settings is None else settings.effective_preset
        print(f"  [ddo] {name}: route={result.route} preset={preset}", flush=True)


def main():
    args = parse_args()
    if args.build_ddo_profile and not args.ddo_profile:
        raise SystemExit("--build-ddo-profile requires --ddo-profile NAME")
    active_steps = max(1, min(args.steps, args.ddo_profile_steps)) if args.build_ddo_profile else args.steps
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
    components = parse_components(args.components)
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
        "pipeline": "LTX2ImagePipeline",
        "components": "auto" if components is None else list(components),
        "width": args.width,
        "height": args.height,
        "seed": seed,
        "num_inference_steps": args.steps,
        "active_inference_steps": active_steps,
        "ddo_profile_name": args.ddo_profile,
        "build_ddo_profile": args.build_ddo_profile,
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

    print(f"Using DDO old-pipeline preset: {settings.effective_preset}", flush=True)
    print(f"  model_path={args.model_path}", flush=True)
    print(f"  components={'auto' if components is None else list(components)}", flush=True)
    print(f"  torch allocated before load: {torch.cuda.memory_allocated(DEVICE) / (1024**3):.2f} GB", flush=True)

    maybe_purge_windows_standby_cache(settings, "before_run", record_event=record_event)

    t0 = tracker.step_start("Load Pipeline")
    event_t0 = time.time()
    transformer_load = from_pretrained_with_dynamic_offload(
        args.model_path,
        model_loader=LTX2ImageTransformer2DModel,
        dynamic_offload_config=settings.config if settings.enabled else None,
        apply_dynamic=False,
        subfolder="transformer",
        torch_dtype=DTYPE,
        device_map="cpu",
    )
    transformer = transformer_load.module
    record_event(
        "load_transformer_with_dynamic_offload",
        time.time() - event_t0,
        source=args.model_path,
        device_map="cpu",
        loader="LTX2ImageTransformer2DModel",
    )

    event_t0 = time.time()
    pipe = LTX2ImagePipeline.from_pretrained(
        args.model_path,
        transformer=transformer,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    )
    record_event(
        "load_pipeline_without_transformer",
        time.time() - event_t0,
        model_path=args.model_path,
    )

    event_t0 = time.time()
    transformer_result = None
    if args.ddo_profile:
        component_names = (
            ("text_encoder", "text_encoder_2", "connectors", "unet", "vae")
            if components is None
            else tuple(name for name in components if name != "transformer")
        )
        offload_results = enable_pipeline_offload(
            pipe,
            settings=settings,
            components=component_names,
            execution_device=DEVICE,
            offload_device=OFFLOAD_DEVICE,
            low_cpu_mem_usage=True,
            record_event=record_event,
        )
        transformer_result = enable_offload(
            pipe.transformer,
            settings=settings,
            component="transformer",
            execution_device=DEVICE,
            offload_device=OFFLOAD_DEVICE,
            low_cpu_mem_usage=True,
            record_event=record_event,
            profile_name=args.ddo_profile,
            build_profile=args.build_ddo_profile,
        )
        offload_results["transformer"] = transformer_result
    else:
        offload_results = enable_pipeline_offload(
            pipe,
            settings=settings,
            components=components,
            execution_device=DEVICE,
            offload_device=OFFLOAD_DEVICE,
            low_cpu_mem_usage=True,
            record_event=record_event,
        )
    record_event("setup_pipeline_dynamic_offload", time.time() - event_t0, components="auto" if components is None else list(components))
    print_routes(offload_results)
    tracker.step_end("Load Pipeline", t0)

    callback = make_denoise_callback(DEVICE, args.steps, args.show_denoise_steps)
    callback.start_time = time.perf_counter()
    callback.last_time = callback.start_time

    t0 = tracker.step_start(f"Generate at {args.width}x{args.height}")
    event_t0 = time.time()
    profile_context = transformer_result.profile_run() if transformer_result is not None else nullcontext(None)
    with profile_context as ddo_profile_report:
        images = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=active_steps,
        guidance_scale=args.guidance_scale,
        guidance_rescale=args.guidance_rescale,
        decode_timestep=args.decode_timestep,
        decode_noise_scale=args.decode_noise_scale,
        pag_scale=args.pag_scale if args.pag else 0.0,
        pag_applied_layers=[int(item.strip()) for item in args.pag_layers.split(",") if item.strip()] if args.pag else None,
        generator=generator,
        output_type="latent" if args.build_ddo_profile else "pil",
        return_dict=False,
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=["latents"],
        )[0]
    run_metrics["ddo_profile"] = ddo_profile_report
    image = images[0] if isinstance(images, list) else images
    record_event("pipeline_call", time.time() - event_t0)
    run_metrics["denoise_step_times"] = callback.step_times
    tracker.step_end(f"Generate at {args.width}x{args.height}", t0)

    if args.build_ddo_profile:
        for result in offload_results.values():
            if result.route == "dynamic_offload":
                remove_dynamic_offload(result.module)
        del pipe
        flush()
        maybe_purge_windows_standby_cache(settings, "after_run", record_event=record_event)
        print(f"  DDO profile calibration complete: {ddo_profile_report}")
        return

    t0 = tracker.step_start("Save Image")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run_slug}.png"
    image.save(output_path)
    record_event("save_image", 0.0, output_path=str(output_path))
    tracker.step_end("Save Image", t0)

    for result in offload_results.values():
        if result.route == "dynamic_offload":
            remove_dynamic_offload(result.module)
    del pipe
    flush()
    maybe_purge_windows_standby_cache(settings, "after_run", record_event=record_event)

    total_time = tracker.total_elapsed()
    run_metrics["total_elapsed_sec"] = round(total_time, 4)
    run_metrics["global_peak_vram_gb"] = round(tracker.global_peak_vram, 4)
    run_metrics["global_peak_ram_gb"] = round(tracker.global_peak_ram, 4)
    run_metrics["output_path"] = str(output_path)

    if save_metrics:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = metrics_dir / f"{run_slug}.json"
        metrics_path.write_text(json.dumps(run_metrics, indent=2), encoding="utf-8")
    else:
        metrics_path = None

    print("\n" + "=" * 70)
    print(f"  TOTAL: {total_time:.1f}s | Peak VRAM: {tracker.global_peak_vram:.2f} GB | Peak RAM: {tracker.global_peak_ram:.2f} GB")
    print(f"  Output: {output_path}")
    if metrics_path is not None:
        print(f"  Metrics JSON: {metrics_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
