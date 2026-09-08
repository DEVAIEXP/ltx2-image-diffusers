"""
Single-call LTX 2 image modular SDNQ text-to-image runner.

This runner uses the custom LTX2ImageAutoBlocks metadata and loads the
quantized transformer from the selected SDNQ model repository. It intentionally
uses Diffusers' standard component loading and group offloading instead of DDO.
"""

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_MODULES_CACHE", str((Path(__file__).parent / ".hf_modules").resolve()))

import torch
from diffusers import ModularPipeline
from diffusers.hooks import apply_group_offloading

from inference_utils import RunTracker, flush, get_sdnq_version


DEVICE = torch.device("cuda:0")
OFFLOAD_DEVICE = torch.device("cpu")
DTYPE = torch.bfloat16

MODEL_TAG = "base_modular_sdnq"
MODEL_PATH = "elismasilva/ltx2.3-image-base"
CUSTOM_BLOCKS_PATH = "elismasilva/ltx2.3_image_custom_blocks"
SDNQ_MODEL_PATHS = {
    4: "elismasilva/ltx2.3-image-base-sdnq-int4",
    8: "elismasilva/ltx2.3-image-base-sdnq-int8",
}

DEFAULT_PROMPT = (
    "Fisheye close-up of a calico cat wearing a tiny flower crown, sniffing the camera lens "
    "in a sunny park, with bright colors, realistic fur detail, and playful viral-pet energy."
)
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, "
    "excessive noise, grainy texture, poor lighting, distorted proportions, artifacts"
)


def parse_args():
    parser = argparse.ArgumentParser(description="LTX 2 image modular SDNQ text-to-image runner.")
    parser.add_argument("--bits", type=int, choices=(4, 8), default=8, help="SDNQ transformer bit depth.")
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
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--guidance-rescale", type=float, default=0.7)
    parser.add_argument("--decode-timestep", type=float, default=0.0)
    parser.add_argument("--decode-noise-scale", type=float, default=None)
    parser.add_argument("--pag", action="store_true")
    parser.add_argument("--pag-scale", type=float, default=0.2)
    parser.add_argument("--pag-layers", default="28")
    parser.add_argument("--output-dir", default="outputs/ltx_image_modular_sdnq")
    parser.add_argument("--save-metrics", action="store_true")
    return parser.parse_args()


def apply_leaf_group_offload(model) -> None:
    if model is None:
        raise RuntimeError("Cannot apply group offload because the component was not loaded.")
    apply_group_offloading(
        model,
        onload_device=DEVICE,
        offload_device=OFFLOAD_DEVICE,
        offload_type="leaf_level",
        use_stream=True,
        record_stream=False,
        low_cpu_mem_usage=True,
    )


def get_required_component(pipe: ModularPipeline, name: str):
    component = pipe.components.get(name)
    if component is None:
        component = getattr(pipe, name, None)
    if component is None:
        available = sorted(key for key, value in pipe.components.items() if value is not None)
        raise RuntimeError(f"Component {name!r} was not loaded. Available components: {available}")
    return component


def build_run_slug(args, *, text_encoder_kind: str) -> str:
    pag_tag = f"pag{args.pag_scale:g}_layers{args.pag_layers.replace(',', '-')}" if args.pag else "nopag"
    return "_".join(
        [
            "ltx23_image",
            MODEL_TAG,
            f"sdnq{args.bits}",
            f"text_encoder_{text_encoder_kind}",
            pag_tag,
            f"{args.width}x{args.height}",
            f"steps{args.steps}",
            f"seed{args.seed}",
        ]
    )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this runner.")
    if args.width % 32 != 0 or args.height % 32 != 0:
        raise ValueError("Width and height must be divisible by 32.")
    if args.text_encoder_bits is not None and args.bits is None:
        raise ValueError("--text-encoder-bits requires --bits so the SDNQ model path is explicit.")

    sdnq_model_path = SDNQ_MODEL_PATHS[args.bits]
    text_encoder_src = sdnq_model_path if args.text_encoder_bits is not None else MODEL_PATH
    text_encoder_kind = f"sdnq{args.text_encoder_bits}" if args.text_encoder_bits is not None else "original"
    transformer_kind = f"sdnq{args.bits}"
    pag_layers = [int(item.strip()) for item in args.pag_layers.split(",") if item.strip()]
    run_slug = build_run_slug(args, text_encoder_kind=text_encoder_kind)
    output_dir = Path(args.output_dir)
    output_path = output_dir / f"{run_slug}.png"
    metrics_dir = output_dir / "metrics"
    metrics_path = metrics_dir / f"{run_slug}.json"

    metrics = {
        "run_slug": run_slug,
        "model_tag": MODEL_TAG,
        "model_path": MODEL_PATH,
        "custom_blocks_path": CUSTOM_BLOCKS_PATH,
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
        "workflow": "text2image",
        "width": args.width,
        "height": args.height,
        "seed": args.seed,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "guidance_rescale": args.guidance_rescale,
        "vae_decode_timestep": args.decode_timestep,
        "vae_decode_noise_scale": args.decode_noise_scale,
        "pag_enabled": args.pag,
        "pag_scale": args.pag_scale if args.pag else 0.0,
        "pag_applied_layers": pag_layers if args.pag else None,
        "dtype": str(DTYPE),
        "offload_backend": "diffusers_group_offloading",
        "group_offload_config": {
            "offload_type": "leaf_level",
            "use_stream": True,
            "record_stream": False,
            "low_cpu_mem_usage": True,
        },
        "events": [],
        "steps": [],
    }

    tracker = RunTracker(str(DEVICE), metrics, interval=0.1)
    record_event = tracker.record_event
    t0 = tracker.step_start(f"Modular SDNQ Text2Image at {args.width}x{args.height}")

    print(
        f"Using modular SDNQ: transformer={transformer_kind} text_encoder={text_encoder_kind} "
        f"offload=leaf_group_offload",
        flush=True,
    )
    print(f"VRAM baseline: {torch.cuda.memory_allocated(DEVICE) / (1024**3):.2f} GB", flush=True)

    event_t0 = time.time()
    pipe = ModularPipeline.from_pretrained(CUSTOM_BLOCKS_PATH, trust_remote_code=True)
    record_event("load_modular_pipeline", time.time() - event_t0, custom_blocks_path=CUSTOM_BLOCKS_PATH)

    event_t0 = time.time()
    pipe.load_components(
        names=["text_encoder"],
        pretrained_model_name_or_path=text_encoder_src,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    )
    record_event(
        "load_text_encoder",
        time.time() - event_t0,
        source=text_encoder_src,
        kind=text_encoder_kind,
        dtype=str(DTYPE),
    )

    event_t0 = time.time()
    base_component_names = ["tokenizer", "connectors", "vae", "scheduler"]
    pipe.load_components(
        names=base_component_names,
        pretrained_model_name_or_path=MODEL_PATH,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    )
    record_event(
        "load_base_runtime_components",
        time.time() - event_t0,
        source=MODEL_PATH,
        names=base_component_names,
        dtype=str(DTYPE),
    )

    event_t0 = time.time()
    pipe.load_components(
        names=["transformer"],
        pretrained_model_name_or_path=sdnq_model_path,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    )
    record_event(
        "load_sdnq_transformer",
        time.time() - event_t0,
        source=sdnq_model_path,
        kind=transformer_kind,
        dtype=str(DTYPE),
    )

    text_encoder = get_required_component(pipe, "text_encoder")
    connectors = get_required_component(pipe, "connectors")
    transformer = get_required_component(pipe, "transformer")
    vae = get_required_component(pipe, "vae")

    event_t0 = time.time()
    apply_leaf_group_offload(text_encoder)
    apply_leaf_group_offload(transformer)
    apply_leaf_group_offload(vae)
    connectors.to(DEVICE)
    record_event("setup_components", time.time() - event_t0, offload_type="leaf_level", use_stream=True)

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    event_t0 = time.time()
    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        guidance_rescale=args.guidance_rescale,
        decode_timestep=args.decode_timestep,
        decode_noise_scale=args.decode_noise_scale,
        pag_scale=args.pag_scale if args.pag else 0.0,
        pag_applied_layers=pag_layers if args.pag else None,
        generator=generator,
        output="images",
    )
    record_event("modular_sdnq_text2image_call", time.time() - event_t0)

    output_dir.mkdir(parents=True, exist_ok=True)
    result[0].save(output_path)
    print(f"Saved image to {output_path}")

    del pipe
    flush()
    tracker.step_end(f"Modular SDNQ Text2Image at {args.width}x{args.height}", t0)

    total_time = tracker.total_elapsed()
    metrics["total_elapsed_sec"] = round(total_time, 4)
    metrics["global_peak_vram_gb"] = round(tracker.global_peak_vram, 4)
    metrics["global_peak_ram_gb"] = round(tracker.global_peak_ram, 4)
    metrics["output_path"] = str(output_path)
    if args.save_metrics:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Metrics JSON: {metrics_path}")

    print(
        f"TOTAL: {total_time:.1f}s | Peak VRAM: {tracker.global_peak_vram:.2f} GB | "
        f"Peak RAM: {tracker.global_peak_ram:.2f} GB",
        flush=True,
    )


if __name__ == "__main__":
    main()
