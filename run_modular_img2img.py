"""
Single-call LTX 2 image modular img2img runner.

This uses the custom LTX2ImageAutoBlocks metadata. The same blocks can run
text2image or image2image; this runner forces the image2image path by passing an
input image.
"""

import argparse
import os
from pathlib import Path
import time

import torch
from diffusers import ModularPipeline
from diffusers.hooks import apply_group_offloading

from inference_utils import RunTracker, flush


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_MODULES_CACHE", str((Path(__file__).parent / ".hf_modules").resolve()))

DEVICE = torch.device("cuda:0")
OFFLOAD_DEVICE = torch.device("cpu")
DTYPE = torch.bfloat16

MODEL_PATH = "elismasilva/ltx2.3-image-base"
CUSTOM_BLOCKS_PATH = "elismasilva/ltx2.3_image_custom_blocks"

DEFAULT_PROMPT = (
    "Fisheye close-up of a calico cat wearing a tiny flower crown, sniffing the camera lens in a sunny park, "
    "with bright colors, realistic fur detail, and playful viral-pet energy. Preserve the exact same composition, "
    "camera framing, subject identity, pose, lighting, background layout, and geometry. Restore crisp fine detail, "
    "natural micro texture, clean edges, realistic material detail, and high-resolution sharpness. Do not change the scene."
)
DEFAULT_NEGATIVE_PROMPT = ""


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
    parser.add_argument("--pag", action="store_true")
    parser.add_argument("--pag-scale", type=float, default=0.2)
    parser.add_argument("--pag-layers", default="28")
    parser.add_argument("--no-input-preprocess", action="store_true")
    parser.add_argument("--output-dir", default="outputs/ltx_image_modular_img2img")
    return parser.parse_args()


def enable_leaf_group_offload(model):
    apply_group_offloading(
        model,
        onload_device=DEVICE,
        offload_device=OFFLOAD_DEVICE,
        offload_type="leaf_level",
        use_stream=True,
        record_stream=False,
        low_cpu_mem_usage=True,
    )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this runner.")
    if args.width % 32 != 0 or args.height % 32 != 0:
        raise ValueError("Width and height must be divisible by 32.")

    effective_input_noise_sigma = 0.0 if args.no_input_preprocess else args.input_noise_sigma
    effective_input_sharpen = 1.0 if args.no_input_preprocess else args.input_sharpen
    effective_phase_cutoff = args.phase_cutoff if args.phase_cutoff and args.phase_cutoff > 0 else None
    pag_layers = [int(item.strip()) for item in args.pag_layers.split(",") if item.strip()]

    run_slug = (
        f"ltx2_image_modular_img2img_strength{args.strength:g}_noise{effective_input_noise_sigma:g}_"
        f"sharp{effective_input_sharpen:g}_phase{effective_phase_cutoff or 'off'}_"
        f"{args.width}x{args.height}_steps{args.steps}_seed{args.seed}_{Path(args.input_image).stem}"
    )
    output_dir = Path(args.output_dir)
    output_path = output_dir / f"{run_slug}.png"

    metrics = {
        "run_slug": run_slug,
        "model_path": MODEL_PATH,
        "custom_blocks_path": str(CUSTOM_BLOCKS_PATH),
        "workflow": "image2image",
        "input_image": args.input_image,
        "width": args.width,
        "height": args.height,
        "seed": args.seed,
        "num_inference_steps": args.steps,
        "strength": args.strength,
        "input_noise_sigma": effective_input_noise_sigma,
        "input_sharpen": effective_input_sharpen,
        "phase_cutoff": effective_phase_cutoff,
        "phase_transition_width": args.phase_transition_width,
        "phase_pad_factor": args.phase_pad_factor,
        "guidance_scale": args.guidance_scale,
        "guidance_rescale": args.guidance_rescale,
        "pag_enabled": args.pag,
        "pag_scale": args.pag_scale if args.pag else 0.0,
        "pag_applied_layers": pag_layers if args.pag else None,
        "events": [],
        "steps": [],
    }
    tracker = RunTracker(str(DEVICE), metrics, interval=0.1)
    record_event = tracker.record_event
    t0 = tracker.step_start(f"Modular Img2Img at {args.width}x{args.height}")

    event_t0 = time.time()
    pipe = ModularPipeline.from_pretrained(CUSTOM_BLOCKS_PATH, trust_remote_code=True)
    record_event("load_modular_pipeline", time.time() - event_t0, custom_blocks_path=str(CUSTOM_BLOCKS_PATH))

    event_t0 = time.time()
    pipe.load_components(
        names=["text_encoder", "connectors", "transformer", "vae"],
        pretrained_model_name_or_path=MODEL_PATH,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    )
    record_event("load_model_components", time.time() - event_t0, model_path=MODEL_PATH, dtype=str(DTYPE))

    event_t0 = time.time()
    pipe.load_components(names=["tokenizer", "scheduler"], pretrained_model_name_or_path=MODEL_PATH)
    record_event("load_tokenizer_scheduler", time.time() - event_t0, model_path=MODEL_PATH)

    event_t0 = time.time()
    enable_leaf_group_offload(pipe.text_encoder)
    enable_leaf_group_offload(pipe.transformer)
    pipe.connectors.to(DEVICE)
    pipe.vae.to(DEVICE)
    record_event("setup_components", time.time() - event_t0, offload_type="leaf_level", use_stream=True)

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    pixel_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    event_t0 = time.time()
    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=args.input_image,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        strength=args.strength,
        input_noise_sigma=effective_input_noise_sigma,
        input_sharpen=effective_input_sharpen,
        phase_cutoff=effective_phase_cutoff,
        phase_transition_width=args.phase_transition_width,
        phase_pad_factor=args.phase_pad_factor,
        guidance_scale=args.guidance_scale,
        guidance_rescale=args.guidance_rescale,
        pag_scale=args.pag_scale if args.pag else 0.0,
        pag_applied_layers=pag_layers if args.pag else None,
        generator=generator,
        pixel_generator=pixel_generator,
        output="images",
    )
    record_event("modular_img2img_call", time.time() - event_t0)

    output_dir.mkdir(parents=True, exist_ok=True)
    result[0].save(output_path)
    print(f"Saved image to {output_path}")

    del pipe
    flush()
    tracker.step_end(f"Modular Img2Img at {args.width}x{args.height}", t0)


if __name__ == "__main__":
    main()
