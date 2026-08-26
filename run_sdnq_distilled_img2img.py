"""
Standalone LTX 2.3 distilled SDNQ image-to-image script.

The img2img logic lives in Diffusers' LTX2ImageImg2ImgPipeline. This runner only
loads the selected components, optionally activates LoRAs, calls the pipeline, and
records local benchmark metrics.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import time
import warnings

import torch
from diffusers.models.transformers import LTX2ImageTransformer2DModel
from diffusers.pipelines.ltx2.pipeline_ltx2_image import LTX2ImagePipeline
from diffusers.pipelines.ltx2.pipeline_ltx2_image_img2img import LTX2ImageImg2ImgPipeline, get_strength_sigmas
from transformers import Gemma3ForConditionalGeneration

from inference_utils import RunTracker, flush, get_sdnq_version

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("diffusers").setLevel(logging.ERROR)

DEVICE = "cuda:0"
OFFLOAD_DEVICE = "cpu"
DTYPE = torch.bfloat16

MODEL_TAG = "distilled_img2img"
MODEL_PATH = r"elismasilva/ltx2.3-image-distilled-1.1"
SDNQ_MODEL_PATHS = {
    4: r"elismasilva/ltx2.3-image-distilled-1.1-sdnq-int4",
    8: r"elismasilva/ltx2.3-image-distilled-1.1-sdnq-int8",
}
LOW_CPU_MEM_USAGE = True
GROUP_OFFLOAD_CONFIG = {
    "onload_device": DEVICE,
    "offload_type": "leaf_level",
    "use_stream": True,
    "low_cpu_mem_usage": LOW_CPU_MEM_USAGE,
}

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


def parse_args():
    parser = argparse.ArgumentParser(description="LTX 2.3 distilled img2img test.")
    parser.add_argument("--input-image", required=True, help="Image to refine.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt used for the I2I pass.")
    parser.add_argument("--negative-prompt", default="", help="Kept for compatibility; distilled runs normally leave it empty.")
    parser.add_argument("--bits", type=int, choices=(4, 8), default=None, help="Use an SDNQ transformer at this bit depth. Leave unset for the original bf16 transformer.")
    parser.add_argument("--text-encoder-bits", type=int, choices=(8,), default=None, help="Use the SDNQ text encoder. Leave unset to use the original text encoder.")
    parser.add_argument("--width", type=int, default=1280, help="Output width. Must be divisible by 32.")
    parser.add_argument("--height", type=int, default=704, help="Output height. Must be divisible by 32.")
    parser.add_argument("--seed", type=int, default=43, help="Seed used for VAE sampling and latent noise.")
    parser.add_argument("--steps", type=int, default=8, help="Base number of denoising steps before strength slicing.")
    parser.add_argument("--strength", type=float, default=0.20, help="Img2img denoise strength in [0, 1].")
    parser.add_argument("--input-noise-sigma", type=float, default=5.0, help="Deterministic pixel-space noise sigma before VAE encode.")
    parser.add_argument("--input-sharpen", type=float, default=1.3, help="PIL sharpness factor before VAE encode.")
    parser.add_argument("--phase-cutoff", type=float, default=9.0, help="Structured latent noise low-frequency phase cutoff. Use 0 to disable.")
    parser.add_argument("--no-input-preprocess", action="store_true", help="Disable input noise and sharpen preprocessing for the I2I pass.")
    parser.add_argument("--phase-transition-width", type=float, default=2.0)
    parser.add_argument("--phase-pad-factor", type=float, default=1.5)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--guidance-rescale", type=float, default=0.7)
    parser.add_argument("--decode-timestep", type=float, default=0.0)
    parser.add_argument("--decode-noise-scale", type=float, default=None)
    parser.add_argument("--pag", action="store_true", help="Enable PAG for the I2I pass.")
    parser.add_argument("--pag-scale", type=float, default=0.2)
    parser.add_argument("--pag-layers", default="28", help="Comma-separated PAG transformer block indices.")

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


    parser.add_argument("--output-dir", default="outputs/ltx_image_img2img")
    return parser.parse_args()


def available_lora_adapters(pipe):
    if pipe is None or not hasattr(pipe, "get_list_adapters"):
        return set()
    adapters = pipe.get_list_adapters()
    if isinstance(adapters, dict):
        names = set()
        for component_adapters in adapters.values():
            names.update(component_adapters)
        return names
    if isinstance(adapters, (list, tuple, set)):
        return set(adapters)
    return set()


def load_lora_adapter(pipe, source, weight_name, requested_adapter_name):
    adapters_before = available_lora_adapters(pipe)
    lora_kwargs = {"adapter_name": requested_adapter_name}
    if weight_name:
        lora_kwargs["weight_name"] = weight_name
    pipe.load_lora_weights(source, **lora_kwargs)
    adapters_after = available_lora_adapters(pipe)
    new_adapters = adapters_after - adapters_before
    if requested_adapter_name in adapters_after:
        return requested_adapter_name, sorted(adapters_after)
    if new_adapters:
        return sorted(new_adapters)[0], sorted(adapters_after)
    return None, sorted(adapters_after)


def maybe_load_lora(pipe, run_metrics, enabled, kind, path, weight_name, adapter_name, scale, record_event):
    if not enabled:
        return None
    event_t0 = time.time()
    actual_adapter_name, available_after = load_lora_adapter(pipe, path, weight_name, adapter_name)
    if actual_adapter_name is not None:
        run_metrics["lora_adapters"].append(
            {
                "kind": kind,
                "path": path,
                "weight_name": weight_name,
                "requested_adapter_name": adapter_name,
                "adapter_name": actual_adapter_name,
                "scale": scale,
            }
        )
    record_event(
        f"load_{kind}_lora",
        time.time() - event_t0,
        path=path,
        weight_name=weight_name,
        requested_adapter_name=adapter_name,
        adapter_name=actual_adapter_name,
        scale=scale,
        available_adapters=available_after,
    )
    return (actual_adapter_name, scale) if actual_adapter_name is not None else None


def make_run_slug(args):
    model_tag = f"sdnq{args.bits}" if args.bits is not None else "bf16"
    text_encoder_tag = f"text_encoder_sdnq{args.text_encoder_bits}" if args.text_encoder_bits is not None else "text_encoder_original"
    pag_tag = f"pag{args.pag_scale:g}_layers{args.pag_layers.replace(',', '-')}" if args.pag else "nopag"
    input_tag = Path(args.input_image).stem.replace(" ", "_")
    lora_tags = []
    for enabled, name, scale in [
        (args.soft_lora, args.soft_lora_adapter_name, args.soft_lora_scale),
        (args.crisp_lora, args.crisp_lora_adapter_name, args.crisp_lora_scale),
    ]:
        if enabled:
            lora_tags.append(f"{name}{scale:g}")
    lora_tag = "lora_" + "-".join(lora_tags) if lora_tags else "nolora"
    return "_".join(
        [
            "ltx23_img2img",
            model_tag,
            text_encoder_tag,
            pag_tag,
            f"strength{args.strength:g}",
            f"noise{effective_input_noise_sigma:g}",
            f"sharp{effective_input_sharpen:g}",
            f"phase{effective_phase_cutoff:g}" if effective_phase_cutoff is not None else "phaseoff",
            f"{args.width}x{args.height}",
            f"steps{args.steps}",
            f"seed{args.seed}",
            lora_tag,
            input_tag,
        ]
    )


def main():
    args = parse_args()

    effective_input_sharpen = 1.0 if args.no_input_preprocess else args.input_sharpen
    effective_input_noise_sigma = 0.0 if args.no_input_preprocess else args.input_noise_sigma
    effective_phase_cutoff = args.phase_cutoff if args.phase_cutoff and args.phase_cutoff > 0 else None
    if args.width % 32 != 0 or args.height % 32 != 0:
        raise ValueError("Width and height must be divisible by 32.")
    if args.text_encoder_bits is not None and args.bits is None:
        raise ValueError("--text-encoder-bits requires --bits so the SDNQ model path is explicit.")

    sdnq_enabled = args.bits is not None
    sdnq_model_path = SDNQ_MODEL_PATHS[args.bits] if sdnq_enabled else None
    transformer_src = sdnq_model_path if sdnq_enabled else MODEL_PATH
    transformer_kind = f"sdnq{args.bits}" if sdnq_enabled else "bf16"
    pag_layers = [int(item.strip()) for item in args.pag_layers.split(",") if item.strip()]
    selected_sigmas, t_start = get_strength_sigmas(args.steps, args.strength)
    run_slug = make_run_slug(args)
    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / "metrics"
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    pixel_generator = torch.Generator(device="cpu").manual_seed(args.seed)

    run_metrics = {
        "run_slug": run_slug,
        "model_tag": MODEL_TAG,
        "model_path": MODEL_PATH,
        "input_image": args.input_image,
        "sdnq_enabled": sdnq_enabled,
        "sdnq_bits": args.bits if sdnq_enabled else None,
        "sdnq_quantized_version": "0.1.6" if sdnq_enabled else None,
        "sdnq_version": get_sdnq_version() if sdnq_enabled else None,
        "sdnq_model_path": sdnq_model_path,
        "sdnq_text_encoder_bits": args.text_encoder_bits,
        "text_encoder_source": None,
        "text_encoder_kind": None,
        "transformer_source": transformer_src,
        "transformer_kind": transformer_kind,
        "width": args.width,
        "height": args.height,
        "seed": args.seed,
        "num_inference_steps": args.steps,
        "strength": args.strength,
        "effective_denoising_steps": len(selected_sigmas),
        "t_start": t_start,
        "input_noise_sigma": effective_input_noise_sigma,
        "input_sharpen": effective_input_sharpen,
        "phase_cutoff": effective_phase_cutoff,
        "phase_transition_width": args.phase_transition_width,
        "phase_pad_factor": args.phase_pad_factor,
        "guidance_scale": args.guidance_scale,
        "guidance_rescale": args.guidance_rescale,
        "vae_decode_timestep": args.decode_timestep,
        "vae_decode_noise_scale": args.decode_noise_scale,
        "pag_enabled": args.pag,
        "pag_scale": args.pag_scale if args.pag else 0.0,
        "pag_applied_layers": pag_layers if args.pag else None,
        "lora_adapters": [],
        "lora_enabled": args.soft_lora or args.crisp_lora,
        "dtype": str(DTYPE),
        "group_offload_config": GROUP_OFFLOAD_CONFIG.copy(),
        "text_encoder_offload_mode": None,
        "diffusion_offload_mode": "model_cpu_offload" if args.bits == 4 else "group_offload",
        "events": [],
        "steps": [],
    }

    tracker = RunTracker(DEVICE, run_metrics, interval=0.1)
    record_event = tracker.record_event
    step_start = tracker.step_start
    step_end = tracker.step_end

    t0 = step_start("Pass 0: Encode prompts")
    text_encoder_src = sdnq_model_path if args.text_encoder_bits is not None else MODEL_PATH
    run_metrics["text_encoder_source"] = text_encoder_src
    run_metrics["text_encoder_kind"] = f"sdnq{args.text_encoder_bits}" if args.text_encoder_bits is not None else "original"

    pipe_kwargs = {"transformer": None, "connectors": None, "vae": None, "scheduler": None, "torch_dtype": DTYPE}
    event_t0 = time.time()
    pipe_kwargs["text_encoder"] = Gemma3ForConditionalGeneration.from_pretrained(
        text_encoder_src,
        subfolder="text_encoder",
        torch_dtype=DTYPE,
    )
    record_event("load_text_encoder", time.time() - event_t0, source=text_encoder_src, kind=run_metrics["text_encoder_kind"])

    event_t0 = time.time()
    embeds_pipe = LTX2ImagePipeline.from_pretrained(MODEL_PATH, **pipe_kwargs)
    record_event("build_prompt_pipeline", time.time() - event_t0, model_path=MODEL_PATH)

    event_t0 = time.time()
    embeds_pipe.enable_sequential_cpu_offload()
    run_metrics["text_encoder_offload_mode"] = "sequential_cpu_offload"
    record_event("setup_prompt_offload", time.time() - event_t0, mode=run_metrics["text_encoder_offload_mode"])

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
    print(f"  prompt_embeds shape: {prompt_embeds.shape}")

    del embeds_pipe
    flush()
    step_end("Pass 0: Encode prompts", t0)

    t0 = step_start(f"Pass 1: Img2Img refine at {args.width}x{args.height}")
    event_t0 = time.time()
    transformer = LTX2ImageTransformer2DModel.from_pretrained(
        transformer_src,
        subfolder="transformer",
        torch_dtype=DTYPE,
        device_map="cpu",
    )
    record_event("load_transformer", time.time() - event_t0, source=transformer_src, kind=run_metrics["transformer_kind"])

    event_t0 = time.time()
    pipe = LTX2ImageImg2ImgPipeline.from_pretrained(
        MODEL_PATH,
        transformer=transformer,
        text_encoder=None,
        tokenizer=None,
        torch_dtype=DTYPE,
    )
    record_event("build_img2img_pipeline", time.time() - event_t0, model_path=MODEL_PATH)

    active_adapters = []
    for item in [
        (args.soft_lora, "soft", args.soft_lora_path, args.soft_lora_weight_name, args.soft_lora_adapter_name, args.soft_lora_scale),
        (args.crisp_lora, "crisp", args.crisp_lora_path, args.crisp_lora_weight_name, args.crisp_lora_adapter_name, args.crisp_lora_scale),
    ]:
        adapter = maybe_load_lora(pipe, run_metrics, *item, record_event=record_event)
        if adapter is not None:
            active_adapters.append(adapter)

    if active_adapters:
        available_adapters = available_lora_adapters(pipe)
        active_adapters = [(name, scale) for name, scale in active_adapters if name in available_adapters]
        if active_adapters:
            names, scales = zip(*active_adapters)
            pipe.set_adapters(list(names), adapter_weights=list(scales))
            record_event("activate_loras", 0.0, adapters=dict(active_adapters), available_adapters=sorted(available_adapters))

    event_t0 = time.time()
    if args.bits == 4:
        pipe.enable_model_cpu_offload()
    else:
        pipe.enable_group_offload(
            onload_device=torch.device(DEVICE),
            offload_type=GROUP_OFFLOAD_CONFIG["offload_type"],
            use_stream=GROUP_OFFLOAD_CONFIG["use_stream"],
            low_cpu_mem_usage=GROUP_OFFLOAD_CONFIG["low_cpu_mem_usage"],
        )
    record_event("setup_img2img_offload", time.time() - event_t0, mode=run_metrics["diffusion_offload_mode"])

    event_t0 = time.time()
    image = pipe(
        image=args.input_image,
        prompt_embeds=prompt_embeds.to(device=DEVICE, dtype=DTYPE),
        prompt_attention_mask=prompt_attention_mask.to(device=DEVICE),
        negative_prompt_embeds=None,
        negative_prompt_attention_mask=None,
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
        decode_timestep=args.decode_timestep,
        decode_noise_scale=args.decode_noise_scale,
        pag_scale=args.pag_scale if args.pag else 0.0,
        pag_applied_layers=pag_layers if args.pag else None,
        generator=generator,
        pixel_generator=pixel_generator,
        output_type="pil",
        return_dict=False,
    )[0][0]
    record_event("img2img_pipe_call", time.time() - event_t0, cold_run=True, includes_possible_compile_or_kernel_warmup=True)

    del prompt_embeds, prompt_attention_mask, pipe, transformer
    flush()
    step_end(f"Pass 1: Img2Img refine at {args.width}x{args.height}", t0)

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
    run_metrics["output_path"] = str(output_path)

    metrics_json_path = metrics_dir / f"{run_slug}.json"
    metrics_txt_path = metrics_dir / f"{run_slug}.txt"
    metrics_json_path.write_text(json.dumps(run_metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    metrics_lines = [
        f"RUN: {run_slug}",
        f"OUTPUT: {output_path}",
        f"INPUT_IMAGE: {args.input_image}",
        f"MODEL_TAG: {MODEL_TAG}",
        f"MODEL_PATH: {MODEL_PATH}",
        f"SDNQ_ENABLED: {sdnq_enabled}",
        f"SDNQ_BITS: {args.bits if sdnq_enabled else 'none'}",
        f"SDNQ_VERSION: {run_metrics['sdnq_version']}",
        f"TEXT_ENCODER_KIND: {run_metrics['text_encoder_kind']}",
        f"TEXT_ENCODER_SOURCE: {run_metrics['text_encoder_source']}",
        f"TRANSFORMER_KIND: {run_metrics['transformer_kind']}",
        f"TRANSFORMER_SOURCE: {run_metrics['transformer_source']}",
        f"STRENGTH: {args.strength}",
        f"EFFECTIVE_DENOISING_STEPS: {run_metrics['effective_denoising_steps']}",
        f"T_START: {run_metrics['t_start']}",
        f"INPUT_NOISE_SIGMA: {effective_input_noise_sigma}",
        f"INPUT_SHARPEN: {effective_input_sharpen}",
        f"PHASE_CUTOFF: {effective_phase_cutoff}",
        f"PAG_ENABLED: {args.pag}",
        f"PAG_SCALE: {args.pag_scale if args.pag else 0.0}",
        f"PAG_APPLIED_LAYERS: {pag_layers if args.pag else None}",
        f"LORA_ENABLED: {run_metrics['lora_enabled']}",
        f"LORA_ADAPTERS: {run_metrics['lora_adapters']}",
        "",
        f"TOTAL_SECONDS: {run_metrics['total_elapsed_sec']}",
        f"GLOBAL_PEAK_VRAM_GB: {run_metrics['global_peak_vram_gb']}",
        f"GLOBAL_PEAK_RAM_GB: {run_metrics['global_peak_ram_gb']}",
    ]
    metrics_txt_path.write_text("\n".join(metrics_lines), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"  TOTAL: {total_time:.1f}s | Peak VRAM: {tracker.global_peak_vram:.2f} GB | Peak RAM: {tracker.global_peak_ram:.2f} GB")
    print(f"  Output: {output_path}")
    print(f"  Metrics JSON: {metrics_json_path}")
    print(f"  Metrics TXT: {metrics_txt_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
