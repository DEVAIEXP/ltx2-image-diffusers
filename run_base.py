"""Standalone LTX 2.3 base text-to-image inference script."""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from diffusers import AutoencoderKLLTX2Video
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.transformers import LTX2ImageTransformer2DModel
from diffusers.pipelines.ltx2.pipeline_ltx2_image import LTX2ImagePipeline
from transformers import Gemma3ForConditionalGeneration

from inference_utils import RunTracker, flush

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEVICE = "cuda:0"
OFFLOAD_DEVICE = "cpu"
DTYPE = torch.bfloat16

MODEL_TAG = "base"
MODEL_PATH = os.getenv("MODEL_PATH", r"elismasilva/ltx2.3-image-base")

LOW_CPU_MEM_USAGE = True
GROUP_OFFLOAD_CONFIG = {
    "onload_device": DEVICE,
    "offload_type": "leaf_level",
    "use_stream": True,
    "low_cpu_mem_usage": LOW_CPU_MEM_USAGE,
}

WIDTH = 1280
HEIGHT = 704
SEED = 43

NUM_INFERENCE_STEPS = 28
GUIDANCE_SCALE = 3.0
GUIDANCE_RESCALE = 0.7
DECODE_TIMESTEP = 0.0
DECODE_NOISE_SCALE = None
PAG_ENABLED = False
PAG_SCALE = 0.2
PAG_APPLIED_LAYERS = [28]
OUTPUT_DIR = Path("outputs/ltx_image")
SHOW_METRICS = True
SAVE_METRICS = True
SHOW_DENOISE_STEPS = True

DEFAULT_PROMPT = """Fisheye close-up of a calico cat wearing a tiny flower crown, sniffing the camera lens in a sunny park, with bright colors, realistic fur detail, and playful viral-pet energy."""
DEFAULT_NEGATIVE_PROMPT = """blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, distorted proportions, unnatural skin tones, deformed features, artifacts, cartoonish rendering, 3D CGI look, unrealistic materials"""


def parse_args():
    parser = argparse.ArgumentParser(description="LTX 2.3 base bf16 text-to-image runner.")
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
    parser.add_argument("--show-metrics", action=argparse.BooleanOptionalAction, default=SHOW_METRICS)
    parser.add_argument("--save-metrics", action=argparse.BooleanOptionalAction, default=SAVE_METRICS)
    parser.add_argument("--show-denoise-steps", action=argparse.BooleanOptionalAction, default=SHOW_DENOISE_STEPS)
    return parser.parse_args()


args = parse_args()
SHOW_METRICS = args.show_metrics
SAVE_METRICS = args.save_metrics
SHOW_DENOISE_STEPS = args.show_denoise_steps
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
OUTPUT_DIR = Path(args.output_dir)
SAVE_METRICS = args.save_metrics
prompt = args.prompt
negative_prompt = args.negative_prompt


def build_run_slug(seed):
    pag_tag = (
        f"pag{PAG_SCALE:g}_layers{'-'.join(map(str, PAG_APPLIED_LAYERS))}"
        if PAG_ENABLED
        else "nopag"
    )
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


RUN_SLUG = build_run_slug(SEED)
METRICS_DIR = OUTPUT_DIR / "metrics"
run_metrics = {
    "run_slug": RUN_SLUG,
    "model_tag": MODEL_TAG,
    "model_path": MODEL_PATH,
    "text_encoder_source": None,
    "text_encoder_kind": None,
    "transformer_source": None,
    "transformer_kind": None,
    "width": WIDTH,
    "height": HEIGHT,
    "seed": SEED,
    "num_inference_steps": NUM_INFERENCE_STEPS,
    "guidance_scale": GUIDANCE_SCALE,
    "guidance_rescale": GUIDANCE_RESCALE,
    "vae_decode_timestep": DECODE_TIMESTEP,
    "vae_decode_noise_scale": DECODE_NOISE_SCALE,
    "pag_enabled": PAG_ENABLED,
    "pag_scale": PAG_SCALE if PAG_ENABLED else 0.0,
    "pag_applied_layers": PAG_APPLIED_LAYERS if PAG_ENABLED else None,
    "dtype": str(DTYPE),
    "group_offload_config": GROUP_OFFLOAD_CONFIG.copy(),
    "text_encoder_offload_mode": None,
    "diffusion_offload_mode": "group_offload",
    "vae_offload_mode": "none_to_cuda",
    "events": [],
    "steps": [],
}


if not SEED:
    SEED = torch.randint(0, 2**32, (1,)).item()
    print(f"Using random seed: {SEED}")
generator = torch.Generator(device="cpu").manual_seed(SEED)


tracker = RunTracker(DEVICE, run_metrics, interval=0.1)
record_event = tracker.record_event
step_start = tracker.step_start
step_end = tracker.step_end


t0 = step_start("Pass 0: Encode prompts")

embeds_pipe_kwargs = {
    "transformer": None,
    "connectors": None,
    "vae": None,
    "scheduler": None,
    "torch_dtype": DTYPE,
}

text_encoder_src = MODEL_PATH
run_metrics["text_encoder_source"] = text_encoder_src
run_metrics["text_encoder_kind"] = "original"

_event_t0 = time.time()
embeds_pipe_kwargs["text_encoder"] = Gemma3ForConditionalGeneration.from_pretrained(
    text_encoder_src,
    subfolder="text_encoder",
    torch_dtype=DTYPE,
)
record_event("load_text_encoder", time.time() - _event_t0, source=text_encoder_src, kind=run_metrics["text_encoder_kind"])

_event_t0 = time.time()
embeds_pipe = LTX2ImagePipeline.from_pretrained(MODEL_PATH, **embeds_pipe_kwargs)
record_event("build_prompt_pipeline", time.time() - _event_t0, model_path=MODEL_PATH)

_event_t0 = time.time()
embeds_pipe.enable_group_offload(
    onload_device=torch.device(DEVICE),
    offload_type=GROUP_OFFLOAD_CONFIG["offload_type"],
    use_stream=GROUP_OFFLOAD_CONFIG["use_stream"],
    low_cpu_mem_usage=GROUP_OFFLOAD_CONFIG["low_cpu_mem_usage"],
)
run_metrics["text_encoder_offload_mode"] = "group_offload"

record_event(
    "setup_prompt_offload",
    time.time() - _event_t0,
    mode=run_metrics["text_encoder_offload_mode"],
    group_offload_config=GROUP_OFFLOAD_CONFIG if run_metrics["text_encoder_offload_mode"] == "group_offload" else None,
)

_event_t0 = time.time()
with torch.inference_mode():
    prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = (
        embeds_pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=True,
        )
    )
record_event("encode_prompt_call", time.time() - _event_t0, classifier_free_guidance=True)

prompt_embeds = prompt_embeds.to(OFFLOAD_DEVICE)
prompt_attention_mask = prompt_attention_mask.to(OFFLOAD_DEVICE)

negative_prompt_embeds = negative_prompt_embeds.to(OFFLOAD_DEVICE)
negative_prompt_attention_mask = negative_prompt_attention_mask.to(OFFLOAD_DEVICE)

print(f"  prompt_embeds shape: {prompt_embeds.shape}")

del embeds_pipe
flush()
step_end("Pass 0: Encode prompts", t0)


t0 = step_start(f"Pass 1: Generate at {WIDTH}x{HEIGHT}")

transformer_src = MODEL_PATH
run_metrics["transformer_source"] = transformer_src
run_metrics["transformer_kind"] = "bf16"
_event_t0 = time.time()

transformer = LTX2ImageTransformer2DModel.from_pretrained(
    transformer_src,
    subfolder="transformer",
    torch_dtype=DTYPE,
    device_map="cpu",
)

record_event("load_transformer", time.time() - _event_t0, source=transformer_src, kind=run_metrics["transformer_kind"])

_event_t0 = time.time()
pipe = LTX2ImagePipeline.from_pretrained(
    MODEL_PATH,
    transformer=transformer,
    text_encoder=None,
    tokenizer=None,
    vae=None,
    torch_dtype=DTYPE,
)
record_event("build_denoise_pipeline", time.time() - _event_t0, model_path=MODEL_PATH)

_event_t0 = time.time()
pipe.enable_group_offload(
    onload_device=torch.device(DEVICE),
    offload_type=GROUP_OFFLOAD_CONFIG["offload_type"],
    use_stream=GROUP_OFFLOAD_CONFIG["use_stream"],
    low_cpu_mem_usage=GROUP_OFFLOAD_CONFIG["low_cpu_mem_usage"],
)
record_event(
    "setup_denoise_offload",
    time.time() - _event_t0,
    mode=run_metrics["diffusion_offload_mode"],
    group_offload_config=GROUP_OFFLOAD_CONFIG if run_metrics["diffusion_offload_mode"] == "group_offload" else None,
)

_event_t0 = time.time()
image_latent = pipe(
    prompt_embeds=prompt_embeds.to(device=DEVICE, dtype=DTYPE),
    prompt_attention_mask=prompt_attention_mask.to(device=DEVICE),
    negative_prompt_embeds=negative_prompt_embeds.to(device=DEVICE, dtype=DTYPE),
    negative_prompt_attention_mask=negative_prompt_attention_mask.to(device=DEVICE),
    width=WIDTH,
    height=HEIGHT,
    num_inference_steps=NUM_INFERENCE_STEPS,
    guidance_scale=GUIDANCE_SCALE,
    guidance_rescale=GUIDANCE_RESCALE,
    decode_timestep=DECODE_TIMESTEP,
    decode_noise_scale=DECODE_NOISE_SCALE,
    pag_scale=PAG_SCALE if PAG_ENABLED else 0.0,
    pag_applied_layers=PAG_APPLIED_LAYERS if PAG_ENABLED else None,
    generator=generator,
    output_type="latent",
    return_dict=False,
)[0]
record_event("denoise_pipe_call", time.time() - _event_t0, cold_run=True, includes_possible_compile_or_kernel_warmup=True)

print(f"  Image latent: {image_latent.shape}")
image_latent = image_latent.to(OFFLOAD_DEVICE)

del prompt_embeds, prompt_attention_mask
del negative_prompt_embeds, negative_prompt_attention_mask
del pipe, transformer
flush()
step_end(f"Pass 1: Generate at {WIDTH}x{HEIGHT}", t0)


t0 = step_start("Pass 2: Decode VAE")

_event_t0 = time.time()
vae = AutoencoderKLLTX2Video.from_pretrained(
    MODEL_PATH, subfolder="vae", torch_dtype=DTYPE
).to(DEVICE)
record_event("load_vae_to_cuda", time.time() - _event_t0, source=MODEL_PATH, offload_mode=run_metrics["vae_offload_mode"])

_event_t0 = time.time()
with torch.no_grad():
    latents_gpu = image_latent.to(device=DEVICE, dtype=DTYPE)
    if not vae.config.timestep_conditioning:
        vae_decode_timestep = None
    else:
        vae_decode_timestep = torch.tensor(
            [DECODE_TIMESTEP] * latents_gpu.shape[0], device=DEVICE, dtype=latents_gpu.dtype
        )
        effective_decode_noise_scale = DECODE_TIMESTEP if DECODE_NOISE_SCALE is None else DECODE_NOISE_SCALE
        if effective_decode_noise_scale != 0.0:
            noise = torch.randn(latents_gpu.shape, generator=generator, device=DEVICE, dtype=latents_gpu.dtype)
            latents_gpu = (1 - effective_decode_noise_scale) * latents_gpu + effective_decode_noise_scale * noise

    latents_mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(DEVICE)
    latents_std = vae.latents_std.view(1, -1, 1, 1, 1).to(DEVICE)
    latents_gpu = (latents_gpu * latents_std) / vae.config.scaling_factor + latents_mean

    decoded = vae.decode(latents_gpu.to(vae.dtype), vae_decode_timestep, return_dict=False)[0]
record_event("vae_decode_call", time.time() - _event_t0, decode_timestep=DECODE_TIMESTEP, decode_noise_scale=DECODE_NOISE_SCALE)

image_processor = VaeImageProcessor(vae_scale_factor=vae.spatial_compression_ratio)
image = image_processor.postprocess(decoded[:, :, 0, :, :].cpu(), output_type="pil")[0]

del vae, image_latent, latents_gpu, decoded, vae_decode_timestep
flush()
step_end("Pass 2: Decode VAE", t0)


t0 = step_start("Save Image")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if SAVE_METRICS:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
output_path = OUTPUT_DIR / f"{RUN_SLUG}.png"
image.save(output_path)
print(f"  Image saved successfully to: {output_path}")
step_end("Save Image", t0)

total_time = tracker.total_elapsed()
run_metrics["total_elapsed_sec"] = round(total_time, 4)
run_metrics["global_peak_vram_gb"] = round(tracker.global_peak_vram, 4)
run_metrics["global_peak_ram_gb"] = round(tracker.global_peak_ram, 4)
run_metrics["output_path"] = str(output_path)
metrics_json_path = METRICS_DIR / f"{RUN_SLUG}.json"
metrics_txt_path = METRICS_DIR / f"{RUN_SLUG}.txt"
if SAVE_METRICS:
    metrics_json_path.write_text(json.dumps(run_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
metrics_lines = [
    f"RUN: {RUN_SLUG}",
    f"OUTPUT: {output_path}",
    f"MODEL_TAG: {MODEL_TAG}",
    f"MODEL_PATH: {MODEL_PATH}",
    f"TEXT_ENCODER_KIND: {run_metrics['text_encoder_kind']}",
    f"TEXT_ENCODER_SOURCE: {run_metrics['text_encoder_source']}",
    f"TEXT_ENCODER_OFFLOAD_MODE: {run_metrics['text_encoder_offload_mode']}",
    f"TRANSFORMER_KIND: {run_metrics['transformer_kind']}",
    f"TRANSFORMER_SOURCE: {run_metrics['transformer_source']}",
    f"DIFFUSION_OFFLOAD_MODE: {run_metrics['diffusion_offload_mode']}",
    f"GUIDANCE_RESCALE: {GUIDANCE_RESCALE}",
    f"VAE_DECODE_TIMESTEP: {DECODE_TIMESTEP}",
    f"VAE_DECODE_NOISE_SCALE: {DECODE_NOISE_SCALE}",
    f"PAG_ENABLED: {PAG_ENABLED}",
    f"PAG_SCALE: {PAG_SCALE if PAG_ENABLED else 0.0}",
    f"PAG_APPLIED_LAYERS: {PAG_APPLIED_LAYERS if PAG_ENABLED else None}",
    f"GROUP_OFFLOAD_CONFIG: {run_metrics['group_offload_config']}",
    "",
    "EVENTS:",
]
for item in run_metrics["events"]:
    extras = {k: v for k, v in item.items() if k not in {"name", "elapsed_sec"}}
    metrics_lines.append(f"- {item['name']}: {item['elapsed_sec']:.4f}s | {extras}")
metrics_lines.append("")
metrics_lines.append("STEPS:")
for item in run_metrics["steps"]:
    metrics_lines.append(
        f"- {item['name']}: {item['elapsed_sec']:.4f}s | peak_vram={item['peak_vram_gb']:.4f} GB | peak_ram={item['peak_ram_gb']:.4f} GB"
    )
metrics_lines.extend(
    [
        "",
        f"TOTAL: {run_metrics['total_elapsed_sec']:.4f}s",
        f"GLOBAL_PEAK_VRAM: {run_metrics['global_peak_vram_gb']:.4f} GB",
        f"GLOBAL_PEAK_RAM: {run_metrics['global_peak_ram_gb']:.4f} GB",
    ]
)
if SAVE_METRICS:
    metrics_txt_path.write_text("\n".join(metrics_lines) + "\n", encoding="utf-8")
print(f"\n{'=' * 70}")
print(f"  TOTAL: {total_time:.1f}s | Peak VRAM: {tracker.global_peak_vram:.2f} GB | Peak RAM: {tracker.global_peak_ram:.2f} GB")
print(f"  Output: {output_path}")
if SAVE_METRICS:
    print(f"  Metrics JSON: {metrics_json_path}")
    print(f"  Metrics TXT: {metrics_txt_path}")
print(f"{'=' * 70}")
