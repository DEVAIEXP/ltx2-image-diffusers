"""
Standalone LTX 2.3 base SDNQ text-to-image inference script.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import time
import warnings

import torch
from diffusers import AutoencoderKLLTX2Video
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.transformers import LTX2ImageTransformer2DModel
from diffusers.pipelines.ltx2.pipeline_ltx2_image import LTX2ImagePipeline
from transformers import Gemma3ForConditionalGeneration

from inference_utils import RunTracker, flush, get_sdnq_version

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEVICE = "cuda:0"
OFFLOAD_DEVICE = "cpu"
DTYPE = torch.bfloat16

MODEL_TAG = "base_sdnq"
MODEL_PATH = r"elismasilva/ltx2.3-image-base"
parser = argparse.ArgumentParser()
parser.add_argument("--bits", type=int, choices=(4, 8), default=8, help="SDNQ transformer bit depth.")
parser.add_argument(
    "--text-encoder-bits",
    type=int,
    choices=(8,),
    default=None,
    help="Use the SDNQ text encoder. Leave unset to use the original text encoder for this model variant.",
)
args = parser.parse_args()

SDNQ_ENABLED = True
SDNQ_BITS = args.bits
SDNQ_MODEL_PATHS = {
    4: r"elismasilva/ltx2.3-image-base-sdnq-int4",
    8: r"elismasilva/ltx2.3-image-base-sdnq-int8",
}
SDNQ_MODEL_PATH = SDNQ_MODEL_PATHS[SDNQ_BITS]
SDNQ_TEXT_ENCODER_BITS = args.text_encoder_bits

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


def build_run_slug(seed):
    sdnq_tag = f"sdnq{SDNQ_BITS}" if SDNQ_ENABLED else "bf16"
    pag_tag = (
        f"pag{PAG_SCALE:g}_layers{'-'.join(map(str, PAG_APPLIED_LAYERS))}"
        if PAG_ENABLED
        else "nopag"
    )
    text_encoder_tag = (
        f"text_encoder_sdnq{SDNQ_TEXT_ENCODER_BITS}"
        if (SDNQ_ENABLED and SDNQ_TEXT_ENCODER_BITS is not None)
        else "text_encoder_original"
    )
    return "_".join(
        [
            "ltx23_image",
            MODEL_TAG,
            sdnq_tag,
            text_encoder_tag,
            pag_tag,
            f"{WIDTH}x{HEIGHT}",
            f"steps{NUM_INFERENCE_STEPS}",
            f"seed{seed}",
        ]
    )


RUN_SLUG = build_run_slug(SEED)
OUTPUT_DIR = Path("outputs/ltx_image")
METRICS_DIR = OUTPUT_DIR / "metrics"
run_metrics = {
    "run_slug": RUN_SLUG,
    "model_tag": MODEL_TAG,
    "model_path": MODEL_PATH,
    "sdnq_enabled": SDNQ_ENABLED,
    "sdnq_bits": SDNQ_BITS if SDNQ_ENABLED else None,
    "sdnq_quantized_version": "0.1.6" if SDNQ_ENABLED else None,
    "sdnq_version": get_sdnq_version(),
    "sdnq_model_path": SDNQ_MODEL_PATH if SDNQ_ENABLED else None,
    "sdnq_text_encoder_bits": SDNQ_TEXT_ENCODER_BITS,
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
    "diffusion_offload_mode": "model_cpu_offload" if SDNQ_BITS == 4 else "group_offload",
    "vae_offload_mode": "none_to_cuda",
    "events": [],
    "steps": [],
}


prompt = """Fisheye close-up of a calico cat wearing a tiny flower crown, sniffing the camera lens in a sunny park, with bright colors, realistic fur detail, and playful viral-pet energy."""
negative_prompt = """blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, distorted proportions, unnatural skin tones, deformed features, artifacts, cartoonish rendering, 3D CGI look, unrealistic materials"""

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

text_encoder_src = SDNQ_MODEL_PATH if (SDNQ_ENABLED and SDNQ_TEXT_ENCODER_BITS is not None) else MODEL_PATH
run_metrics["text_encoder_source"] = text_encoder_src
run_metrics["text_encoder_kind"] = (
    f"sdnq{SDNQ_TEXT_ENCODER_BITS}" if (SDNQ_ENABLED and SDNQ_TEXT_ENCODER_BITS is not None) else "original"
)

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
if SDNQ_ENABLED:
    embeds_pipe.enable_sequential_cpu_offload()
    run_metrics["text_encoder_offload_mode"] = "sequential_cpu_offload"
else:
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

transformer_src = SDNQ_MODEL_PATH if SDNQ_ENABLED else MODEL_PATH
run_metrics["transformer_source"] = transformer_src
run_metrics["transformer_kind"] = f"sdnq{SDNQ_BITS}" if SDNQ_ENABLED else "bf16"
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
if SDNQ_BITS == 4:
    pipe.enable_model_cpu_offload()
else:
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
metrics_json_path.write_text(json.dumps(run_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
metrics_lines = [
    f"RUN: {RUN_SLUG}",
    f"OUTPUT: {output_path}",
    f"MODEL_TAG: {MODEL_TAG}",
    f"MODEL_PATH: {MODEL_PATH}",
    f"SDNQ_ENABLED: {SDNQ_ENABLED}",
    f"SDNQ_BITS: {SDNQ_BITS if SDNQ_ENABLED else 'none'}",
    f"SDNQ_VERSION: {run_metrics['sdnq_version']}",
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
metrics_txt_path.write_text("\n".join(metrics_lines) + "\n", encoding="utf-8")
print(f"\n{'=' * 70}")
print(f"  TOTAL: {total_time:.1f}s | Peak VRAM: {tracker.global_peak_vram:.2f} GB | Peak RAM: {tracker.global_peak_ram:.2f} GB")
print(f"  Output: {output_path}")
print(f"  Metrics JSON: {metrics_json_path}")
print(f"  Metrics TXT: {metrics_txt_path}")
print(f"{'=' * 70}")
