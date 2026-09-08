"""
Modular LTX 2.3 distilled image runner using official Diffusers group offloading.
"""

import json
import os
from pathlib import Path
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from diffusers import AutoencoderKLLTX2Video, FlowMatchEulerDiscreteScheduler
from diffusers.hooks import apply_group_offloading
from transformers import Gemma3ForConditionalGeneration, GemmaTokenizerFast

from custom_blocks.ltx2_image import LTX2ImageDistilledBlocks, LTX2ImageTextEncoderStep
from custom_blocks.ltx2_image.connectors_ltx2_image import LTX2ImageTextConnectors
from custom_blocks.ltx2_image.modular_blocks_ltx2_image import (
    LTX2ImageConnectorStep,
    LTX2ImageDenoiseStep,
    LTX2ImagePrepareLatentsStep,
)
from custom_blocks.ltx2_image.transformer_ltx2_image import LTX2ImageTransformer2DModel
from inference_utils import RunTracker, flush


DEVICE = "cuda:0"
OFFLOAD_DEVICE = "cpu"
DTYPE = torch.bfloat16

MODEL_TAG = "distilled_modular_diffusers_group_offload"
MODEL_PATH = r"E:\model\ltx2.3-image-distilled-1.1"
LOW_CPU_MEM_USAGE = True

GROUP_OFFLOAD_CONFIG = {
    "text_encoder_offload_type": "leaf_level",
    "text_encoder_use_stream": True,
    "text_encoder_record_stream": False,
    "text_encoder_low_cpu_mem_usage": True,
    "text_encoder_num_blocks_per_group": 1,
    "transformer_offload_type": "leaf_level",
    "transformer_use_stream": True,
    "transformer_record_stream": False,
    "transformer_low_cpu_mem_usage": True,
    "transformer_num_blocks_per_group": 1,
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
GENERATION_REPEATS = 1

SHOW_METRICS = True
SAVE_METRICS = True
SHOW_DENOISE_STEPS = True

prompt = "Fisheye close-up of a calico cat wearing a tiny flower crown, sniffing the camera lens in a sunny park, with bright colors, realistic fur detail, and playful viral-pet energy."
negative_prompt = ""


def build_run_slug(seed):
    pag_tag = f"pag{PAG_SCALE:g}_layers{'-'.join(map(str, PAG_APPLIED_LAYERS))}" if PAG_ENABLED else "nopag"
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


def apply_model_group_offload(model, *, prefix: str):
    offload_type = GROUP_OFFLOAD_CONFIG[f"{prefix}_offload_type"]
    kwargs = {
        "onload_device": torch.device(DEVICE),
        "offload_device": torch.device(OFFLOAD_DEVICE),
        "offload_type": offload_type,
        "use_stream": GROUP_OFFLOAD_CONFIG[f"{prefix}_use_stream"],
        "record_stream": GROUP_OFFLOAD_CONFIG[f"{prefix}_record_stream"],
        "low_cpu_mem_usage": GROUP_OFFLOAD_CONFIG[f"{prefix}_low_cpu_mem_usage"],
    }
    if offload_type == "block_level":
        kwargs["num_blocks_per_group"] = GROUP_OFFLOAD_CONFIG[f"{prefix}_num_blocks_per_group"]
    apply_group_offloading(model, **kwargs)


def cleanup_runtime_state(record_event, event_name: str):
    event_t0 = time.time()
    flush()
    record_event(event_name, time.time() - event_t0)


def make_denoise_progress_callback(tracker):
    def callback(pipe, step_index, timestep, callback_kwargs):
        now = time.perf_counter()
        elapsed = now - callback.last_time
        callback.last_time = now
        callback.step_times.append(elapsed)
        if SHOW_DENOISE_STEPS:
            avg = (now - callback.start_time) / (step_index + 1)
            allocated = torch.cuda.memory_allocated(DEVICE) / (1024**3)
            reserved = torch.cuda.memory_reserved(DEVICE) / (1024**3)
            total_steps = len(callback.timesteps) if callback.timesteps is not None else NUM_INFERENCE_STEPS
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
    return callback


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this runner.")

    seed = SEED or torch.randint(0, 2**32, (1,)).item()
    if not SEED:
        print(f"  Using random seed: {seed}")
    generator = torch.Generator(device="cpu").manual_seed(seed)

    run_slug = build_run_slug(seed)
    output_dir = Path("outputs/ltx_image_modular")
    metrics_dir = output_dir / "metrics"
    run_metrics = {
        "run_slug": run_slug,
        "model_tag": MODEL_TAG,
        "model_path": MODEL_PATH,
        "text_encoder_kind": "original",
        "transformer_kind": "custom_modular",
        "width": WIDTH,
        "height": HEIGHT,
        "seed": seed,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "guidance_rescale": GUIDANCE_RESCALE,
        "vae_decode_timestep": DECODE_TIMESTEP,
        "vae_decode_noise_scale": DECODE_NOISE_SCALE,
        "pag_enabled": PAG_ENABLED,
        "pag_scale": PAG_SCALE if PAG_ENABLED else 0.0,
        "pag_applied_layers": PAG_APPLIED_LAYERS if PAG_ENABLED else None,
        "dtype": str(DTYPE),
        "offload_backend": "diffusers_group_offloading",
        "group_offload_config": GROUP_OFFLOAD_CONFIG.copy(),
        "events": [],
        "steps": [],
    }

    tracker = RunTracker(DEVICE, run_metrics, interval=0.1, show_metrics=SHOW_METRICS)
    record_event = tracker.record_event
    step_start = tracker.step_start
    step_end = tracker.step_end
    denoise_progress_callback = make_denoise_progress_callback(tracker)

    print("Using modular Diffusers group offload baseline", flush=True)
    print(
        f"  text_encoder=leaf group offload transformer=leaf group offload "
        f"model_path={MODEL_PATH}",
        flush=True,
    )
    print(f"  VRAM baseline: {torch.cuda.memory_allocated(DEVICE) / (1024**3):.2f} GB", flush=True)

    t0 = step_start("Pass 0: Encode prompts")

    event_t0 = time.time()
    text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        subfolder="text_encoder",
        torch_dtype=DTYPE,
        low_cpu_mem_usage=LOW_CPU_MEM_USAGE,
    )
    record_event("load_text_encoder", time.time() - event_t0, source=MODEL_PATH)

    event_t0 = time.time()
    apply_model_group_offload(text_encoder, prefix="text_encoder")
    record_event(
        "setup_text_encoder_group_offload",
        time.time() - event_t0,
        offload_type=GROUP_OFFLOAD_CONFIG["text_encoder_offload_type"],
        use_stream=GROUP_OFFLOAD_CONFIG["text_encoder_use_stream"],
        record_stream=GROUP_OFFLOAD_CONFIG["text_encoder_record_stream"],
    )

    event_t0 = time.time()
    tokenizer = GemmaTokenizerFast.from_pretrained(MODEL_PATH, subfolder="tokenizer")
    record_event("load_tokenizer", time.time() - event_t0, source=MODEL_PATH)

    event_t0 = time.time()
    prompt_pipe = LTX2ImageTextEncoderStep().init_pipeline()
    prompt_pipe.update_components(text_encoder=text_encoder, tokenizer=tokenizer)
    record_event("build_prompt_modular_pipeline", time.time() - event_t0, model_path=MODEL_PATH)

    event_t0 = time.time()
    with torch.inference_mode():
        prompt_state = prompt_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=GUIDANCE_SCALE,
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
    record_event("encode_prompt_call", time.time() - event_t0, classifier_free_guidance=False)

    prompt_embeds = prompt_state["prompt_embeds"].to(OFFLOAD_DEVICE)
    prompt_attention_mask = prompt_state["prompt_attention_mask"].to(OFFLOAD_DEVICE)
    latent_batch_size = prompt_state["batch_size"]
    prompt_dtype = prompt_state["dtype"]
    do_classifier_free_guidance = prompt_state["do_classifier_free_guidance"]
    if SHOW_METRICS:
        print(f"  prompt_embeds shape: {prompt_embeds.shape}")

    del prompt_state, prompt_pipe, text_encoder, tokenizer
    cleanup_runtime_state(record_event, "cleanup_after_text_encoder")
    step_end("Pass 0: Encode prompts", t0)

    t0 = step_start(f"Pass 1: Generate at {WIDTH}x{HEIGHT}")

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
    record_event("build_connector_modular_pipeline", time.time() - event_t0, model_path=MODEL_PATH)

    event_t0 = time.time()
    connector_state = connector_pipe(
        prompt_embeds=prompt_embeds.to(device=DEVICE, dtype=DTYPE),
        prompt_attention_mask=prompt_attention_mask.to(device=DEVICE),
        negative_prompt_embeds=None,
        negative_prompt_attention_mask=None,
        do_classifier_free_guidance=do_classifier_free_guidance,
        pag_scale=PAG_SCALE if PAG_ENABLED else 0.0,
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

    del connector_state, connector_pipe, connectors, prompt_embeds, prompt_attention_mask
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
    record_event(
        "load_transformer",
        time.time() - event_t0,
        source=MODEL_PATH,
        low_cpu_mem_usage=LOW_CPU_MEM_USAGE,
        device_map=transformer_load_kwargs.get("device_map"),
    )

    event_t0 = time.time()
    apply_model_group_offload(transformer, prefix="transformer")
    record_event(
        "setup_transformer_group_offload",
        time.time() - event_t0,
        offload_type=GROUP_OFFLOAD_CONFIG["transformer_offload_type"],
        use_stream=GROUP_OFFLOAD_CONFIG["transformer_use_stream"],
        record_stream=GROUP_OFFLOAD_CONFIG["transformer_record_stream"],
        low_cpu_mem_usage=GROUP_OFFLOAD_CONFIG["transformer_low_cpu_mem_usage"],
    )

    event_t0 = time.time()
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")
    record_event("load_scheduler", time.time() - event_t0, source=MODEL_PATH)

    event_t0 = time.time()
    prepare_pipe = LTX2ImagePrepareLatentsStep().init_pipeline()
    prepare_pipe.update_components(transformer=transformer, scheduler=scheduler)
    record_event("build_prepare_latents_modular_pipeline", time.time() - event_t0, model_path=MODEL_PATH)

    event_t0 = time.time()
    denoise_pipe = LTX2ImageDenoiseStep().init_pipeline()
    denoise_pipe.update_components(transformer=transformer, scheduler=scheduler)
    record_event("build_denoise_modular_pipeline", time.time() - event_t0, model_path=MODEL_PATH)

    image_latent = None
    latent_height = None
    latent_width = None
    latent_channels = None
    run_metrics["denoise_step_times_by_repeat"] = []

    for repeat_index in range(GENERATION_REPEATS):
        repeat_suffix = "" if GENERATION_REPEATS == 1 else f"_repeat_{repeat_index + 1}"
        if GENERATION_REPEATS > 1:
            print(f"  Warm generation repeat {repeat_index + 1}/{GENERATION_REPEATS}", flush=True)

        event_t0 = time.time()
        prepare_state = prepare_pipe(
            width=WIDTH,
            height=HEIGHT,
            num_inference_steps=NUM_INFERENCE_STEPS,
            batch_size=latent_batch_size,
            transformer_batch_multiplier=transformer_batch_multiplier,
            generator=generator,
            output=["latents", "timesteps", "latent_height", "latent_width", "in_channels", "video_rotary_emb"],
        )
        record_event(f"prepare_latents_modular_pipe_call{repeat_suffix}", time.time() - event_t0)

        denoise_progress_callback.timesteps = prepare_state["timesteps"]
        denoise_progress_callback.start_time = time.perf_counter()
        denoise_progress_callback.last_time = denoise_progress_callback.start_time
        denoise_progress_callback.step_times = []
        print(f"  Starting denoise loop{repeat_suffix}", flush=True)
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
            do_classifier_free_guidance=False,
            do_perturbed_attention_guidance=do_perturbed_attention_guidance,
            guidance_scale=GUIDANCE_SCALE,
            guidance_rescale=GUIDANCE_RESCALE,
            pag_scale=PAG_SCALE if PAG_ENABLED else 0.0,
            pag_applied_layers=PAG_APPLIED_LAYERS if PAG_ENABLED else None,
            callback_on_step_end=denoise_progress_callback,
            callback_on_step_end_tensor_inputs=["latents"],
            output="latents",
        )
        record_event(
            f"denoise_modular_pipe_call{repeat_suffix}",
            time.time() - event_t0,
            repeat_index=repeat_index,
        )
        run_metrics["denoise_step_times_by_repeat"].append(list(denoise_progress_callback.step_times))
        run_metrics["denoise_step_times"] = denoise_progress_callback.step_times

        if image_latent is not None:
            del image_latent
        image_latent = denoise_state.to(OFFLOAD_DEVICE)
        latent_height = prepare_state["latent_height"]
        latent_width = prepare_state["latent_width"]
        latent_channels = prepare_state["in_channels"]
        if SHOW_METRICS:
            print(f"  Image latent: {image_latent.shape}")
        del prepare_state, denoise_state

    del connector_prompt_embeds, connector_attention_mask
    del prepare_pipe, denoise_pipe, transformer, scheduler
    cleanup_runtime_state(record_event, "cleanup_before_vae_decode")
    step_end(f"Pass 1: Generate at {WIDTH}x{HEIGHT}", t0)

    t0 = step_start("Pass 2: Decode VAE")

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
        decode_timestep=DECODE_TIMESTEP,
        decode_noise_scale=DECODE_NOISE_SCALE,
        generator=generator,
        output_type="pil",
        output="images",
    )[0]
    record_event("vae_decode_modular_call", time.time() - event_t0)

    del decode_pipe, vae, image_latent
    cleanup_runtime_state(record_event, "cleanup_after_vae_decode")
    step_end("Pass 2: Decode VAE", t0)

    t0 = step_start("Save Image")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run_slug}.png"
    image.save(output_path)
    print(f"  Image saved successfully to: {output_path}")
    step_end("Save Image", t0)

    total_time = tracker.total_elapsed()
    run_metrics["total_elapsed_sec"] = round(total_time, 4)
    run_metrics["global_peak_vram_gb"] = round(tracker.global_peak_vram, 4)
    run_metrics["global_peak_ram_gb"] = round(tracker.global_peak_ram, 4)

    metrics_path = metrics_dir / f"{run_slug}.json"
    if SAVE_METRICS:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(run_metrics, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"  TOTAL: {total_time:.1f}s | Peak VRAM: {tracker.global_peak_vram:.2f} GB | Peak RAM: {tracker.global_peak_ram:.2f} GB")
    print(f"  Output: {output_path}")
    if SAVE_METRICS:
        print(f"  Metrics JSON: {metrics_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
