"""
Gradio app for LTX 2.3 distilled T2I generation with an optional I2I pass.

This file is intentionally independent from pyproject.toml so it can be copied to a
Hugging Face Space and tested with a manually installed Gradio 6 environment.
"""

import gc
import os
import time
from pathlib import Path

import gradio as gr
import torch
from diffusers import AutoencoderKLLTX2Video
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.transformers import LTX2ImageTransformer2DModel
from diffusers.pipelines.ltx2.pipeline_ltx2_image import LTX2ImagePipeline
from diffusers.pipelines.ltx2.pipeline_ltx2_image_img2img import LTX2ImageImg2ImgPipeline
from transformers import Gemma3ForConditionalGeneration

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DEVICE = os.getenv("LTX_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
OFFLOAD_DEVICE = "cpu"
DTYPE = torch.bfloat16

MODEL_PATH = os.getenv("LTX_DISTILLED_MODEL_PATH", r"elismasilva/ltx2.3-image-distilled-1.1")
SDNQ_MODEL_PATHS = {
    "int4": os.getenv("LTX_DISTILLED_SDNQ_INT4_PATH", r"elismasilva/ltx2.3-image-distilled-1.1-sdnq-int4"),
    "int8": os.getenv("LTX_DISTILLED_SDNQ_INT8_PATH", r"elismasilva/ltx2.3-image-distilled-1.1-sdnq-int8"),
}

CRISP_LORA_PATH = os.getenv("LTX_CRISP_LORA_PATH", "vrgamedevgirl84/LTX_2.3_Crisp_Enhance_Style_LoRa")
CRISP_LORA_WEIGHT_NAME = os.getenv("LTX_CRISP_LORA_WEIGHT_NAME", "LTX2.3_Crisp_Enhance.safetensors")
CRISP_LORA_ADAPTER_NAME = "crisp"
SOFT_LORA_PATH = os.getenv("LTX_SOFT_LORA_PATH", "vrgamedevgirl84/LTX_2.3_Soft_Enhance_Style_LoRa")
SOFT_LORA_WEIGHT_NAME = os.getenv("LTX_SOFT_LORA_WEIGHT_NAME", "LTX2.3_Soft_Enhance.safetensors")
SOFT_LORA_ADAPTER_NAME = "soft"

DEFAULT_PROMPT = (
    "Fisheye close-up of a calico cat wearing a tiny flower crown, sniffing the camera lens in a sunny park, "
    "with bright colors, realistic fur detail, and playful viral-pet energy."
)
DEFAULT_IMG2IMG_SUFFIX = (
    " Preserve the exact same composition, camera framing, subject identity, pose, lighting, background layout, "
    "and geometry. Preserve the exact same composition while improving natural detail, clean edges, realistic material detail, "
    "and high-resolution sharpness. Do not change the scene."
)
OUTPUT_DIR = Path(os.getenv("LTX_GRADIO_OUTPUT_DIR", "outputs/gradio_distilled"))

GROUP_OFFLOAD_CONFIG = {
    "onload_device": torch.device(DEVICE),
    "offload_type": "leaf_level",
    "use_stream": True,
    "low_cpu_mem_usage": True,
}


def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def normalize_bits(bits):
    return None if bits == "bf16" else bits


def transformer_source(bits):
    normalized = normalize_bits(bits)
    return MODEL_PATH if normalized is None else SDNQ_MODEL_PATHS[normalized]


def ensure_size(width, height):
    width = int(width)
    height = int(height)
    if width % 32 != 0 or height % 32 != 0:
        raise gr.Error("Width and height must be divisible by 32.")
    return width, height


def parse_pag_layers(layers_text):
    if not layers_text or not str(layers_text).strip():
        return [28]
    try:
        return [int(item.strip()) for item in str(layers_text).split(",") if item.strip()]
    except ValueError as exc:
        raise gr.Error("PAG layers must be a comma-separated list of integers, for example: 28 or 16,27") from exc
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
        return requested_adapter_name
    if new_adapters:
        return sorted(new_adapters)[0]
    return requested_adapter_name


def activate_loras(pipe, crisp_enabled, crisp_scale, soft_enabled, soft_scale):
    active_adapters = []
    if crisp_enabled:
        adapter_name = load_lora_adapter(pipe, CRISP_LORA_PATH, CRISP_LORA_WEIGHT_NAME, CRISP_LORA_ADAPTER_NAME)
        active_adapters.append((adapter_name, float(crisp_scale)))
    if soft_enabled:
        adapter_name = load_lora_adapter(pipe, SOFT_LORA_PATH, SOFT_LORA_WEIGHT_NAME, SOFT_LORA_ADAPTER_NAME)
        active_adapters.append((adapter_name, float(soft_scale)))
    if active_adapters:
        available = available_lora_adapters(pipe)
        active_adapters = [(name, scale) for name, scale in active_adapters if name in available]
        if active_adapters:
            names, scales = zip(*active_adapters, strict=False)
            pipe.set_adapters(list(names), adapter_weights=list(scales))
    return active_adapters

def encode_prompt(prompt, negative_prompt=""):
    text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        subfolder="text_encoder",
        torch_dtype=DTYPE,
    )
    pipe = LTX2ImagePipeline.from_pretrained(
        MODEL_PATH,
        transformer=None,
        connectors=None,
        vae=None,
        scheduler=None,
        text_encoder=text_encoder,
        torch_dtype=DTYPE,
    )
    if torch.cuda.is_available():
        pipe.enable_sequential_cpu_offload()
    with torch.inference_mode():
        prompt_embeds, prompt_attention_mask, _, _ = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=False,
        )
    del pipe, text_encoder
    flush()
    return prompt_embeds.to(OFFLOAD_DEVICE), prompt_attention_mask.to(OFFLOAD_DEVICE)


def apply_offload(pipe, bits):
    if not torch.cuda.is_available():
        return
    if bits == "int4":
        pipe.enable_model_cpu_offload()
    else:
        pipe.enable_group_offload(**GROUP_OFFLOAD_CONFIG)


def decode_latents(latents, seed, decode_timestep=0.0, decode_noise_scale=None):
    vae = AutoencoderKLLTX2Video.from_pretrained(MODEL_PATH, subfolder="vae", torch_dtype=DTYPE).to(DEVICE)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    with torch.no_grad():
        latents_gpu = latents.to(device=DEVICE, dtype=DTYPE)
        if not vae.config.timestep_conditioning:
            vae_decode_timestep = None
        else:
            vae_decode_timestep = torch.tensor([decode_timestep] * latents_gpu.shape[0], device=DEVICE, dtype=latents_gpu.dtype)
            effective_decode_noise_scale = decode_timestep if decode_noise_scale is None else decode_noise_scale
            if effective_decode_noise_scale != 0.0:
                noise = torch.randn(latents_gpu.shape, generator=generator, device=DEVICE, dtype=latents_gpu.dtype)
                latents_gpu = (1 - effective_decode_noise_scale) * latents_gpu + effective_decode_noise_scale * noise

        latents_mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(DEVICE)
        latents_std = vae.latents_std.view(1, -1, 1, 1, 1).to(DEVICE)
        latents_gpu = (latents_gpu * latents_std) / vae.config.scaling_factor + latents_mean
        decoded = vae.decode(latents_gpu.to(vae.dtype), vae_decode_timestep, return_dict=False)[0]

    image_processor = VaeImageProcessor(vae_scale_factor=vae.spatial_compression_ratio)
    image = image_processor.postprocess(decoded[:, :, 0, :, :].cpu(), output_type="pil")[0]
    del vae, latents_gpu, decoded
    flush()
    return image


def run_stage1(
    prompt,
    width,
    height,
    seed,
    bits,
    pag_enabled,
    pag_scale,
    pag_layers,
    crisp_enabled,
    crisp_scale,
    soft_enabled,
    soft_scale,
    progress=None,
):
    progress = progress or gr.Progress()
    width, height = ensure_size(width, height)
    t0 = time.time()
    progress(0.05, desc="Encoding prompt")
    prompt_embeds, prompt_attention_mask = encode_prompt(prompt)

    progress(0.20, desc="Loading T2I transformer")
    source = transformer_source(bits)
    transformer = LTX2ImageTransformer2DModel.from_pretrained(
        source,
        subfolder="transformer",
        torch_dtype=DTYPE,
        device_map="cpu",
    )
    pipe = LTX2ImagePipeline.from_pretrained(
        MODEL_PATH,
        transformer=transformer,
        text_encoder=None,
        tokenizer=None,
        vae=None,
        torch_dtype=DTYPE,
    )
    active_loras = activate_loras(pipe, crisp_enabled, crisp_scale, soft_enabled, soft_scale)
    apply_offload(pipe, bits)

    progress(0.42, desc=f"Generating T2I latents at {width}x{height}")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    latents = pipe(
        prompt_embeds=prompt_embeds.to(device=DEVICE, dtype=DTYPE),
        prompt_attention_mask=prompt_attention_mask.to(device=DEVICE),
        negative_prompt_embeds=None,
        negative_prompt_attention_mask=None,
        width=width,
        height=height,
        num_inference_steps=8,
        guidance_scale=1.0,
        guidance_rescale=0.7,
        decode_timestep=0.0,
        decode_noise_scale=None,
        pag_scale=float(pag_scale) if pag_enabled else 0.0,
        pag_applied_layers=parse_pag_layers(pag_layers) if pag_enabled else None,
        generator=generator,
        output_type="latent",
        return_dict=False,
    )[0].to(OFFLOAD_DEVICE)

    del pipe, transformer
    flush()

    del prompt_embeds, prompt_attention_mask
    flush()

    progress(0.90, desc="Decoding T2I image")
    image = decode_latents(latents, seed)
    del latents
    flush()
    elapsed = time.time() - t0
    return image, {
        "stage": "stage1",
        "bits": bits,
        "loras": active_loras,
        "pag_enabled": pag_enabled,
        "pag_scale": float(pag_scale) if pag_enabled else 0.0,
        "pag_layers": parse_pag_layers(pag_layers) if pag_enabled else None,
        "elapsed_sec": round(elapsed, 2),
    }


def run_stage2(
    image,
    prompt,
    stage2_prompt_mode,
    stage2_custom_prompt,
    stage2_normalization_prompt,
    width,
    height,
    seed,
    bits,
    pag_enabled,
    pag_scale,
    pag_layers,
    strength,
    input_sharpen,
    input_noise_sigma,
    phase_enabled,
    phase_cutoff,
    crisp_enabled,
    crisp_scale,
    soft_enabled,
    soft_scale,
    progress=None,
):
    progress = progress or gr.Progress()
    if image is None:
        raise gr.Error("I2I needs either the T2I result or an uploaded reference image.")
    width, height = ensure_size(width, height)
    t0 = time.time()
    normalization = (stage2_normalization_prompt or "").strip()
    custom_prompt = (stage2_custom_prompt or "").strip()
    if stage2_prompt_mode == "T2I prompt only":
        normalize_prompt = prompt
    elif stage2_prompt_mode == "I2I prompt only":
        normalize_prompt = normalization or DEFAULT_IMG2IMG_SUFFIX.strip()
    elif stage2_prompt_mode == "Custom I2I prompt":
        normalize_prompt = custom_prompt or prompt
    else:
        normalize_prompt = (prompt.rstrip() + " " + normalization).strip() if normalization else prompt

    progress(0.05, desc="Encoding image-to-image prompt")
    prompt_embeds, prompt_attention_mask = encode_prompt(normalize_prompt)

    progress(0.20, desc="Loading image-to-image transformer")
    source = transformer_source(bits)
    transformer = LTX2ImageTransformer2DModel.from_pretrained(
        source,
        subfolder="transformer",
        torch_dtype=DTYPE,
        device_map="cpu",
    )
    pipe = LTX2ImageImg2ImgPipeline.from_pretrained(
        MODEL_PATH,
        transformer=transformer,
        text_encoder=None,
        tokenizer=None,
        torch_dtype=DTYPE,
    )
    active_loras = activate_loras(pipe, crisp_enabled, crisp_scale, soft_enabled, soft_scale)
    apply_offload(pipe, bits)

    progress(0.45, desc="Running image to image")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    pixel_generator = torch.Generator(device="cpu").manual_seed(int(seed))
    output = pipe(
        image=image,
        prompt_embeds=prompt_embeds.to(device=DEVICE, dtype=DTYPE),
        prompt_attention_mask=prompt_attention_mask.to(device=DEVICE),
        negative_prompt_embeds=None,
        negative_prompt_attention_mask=None,
        width=width,
        height=height,
        num_inference_steps=8,
        strength=float(strength),
        input_noise_sigma=float(input_noise_sigma),
        input_sharpen=float(input_sharpen),
        phase_cutoff=float(phase_cutoff) if phase_enabled else None,
        phase_transition_width=2.0,
        phase_pad_factor=1.5,
        guidance_scale=1.0,
        guidance_rescale=0.7,
        decode_timestep=0.0,
        decode_noise_scale=None,
        pag_scale=float(pag_scale) if pag_enabled else 0.0,
        pag_applied_layers=parse_pag_layers(pag_layers) if pag_enabled else None,
        generator=generator,
        pixel_generator=pixel_generator,
        output_type="pil",
        return_dict=False,
    )[0][0]

    del prompt_embeds, prompt_attention_mask, pipe, transformer
    flush()
    elapsed = time.time() - t0
    return output, {"stage": "stage2", "bits": bits, "loras": active_loras, "pag_enabled": pag_enabled, "pag_scale": float(pag_scale) if pag_enabled else 0.0, "pag_layers": parse_pag_layers(pag_layers) if pag_enabled else None, "elapsed_sec": round(elapsed, 2)}


def save_image(image, prefix):
    if image is None:
        return None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{prefix}_{int(time.time())}.png"
    image.save(path)
    return str(path)


def comparison_value(before, after):
    if before is None or after is None:
        return None
    return [before, after]


def run_app(
    prompt,
    width,
    height,
    seed,
    stage2_only,
    stage1_pag,
    stage1_pag_scale,
    stage1_pag_layers,
    stage1_crisp_lora,
    stage1_crisp_scale,
    stage1_enhance_lora,
    stage1_soft_scale,
    enable_stage2,
    stage2_input_image,
    stage2_prompt_mode,
    stage2_custom_prompt,
    stage2_normalization_prompt,
    stage2_pag,
    stage2_pag_scale,
    stage2_pag_layers,
    stage2_strength,
    stage2_preprocess_enabled,
    stage2_input_sharpen,
    stage2_input_noise_sigma,
    stage2_phase_enabled,
    stage2_phase_cutoff,
    stage2_crisp_lora,
    stage2_crisp_scale,
    stage2_enhance_lora,
    stage2_soft_scale,
    progress=None,
):
    progress = progress or gr.Progress()
    if not prompt or not prompt.strip():
        raise gr.Error("Prompt is required.")

    stage1_image = None
    stage2_image = None
    logs = []

    if stage2_only:
        if stage2_input_image is None:
            raise gr.Error("Upload an I2I reference image when I2I only is enabled.")
        stage1_source = stage2_input_image
        logs.append("T2I skipped; using uploaded I2I input image.")
    else:
        stage1_image, metrics = run_stage1(
            prompt=prompt,
            width=width,
            height=height,
            seed=seed,
            bits="bf16",
            pag_enabled=stage1_pag,
            pag_scale=stage1_pag_scale,
            pag_layers=stage1_pag_layers,
            crisp_enabled=stage1_crisp_lora,
            crisp_scale=stage1_crisp_scale,
            soft_enabled=stage1_enhance_lora,
            soft_scale=stage1_soft_scale,
            progress=progress,
        )
        stage1_source = stage1_image
        save_image(stage1_image, "stage1")
        logs.append(f"T2I done in {metrics['elapsed_sec']}s using {metrics['bits']} | PAG: {metrics['pag_scale']} {metrics['pag_layers']}.")

    if enable_stage2 or stage2_only:
        stage2_image, metrics = run_stage2(
            image=stage1_source,
            prompt=prompt,
            stage2_prompt_mode=stage2_prompt_mode,
            stage2_custom_prompt=stage2_custom_prompt,
            stage2_normalization_prompt=stage2_normalization_prompt,
            width=width,
            height=height,
            seed=seed,
            bits="bf16",
            pag_enabled=stage2_pag,
            pag_scale=stage2_pag_scale,
            pag_layers=stage2_pag_layers,
            strength=stage2_strength,
            input_sharpen=stage2_input_sharpen if stage2_preprocess_enabled else 1.0,
            input_noise_sigma=stage2_input_noise_sigma if stage2_preprocess_enabled else 0.0,
            phase_enabled=stage2_phase_enabled,
            phase_cutoff=stage2_phase_cutoff,
            crisp_enabled=stage2_crisp_lora,
            crisp_scale=stage2_crisp_scale,
            soft_enabled=stage2_enhance_lora,
            soft_scale=stage2_soft_scale,
            progress=progress,
        )
        save_image(stage2_image, "stage2")
        logs.append(f"I2I done in {metrics['elapsed_sec']}s using {metrics['bits']} | PAG: {metrics['pag_scale']} {metrics['pag_layers']}.")
    else:
        logs.append("I2I disabled.")

    compare_before = stage1_source if stage2_image is not None else None
    compare = comparison_value(compare_before, stage2_image)
    return stage1_image, stage2_image, compare, "\n".join(logs)

theme = gr.themes.Default(
    primary_hue="red", secondary_hue="orange", neutral_hue="gray"
).set(
    body_background_fill="*neutral_100",
    body_background_fill_dark="*neutral_900",
    body_text_color="*neutral_900",
    body_text_color_dark="*neutral_100",
    panel_background_fill="*neutral_800",
    panel_background_fill_dark="*neutral_900",
    input_background_fill="white",
    input_background_fill_dark="*neutral_800",
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_dark="*primary_700",
    button_primary_text_color="white",
    button_primary_text_color_dark="white",
    button_secondary_background_fill="*secondary_500",
    button_secondary_background_fill_dark="*secondary_700",
    button_secondary_text_color="white",
    button_secondary_text_color_dark="white",
)

with gr.Blocks(title="LTX 2.3 Image - Distilled Model") as demo:
    gr.Markdown("# LTX 2.3 Image - Distilled Model")
    gr.Markdown("T2I + optional I2I distilled image generation with optional SDNQ transformers and Enhance/Crisp LoRAs.")

    prompt = gr.Textbox(label="Prompt", value=DEFAULT_PROMPT, lines=4)
    with gr.Row():
        width = gr.Number(label="Width", value=1920, precision=0)
        height = gr.Number(label="Height", value=1056, precision=0)
        seed = gr.Number(label="Seed", value=43, precision=0)
        stage2_only = gr.Checkbox(label="I2I only", value=False)

    with gr.Accordion("Text-to-Image", open=True), gr.Row():
        #stage1_bits = gr.Radio(label="Transformer", choices=["int8", "int4", "bf16"], value="int8")
        stage1_pag = gr.Checkbox(label="PAG", value=False)
        stage1_pag_scale = gr.Slider(label="PAG Scale", minimum=0.0, maximum=1.5, value=0.2, step=0.05)
        stage1_pag_layers = gr.Textbox(label="PAG Layers", value="28")
        stage1_crisp_lora = gr.Checkbox(label="Crisp LoRA", value=False)
        stage1_crisp_scale = gr.Slider(label="Crisp Scale", minimum=0.0, maximum=2.0, value=0.3, step=0.05)
        stage1_enhance_lora = gr.Checkbox(label="Enhance LoRA", value=False)
        stage1_soft_scale = gr.Slider(label="Enhance Scale", minimum=0.0, maximum=2.0, value=0.15, step=0.05)
    with gr.Accordion("Image-to-Image", open=False):
        with gr.Row():
            enable_stage2 = gr.Checkbox(label="Enable I2I", value=False)
            #stage2_bits = gr.Radio(label="Transformer", choices=["bf16", "int8", "int4"], value="bf16")
            stage2_pag = gr.Checkbox(label="PAG", value=False)
            stage2_pag_scale = gr.Slider(label="PAG Scale", minimum=0.0, maximum=1.5, value=0.2, step=0.05)
            stage2_pag_layers = gr.Textbox(label="PAG Layers", value="28")
        stage2_input_image = gr.Image(label="Optional I2I Input", type="pil")
        stage2_prompt_mode = gr.Radio(label="Prompt Mode", choices=["T2I prompt only", "T2I prompt + I2I prompt", "I2I prompt only", "Custom I2I prompt"], value="T2I prompt only")
        stage2_custom_prompt = gr.Textbox(label="Custom I2I Prompt", value="", lines=3)
        stage2_normalization_prompt = gr.Textbox(label="I2I Prompt", value=DEFAULT_IMG2IMG_SUFFIX.strip(), lines=3)
        with gr.Row():
            stage2_strength = gr.Slider(label="Strength", minimum=0.0, maximum=1.0, value=0.2, step=0.01)
            stage2_preprocess_enabled = gr.Checkbox(label="Input Preprocess", value=True)
            stage2_input_sharpen = gr.Slider(label="Input Sharpen", minimum=0.0, maximum=2.0, value=1.3, step=0.05)
            stage2_input_noise_sigma = gr.Slider(label="Input Noise Sigma", minimum=0.0, maximum=15.0, value=5.0, step=0.5)
        with gr.Row():
            stage2_phase_enabled = gr.Checkbox(label="Structured Noise Cutoff", value=True)
            stage2_phase_cutoff = gr.Slider(label="Phase Cutoff Radius", minimum=0.0, maximum=15.0, value=9.0, step=0.5)
        with gr.Row():
            stage2_crisp_lora = gr.Checkbox(label="Crisp LoRA", value=False)
            stage2_crisp_scale = gr.Slider(label="Crisp Scale", minimum=0.0, maximum=2.0, value=0.5, step=0.05)
            stage2_enhance_lora = gr.Checkbox(label="Enhance LoRA", value=False)
            stage2_soft_scale = gr.Slider(label="Enhance Scale", minimum=0.0, maximum=2.0, value=0.5, step=0.05)

    with gr.Row():
        run_button = gr.Button("Generate", variant="primary")
        clear_button = gr.Button("Clear Outputs")

    with gr.Row():
        stage1_output = gr.Image(label="T2I Output", type="pil")
        stage2_output = gr.Image(label="I2I Output", type="pil")
    comparison_output = gr.ImageSlider(label="I2I Comparison", type="pil")
    status = gr.Textbox(label="Status", lines=4)

    inputs = [
        prompt,
        width,
        height,
        seed,
        stage2_only,
        stage1_pag,
        stage1_pag_scale,
        stage1_pag_layers,
        stage1_crisp_lora,
        stage1_crisp_scale,
        stage1_enhance_lora,
        stage1_soft_scale,
        enable_stage2,
        stage2_input_image,
        stage2_prompt_mode,
        stage2_custom_prompt,
        stage2_normalization_prompt,
        stage2_pag,
        stage2_pag_scale,
        stage2_pag_layers,
        stage2_strength,
        stage2_preprocess_enabled,
        stage2_input_sharpen,
        stage2_input_noise_sigma,
        stage2_phase_enabled,
        stage2_phase_cutoff,
        stage2_crisp_lora,
        stage2_crisp_scale,
        stage2_enhance_lora,
        stage2_soft_scale,
    ]
    run_button.click(run_app, inputs=inputs, outputs=[stage1_output, stage2_output, comparison_output, status], show_progress_on=[status])
    clear_button.click(lambda: (None, None, None), outputs=[stage1_output, stage2_output, comparison_output])

    gr.Examples(
        examples=[
            [DEFAULT_PROMPT],
            ["Macro photograph of a blue butterfly resting on glossy wet jungle leaves, realistic detail, natural light, shallow depth of field."],
            ["Cinematic realistic jungle scene with a young explorer and a calm gorilla exchanging a small stone tool, lush forest, soft natural light, emotional eye contact."],
            ["A woman in her 30s sits by the window of a small Parisian cafe. Rain runs down the glass behind her. Warm tungsten interior lighting. She slowly stirs her coffee while glancing at her phone. Background softly out of focus. The ambient sound of rain pattering against the glass and soft cafe chatter fills the air."],
        ],
        inputs=[prompt],
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(theme=theme)
