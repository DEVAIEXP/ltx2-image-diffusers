# Copyright 2026 The HuggingFace Team and DEVAIEXP.
#
# Licensed under the Apache License, Version 2.0.

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from transformers import Gemma3ForConditionalGeneration, PreTrainedModel, PreTrainedTokenizerBase

from diffusers import AutoencoderKLLTX2Video, FlowMatchEulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.modular_pipelines import ComponentSpec, InputParam, ModularPipelineBlocks, OutputParam, PipelineState
from diffusers.modular_pipelines.modular_pipeline import SequentialPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import InsertableDict
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor

try:
    from .connectors_ltx2_image import LTX2ImageTextConnectors
    from .transformer_ltx2_image import LTX2ImageTransformer2DModel
except ImportError:
    from diffusers.models.transformers import LTX2ImageTransformer2DModel
    from diffusers.pipelines.ltx2.connectors import LTX2ImageTextConnectors


@dataclass
class LTX2ImagePipelineOutput(BaseOutput):
    """Output container matching the LTX image pipeline image result."""

    images: Union[List[PIL.Image.Image], np.ndarray, torch.Tensor]


# Copied from diffusers.pipelines.ltx2.pipeline_ltx2_image.calculate_shift.
def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Calculate FlowMatch timestep shift from the image sequence length."""

    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


# Copied from diffusers.pipelines.ltx2.pipeline_ltx2_image.retrieve_timesteps.
def retrieve_timesteps(
    scheduler: FlowMatchEulerDiscreteScheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, int]:
    """Set scheduler timesteps from custom timesteps, sigmas, or a step count."""

    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# Copied from diffusers.pipelines.ltx2.pipeline_ltx2_image.rescale_noise_cfg.
def rescale_noise_cfg(
    noise_cfg: torch.Tensor, noise_pred_text: torch.Tensor, guidance_rescale: float = 0.0
) -> torch.Tensor:
    """Rescale guided noise using the text prediction standard deviation."""

    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    return guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg


def normalize_vae_latents(vae: AutoencoderKLLTX2Video, latents: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Apply LTX VAE latent normalization."""

    latents_mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(device=device, dtype=latents.dtype)
    latents_std = vae.latents_std.view(1, -1, 1, 1, 1).to(device=device, dtype=latents.dtype)
    return (latents - latents_mean) * vae.config.scaling_factor / latents_std


def get_strength_sigmas(num_inference_steps: int, strength: float) -> tuple[list[float], int]:
    """Slice the default sigma schedule according to an img2img strength value."""

    if strength < 0 or strength > 1:
        raise ValueError(f"`strength` must be in [0.0, 1.0], got {strength}.")
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps, dtype=np.float32)
    init_timestep = min(num_inference_steps * strength, num_inference_steps)
    t_start = int(max(num_inference_steps - init_timestep, 0))
    selected_sigmas = sigmas[t_start:]
    if len(selected_sigmas) == 0:
        raise ValueError("The chosen strength leaves zero denoising steps. Increase `strength`.")
    return selected_sigmas.tolist(), t_start


def get_default_sigmas(
    num_inference_steps: int, timesteps: Optional[List[int]], sigmas: Optional[List[float]]
) -> Optional[Union[List[float], np.ndarray]]:
    """Return the default sigma schedule unless custom timesteps or sigmas were provided."""

    if timesteps is not None:
        return sigmas
    if sigmas is not None:
        return sigmas
    return np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)


def add_deterministic_noise(
    image: PIL.Image.Image, sigma: float, generator: Optional[torch.Generator] = None
) -> PIL.Image.Image:
    """Add deterministic RGB pixel noise before VAE encoding."""

    if sigma is None or sigma <= 0:
        return image
    image_np = np.array(image.convert("RGB"))
    image_tensor = torch.from_numpy(image_np).float()
    noise = torch.randn(image_tensor.shape, generator=generator, dtype=image_tensor.dtype) * sigma
    noisy_tensor = torch.clamp(image_tensor + noise, 0, 255).to(torch.uint8)
    return Image.fromarray(noisy_tensor.numpy())


def preprocess_refinement_image(
    image: Union[str, PIL.Image.Image],
    width: int,
    height: int,
    noise_sigma: float = 0.0,
    sharpen: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> PIL.Image.Image:
    """Load, resize, and optionally perturb a reference image for img2img."""

    if isinstance(image, str):
        image = Image.open(image)
    if not isinstance(image, PIL.Image.Image):
        raise ValueError("`image` must be a PIL image or a local image path for LTX2 image img2img.")

    image = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    image = add_deterministic_noise(image, noise_sigma, generator)
    if sharpen != 1.0:
        image = ImageEnhance.Sharpness(image).enhance(sharpen)
    return image


def create_frequency_soft_cutoff_mask(
    height: int,
    width: int,
    cutoff_radius: float,
    transition_width: float = 5.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Create a radial soft cutoff mask in frequency space."""

    u = torch.arange(height, device=device)
    v = torch.arange(width, device=device)
    u, v = torch.meshgrid(u, v, indexing="ij")
    center_u, center_v = height // 2, width // 2
    frequency_radius = torch.sqrt((u - center_u) ** 2 + (v - center_v) ** 2)
    mask = torch.exp(-((frequency_radius - cutoff_radius) ** 2) / (2 * transition_width**2))
    return torch.where(frequency_radius <= cutoff_radius, torch.ones_like(mask), mask)


def generate_structured_noise(
    latents: torch.Tensor,
    input_noise: torch.Tensor,
    cutoff_radius: Optional[float],
    transition_width: float = 2.0,
    pad_factor: float = 1.5,
) -> torch.Tensor:
    """Mix low-frequency image phase into the latent noise for img2img preservation."""

    if cutoff_radius is None or cutoff_radius <= 0:
        return input_noise

    batch, channels, frames, height, width = latents.shape
    latent_4d = latents.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width).float()
    noise_4d = input_noise.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width).float()

    pad_h = int(height * (pad_factor - 1)) // 2 * 2
    pad_w = int(width * (pad_factor - 1)) // 2 * 2
    padded_latents = F.pad(latent_4d, (pad_w // 2, pad_w // 2, pad_h // 2, pad_h // 2), mode="reflect")
    padded_noise = F.pad(noise_4d, (pad_w // 2, pad_w // 2, pad_h // 2, pad_h // 2), mode="reflect")

    padded_height, padded_width = padded_latents.shape[-2:]
    fft_latents = torch.fft.fftshift(torch.fft.fft2(padded_latents, dim=(-2, -1)), dim=(-2, -1))
    fft_noise = torch.fft.fftshift(torch.fft.fft2(padded_noise, dim=(-2, -1)), dim=(-2, -1))
    mask = create_frequency_soft_cutoff_mask(
        padded_height, padded_width, cutoff_radius, transition_width, latents.device
    ).unsqueeze(0).unsqueeze(0)
    mixed_phase = mask * torch.angle(fft_latents) + (1 - mask) * torch.angle(fft_noise)
    fft_combined = torch.abs(fft_noise) * torch.exp(1j * mixed_phase)
    structured_noise = torch.real(torch.fft.ifft2(torch.fft.ifftshift(fft_combined, dim=(-2, -1)), dim=(-2, -1)))
    structured_noise = structured_noise[..., pad_h // 2 : pad_h // 2 + height, pad_w // 2 : pad_w // 2 + width]
    return structured_noise.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4).to(input_noise.dtype)


def _get_gemma_prompt_embeds(
    components: Any,
    prompt: Union[str, List[str]],
    max_sequence_length: int = 1024,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode prompts with Gemma and pack hidden states for the LTX connectors."""

    device = device or components._execution_device
    dtype = dtype or components.text_encoder.dtype

    prompt = [prompt] if isinstance(prompt, str) else prompt
    components.tokenizer.padding_side = "left"
    if components.tokenizer.pad_token is None:
        components.tokenizer.pad_token = components.tokenizer.eos_token

    prompt = [p.strip() for p in prompt]
    text_inputs = components.tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)
    prompt_attention_mask = text_inputs.attention_mask.to(device)

    text_encoder_outputs = components.text_encoder(
        input_ids=text_input_ids, attention_mask=prompt_attention_mask, output_hidden_states=True
    )
    text_encoder_hidden_states = torch.stack(text_encoder_outputs.hidden_states, dim=-1)
    prompt_embeds = text_encoder_hidden_states.flatten(2, 3).to(dtype=dtype)

    return prompt_embeds, prompt_attention_mask


class LTX2ImageTextEncoderStep(ModularPipelineBlocks):
    """Encode prompts into packed Gemma hidden states for LTX 2 image."""

    model_name = None

    @property
    def description(self) -> str:
        return "Encodes text prompts into packed per-layer Gemma hidden states for the LTX image transformer."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", PreTrainedModel),
            ComponentSpec("tokenizer", PreTrainedTokenizerBase),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt"),
            InputParam.template("negative_prompt"),
            InputParam.template("max_sequence_length", default=1024),
            InputParam("guidance_scale", type_hint=float, default=1.0),
            InputParam("prompt_embeds", type_hint=torch.Tensor, default=None),
            InputParam("prompt_attention_mask", type_hint=torch.Tensor, default=None),
            InputParam("negative_prompt_embeds", type_hint=torch.Tensor, default=None),
            InputParam("negative_prompt_attention_mask", type_hint=torch.Tensor, default=None),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("prompt_embeds", type_hint=torch.Tensor, description="Prompt embeddings."),
            OutputParam("prompt_attention_mask", type_hint=torch.Tensor, description="Prompt attention mask."),
            OutputParam("negative_prompt_embeds", type_hint=torch.Tensor, description="Negative prompt embeddings."),
            OutputParam(
                "negative_prompt_attention_mask", type_hint=torch.Tensor, description="Negative prompt attention mask."
            ),
            OutputParam("batch_size", type_hint=int, description="Prompt batch size."),
            OutputParam("dtype", type_hint=torch.dtype, description="Transformer dtype."),
            OutputParam("do_classifier_free_guidance", type_hint=bool, description="Whether CFG is active."),
        ]

    @torch.no_grad()
    def __call__(self, components: Any, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)
        device = components._execution_device
        dtype = components.text_encoder.dtype if getattr(components, "text_encoder", None) is not None else torch.bfloat16
        do_cfg = block_state.guidance_scale > 1.0

        if block_state.prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = _get_gemma_prompt_embeds(
                components,
                block_state.prompt,
                max_sequence_length=block_state.max_sequence_length,
                device=device,
                dtype=dtype,
            )
            block_state.prompt_embeds = prompt_embeds
            block_state.prompt_attention_mask = prompt_attention_mask

        if do_cfg and block_state.negative_prompt_embeds is None:
            prompt = [block_state.prompt] if isinstance(block_state.prompt, str) else block_state.prompt
            batch_size = len(prompt) if prompt is not None else block_state.prompt_embeds.shape[0]
            negative_prompt = block_state.negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_embeds, negative_attention_mask = _get_gemma_prompt_embeds(
                components,
                negative_prompt,
                max_sequence_length=block_state.max_sequence_length,
                device=device,
                dtype=dtype,
            )
            block_state.negative_prompt_embeds = negative_embeds
            block_state.negative_prompt_attention_mask = negative_attention_mask

        block_state.do_classifier_free_guidance = do_cfg and block_state.negative_prompt_embeds is not None
        block_state.batch_size = block_state.prompt_embeds.shape[0]
        block_state.dtype = dtype

        self.set_block_state(state, block_state)
        return components, state


class LTX2ImageConnectorStep(ModularPipelineBlocks):
    """Run the LTX 2 image text connectors and expand batches for CFG/PAG."""

    model_name = None

    @property
    def description(self) -> str:
        return "Adapts Gemma hidden states with the image-only text connectors."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("connectors", LTX2ImageTextConnectors)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("prompt_embeds", type_hint=torch.Tensor, required=True),
            InputParam("prompt_attention_mask", type_hint=torch.Tensor, required=True),
            InputParam("negative_prompt_embeds", type_hint=torch.Tensor, default=None),
            InputParam("negative_prompt_attention_mask", type_hint=torch.Tensor, default=None),
            InputParam("do_classifier_free_guidance", type_hint=bool, required=True),
            InputParam("pag_scale", type_hint=float, default=0.0),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("connector_prompt_embeds", type_hint=torch.Tensor, description="Connector prompt embeddings."),
            OutputParam("connector_attention_mask", type_hint=torch.Tensor, description="Connector attention mask."),
            OutputParam("batch_size", type_hint=int, description="Prompt batch size after guidance expansion."),
            OutputParam("transformer_batch_multiplier", type_hint=int, description="Number of denoiser passes per item."),
            OutputParam("do_perturbed_attention_guidance", type_hint=bool, description="Whether PAG is active."),
        ]

    @torch.no_grad()
    def __call__(self, components: Any, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)
        do_pag = block_state.pag_scale > 0.0
        prompt_embeds = block_state.prompt_embeds
        prompt_attention_mask = block_state.prompt_attention_mask

        if do_pag:
            if block_state.do_classifier_free_guidance:
                prompt_embeds = torch.cat([block_state.negative_prompt_embeds, prompt_embeds, prompt_embeds], dim=0)
                prompt_attention_mask = torch.cat(
                    [
                        block_state.negative_prompt_attention_mask,
                        prompt_attention_mask,
                        prompt_attention_mask,
                    ],
                    dim=0,
                )
            else:
                prompt_embeds = torch.cat([prompt_embeds, prompt_embeds], dim=0)
                prompt_attention_mask = torch.cat([prompt_attention_mask, prompt_attention_mask], dim=0)
        elif block_state.do_classifier_free_guidance:
            prompt_embeds = torch.cat([block_state.negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([block_state.negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        padding_side = getattr(getattr(components, "tokenizer", None), "padding_side", "left")
        if getattr(components, "connectors", None) is not None:
            block_state.connector_prompt_embeds, block_state.connector_attention_mask = components.connectors(
                prompt_embeds, prompt_attention_mask, padding_side=padding_side
            )
        else:
            block_state.connector_prompt_embeds = prompt_embeds
            block_state.connector_attention_mask = prompt_attention_mask

        if do_pag:
            multiplier = 3 if block_state.do_classifier_free_guidance else 2
            batch_size = prompt_embeds.shape[0] // multiplier
        elif block_state.do_classifier_free_guidance:
            multiplier = 2
            batch_size = prompt_embeds.shape[0] // 2
        else:
            multiplier = 1
            batch_size = prompt_embeds.shape[0]

        block_state.batch_size = batch_size
        block_state.transformer_batch_multiplier = multiplier
        block_state.do_perturbed_attention_guidance = do_pag

        self.set_block_state(state, block_state)
        return components, state


class LTX2ImageVaeEncoderStep(ModularPipelineBlocks):
    """Encode an optional reference image into normalized one-frame VAE latents."""

    model_name = None

    @property
    def description(self) -> str:
        return "Optionally encodes an input image into normalized LTX image latents for img2img."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("vae", AutoencoderKLLTX2Video)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("image", type_hint=Union[str, PIL.Image.Image], default=None),
            InputParam.template("height", default=544),
            InputParam.template("width", default=960),
            InputParam.template("generator"),
            InputParam("pixel_generator", type_hint=torch.Generator, default=None),
            InputParam("image_latents", type_hint=torch.Tensor, default=None),
            InputParam("input_noise_sigma", type_hint=float, default=0.0),
            InputParam("input_sharpen", type_hint=float, default=1.0),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("image_latents", type_hint=torch.Tensor, description="Encoded img2img latents.")]

    @torch.no_grad()
    def __call__(self, components: Any, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)
        if block_state.image_latents is not None:
            block_state.image_latents = block_state.image_latents.to(components._execution_device, dtype=torch.float32)
            self.set_block_state(state, block_state)
            return components, state

        if block_state.image is None:
            block_state.image_latents = None
            self.set_block_state(state, block_state)
            return components, state

        if block_state.height % 32 != 0 or block_state.width % 32 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 32 but are {block_state.height} and {block_state.width}."
            )

        device = components._execution_device
        init_image = preprocess_refinement_image(
            block_state.image,
            block_state.width,
            block_state.height,
            noise_sigma=block_state.input_noise_sigma,
            sharpen=block_state.input_sharpen,
            generator=block_state.pixel_generator,
        )
        image_processor = VaeImageProcessor(vae_scale_factor=components.vae.spatial_compression_ratio)
        image_tensor = image_processor.preprocess(init_image, height=block_state.height, width=block_state.width)
        video_tensor = image_tensor.unsqueeze(2).to(device=device, dtype=components.vae.dtype)
        image_latents = components.vae.encode(video_tensor, return_dict=False)[0].sample(generator=block_state.generator)
        block_state.image_latents = normalize_vae_latents(components.vae, image_latents, device).to(dtype=torch.float32)

        self.set_block_state(state, block_state)
        return components, state


class LTX2ImagePrepareLatentsStep(ModularPipelineBlocks):
    """Prepare random or img2img latents, schedule timesteps, and rotary embeddings."""

    model_name = None

    @property
    def description(self) -> str:
        return "Prepares one-frame image latents and FlowMatch timesteps."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
            ComponentSpec("transformer", LTX2ImageTransformer2DModel),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("height", default=544),
            InputParam.template("width", default=960),
            InputParam.template("num_inference_steps", default=30),
            InputParam.template("sigmas"),
            InputParam.template("timesteps"),
            InputParam.template("generator"),
            InputParam("latents", type_hint=torch.Tensor, default=None),
            InputParam("image_latents", type_hint=torch.Tensor, default=None),
            InputParam("strength", type_hint=float, default=0.55),
            InputParam("phase_cutoff", type_hint=float, default=None),
            InputParam("phase_transition_width", type_hint=float, default=2.0),
            InputParam("phase_pad_factor", type_hint=float, default=1.5),
            InputParam("batch_size", type_hint=int, required=True),
            InputParam("transformer_batch_multiplier", type_hint=int, required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("latents", type_hint=torch.Tensor, description="Flattened image latents."),
            OutputParam("timesteps", type_hint=torch.Tensor, description="Denoising timesteps."),
            OutputParam("num_inference_steps", type_hint=int, description="Effective number of denoising steps."),
            OutputParam("latent_height", type_hint=int, description="Latent height."),
            OutputParam("latent_width", type_hint=int, description="Latent width."),
            OutputParam("in_channels", type_hint=int, description="Latent channel count."),
            OutputParam("video_rotary_emb", type_hint=tuple, description="Rotary embeddings for one-frame image tokens."),
        ]

    @torch.no_grad()
    def __call__(self, components: Any, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)
        if block_state.height % 32 != 0 or block_state.width % 32 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 32 but are {block_state.height} and {block_state.width}."
            )

        device = components._execution_device
        vae = getattr(components, "vae", None)
        vae_scale = getattr(vae, "spatial_compression_ratio", 32)
        latent_height = block_state.height // vae_scale
        latent_width = block_state.width // vae_scale
        in_channels = components.transformer.config.in_channels
        latent_shape = (block_state.batch_size, in_channels, 1, latent_height, latent_width)

        if block_state.image_latents is not None:
            if block_state.timesteps is None and block_state.sigmas is None:
                sigmas, _ = get_strength_sigmas(block_state.num_inference_steps, block_state.strength)
            else:
                sigmas = block_state.sigmas
            scheduler_num_steps = len(sigmas) if sigmas is not None else block_state.num_inference_steps

            init_latents = block_state.image_latents.to(device=device, dtype=torch.float32)
            if init_latents.shape[0] != block_state.batch_size:
                if block_state.batch_size % init_latents.shape[0] != 0:
                    raise ValueError(
                        f"`image_latents` batch size {init_latents.shape[0]} does not match prompt batch size "
                        f"{block_state.batch_size}."
                    )
                init_latents = init_latents.repeat(block_state.batch_size // init_latents.shape[0], 1, 1, 1, 1)
            if tuple(init_latents.shape) != latent_shape:
                raise ValueError(f"`image_latents` must have shape {latent_shape}, got {tuple(init_latents.shape)}.")

            seq_len = latent_height * latent_width
            mu = calculate_shift(
                seq_len,
                components.scheduler.config.get("base_image_seq_len", 1024),
                components.scheduler.config.get("max_image_seq_len", 4096),
                components.scheduler.config.get("base_shift", 0.95),
                components.scheduler.config.get("max_shift", 2.05),
            )
            prepare_timesteps, _ = retrieve_timesteps(
                components.scheduler,
                scheduler_num_steps,
                device,
                block_state.timesteps,
                sigmas=sigmas,
                mu=mu,
            )
            noise = randn_tensor(init_latents.shape, generator=block_state.generator, device=device, dtype=torch.float32)
            noise = generate_structured_noise(
                init_latents,
                noise,
                block_state.phase_cutoff,
                transition_width=block_state.phase_transition_width,
                pad_factor=block_state.phase_pad_factor,
            )
            latent_timestep = prepare_timesteps[:1].repeat(init_latents.shape[0])
            latents = components.scheduler.scale_noise(init_latents, latent_timestep, noise)
        elif block_state.latents is None:
            latents = randn_tensor(latent_shape, generator=block_state.generator, device=device, dtype=torch.float32)
            sigmas = get_default_sigmas(
                block_state.num_inference_steps, block_state.timesteps, block_state.sigmas
            )
            scheduler_num_steps = len(sigmas) if sigmas is not None else block_state.num_inference_steps
        else:
            latents = block_state.latents.to(device=device, dtype=torch.float32)
            sigmas = get_default_sigmas(
                block_state.num_inference_steps, block_state.timesteps, block_state.sigmas
            )
            scheduler_num_steps = len(sigmas) if sigmas is not None else block_state.num_inference_steps

        latents = latents.permute(0, 2, 3, 4, 1).flatten(1, 3)

        mu = calculate_shift(
            latents.shape[1],
            components.scheduler.config.get("base_image_seq_len", 1024),
            components.scheduler.config.get("max_image_seq_len", 4096),
            components.scheduler.config.get("base_shift", 0.95),
            components.scheduler.config.get("max_shift", 2.05),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            components.scheduler,
            scheduler_num_steps,
            device,
            block_state.timesteps,
            sigmas=sigmas,
            mu=mu,
        )
        components.scheduler.set_begin_index(0)

        transformer_batch_size = block_state.batch_size * block_state.transformer_batch_multiplier
        video_coords = components.transformer.rope.prepare_video_coords(
            transformer_batch_size, 1, latent_height, latent_width, device, fps=24.0
        )

        block_state.latents = latents
        block_state.timesteps = timesteps
        block_state.num_inference_steps = num_inference_steps
        block_state.latent_height = latent_height
        block_state.latent_width = latent_width
        block_state.in_channels = in_channels
        block_state.video_rotary_emb = components.transformer.rope(video_coords, device=device)

        self.set_block_state(state, block_state)
        return components, state


class LTX2ImageDenoiseStep(ModularPipelineBlocks):
    """Denoise flattened one-frame LTX 2 image latents."""

    model_name = None

    @property
    def description(self) -> str:
        return "Runs the image-only denoising loop."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
            ComponentSpec("transformer", LTX2ImageTransformer2DModel),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latents", type_hint=torch.Tensor, required=True),
            InputParam("timesteps", type_hint=torch.Tensor, required=True),
            InputParam("connector_prompt_embeds", type_hint=torch.Tensor, required=True),
            InputParam("connector_attention_mask", type_hint=torch.Tensor, required=True),
            InputParam("latent_height", type_hint=int, required=True),
            InputParam("latent_width", type_hint=int, required=True),
            InputParam("video_rotary_emb", type_hint=tuple, required=True),
            InputParam("batch_size", type_hint=int, required=True),
            InputParam("transformer_batch_multiplier", type_hint=int, required=True),
            InputParam("do_classifier_free_guidance", type_hint=bool, required=True),
            InputParam("do_perturbed_attention_guidance", type_hint=bool, required=True),
            InputParam("guidance_scale", type_hint=float, default=1.0),
            InputParam("guidance_rescale", type_hint=float, default=0.7),
            InputParam("pag_scale", type_hint=float, default=0.0),
            InputParam("pag_applied_layers", type_hint=list, default=None),
            InputParam("callback_on_step_end", type_hint=Callable, default=None),
            InputParam("callback_on_step_end_tensor_inputs", type_hint=list, default=["latents"]),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("latents", type_hint=torch.Tensor, description="Denoised flattened image latents.")]

    @staticmethod
    def convert_velocity_to_x0(
        sample: torch.Tensor, denoised_output: torch.Tensor, step_idx: int, scheduler: FlowMatchEulerDiscreteScheduler
    ) -> torch.Tensor:
        return sample - denoised_output * scheduler.sigmas[step_idx]

    @staticmethod
    def convert_x0_to_velocity(
        sample: torch.Tensor, denoised_output: torch.Tensor, step_idx: int, scheduler: FlowMatchEulerDiscreteScheduler
    ) -> torch.Tensor:
        return (sample - denoised_output) / scheduler.sigmas[step_idx]

    @torch.no_grad()
    def __call__(self, components: Any, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)
        dtype = components.transformer.dtype
        device = components._execution_device
        pag_applied_layers = block_state.pag_applied_layers

        if block_state.do_perturbed_attention_guidance and pag_applied_layers is None:
            num_transformer_blocks = len(components.transformer.transformer_blocks)
            pag_applied_layers = [28 if num_transformer_blocks > 28 else num_transformer_blocks // 2]

        for i, t in enumerate(block_state.timesteps):
            latent_model_input = (
                torch.cat([block_state.latents] * block_state.transformer_batch_multiplier)
                if block_state.transformer_batch_multiplier > 1
                else block_state.latents
            )
            latent_model_input = latent_model_input.to(dtype)
            perturbation_mask = None
            if block_state.do_perturbed_attention_guidance:
                perturbation_mask = torch.ones((latent_model_input.shape[0],), device=device, dtype=dtype)
                perturbation_mask[-block_state.batch_size :] = 0
            t_input = t.expand(latent_model_input.shape[0]).to(device)

            noise_pred = components.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=block_state.connector_prompt_embeds,
                timestep=t_input,
                encoder_attention_mask=block_state.connector_attention_mask,
                height=block_state.latent_height,
                width=block_state.latent_width,
                video_rotary_emb=block_state.video_rotary_emb,
                pag_applied_layers=pag_applied_layers,
                perturbation_mask=perturbation_mask,
                return_dict=False,
            )[0].float()

            if block_state.do_perturbed_attention_guidance:
                if block_state.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text, noise_pred_perturb = noise_pred.chunk(3)
                    noise_pred_uncond = self.convert_velocity_to_x0(
                        block_state.latents, noise_pred_uncond, i, components.scheduler
                    )
                    noise_pred_text = self.convert_velocity_to_x0(
                        block_state.latents, noise_pred_text, i, components.scheduler
                    )
                    noise_pred_perturb = self.convert_velocity_to_x0(
                        block_state.latents, noise_pred_perturb, i, components.scheduler
                    )
                    noise_pred = (
                        noise_pred_text
                        + (block_state.guidance_scale - 1.0) * (noise_pred_text - noise_pred_uncond)
                        + block_state.pag_scale * (noise_pred_text - noise_pred_perturb)
                    )
                    if block_state.guidance_rescale > 0.0:
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, block_state.guidance_rescale)
                else:
                    noise_pred_text, noise_pred_perturb = noise_pred.chunk(2)
                    noise_pred_text = self.convert_velocity_to_x0(
                        block_state.latents, noise_pred_text, i, components.scheduler
                    )
                    noise_pred_perturb = self.convert_velocity_to_x0(
                        block_state.latents, noise_pred_perturb, i, components.scheduler
                    )
                    noise_pred = noise_pred_text + block_state.pag_scale * (noise_pred_text - noise_pred_perturb)
                    if block_state.guidance_rescale > 0.0:
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, block_state.guidance_rescale)
                noise_pred = self.convert_x0_to_velocity(block_state.latents, noise_pred, i, components.scheduler)
            elif block_state.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred_uncond = self.convert_velocity_to_x0(
                    block_state.latents, noise_pred_uncond, i, components.scheduler
                )
                noise_pred_text = self.convert_velocity_to_x0(block_state.latents, noise_pred_text, i, components.scheduler)
                noise_pred = noise_pred_text + (block_state.guidance_scale - 1.0) * (
                    noise_pred_text - noise_pred_uncond
                )
                if block_state.guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, block_state.guidance_rescale)
                noise_pred = self.convert_x0_to_velocity(block_state.latents, noise_pred, i, components.scheduler)

            block_state.latents = components.scheduler.step(
                noise_pred, t, block_state.latents, return_dict=False
            )[0]

            if block_state.callback_on_step_end is not None:
                callback_kwargs = {}
                for tensor_name in block_state.callback_on_step_end_tensor_inputs:
                    callback_kwargs[tensor_name] = getattr(block_state, tensor_name)
                callback_outputs = block_state.callback_on_step_end(components, i, t, callback_kwargs)
                callback_outputs = callback_outputs or {}
                block_state.latents = callback_outputs.get("latents", block_state.latents)

        self.set_block_state(state, block_state)
        return components, state


class LTX2ImageDecodeStep(ModularPipelineBlocks):
    """Decode flattened one-frame LTX 2 image latents into images."""

    model_name = None

    @property
    def description(self) -> str:
        return "Decodes one-frame LTX image latents into PIL images or returns latent output."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("vae", AutoencoderKLLTX2Video)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latents", type_hint=torch.Tensor, required=True),
            InputParam("batch_size", type_hint=int, required=True),
            InputParam("latent_height", type_hint=int, required=True),
            InputParam("latent_width", type_hint=int, required=True),
            InputParam("in_channels", type_hint=int, required=True),
            InputParam("decode_timestep", type_hint=Union[float, list], default=0.0),
            InputParam("decode_noise_scale", type_hint=Union[float, list], default=None),
            InputParam.template("generator"),
            InputParam.template("output_type", default="pil"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("images", type_hint=Union[list, np.ndarray, torch.Tensor], description="Generated images.")]

    @torch.no_grad()
    def __call__(self, components: Any, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)
        device = components._execution_device
        dtype = components.vae.dtype
        latents = block_state.latents.reshape(
            block_state.batch_size, 1, block_state.latent_height, block_state.latent_width, block_state.in_channels
        ).permute(0, 4, 1, 2, 3)

        if block_state.output_type == "latent":
            block_state.images = latents
            self.set_block_state(state, block_state)
            return components, state

        latents = latents.to(dtype)
        if not components.vae.config.timestep_conditioning:
            vae_timestep = None
        else:
            decode_timestep = block_state.decode_timestep
            decode_noise_scale = block_state.decode_noise_scale
            if not isinstance(decode_timestep, list):
                decode_timestep = [decode_timestep] * block_state.batch_size
            if decode_noise_scale is None:
                decode_noise_scale = decode_timestep
            elif not isinstance(decode_noise_scale, list):
                decode_noise_scale = [decode_noise_scale] * block_state.batch_size

            vae_timestep = torch.tensor(decode_timestep, device=device, dtype=latents.dtype)
            decode_noise_scale = torch.tensor(decode_noise_scale, device=device, dtype=latents.dtype)[
                :, None, None, None, None
            ]
            noise = randn_tensor(latents.shape, generator=block_state.generator, device=device, dtype=latents.dtype)
            latents = (1 - decode_noise_scale) * latents + decode_noise_scale * noise

        latents_mean = components.vae.latents_mean.view(1, -1, 1, 1, 1).to(device)
        latents_std = components.vae.latents_std.view(1, -1, 1, 1, 1).to(device)
        latents = (latents.to(device) * latents_std) / components.vae.config.scaling_factor + latents_mean

        image_tensor = components.vae.decode(latents.to(components.vae.dtype), vae_timestep, return_dict=False)[0]
        image_tensor = image_tensor[:, :, 0, :, :]
        image_processor = VaeImageProcessor(vae_scale_factor=components.vae.spatial_compression_ratio)
        block_state.images = image_processor.postprocess(image_tensor.cpu(), output_type=block_state.output_type)

        self.set_block_state(state, block_state)
        return components, state


class LTX2ImageDistilledBlocks(SequentialPipelineBlocks):
    """Sequential text-to-image blocks kept for step-by-step modular runners."""

    block_classes = [
        LTX2ImageTextEncoderStep,
        LTX2ImageConnectorStep,
        LTX2ImagePrepareLatentsStep,
        LTX2ImageDenoiseStep,
        LTX2ImageDecodeStep,
    ]
    block_names = ["text_encoder", "connectors", "prepare_latents", "denoise", "decode"]

    @property
    def description(self) -> str:
        return "Experimental LTX 2.3 image-only distilled text-to-image modular pipeline blocks."

    @property
    def outputs(self) -> list[OutputParam]:
        return [OutputParam("images", type_hint=Union[list, np.ndarray, torch.Tensor], description="Generated images.")]


LTX2_IMAGE_AUTO_BLOCKS = InsertableDict(
    [
        ("text_encoder", LTX2ImageTextEncoderStep()),
        ("connectors", LTX2ImageConnectorStep()),
        ("vae_encoder", LTX2ImageVaeEncoderStep()),
        ("prepare_latents", LTX2ImagePrepareLatentsStep()),
        ("denoise", LTX2ImageDenoiseStep()),
        ("decode", LTX2ImageDecodeStep()),
    ]
)


class LTX2ImageAutoBlocks(SequentialPipelineBlocks):
    """Unified AutoBlocks for text-to-image and image-to-image LTX 2 image workflows."""

    model_name = "ltx2_image"
    block_classes = list(LTX2_IMAGE_AUTO_BLOCKS.values())
    block_names = list(LTX2_IMAGE_AUTO_BLOCKS.keys())
    _workflow_map = {
        "text2image": {"prompt": True},
        "image2image": {"prompt": True, "image": True},
    }

    @property
    def description(self) -> str:
        return "Unified LTX 2 image modular blocks for text-to-image and image-to-image."

    @property
    def outputs(self) -> list[OutputParam]:
        return [OutputParam("images", type_hint=Union[list, np.ndarray, torch.Tensor], description="Generated images.")]


class LTX2ImageDenoiseBlocks(SequentialPipelineBlocks):
    """Denoise-only block group for runners with precomputed prompt embeddings."""

    block_classes = [
        LTX2ImageConnectorStep,
        LTX2ImagePrepareLatentsStep,
        LTX2ImageDenoiseStep,
    ]
    block_names = ["connectors", "prepare_latents", "denoise"]

    @property
    def description(self) -> str:
        return "Experimental LTX 2.3 image-only distilled denoising blocks for precomputed prompt embeddings."

    @property
    def outputs(self) -> list[OutputParam]:
        return [
            OutputParam("latents", type_hint=torch.Tensor, description="Denoised flattened image latents."),
            OutputParam("batch_size", type_hint=int, description="Prompt batch size."),
            OutputParam("latent_height", type_hint=int, description="Latent height."),
            OutputParam("latent_width", type_hint=int, description="Latent width."),
            OutputParam("in_channels", type_hint=int, description="Latent channel count."),
        ]
