# Copyright 2025 The Lightricks team and The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import torch.nn as nn
from contextlib import nullcontext
from typing import List, Optional, Tuple, Union

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.attention import AttentionMixin, FeedForward
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.normalization import RMSNorm
from diffusers.models.transformers.transformer_ltx2 import (
    LTX2AdaLayerNormSingle,
    LTX2Attention,
    LTX2AudioVideoAttnProcessor,
    LTX2AudioVideoRotaryPosEmbed,
    LTX2PerturbedAttnProcessor,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import apply_lora_scale, logging

logger = logging.get_logger(__name__)

_LOG_ATTENTION_MASK = os.environ.get("LTX_IMAGE_LOG_ATTENTION_MASK", "0") == "1"
_LOG_ATTENTION_MASK_LIMIT = int(os.environ.get("LTX_IMAGE_LOG_ATTENTION_MASK_LIMIT", "8"))
_DROP_TRIVIAL_ATTENTION_MASK_DEFAULT = os.environ.get("LTX_IMAGE_DROP_TRIVIAL_ATTENTION_MASK", "0") == "1"
_attention_mask_log_count = 0


def _mask_stats(mask: Optional[torch.Tensor]) -> dict:
    if mask is None:
        return {"is_none": True, "trivial": True}

    with torch.no_grad():
        sample = mask.detach()
        numel = sample.numel()
        stats = {
            "is_none": False,
            "shape": list(sample.shape),
            "dtype": str(sample.dtype),
            "device": str(sample.device),
            "numel": int(numel),
        }
        if numel == 0:
            stats["trivial"] = True
            return stats

        if sample.dtype == torch.bool:
            true_count = int(sample.sum().item())
            stats.update({"true": true_count, "false": int(numel - true_count), "trivial": true_count == numel})
            return stats

        stat_sample = sample.float()
        min_value = float(stat_sample.min().item())
        max_value = float(stat_sample.max().item())
        nonzero = int((stat_sample != 0).sum().item())
        all_one = bool(torch.all(stat_sample == 1).item())
        all_zero = bool(torch.all(stat_sample == 0).item())
        finite = bool(torch.isfinite(stat_sample).all().item())
        stats.update(
            {
                "min": min_value,
                "max": max_value,
                "nonzero": nonzero,
                "all_one": all_one,
                "all_zero": all_zero,
                "finite": finite,
                "trivial": all_one or all_zero,
            }
        )
        return stats


def _log_attention_mask(raw_mask: Optional[torch.Tensor], prepared_mask: Optional[torch.Tensor], *, block_idx: Optional[int] = None):
    global _attention_mask_log_count
    if not _LOG_ATTENTION_MASK or _attention_mask_log_count >= _LOG_ATTENTION_MASK_LIMIT:
        return
    print(
        f"  [attention_mask] call={_attention_mask_log_count + 1} block={block_idx} "
        f"raw={_mask_stats(raw_mask)} prepared={_mask_stats(prepared_mask)}",
        flush=True,
    )
    _attention_mask_log_count += 1


class LTX2ImageTransformerBlock(nn.Module):
    r"""
    Transformer block used in LTX-2 Image (Visual-only) model.
    """
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        video_gated_attn: bool = False,
        video_cross_attn_adaln: bool = True,
        qk_norm: str = "rms_norm_across_heads",
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
        rope_type: str = "interleaved",
        ff_bias: bool = True,
    ):
        super().__init__()

        self.video_cross_attn_adaln = video_cross_attn_adaln

        # 1. Self-Attention
        self.norm1 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn1 = LTX2Attention(
            query_dim=dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            cross_attention_dim=None,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            rope_type=rope_type,
            apply_gated_attention=video_gated_attn,
            processor=LTX2PerturbedAttnProcessor(),
        )

        # 2. Prompt Cross-Attention
        self.norm2 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn2 = LTX2Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            rope_type=rope_type,
            apply_gated_attention=video_gated_attn,
            processor=LTX2AudioVideoAttnProcessor(),
        )

        # 3. Feedforward
        self.norm3 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.ff = FeedForward(dim, activation_fn=activation_fn, bias=ff_bias)

        # 4. Modulations
        video_mod_param_num = 9 if self.video_cross_attn_adaln else 6
        self.scale_shift_table = nn.Parameter(torch.randn(video_mod_param_num, dim) / dim**0.5)
        self.prompt_scale_shift_table = nn.Parameter(torch.randn(2, dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        temb_prompt: Optional[torch.Tensor],
        video_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        encoder_attention_mask: Optional[torch.Tensor] = None,
        perturbation_mask: Optional[torch.Tensor] = None,
        all_perturbed: bool = False,
    ) -> torch.Tensor:
        batch_size = hidden_states.size(0)
        block_idx = getattr(self, "_ltx2_block_idx", -1)
        memory_manager = getattr(self, "_ltx2_memory_manager", None)
        profile_layer = memory_manager.profile_layer if memory_manager is not None else None

        with profile_layer(block_idx, "adaln") if profile_layer is not None else nullcontext():
            num_ada_params = self.scale_shift_table.shape[0]
            ada_values = self.scale_shift_table[None, None].to(temb.device) + temb.view(
                batch_size, temb.size(1), num_ada_params, -1
            )
            video_ada_params = ada_values.unbind(dim=2)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = video_ada_params[:6]

            if self.video_cross_attn_adaln:
                shift_text_q, scale_text_q, gate_text_q = video_ada_params[6:9]

            if temb_prompt is not None:
                num_prompt_params = self.prompt_scale_shift_table.shape[0]
                prompt_ada_values = self.prompt_scale_shift_table[None, None].to(temb_prompt.device) + temb_prompt.view(
                    batch_size, temb_prompt.size(1), num_prompt_params, -1
                )
                shift_text_kv, scale_text_kv = prompt_ada_values.unbind(dim=2)
            else:
                shift_text_kv, scale_text_kv = (
                    self.prompt_scale_shift_table[None, None]
                    .to(device=hidden_states.device, dtype=hidden_states.dtype)
                    .unbind(dim=2)
                )

        # 1. Self-Attention
        with profile_layer(block_idx, "self_attn") if profile_layer is not None else nullcontext():
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            attn_hidden_states = self.attn1(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=None,
                query_rotary_emb=video_rotary_emb,
                perturbation_mask=perturbation_mask,
                all_perturbed=all_perturbed,
            )
            hidden_states = hidden_states + attn_hidden_states * gate_msa

        # 2. Cross-Attention
        with profile_layer(block_idx, "cross_attn") if profile_layer is not None else nullcontext():
            norm_hidden_states = self.norm2(hidden_states)
            if self.video_cross_attn_adaln:
                norm_hidden_states = norm_hidden_states * (1 + scale_text_q) + shift_text_q

            encoder_hidden_states = encoder_hidden_states * (1 + scale_text_kv) + shift_text_kv

            if _LOG_ATTENTION_MASK:
                prepared_attention_mask = None
                if encoder_attention_mask is not None:
                    sequence_length = encoder_hidden_states.shape[1]
                    prepared_attention_mask = self.attn2.prepare_attention_mask(
                        encoder_attention_mask, sequence_length, batch_size
                    )
                    prepared_attention_mask = prepared_attention_mask.view(
                        batch_size, self.attn2.heads, -1, prepared_attention_mask.shape[-1]
                    )
                _log_attention_mask(encoder_attention_mask, prepared_attention_mask, block_idx=block_idx)

            attention_mask_for_attn = encoder_attention_mask
            if getattr(self, "drop_trivial_attention_mask", _DROP_TRIVIAL_ATTENTION_MASK_DEFAULT) and attention_mask_for_attn is not None:
                if bool(torch.all(attention_mask_for_attn == 0).item()):
                    attention_mask_for_attn = None

            attn_hidden_states = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask_for_attn,
            )
            if self.video_cross_attn_adaln:
                attn_hidden_states = attn_hidden_states * gate_text_q
            hidden_states = hidden_states + attn_hidden_states

        # 3. Feedforward
        with profile_layer(block_idx, "ff") if profile_layer is not None else nullcontext():
            norm_hidden_states = self.norm3(hidden_states) * (1 + scale_mlp) + shift_mlp
            hidden_states = hidden_states + self.ff(norm_hidden_states) * gate_mlp

        return hidden_states


class LTX2ImageTransformer2DModel(ModelMixin, ConfigMixin, AttentionMixin, FromOriginalModelMixin, PeftAdapterMixin, CacheMixin):
    r"""
    A 2D Transformer model for image generation derived from the LTX-2.X Video architecture.
    """
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 128,
        out_channels: Optional[int] = 128,
        patch_size: int = 1,
        patch_size_t: int = 1,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        cross_attention_dim: int = 4096,
        num_layers: int = 48,
        activation_fn: str = "gelu-approximate",
        qk_norm: str = "rms_norm_across_heads",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        rope_theta: float = 10000.0,
        rope_double_precision: bool = True,
        causal_offset: int = 1,
        gated_attn: bool = False,
        cross_attn_mod: bool = True,
        ff_bias: bool = True,
        rope_type: str = "interleaved",
        vae_scale_factors: Tuple[int, int, int] = (8, 32, 32),
        pos_embed_max_pos: int = 20,
        base_height: int = 2048,
        base_width: int = 2048,
        use_prompt_adaln_single: bool = True,
        **kwargs,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        inner_dim = num_attention_heads * attention_head_dim

        self.proj_in = nn.Linear(in_channels, inner_dim)

        video_time_emb_mod_params = 9 if cross_attn_mod else 6
        self.time_embed = LTX2AdaLayerNormSingle(
            inner_dim, num_mod_params=video_time_emb_mod_params, use_additional_conditions=False
        )
        self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)

        self.prompt_modulation = cross_attn_mod
        if self.prompt_modulation and use_prompt_adaln_single:
            self.prompt_adaln = LTX2AdaLayerNormSingle(
                inner_dim, num_mod_params=2, use_additional_conditions=False
            )
        else:
            self.prompt_adaln = None

        self.rope = LTX2AudioVideoRotaryPosEmbed(
            dim=inner_dim,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            base_num_frames=pos_embed_max_pos,
            base_height=base_height,
            base_width=base_width,
            scale_factors=vae_scale_factors,
            theta=rope_theta,
            causal_offset=causal_offset,
            modality="video",
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=num_attention_heads,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                LTX2ImageTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    video_gated_attn=gated_attn,
                    video_cross_attn_adaln=cross_attn_mod,
                    qk_norm=qk_norm,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    attention_out_bias=attention_out_bias,
                    eps=norm_eps,
                    elementwise_affine=norm_elementwise_affine,
                    rope_type=rope_type,
                    ff_bias=ff_bias,
                )
                for _ in range(num_layers)
            ]
        )

        for block_idx, block in enumerate(self.transformer_blocks):
            block._ltx2_block_idx = block_idx

        self.norm_out = nn.LayerNorm(inner_dim, eps=1e-6, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels)
        self.gradient_checkpointing = False
        self.memory_manager = None
        self.drop_trivial_attention_mask = _DROP_TRIVIAL_ATTENTION_MASK_DEFAULT

    def set_memory_manager(self, memory_manager):
        self.memory_manager = memory_manager

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        video_coords: Optional[torch.Tensor] = None,
        video_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        pag_applied_layers: Optional[List[int]] = None,
        perturbation_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[Transformer2DModelOutput, Tuple[torch.Tensor]]:
        batch_size = hidden_states.size(0)

        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
        if getattr(self, "drop_trivial_attention_mask", _DROP_TRIVIAL_ATTENTION_MASK_DEFAULT) and encoder_attention_mask is not None:
            if bool(torch.all(encoder_attention_mask == 0).item()):
                if _LOG_ATTENTION_MASK:
                    print("  [attention_mask] dropping trivial all-zero attention mask", flush=True)
                encoder_attention_mask = None

        if video_rotary_emb is None:
            if video_coords is None:
                video_coords = self.rope.prepare_video_coords(
                    batch_size, 1, height, width, hidden_states.device, fps=24.0
                )
            video_rotary_emb = self.rope(video_coords, device=hidden_states.device)

        hidden_states = self.proj_in(hidden_states)

        temb, embedded_timestep = self.time_embed(
            timestep.flatten(), batch_size=batch_size, hidden_dtype=hidden_states.dtype
        )
        temb = temb.view(batch_size, -1, temb.size(-1))
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))

        if self.prompt_adaln is not None:
            temb_prompt, _ = self.prompt_adaln(
                timestep.flatten(), batch_size=batch_size, hidden_dtype=hidden_states.dtype
            )
            temb_prompt = temb_prompt.view(batch_size, -1, temb_prompt.size(-1))
        else:
            temb_prompt = None

        pag_applied_layers = pag_applied_layers or []
        if len(pag_applied_layers) > 0 and perturbation_mask is None:
            perturbation_mask = torch.zeros((batch_size,), device=hidden_states.device)
        if perturbation_mask is not None and perturbation_mask.ndim == 1:
            perturbation_mask = perturbation_mask[:, None, None]
        all_perturbed = torch.all(perturbation_mask == 0) if perturbation_mask is not None else False
        pag_blocks = set(pag_applied_layers)

        for block_idx, block in enumerate(self.transformer_blocks):
            block.drop_trivial_attention_mask = getattr(
                self, "drop_trivial_attention_mask", _DROP_TRIVIAL_ATTENTION_MASK_DEFAULT
            )
            block_perturbation_mask = perturbation_mask if block_idx in pag_blocks else None
            block_all_perturbed = all_perturbed if block_idx in pag_blocks else False

            block_context = (
                self.memory_manager.use_block(block_idx, block)
                if self.memory_manager is not None
                else nullcontext(block)
            )
            with block_context as active_block:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    hidden_states = self._gradient_checkpointing_func(
                        active_block,
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        temb_prompt,
                        video_rotary_emb,
                        encoder_attention_mask,
                        block_perturbation_mask,
                        block_all_perturbed,
                    )
                else:
                    hidden_states = active_block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        temb_prompt=temb_prompt,
                        video_rotary_emb=video_rotary_emb,
                        encoder_attention_mask=encoder_attention_mask,
                        perturbation_mask=block_perturbation_mask,
                        all_perturbed=block_all_perturbed,
                    )

        scale_shift_values = self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
