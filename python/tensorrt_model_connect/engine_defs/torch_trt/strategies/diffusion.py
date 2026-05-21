"""Diffusion build strategy — multi-component models (text encoder + denoiser + VAE).

Unlike decoder/encoder strategies that produce a single TRT engine, diffusion
models require separate engines for each component. The family plugin's
build_components() method handles component-specific loading and wrapping,
calling compile_model() for each.

CPU-side export: Large text encoders (e.g., T5-XXL at 10.75 GB fp16) are
exported on CPU to avoid GPU OOM. The TRT builder only needs workspace memory
(~2-4 GB) on GPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class T5EncoderWrapper(nn.Module):
    """Wraps T5EncoderModel for torch-trt compilation.

    Inputs:
      - input_ids: int32 [1, seq_len]
      - attention_mask: int32 [1, seq_len]  (1 = real token, 0 = padding)

    Outputs:
      - hidden_states: float32 [1, seq_len, d_model]
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        ids = input_ids.to(torch.int64)
        mask = attention_mask.to(torch.int64)
        outputs = self.model(input_ids=ids, attention_mask=mask)
        return (outputs.last_hidden_state.to(torch.float32),)


class _TrtSafeAttnProcessor:
    """Attention processor compatible with TRT compilation.

    Uses basic matmul + add + softmax instead of SDPA or baddbmm,
    which TRT can compile reliably including attention mask support.
    """

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, *args, **kwargs):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size = hidden_states.shape[0]

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(
                hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        kv_input = (encoder_hidden_states
                     if encoder_hidden_states is not None
                     else hidden_states)
        key = attn.to_k(kv_input)
        value = attn.to_v(kv_input)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Attention: Q @ K^T / sqrt(d) + mask -> softmax -> @ V.
        # Accumulate the score path in fp32. PixArt cross-attention can
        # overflow in pure fp16 and silently produce NaNs after compilation.
        query_f = query.float()
        key_f = key.float()
        value_f = value.float()
        attn_scores = torch.matmul(query_f, key_f.transpose(-2, -1))
        attn_scores = attn_scores * (head_dim ** -0.5)

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask.float()

        attn_probs = attn_scores.softmax(dim=-1)
        hidden_states = torch.matmul(attn_probs, value_f).to(query.dtype)

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # Dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


class PixArtDiTWrapper(nn.Module):
    """Wraps PixArtTransformer2DModel for torch-trt compilation.

    Runs the full transformer including patch embedding, positional encoding,
    caption projection, timestep embedding, transformer blocks, and output
    projection.

    Inputs:
      - sample: float16 [1, C, H_lat, W_lat] (noisy latent)
      - encoder_hidden_states: float16 [1, seq_len, text_dim] (T5 output)
      - timestep: float16 [1] (diffusion timestep)
      - encoder_attention_mask: float16 [1, seq_len] (1=real, 0=padding)

    Outputs:
      - output: float16 [1, out_channels, H_lat, W_lat] (noise prediction)

    Masking strategy: The encoder_attention_mask ({0,1}) is converted to an
    additive attention bias (0/-10000) and passed as a 3D tensor to the model.
    The model uses a TRT-safe attention processor (basic matmul + softmax)
    instead of SDPA, which TRT can compile with attention mask support.
    """

    def __init__(self, model: nn.Module, in_channels: int, out_channels: int):
        super().__init__()
        # Use TRT-safe attention processor for reliable mask compilation.
        # SDPA fused kernels cause NaN when compiled with attention masks.
        model.set_attn_processor(_TrtSafeAttnProcessor())
        self.model = model
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, sample, encoder_hidden_states, timestep,
                encoder_attention_mask):
        # Convert {0,1} mask to additive attention bias: keep=0, discard=-10000.
        # Pass as 3D [B,1,seq] so the model's ndim==2 check is skipped
        # (it only converts 2D masks; 3D masks are used as-is).
        # This ensures cross-attention softmax gives ~0 weight to padding.
        additive_mask = (1.0 - encoder_attention_mask) * (-10000.0)
        additive_mask = additive_mask.unsqueeze(1)  # [1, 1, seq_len]

        outputs = self.model(
            hidden_states=sample,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            encoder_attention_mask=additive_mask,
        )
        return (outputs.sample,)


class VAEDecoderWrapper(nn.Module):
    """Wraps AutoencoderKL decoder for torch-trt compilation.

    Applies inverse scaling before decoding.

    Inputs:
      - latent: float16 [1, C_lat, H_lat, W_lat]

    Outputs:
      - image: float32 [1, 3, H, W]
    """

    def __init__(self, model: nn.Module, scaling_factor: float):
        super().__init__()
        self.model = model
        self.scaling_factor = scaling_factor

    def forward(self, latent):
        scaled = latent / self.scaling_factor
        decoded = self.model.decode(scaled).sample
        return (decoded.to(torch.float32),)


class DiffusionBuildStrategy:
    """Build strategy for diffusion models (PixArt, Wan, FLUX, etc.).

    Diffusion models have multiple components (text encoder, denoiser, VAE)
    that are compiled separately. The family plugin's build_components()
    handles component-specific loading and wrapping.

    The standard wrap_model/make_export_args methods are not used — instead
    the compiler detects hasattr(plugin, 'build_components') and uses the
    multi-engine path.
    """

    name = "diffusion"
    runtime_strategy = "torchtrt_diffusion"

    def wrap_model(self, model, config, max_cache_length, *, compute_dtype=None):
        raise NotImplementedError(
            "Diffusion strategy uses build_components(), not wrap_model()")

    def make_export_args(self, config, max_cache_length, *, precision="fp16"):
        raise NotImplementedError(
            "Diffusion strategy uses build_components(), not make_export_args()")

    def pre_export_setup(self):
        pass
