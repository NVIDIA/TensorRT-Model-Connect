# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image 2.1 family build.

Checkpoint layout (diffusers ``QwenImage21Pipeline``):
  text_encoder: Qwen3-VL (language model only for text-to-image)
  transformer:  QwenImage21Transformer2DModel, 32 single-stream blocks, block-causal attention
  vae:          AutoencoderKLQwenImage21, 16x spatial, 64 latent channels, RGBA output
  scheduler:    FlowMatchEulerDiscreteScheduler with exponential dynamic shifting

A bundle is built for one output size and one text budget; every plan is static.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .bundle_config import (
    DEFAULT_NUM_STEPS,
    prompt_layout,
    resolve_geometry,
    scheduler_settings,
)

if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one Qwen-Image 2.1 text-to-image bundle."""
    if request.task != "text_to_image":
        raise ValueError("qwen_image2_1 supports only task=text_to_image")
    if request.backend != "trt":
        raise ValueError("qwen_image2_1 supports only backend=trt")
    if request.precision != "bf16":
        raise ValueError("qwen_image2_1 requires precision=bf16")
    if request.dynamic_kv_cache:
        raise NotImplementedError("qwen_image2_1 does not support dynamic_kv_cache")
    if request.video_num_frames is not None:
        raise NotImplementedError("qwen_image2_1 does not support video_num_frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("qwen_image2_1 does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("qwen_image2_1 does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("qwen_image2_1 does not support context parallelism")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("qwen_image2_1 does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("qwen_image2_1 does not support fp32_layers")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("qwen_image2_1 does not support MTP speculative decoding")

    model_dir = Path(request.model_dir)
    verbose = bool(request.verbose)

    # Heavy TensorRT / NumPy modules load only when a bundle is actually built.
    from .dit_builder import build_denoiser_plan, load_denoiser
    from .text_encoder_builder import build_text_encoder_plan, load_text_encoder
    from .vae_builder import build_vae_decoder_plan, load_vae

    scheduler = scheduler_settings(model_dir)
    prompt = prompt_layout(model_dir / "processor")
    tokenizer_path = model_dir / "processor" / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Qwen-Image 2.1 checkpoint is missing {tokenizer_path}")

    vae_config, vae_weights = load_vae(model_dir / "vae")
    geometry = resolve_geometry(
        request.image_height,
        request.image_width,
        request.max_sequence_length,
        vae_scale=vae_config.spatial_scale,
        max_image_tokens=scheduler["max_image_seq_len"],
        frame_tokens=prompt["frame_tokens"],
    )

    writer.set_header(family="qwen_image2_1", task=request.task, backend=request.backend)

    text_config, text_weights = load_text_encoder(model_dir / "text_encoder")
    if not 0 <= prompt["pad_token_id"] < text_config.vocab_size:
        raise ValueError("the tokenizer's padding token is outside the text encoder's vocabulary")
    with writer.open_section("text_encoder.plan") as section:
        build_text_encoder_plan(
            text_config,
            text_weights,
            section,
            prefix_tokens=prompt["prefix_tokens"],
            text_tokens=geometry.text_tokens,
            verbose=verbose,
        )
    del text_weights

    denoiser_config, denoiser_weights = load_denoiser(model_dir / "transformer")
    if denoiser_config.context_dim != text_config.hidden:
        raise ValueError("transformer context width does not match the text encoder")
    if denoiser_config.in_channels != vae_config.z_dim:
        raise ValueError("transformer latent channels do not match the VAE")
    if denoiser_config.out_channels != denoiser_config.in_channels:
        raise ValueError("transformer output channels do not match its latent channels")
    with writer.open_section("denoiser.plan") as section:
        build_denoiser_plan(
            denoiser_config,
            denoiser_weights,
            section,
            latent_height=geometry.latent_height,
            latent_width=geometry.latent_width,
            text_tokens=geometry.text_tokens,
            verbose=verbose,
        )
    del denoiser_weights

    with writer.open_section("vae.plan") as section:
        build_vae_decoder_plan(
            vae_config,
            vae_weights,
            section,
            latent_height=geometry.latent_height,
            latent_width=geometry.latent_width,
            verbose=verbose,
        )

    writer.add_bytes("tokenizer.json", tokenizer_path.read_bytes())
    writer.add_json(
        "runtime.json",
        {
            "tensor_parallel_size": 1,
            "image_height": geometry.image_height,
            "image_width": geometry.image_width,
            "image_channels": vae_config.out_channels,
            "latent_channels": vae_config.z_dim,
            "vae_scale_factor": vae_config.spatial_scale,
            "text_tokens": geometry.text_tokens,
            "text_hidden_size": text_config.hidden,
            "prefix_tokens": prompt["prefix_tokens"],
            "pad_token_id": prompt["pad_token_id"],
            "prompt_prefix": prompt["prompt_prefix"],
            "prompt_suffix": prompt["prompt_suffix"],
            **scheduler,
            "default_num_steps": DEFAULT_NUM_STEPS,
        },
    )
