# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build one flux3 text-to-image bundle (FLUX 3 Image Klein, step-distilled)."""

from __future__ import annotations

import sys
from pathlib import Path

import tensorrt as trt

from . import bundle_config as bc
from .dit_builder import build_denoiser
from .text_encoder_builder import PREFIX as TEXT_PREFIX
from .text_encoder_builder import build_text_encoder
from .vae_builder import build_vae_decoder
from .weights import ComponentWeights

_OPTIMIZATION_LEVEL = 3
_OPT_TEXT_TOKENS = 320


def build(request, writer) -> None:
    """Build the text encoder, denoiser and VAE engines plus the runtime sections."""

    if request.task != "text_to_image":
        raise ValueError("flux3 supports only task=text_to_image")
    if request.backend != "trt":
        raise ValueError("flux3 supports only the trt backend")
    if request.precision != "bf16":
        raise ValueError("flux3 builds the checkpoint's BF16 graph only (precision=bf16)")
    if request.quantization not in (None, "none"):
        raise NotImplementedError(
            "flux3 does not support quantized builds (variants/fp8r and variants/nvfp4 are not read)"
        )
    if request.max_batch_size != 1:
        raise NotImplementedError("flux3 builds one image per request (max_batch_size=1)")
    if request.tensor_parallel_size != 1 or request.context_parallel_size != 1:
        raise NotImplementedError("flux3 runs on one GPU (no tensor or context parallelism)")
    if request.video_num_frames is not None:
        raise NotImplementedError("flux3 does not support video_num_frames")
    if request.fp32_layers:
        raise NotImplementedError("flux3 does not support fp32_layers")
    if request.dynamic_kv_cache:
        raise NotImplementedError("flux3 does not support dynamic_kv_cache")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("flux3 does not support mtp_seq_len")
    if request.graph_transform is not None:
        raise NotImplementedError("flux3 does not support graph transforms")

    checkpoint = bc.load_checkpoint(request.model_dir)
    height, width = bc.image_geometry(request.image_height, request.image_width)
    budget = bc.text_token_budget(request.max_sequence_length)
    latent_height, latent_width = height // bc.SIDE_MULTIPLE, width // bc.SIDE_MULTIPLE
    prefix, suffix = bc.render_prompt_frame(request.model_dir)
    pad_id = bc.pad_token_id(request.model_dir)
    model_dir = Path(request.model_dir)
    opt_tokens = min(budget, _OPT_TEXT_TOKENS)
    verbose = bool(request.verbose)

    def log(message: str) -> None:
        print(f"[flux3] {message}", file=sys.stderr, flush=True)

    log(f"text encoder: Qwen3-VL layers 0..{max(checkpoint.text_layers) - 1}, window 80..{budget}")
    text_plan = build_text_encoder(
        ComponentWeights(model_dir / "text_encoder", prefix=TEXT_PREFIX),
        checkpoint.text,
        output_layers=checkpoint.text_layers,
        max_tokens=budget,
        opt_tokens=opt_tokens,
        min_tokens=bc.PAD_MULTIPLE,
        compute=trt.bfloat16,
        optimization_level=_OPTIMIZATION_LEVEL,
        verbose=verbose,
    )
    log(f"denoiser: {latent_height}x{latent_width} latent tokens")
    denoiser_plan = build_denoiser(
        ComponentWeights(model_dir / "transformer"),
        checkpoint.transformer,
        latent_height=latent_height,
        latent_width=latent_width,
        rope_scale=bc.IMAGE_ROPE_SCALE,
        max_tokens=budget,
        opt_tokens=opt_tokens,
        min_tokens=bc.PAD_MULTIPLE,
        compute=trt.bfloat16,
        optimization_level=_OPTIMIZATION_LEVEL,
        verbose=verbose,
    )
    log(f"VAE decoder: {height}x{width}")
    vae_plan = build_vae_decoder(
        ComponentWeights(model_dir / "vae"),
        checkpoint.vae,
        latent_height=latent_height,
        latent_width=latent_width,
        latent_dtype=trt.bfloat16,
        optimization_level=_OPTIMIZATION_LEVEL,
        verbose=verbose,
    )

    writer.set_header(family="flux3", task=request.task, backend=request.backend)
    writer.add_bytes("text_encoder.plan", text_plan)
    writer.add_bytes("denoiser.plan", denoiser_plan)
    writer.add_bytes("vae.plan", vae_plan)
    writer.add_bytes("tokenizer.json", (model_dir / "text_encoder" / "tokenizer.json").read_bytes())
    writer.add_json(
        "runtime.json",
        bc.runtime_config(
            height=height,
            width=width,
            text_budget=budget,
            prompt_prefix=prefix,
            prompt_suffix=suffix,
            pad_id=pad_id,
            embed_dim=int(checkpoint.transformer["joint_attention_dim"]),
            latent_channels=int(checkpoint.transformer["in_channels"]),
        ),
    )
