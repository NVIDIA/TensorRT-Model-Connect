# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Component orchestration for the native Cosmos3-Nano T2V bundle."""

from __future__ import annotations

from pathlib import Path

from .model_config import select_generation_profile


def build_cosmos3_components(
    model_dir: str,
    *,
    config,
    weights: dict,
    precision: str = "bf16",
    verbose: bool = False,
    parallel_config=None,
    **_kwargs,
) -> dict:
    """Build the dual-stream denoiser and recurrent video decoder plans."""

    if precision.lower() not in {"bf16", "bfloat16"}:
        raise ValueError("Cosmos3-Nano requires BF16 precision")
    profile = select_generation_profile(config.raw)

    from .transformer_builder import build_cosmos3_transformer_engine
    from .vae_step_builder import (
        Cosmos3VaeStepProfile,
        build_vae_step_engine,
        load_vae_step_weights,
    )

    denoiser = build_cosmos3_transformer_engine(
        weights["_transformer_dir"],
        profile=profile,
        parallel_config=parallel_config,
        verbose=verbose,
    )
    vae_weights = load_vae_step_weights(weights["_vae_dir"])
    vae_profile = Cosmos3VaeStepProfile(profile.latent_height, profile.latent_width)
    vae_decoder = build_vae_step_engine(
        vae_weights,
        profile=vae_profile,
        first_frame_only=False,
        verbose=verbose,
    )
    vae_decoder_first_frame = build_vae_step_engine(
        vae_weights,
        profile=vae_profile,
        first_frame_only=True,
        verbose=verbose,
    )
    tokenizer_dir = Path(weights["_tokenizer_dir"])
    tokenizer_json = (tokenizer_dir / "tokenizer.json").read_bytes()
    tokenizer_config_json = (tokenizer_dir / "tokenizer_config.json").read_bytes()
    negative_prompt = (Path(model_dir) / "assets" / "negative_prompt.json").read_text(
        encoding="utf-8"
    )
    return {
        "denoiser": bytes(denoiser),
        "vae_decoder": bytes(vae_decoder),
        "vae_decoder_first_frame": bytes(vae_decoder_first_frame),
        "tokenizer_json": tokenizer_json,
        "tokenizer_config_json": tokenizer_config_json,
        "negative_prompt": negative_prompt,
    }
