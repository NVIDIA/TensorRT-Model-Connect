# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure TensorRT component builder for Wan2.2 TI2V-5B.

Python and PyTorch are build-time checkpoint readers only. Every component is
built from the requested checkpoint. The resulting bundle contains four
native TensorRT plans, the tokenizer and config; it contains no executable
companion or custom plugin artifact.
"""

from __future__ import annotations

from pathlib import Path

from .model_config import WAN22_TI2V_5B, select_generation_profile


# Heavy TensorRT/NumPy/PyTorch conversion modules are loaded only when the
# corresponding component is built.
def build_native_umt5_encoder_engine(*args, **kwargs):
    from .umt5_encoder_builder import build_native_umt5_encoder_engine as implementation

    return implementation(*args, **kwargs)


def build_dit_engine(*args, **kwargs):
    from .dit_builder import build_dit_engine as implementation

    return implementation(*args, **kwargs)


def build_vae_step_engine(*args, **kwargs):
    from .vae_step_builder import build_vae_step_engine as implementation

    return implementation(*args, **kwargs)


def load_vae_step_weights(*args, **kwargs):
    from .vae_step_builder import load_vae_step_weights as implementation

    return implementation(*args, **kwargs)


def _qualified_fp8_scale_names(profile) -> tuple[set[str], set[str]]:
    ffn = {
        name
        for index in range(profile.num_layers)
        for name in (
            f"blocks.{index}.ffn.net.0.proj",
            f"blocks.{index}.ffn.net.2",
        )
    }
    cross_qo = {
        name
        for index in range(profile.num_layers)
        for name in (
            f"blocks.{index}.attn2.to_q",
            f"blocks.{index}.attn2.to_out.0",
        )
    }
    return ffn, cross_qo


def _split_qualified_fp8_scales(
    fp8_scales: dict | None,
    profile,
) -> tuple[dict | None, dict | None]:
    if fp8_scales is None:
        return None, None
    if not isinstance(fp8_scales, dict):
        raise TypeError("Wan2.2 FP8 scales must be a dictionary")

    expected_ffn, expected_cross_qo = _qualified_fp8_scale_names(profile)
    provided = set(fp8_scales)
    if provided == expected_ffn:
        return fp8_scales, None
    if provided == expected_ffn | expected_cross_qo:
        return (
            {name: fp8_scales[name] for name in sorted(expected_ffn)},
            {name: fp8_scales[name] for name in sorted(expected_cross_qo)},
        )

    raise ValueError(
        "Wan2.2 FP8 scales must contain either the complete FFN map or the "
        "complete FFN+cross-Q/O map; "
        f"missing_ffn={sorted(expected_ffn - provided)}, "
        f"missing_cross_qo={sorted(expected_cross_qo - provided)}, "
        f"unexpected={sorted(provided - expected_ffn - expected_cross_qo)}"
    )


def build_wan22_components(
    model_dir: str,
    *,
    config,
    weights: dict,
    precision: str = "bf16",
    verbose: bool = False,
    fp8_scales: dict | None = None,
    **_kwargs,
) -> dict:
    """Build all four plans for one qualified checkpoint profile."""

    if precision.lower() not in {"bf16", "bfloat16"}:
        raise ValueError("Wan2.2-TI2V-5B requires BF16 DiT/T5 precision")
    generation_profile = select_generation_profile(config.raw)
    if fp8_scales is not None and generation_profile != WAN22_TI2V_5B:
        raise ValueError(
            "Wan2.2 FP8 scales are qualified only for the official "
            "1280x704, 121-frame, 50-step profile"
        )
    ffn_fp8_scales, cross_qo_fp8_scales = _split_qualified_fp8_scales(
        fp8_scales,
        generation_profile,
    )

    text_encoder = build_native_umt5_encoder_engine(
        weights["_text_encoder_checkpoint"],
        verbose=verbose,
    )
    denoiser = build_dit_engine(
        model_dir,
        profile=generation_profile,
        ffn_fp8_scales=ffn_fp8_scales,
        cross_qo_fp8_scales=cross_qo_fp8_scales,
        verbose=verbose,
    )

    vae_weights = load_vae_step_weights(weights["_vae_checkpoint"])
    from .vae_step_builder import Wan22VaeStepProfile

    vae_profile = Wan22VaeStepProfile(
        generation_profile.latent_height,
        generation_profile.latent_width,
    )
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

    tokenizer_path = Path(weights["_tokenizer_dir"]) / "tokenizer.json"
    tokenizer_json = tokenizer_path.read_bytes()
    return {
        "text_encoders": [("umt5_xxl", bytes(text_encoder))],
        "denoiser": bytes(denoiser),
        "vae_decoder": bytes(vae_decoder),
        "vae_decoder_first_frame": bytes(vae_decoder_first_frame),
        "tokenizer_json": tokenizer_json,
    }
