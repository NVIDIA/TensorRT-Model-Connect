# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualified architecture and generation contract for Cosmos3-Nano T2V."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class Cosmos3NanoConfig:
    """Exact public Diffusers architecture of ``nvidia/Cosmos3-Nano``."""

    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 151936
    latent_channel: int = 48
    latent_patch_size: int = 2
    patch_latent_dim: int = 192
    timestep_dim: int = 256
    timestep_scale: float = 0.001
    rms_norm_eps: float = 1.0e-6
    rope_theta: float = 5_000_000.0
    rope_axes_dim: tuple[int, int, int] = (24, 20, 20)
    temporal_modality_margin: int = 15_000
    base_fps: int = 24
    qk_norm_for_text: bool = True
    qk_norm_for_diffusion: bool = True
    hidden_act: str = "silu"
    attention_bias: bool = False

    vae_spatial_scale: int = 16
    vae_temporal_scale: int = 4
    # Preserve the official long-form prompts in every supported CP topology.
    max_text_seq_len: int = 4096

    video_height: int = 720
    video_width: int = 1280
    video_num_frames: int = 189
    frame_rate: int = 24
    num_inference_steps: int = 35
    guidance_scale: float = 6.0
    flow_shift: float = 10.0
    seed: int = 42

    @property
    def latent_frames(self) -> int:
        return (self.video_num_frames - 1) // self.vae_temporal_scale + 1

    @property
    def latent_height(self) -> int:
        return self.video_height // self.vae_spatial_scale

    @property
    def latent_width(self) -> int:
        return self.video_width // self.vae_spatial_scale

    @property
    def patch_height(self) -> int:
        return _ceil_div(self.latent_height, self.latent_patch_size)

    @property
    def patch_width(self) -> int:
        return _ceil_div(self.latent_width, self.latent_patch_size)

    @property
    def num_vision_tokens(self) -> int:
        return self.latent_frames * self.patch_height * self.patch_width


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


COSMOS3_NANO = Cosmos3NanoConfig()
COSMOS3_NANO_NEGATIVE_PROMPT = "blurry, distorted, low quality, jittery, deformed"


def select_generation_profile(raw: Mapping[str, object]) -> Cosmos3NanoConfig:
    """Accept only the fixed full-quality T2V recipe for this TensorRT lane."""

    profile = COSMOS3_NANO
    requested = {
        "video_height": int(raw.get("video_height", profile.video_height)),
        "video_width": int(raw.get("video_width", profile.video_width)),
        "video_num_frames": int(raw.get("video_num_frames", profile.video_num_frames)),
        "frame_rate": int(raw.get("frame_rate", profile.frame_rate)),
        "num_inference_steps": int(raw.get("num_inference_steps", profile.num_inference_steps)),
        "guidance_scale": float(raw.get("guidance_scale", profile.guidance_scale)),
        "flow_shift": float(raw.get("flow_shift", profile.flow_shift)),
    }
    expected = {name: getattr(profile, name) for name in requested}
    if requested != expected:
        raise ValueError(
            "Cosmos3-Nano supports the fixed full-quality T2V profile only: "
            "1280x720, 189 frames, 35 denoising steps, guidance 6.0, "
            f"flow shift 10.0 at 24 FPS; requested {requested}"
        )
    return profile


def validate_transformer_config(raw: Mapping[str, object]) -> None:
    """Reject a transformer that is not the exact supported Nano backbone."""

    expected = {
        "hidden_size": COSMOS3_NANO.hidden_size,
        "intermediate_size": COSMOS3_NANO.intermediate_size,
        "num_hidden_layers": COSMOS3_NANO.num_hidden_layers,
        "num_attention_heads": COSMOS3_NANO.num_attention_heads,
        "num_key_value_heads": COSMOS3_NANO.num_key_value_heads,
        "head_dim": COSMOS3_NANO.head_dim,
        "vocab_size": COSMOS3_NANO.vocab_size,
        "latent_channel": COSMOS3_NANO.latent_channel,
        "latent_patch_size": COSMOS3_NANO.latent_patch_size,
        "patch_latent_dim": COSMOS3_NANO.patch_latent_dim,
        "rms_norm_eps": COSMOS3_NANO.rms_norm_eps,
        "rope_theta": COSMOS3_NANO.rope_theta,
        "timestep_scale": COSMOS3_NANO.timestep_scale,
        "hidden_act": COSMOS3_NANO.hidden_act,
        "attention_bias": COSMOS3_NANO.attention_bias,
        "qk_norm_for_text": COSMOS3_NANO.qk_norm_for_text,
        "qk_norm_for_diffusion": COSMOS3_NANO.qk_norm_for_diffusion,
    }
    mismatches = {
        key: (raw.get(key), value) for key, value in expected.items() if raw.get(key) != value
    }
    rope_scaling = raw.get("rope_scaling")
    rope_axes = (
        tuple(rope_scaling.get("mrope_section", ())) if isinstance(rope_scaling, Mapping) else ()
    )
    if rope_axes != COSMOS3_NANO.rope_axes_dim:
        mismatches["rope_axes_dim"] = (rope_axes, COSMOS3_NANO.rope_axes_dim)
    if mismatches:
        details = ", ".join(
            f"{key}={actual!r} (expected {wanted!r})"
            for key, (actual, wanted) in sorted(mismatches.items())
        )
        raise ValueError(f"Checkpoint is not supported Cosmos3-Nano: {details}")


def validate_vae_config(raw: Mapping[str, object]) -> None:
    """Validate the public Cosmos3 causal 3D autoencoder geometry."""

    expected = {
        "z_dim": COSMOS3_NANO.latent_channel,
        "scale_factor_spatial": COSMOS3_NANO.vae_spatial_scale,
        "scale_factor_temporal": COSMOS3_NANO.vae_temporal_scale,
        "patch_size": COSMOS3_NANO.latent_patch_size,
    }
    mismatches = {
        key: (raw.get(key), value) for key, value in expected.items() if raw.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}={actual!r} (expected {wanted!r})"
            for key, (actual, wanted) in sorted(mismatches.items())
        )
        raise ValueError(f"Checkpoint has an unsupported Cosmos3 VAE: {details}")
