# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authoritative architecture constants and profiles for Wan2.2-TI2V-5B.

The values below are defined by the upstream Wan2.2 ``ti2v_5B`` task and its
native ``config.json``.  Keeping them in this family makes an accidental match
to another Wan generation fail loudly at build time.  CI adds one reduced
generation profile while retaining the exact same checkpoint architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class Wan22TI2VConfig:
    model_type: str = "ti2v"
    in_channels: int = 48
    out_channels: int = 48
    dim: int = 3072
    ffn_dim: int = 14336
    freq_dim: int = 256
    num_heads: int = 24
    num_layers: int = 30
    head_dim: int = 128
    text_dim: int = 4096
    text_seq_len: int = 512
    eps: float = 1.0e-6
    patch_size: tuple[int, int, int] = (1, 2, 2)

    z_dim: int = 48
    scale_factor_temporal: int = 4
    scale_factor_spatial: int = 16

    video_height: int = 704
    video_width: int = 1280
    video_num_frames: int = 121
    frame_rate: int = 24
    num_inference_steps: int = 50
    guidance_scale: float = 5.0
    flow_shift: float = 5.0
    train_timesteps: int = 1000

    @property
    def latent_frames(self) -> int:
        return (self.video_num_frames - 1) // self.scale_factor_temporal + 1

    @property
    def latent_height(self) -> int:
        return self.video_height // self.scale_factor_spatial

    @property
    def latent_width(self) -> int:
        return self.video_width // self.scale_factor_spatial

    @property
    def num_patches(self) -> int:
        pt, ph, pw = self.patch_size
        return self.latent_frames // pt * self.latent_height // ph * self.latent_width // pw


WAN22_TI2V_5B = Wan22TI2VConfig()
WAN22_TI2V_5B_L0 = Wan22TI2VConfig(
    video_height=384,
    video_width=672,
    video_num_frames=5,
    num_inference_steps=15,
)

SUPPORTED_GENERATION_PROFILES = (
    WAN22_TI2V_5B,
    WAN22_TI2V_5B_L0,
)


def artifact_profile(arch: Wan22TI2VConfig) -> dict[str, object]:
    """Return the source-bound artifact contract for one qualified profile."""

    return {
        "video_width": arch.video_width,
        "video_height": arch.video_height,
        "video_num_frames": arch.video_num_frames,
        "latent_shape": [
            1,
            arch.z_dim,
            arch.latent_frames,
            arch.latent_height,
            arch.latent_width,
        ],
        "architecture": {
            "model_type": arch.model_type,
            "in_channels": arch.in_channels,
            "out_channels": arch.out_channels,
            "dim": arch.dim,
            "ffn_dim": arch.ffn_dim,
            "freq_dim": arch.freq_dim,
            "num_heads": arch.num_heads,
            "num_layers": arch.num_layers,
            "head_dim": arch.head_dim,
            "text_dim": arch.text_dim,
            "text_seq_len": arch.text_seq_len,
            "eps": arch.eps,
            "patch_size": list(arch.patch_size),
            "z_dim": arch.z_dim,
            "scale_factor_temporal": arch.scale_factor_temporal,
            "scale_factor_spatial": arch.scale_factor_spatial,
            "frame_rate": arch.frame_rate,
            "num_inference_steps": arch.num_inference_steps,
            "guidance_scale": arch.guidance_scale,
            "flow_shift": arch.flow_shift,
            "train_timesteps": arch.train_timesteps,
        },
        "text_seq_len": arch.text_seq_len,
        "text_encoder_dim": arch.text_dim,
        "text_encoder_numerics": {
            "shape": [1, 512, 4096],
            "num_heads": 64,
            "epsilon": 1e-6,
            "source_softmax": True,
            "source_rmsnorm": True,
        },
        "precision": "bf16",
    }


def official_artifact_profile() -> dict[str, object]:
    """Return the complete official 720p architecture and generation contract."""

    return artifact_profile(WAN22_TI2V_5B)


def select_generation_profile(raw: Mapping[str, object]) -> Wan22TI2VConfig:
    """Select one exact qualified generation profile and reject hybrid shapes."""

    official = WAN22_TI2V_5B
    requested = {
        "video_width": int(raw.get("video_width", official.video_width)),
        "video_height": int(raw.get("video_height", official.video_height)),
        "video_num_frames": int(raw.get("video_num_frames", official.video_num_frames)),
        "num_inference_steps": int(raw.get("num_inference_steps", official.num_inference_steps)),
        "guidance_scale": float(raw.get("guidance_scale", official.guidance_scale)),
        "flow_shift": float(raw.get("flow_shift", official.flow_shift)),
        "frame_rate": int(raw.get("frame_rate", official.frame_rate)),
    }
    for profile in SUPPORTED_GENERATION_PROFILES:
        expected = {name: getattr(profile, name) for name in requested}
        if requested == expected:
            return profile

    supported = ", ".join(
        f"{profile.video_width}x{profile.video_height}/{profile.video_num_frames} frames/"
        f"{profile.num_inference_steps} steps"
        for profile in SUPPORTED_GENERATION_PROFILES
    )
    raise ValueError(
        "Wan2.2-TI2V-5B requires one exact qualified generation profile; "
        f"requested {requested}. Supported profiles: {supported}"
    )


def validate_artifact_profile(profile: object) -> Wan22TI2VConfig:
    """Return the qualified config represented by an exact artifact profile."""

    for candidate in SUPPORTED_GENERATION_PROFILES:
        if profile == artifact_profile(candidate):
            return candidate
    raise ValueError("Wan2.2 artifact profile is not one of the qualified profiles")


OFFICIAL_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def validate_native_config(raw: dict) -> None:
    """Reject checkpoints that are not the exact Wan2.2 TI2V-5B shape."""

    expected = {
        "model_type": WAN22_TI2V_5B.model_type,
        "in_dim": WAN22_TI2V_5B.in_channels,
        "out_dim": WAN22_TI2V_5B.out_channels,
        "dim": WAN22_TI2V_5B.dim,
        "ffn_dim": WAN22_TI2V_5B.ffn_dim,
        "freq_dim": WAN22_TI2V_5B.freq_dim,
        "num_heads": WAN22_TI2V_5B.num_heads,
        "num_layers": WAN22_TI2V_5B.num_layers,
        "text_len": WAN22_TI2V_5B.text_seq_len,
    }
    mismatches = {
        key: (raw.get(key), value) for key, value in expected.items() if raw.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}={actual!r} (expected {wanted!r})"
            for key, (actual, wanted) in sorted(mismatches.items())
        )
        raise ValueError(f"Checkpoint is not Wan2.2-TI2V-5B: {details}")
