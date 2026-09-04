# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated native TensorRT profiles for MiniMax-H3.

The fixed profile preserves the 124-frame, 1344x768 shape used by the public
Sol-Engine H3 benchmark.  The production media profile covers every released
5--15 second geometry, the public continuous 1:4--4:1 canvas resolver, and the
documented explicit 960x544 performance canvas in both orientations.
Structural row counts are explicit because prompt/media packing is part of
the engine ABI and must match the Hugging Face reference.
"""

from __future__ import annotations

from dataclasses import dataclass


TEXT_ENCODER_DEFAULT_WORKSPACE_BYTES = 96 << 30
VISION_ENCODER_DEFAULT_WORKSPACE_BYTES = 32 << 30
ADALN_PRECOMPUTE_DEFAULT_WORKSPACE_BYTES = 64 << 30
DENOISER_DEFAULT_WORKSPACE_BYTES = 96 << 30
KEYFRAME_VAE_ENCODER_DEFAULT_WORKSPACE_BYTES = 32 << 30
VAE_TILE_DECODER_DEFAULT_WORKSPACE_BYTES = 96 << 30
AUDIO_VAE_DECODER_DEFAULT_WORKSPACE_BYTES = 96 << 30
AUDIO_LATENT_FRAMES_MIN = 207
AUDIO_LATENT_FRAMES_OPT = 207
AUDIO_LATENT_FRAMES_MAX = 575
VIDEO_NUM_FRAMES_MIN = 124
VIDEO_NUM_FRAMES_OPT = 124
VIDEO_NUM_FRAMES_MAX = 345
VIDEO_ROWS_MIN = 18_870
VIDEO_ROWS_OPT = 37_296
TARGET_VIDEO_ROWS_MAX = 106_488
VIDEO_ROWS_MAX = 108_576
CANVAS_MULTIPLE = 32
CANVAS_SHORT_EDGE = 768
CANVAS_MAX_PIXELS = 768 * 1344
CANVAS_MIN_ASPECT_RATIO = 0.25
CANVAS_MAX_ASPECT_RATIO = 4.0
# Extra explicit Diffusers performance canvas, stored as (height, width). The
# TensorRT runtime remains a finite allowlist: these two orientations are in
# addition to, not a replacement for, the 95 resolver-produced canvases.
NATIVE_EXPLICIT_CANVAS_SIZES = ((544, 960), (960, 544))
# The RTX path builds each plan in a fresh process, so one conservative
# workspace and runtime budget cover every stage without coupling the public
# artifact to a particular workstation identity.
RTX_STAGED_WORKSPACE_BYTES = 16 << 30
RTX_WEIGHT_STREAMING_BUDGET_BYTES = 32 << 30
RTX_CUDA_MAJOR = 12
TRT_DEFAULT_WORKSPACE_POLICY = "trt_default_max"

DEFAULT_WORKSPACE_LIMIT_BYTES = {
    "text_encoder.plan": TEXT_ENCODER_DEFAULT_WORKSPACE_BYTES,
    "vision_encoder.plan": VISION_ENCODER_DEFAULT_WORKSPACE_BYTES,
    "adaln_precompute.plan": ADALN_PRECOMPUTE_DEFAULT_WORKSPACE_BYTES,
    "fl2va_keyframe_vae_encoder.plan": KEYFRAME_VAE_ENCODER_DEFAULT_WORKSPACE_BYTES,
    "vae_tile_decoder.plan": VAE_TILE_DECODER_DEFAULT_WORKSPACE_BYTES,
    "audio_vae_decoder.plan": AUDIO_VAE_DECODER_DEFAULT_WORKSPACE_BYTES,
}

FIRST_BLOCK_CACHE_DENOISER_PLAN_FILENAMES = (
    "denoiser_head.plan",
    "denoiser_tail.plan",
    "denoiser_finish.plan",
)


def native_plan_filenames() -> tuple[str, ...]:
    """Return the singular original-weight dense FirstBlockCache plan set."""

    return (
        "text_encoder.plan",
        "vision_encoder.plan",
        "adaln_precompute.plan",
        *FIRST_BLOCK_CACHE_DENOISER_PLAN_FILENAMES,
        "fl2va_keyframe_vae_encoder.plan",
        "vae_tile_decoder.plan",
        "audio_vae_decoder.plan",
    )


def default_workspace_limit_bytes() -> dict[str, int | str]:
    """Return per-plan tactic workspace limits for the dense FBC layout."""

    return {
        filename: (
            TRT_DEFAULT_WORKSPACE_POLICY
            if filename.startswith("denoiser_") or filename.startswith("adaln_")
            else DEFAULT_WORKSPACE_LIMIT_BYTES[filename]
        )
        for filename in native_plan_filenames()
    }


def resolve_workspace_bytes(workspace_bytes: int | None, *, default_bytes: int) -> int:
    """Resolve a positive tactic-workspace limit without silently coercing values."""

    resolved = default_bytes if workspace_bytes is None else workspace_bytes
    if not isinstance(resolved, int) or isinstance(resolved, bool) or resolved <= 0:
        raise ValueError("MiniMax-H3 TensorRT workspace_bytes must be a positive integer")
    return resolved


@dataclass(frozen=True)
class MiniMaxH3Config:
    hidden_size: int = 5376
    num_layers: int = 50
    num_refiner_layers: int = 2
    num_heads: int = 56
    head_dim: int = 128
    ffn_dim: int = 14336
    video_in_channels: int = 24
    video_patch: tuple[int, int, int] = (1, 2, 2)
    audio_in_channels: int = 32
    text_dim: int = 5120
    timestep_input_dim: int = 256
    timestep_hidden_size: int = 5376
    timestep_embed_dim: int = 2688
    rope_freq_dim: int = 16
    norm_eps: float = 1.0e-5
    min_video_rows: int = 37296
    opt_video_rows: int = 37296
    video_rows: int = 37296
    min_audio_rows: int = 414
    opt_audio_rows: int = 414
    audio_rows: int = 414
    min_text_rows: int = 1
    opt_text_rows: int = 128
    text_rows: int = 537
    padded_sequence_length: int = 38247
    max_timestep_count: int = 4
    context_parallel_size: int = 1
    first_block_cache: bool = True

    @property
    def sequence_length(self) -> int:
        return self.video_rows + self.audio_rows + self.text_rows

    @property
    def min_sequence_length(self) -> int:
        return self.min_video_rows + self.min_audio_rows + self.min_text_rows

    @property
    def opt_sequence_length(self) -> int:
        return self.opt_video_rows + self.opt_audio_rows + self.opt_text_rows

    @property
    def video_row_profile(self) -> tuple[int, int, int]:
        return self.min_video_rows, self.opt_video_rows, self.video_rows

    @property
    def audio_row_profile(self) -> tuple[int, int, int]:
        return self.min_audio_rows, self.opt_audio_rows, self.audio_rows

    @property
    def packed_row_profile(self) -> tuple[int, int, int]:
        return self.min_sequence_length, self.opt_sequence_length, self.sequence_length

    @property
    def text_row_profile(self) -> tuple[int, int, int]:
        return self.min_text_rows, self.opt_text_rows, self.text_rows

    @property
    def padding_rows(self) -> int:
        return self.padded_sequence_length - self.sequence_length

    @property
    def video_patch_dim(self) -> int:
        pt, ph, pw = self.video_patch
        return self.video_in_channels * pt * ph * pw

    @property
    def attention_size(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def adaln_table_rows(self) -> int:
        return self.max_timestep_count * 3

    def validate(self) -> None:
        if self.hidden_size <= 0 or self.num_layers <= 0:
            raise ValueError("MiniMax-H3 hidden_size and num_layers must be positive")
        if self.context_parallel_size != 1:
            raise ValueError("MiniMax-H3 native runtime currently requires context_parallel_size=1")
        if self.attention_size <= self.hidden_size:
            raise ValueError("MiniMax-H3 attention width must exceed its residual width")
        if not 1 <= self.min_text_rows <= self.opt_text_rows <= self.text_rows:
            raise ValueError("MiniMax-H3 text rows must satisfy 1 <= min <= opt <= max")
        if not 0 < self.min_video_rows <= self.opt_video_rows <= self.video_rows:
            raise ValueError("MiniMax-H3 video rows must satisfy 0 < min <= opt <= max")
        if not 0 < self.min_audio_rows <= self.opt_audio_rows <= self.audio_rows:
            raise ValueError("MiniMax-H3 audio rows must satisfy 0 < min <= opt <= max")
        if any(rows % 2 for rows in self.audio_row_profile):
            raise ValueError("MiniMax-H3 audio_rows must contain two equal stereo channels")
        if self.sequence_length != self.padded_sequence_length:
            raise ValueError(
                "MiniMax-H3 requires no packed-sequence padding: "
                "padded_sequence_length must equal the maximum packed sequence"
            )
        if self.rope_freq_dim * 6 > self.head_dim:
            raise ValueError("MiniMax-H3 rotary channels exceed head_dim")
        if not isinstance(self.first_block_cache, bool):
            raise ValueError("MiniMax-H3 first_block_cache must be a boolean")


SOL_ENGINE_1344X768_124F = MiniMaxH3Config()

# Profile zero in the production FirstBlockCache engines exactly specializes
# the 537-token reference request used by the qualified 8--9 minute path.
# Profile one retains variable prompts and the complete public 5--15 second
# envelope below, so both routes remain contexts of the same three engines.
SOL_ENGINE_1344X768_124F_FAST_FBC = MiniMaxH3Config(
    min_text_rows=537,
    opt_text_rows=537,
)

# The released local pipeline aligns requested frame counts to ``17 * n + 5``.
# At 24 fps its supported 5--15 second endpoints are therefore 124 and 345
# frames.  Video tokens use 1,008 rows per latent frame at 1344x768; audio is
# packed as two stereo row groups of 207 through 575 latent frames.
SOL_ENGINE_1344X768_124_TO_345F = MiniMaxH3Config(
    min_video_rows=VIDEO_ROWS_MIN,
    opt_video_rows=VIDEO_ROWS_OPT,
    video_rows=VIDEO_ROWS_MAX,
    audio_rows=1150,
    text_rows=2641,
    padded_sequence_length=112367,
)
