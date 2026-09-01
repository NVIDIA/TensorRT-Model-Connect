# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated native TensorRT profiles for MiniMax-H3.

The default profile is the 124-frame, 1344x768 shape used by the public
Sol-Engine H3 benchmark. Structural row counts are explicit because prompt
packing is part of the engine ABI and must match the Hugging Face reference.
Text rows are dynamic; ``text_rows`` remains the maximum for compatible
bundle metadata.
"""

from __future__ import annotations

from dataclasses import dataclass


TEXT_ENCODER_DEFAULT_WORKSPACE_BYTES = 96 << 30
ADALN_PRECOMPUTE_DEFAULT_WORKSPACE_BYTES = 64 << 30
DENOISER_DEFAULT_WORKSPACE_BYTES = 96 << 30
VAE_TILE_DECODER_DEFAULT_WORKSPACE_BYTES = 96 << 30

DEFAULT_WORKSPACE_LIMIT_BYTES = {
    "text_encoder.plan": TEXT_ENCODER_DEFAULT_WORKSPACE_BYTES,
    "adaln_precompute.plan": ADALN_PRECOMPUTE_DEFAULT_WORKSPACE_BYTES,
    "denoiser.plan": DENOISER_DEFAULT_WORKSPACE_BYTES,
    "vae_tile_decoder.plan": VAE_TILE_DECODER_DEFAULT_WORKSPACE_BYTES,
}

FIRST_BLOCK_CACHE_DENOISER_PLAN_FILENAMES = (
    "denoiser_head.plan",
    "denoiser_tail.plan",
    "denoiser_finish.plan",
)


def native_plan_filenames(*, first_block_cache: bool) -> tuple[str, ...]:
    """Return the exact plan set selected by the native denoiser profile."""

    if not isinstance(first_block_cache, bool):
        raise ValueError("MiniMax-H3 first_block_cache must be a boolean")
    denoiser_plans = (
        FIRST_BLOCK_CACHE_DENOISER_PLAN_FILENAMES if first_block_cache else ("denoiser.plan",)
    )
    return (
        "text_encoder.plan",
        "adaln_precompute.plan",
        *denoiser_plans,
        "vae_tile_decoder.plan",
    )


def default_workspace_limit_bytes(*, first_block_cache: bool) -> dict[str, int]:
    """Return per-plan tactic workspace limits for one denoiser layout."""

    return {
        filename: (
            DENOISER_DEFAULT_WORKSPACE_BYTES
            if filename.startswith("denoiser_") or filename == "denoiser.plan"
            else DEFAULT_WORKSPACE_LIMIT_BYTES[filename]
        )
        for filename in native_plan_filenames(first_block_cache=first_block_cache)
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
    video_rows: int = 37296
    audio_rows: int = 414
    min_text_rows: int = 1
    opt_text_rows: int = 128
    text_rows: int = 537
    padded_sequence_length: int = 38247
    max_timestep_count: int = 4
    context_parallel_size: int = 1
    first_block_cache: bool = False

    @property
    def sequence_length(self) -> int:
        return self.video_rows + self.audio_rows + self.text_rows

    @property
    def min_sequence_length(self) -> int:
        return self.video_rows + self.audio_rows + self.min_text_rows

    @property
    def opt_sequence_length(self) -> int:
        return self.video_rows + self.audio_rows + self.opt_text_rows

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
        if self.sequence_length != self.padded_sequence_length:
            raise ValueError(
                "MiniMax-H3 padded_sequence_length must equal the maximum packed sequence"
            )
        if self.rope_freq_dim * 6 > self.head_dim:
            raise ValueError("MiniMax-H3 rotary channels exceed head_dim")
        if not isinstance(self.first_block_cache, bool):
            raise ValueError("MiniMax-H3 first_block_cache must be a boolean")


SOL_ENGINE_1344X768_124F = MiniMaxH3Config()
