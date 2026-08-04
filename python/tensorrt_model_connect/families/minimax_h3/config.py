# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated native TensorRT profiles for MiniMax-H3.

The default profile is the 124-frame, 1344x768 shape used by the public
Sol-Engine H3 benchmark. Structural row counts are explicit because prompt
packing is part of the engine ABI and must match the Hugging Face reference.
"""

from __future__ import annotations

from dataclasses import dataclass


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
    text_rows: int = 537
    padded_sequence_length: int = 38272
    max_timestep_count: int = 4
    context_parallel_size: int = 8

    @property
    def sequence_length(self) -> int:
        return self.video_rows + self.audio_rows + self.text_rows

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
        if self.attention_size <= self.hidden_size:
            raise ValueError("MiniMax-H3 attention width must exceed its residual width")
        if self.sequence_length > self.padded_sequence_length:
            raise ValueError("MiniMax-H3 packed rows exceed padded_sequence_length")
        if self.padded_sequence_length % self.context_parallel_size:
            raise ValueError(
                "MiniMax-H3 padded_sequence_length must divide evenly across context-parallel ranks"
            )
        if self.num_heads % self.context_parallel_size:
            raise ValueError(
                "MiniMax-H3 num_heads must divide evenly across context-parallel ranks"
            )
        if self.rope_freq_dim * 6 > self.head_dim:
            raise ValueError("MiniMax-H3 rotary channels exceed head_dim")


SOL_ENGINE_1344X768_124F = MiniMaxH3Config()
