# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rotary-embedding constants of Apertus, computed like transformers on the CPU in FP32."""

from __future__ import annotations

import math

import numpy as np

from .config import RopeConfig


def rope_inverse_frequencies(head_dim: int, rope: RopeConfig) -> np.ndarray:
    """``inv_freq`` of the default or ``llama3`` schedule, in transformers' operation order."""
    import torch

    with torch.no_grad():
        exponents = torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float) / head_dim
        inv_freq = 1.0 / (rope.theta**exponents)
        if rope.rope_type == "llama3":
            old_context = rope.original_max_position_embeddings
            low_wavelen = old_context / rope.low_freq_factor
            high_wavelen = old_context / rope.high_freq_factor
            wavelen = 2 * math.pi / inv_freq
            scaled = torch.where(wavelen > low_wavelen, inv_freq / rope.factor, inv_freq)
            smooth = (old_context / wavelen - rope.low_freq_factor) / (
                rope.high_freq_factor - rope.low_freq_factor
            )
            smoothed = (1 - smooth) * scaled / rope.factor + smooth * scaled
            medium = ~(wavelen < high_wavelen) * ~(wavelen > low_wavelen)
            inv_freq = torch.where(medium, smoothed, scaled)
    return np.asarray(inv_freq.to(torch.float32).numpy(), dtype=np.float32).copy()


def rope_tables(head_dim: int, rope: RopeConfig, positions: int) -> tuple[np.ndarray, np.ndarray]:
    """FP32 ``[positions, head_dim / 2]`` cos and sin tables (both schedules scale by 1)."""
    import torch

    inv_freq = torch.from_numpy(rope_inverse_frequencies(head_dim, rope))
    with torch.no_grad():
        angles = torch.arange(positions, dtype=torch.float32)[:, None] * inv_freq[None, :]
        return angles.cos().numpy().copy(), angles.sin().numpy().copy()
