# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rotary tables and the position-dependent query scale of Ministral 3.

The inverse frequencies follow the Transformers YaRN formula in float32, in the
same operation order, so the angles ``position * inv_freq`` are the float32
products the reference forms. The query scale is
``1 + beta * log(1 + floor(position / original_max_position_embeddings))``.
"""

from __future__ import annotations

import math

import numpy as np

from .config import RopeConfig


def attention_factor(rope: RopeConfig) -> float:
    """The cos/sin multiplier of the YaRN formula (1.0 for default RoPE)."""
    if rope.rope_type == "default":
        return 1.0
    if rope.attention_factor is not None:
        return float(rope.attention_factor)

    def mscale(scale: float, value: float = 1.0) -> float:
        return 1.0 if scale <= 1 else 0.1 * value * math.log(scale) + 1.0

    if rope.mscale and rope.mscale_all_dim:
        return float(mscale(rope.factor, rope.mscale) / mscale(rope.factor, rope.mscale_all_dim))
    return mscale(rope.factor)


def inverse_frequencies(rope: RopeConfig, head_dim: int) -> np.ndarray:
    """Float32 inverse frequencies for the ``head_dim // 2`` rotary pairs."""
    dim = int(head_dim)
    exponents = np.arange(0, dim, 2, dtype=np.int64).astype(np.float32) / np.float32(dim)
    pos_freqs = np.power(np.float32(rope.theta), exponents, dtype=np.float32)
    extrapolation = (np.float32(1.0) / pos_freqs).astype(np.float32)
    if rope.rope_type == "default":
        return extrapolation
    interpolation = (np.float32(1.0) / (np.float32(rope.factor) * pos_freqs)).astype(np.float32)

    def correction_dim(rotations: float) -> float:
        return (
            dim * math.log(rope.original_max_position_embeddings / (rotations * 2 * math.pi))
        ) / (2 * math.log(rope.theta))

    low = correction_dim(rope.beta_fast)
    high = correction_dim(rope.beta_slow)
    if rope.truncate:
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001
    ramp = (np.arange(dim // 2, dtype=np.float32) - np.float32(low)) / np.float32(high - low)
    ramp = np.clip(ramp, np.float32(0.0), np.float32(1.0)).astype(np.float32)
    extrapolation_factor = (np.float32(1.0) - ramp).astype(np.float32)
    blended = (
        interpolation * (np.float32(1.0) - extrapolation_factor)
        + extrapolation * extrapolation_factor
    )
    return blended.astype(np.float32)


def rotary_tables(rope: RopeConfig, head_dim: int, positions: int) -> tuple[np.ndarray, np.ndarray]:
    """Return float32 ``cos`` and ``sin`` tables of shape ``[positions, head_dim // 2]``."""
    inv_freq = inverse_frequencies(rope, head_dim)
    angles = np.arange(positions, dtype=np.float32)[:, None] * inv_freq[None, :]
    angles = angles.astype(np.float32).astype(np.float64)
    scale = attention_factor(rope)
    cos = (np.cos(angles) * scale).astype(np.float32)
    sin = (np.sin(angles) * scale).astype(np.float32)
    return cos, sin


def query_scales(rope: RopeConfig, positions: int) -> np.ndarray:
    """Float32 per-position query multiplier, shape ``[positions]``."""
    position = np.arange(positions, dtype=np.float32)
    periods = np.floor(position / np.float32(rope.original_max_position_embeddings))
    logs = np.log((np.float32(1.0) + periods).astype(np.float32)).astype(np.float32)
    return (np.float32(1.0) + np.float32(rope.llama_4_scaling_beta) * logs).astype(np.float32)


def needs_query_scale(rope: RopeConfig, positions: int) -> bool:
    """True when some position below ``positions`` has a scale other than 1."""
    return rope.llama_4_scaling_beta != 0.0 and positions > rope.original_max_position_embeddings
