# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Laguna rotary-embedding tables for the TensorRT RoPE layer."""

from __future__ import annotations

import math

import numpy as np

from .config import RopeSpec


def _yarn_inverse_frequencies(spec: RopeSpec, pos_freqs: np.ndarray) -> np.ndarray:
    dim = spec.rotary_dim
    extrapolation = (np.float32(1.0) / pos_freqs).astype(np.float32)
    interpolation = (np.float32(1.0) / (np.float32(spec.factor) * pos_freqs)).astype(np.float32)

    def correction_dim(rotations: float) -> float:
        return (
            dim
            * math.log(spec.original_max_position_embeddings / (rotations * 2 * math.pi))
            / (2 * math.log(spec.theta))
        )

    low = correction_dim(spec.beta_fast)
    high = correction_dim(spec.beta_slow)
    if spec.truncate:
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001
    ramp = np.clip(
        (np.arange(dim // 2, dtype=np.float32) - np.float32(low)) / np.float32(high - low),
        0.0,
        1.0,
    ).astype(np.float32)
    keep = (np.float32(1.0) - ramp).astype(np.float32)
    return (interpolation * (np.float32(1.0) - keep) + extrapolation * keep).astype(np.float32)


def rope_tables(spec: RopeSpec, positions: int) -> tuple[np.ndarray, np.ndarray]:
    """Half-width cos/sin tables ``[positions, rotary_dim // 2]``.

    Mirrors the transformers ``default`` and ``yarn`` (``truncate=True``)
    inverse frequencies and evaluates angles and cos/sin in float32, scaled by
    the YaRN attention factor, as the reference rotary embedding does.
    """
    exponents = np.arange(0, spec.rotary_dim, 2, dtype=np.float32) / np.float32(spec.rotary_dim)
    pos_freqs = np.power(np.float32(spec.theta), exponents, dtype=np.float32)
    scaling = 1.0
    if spec.rope_type == "default":
        inv_freq = (np.float32(1.0) / pos_freqs).astype(np.float32)
    else:
        inv_freq = _yarn_inverse_frequencies(spec, pos_freqs)
        if spec.attention_factor is not None:
            scaling = spec.attention_factor
        elif spec.factor > 1.0:
            scaling = 0.1 * math.log(spec.factor) + 1.0
    angles = np.arange(positions, dtype=np.float32)[:, None] * inv_freq[None, :]
    cos = (np.cos(angles).astype(np.float32) * np.float32(scaling)).astype(np.float32)
    sin = (np.sin(angles).astype(np.float32) * np.float32(scaling)).astype(np.float32)
    return cos, sin
