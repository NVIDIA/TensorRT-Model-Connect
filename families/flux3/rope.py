# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rotary position tables for flux3 (pure NumPy, computed at build time).

The transformer uses four interleaved RoPE axes ``(t, h, w, l)`` of 32 channels
each, theta 10000, with frequencies in float64 and tables in float32
(Diffusers ``Flux3PosEmbed``). Image tokens sit at ``(0, 0.5 * row,
0.5 * col, 0)`` and text tokens at ``(0, 0, 0, l)``. The Qwen3-VL text encoder
uses rotate-half RoPE with theta 5e6; for text-only input its three M-RoPE
axes carry the same position, which makes it ordinary RoPE.
"""

from __future__ import annotations

import numpy as np


def image_ids(latent_height: int, latent_width: int, rope_scale: float) -> np.ndarray:
    """``(t, h, w, l)`` ids of the generated image tokens, row-major, float32."""

    rows, cols = np.meshgrid(
        np.arange(latent_height, dtype=np.float32),
        np.arange(latent_width, dtype=np.float32),
        indexing="ij",
    )
    ids = np.zeros((latent_height * latent_width, 4), dtype=np.float32)
    ids[:, 1] = rows.reshape(-1) * np.float32(rope_scale)
    ids[:, 2] = cols.reshape(-1) * np.float32(rope_scale)
    return ids


def text_ids(length: int) -> np.ndarray:
    """``(0, 0, 0, l)`` ids of ``length`` text tokens, float32."""

    ids = np.zeros((length, 4), dtype=np.float32)
    ids[:, 3] = np.arange(length, dtype=np.float32)
    return ids


def interleaved_tables(
    ids: np.ndarray, axes_dims: tuple[int, ...], theta: float
) -> tuple[np.ndarray, np.ndarray]:
    """``(cos, sin)`` of shape ``[rows, sum(axes_dims)]`` with each frequency repeated twice."""

    cos_parts, sin_parts = [], []
    for axis, dim in enumerate(axes_dims):
        exponents = np.arange(0, dim, 2, dtype=np.float64)[: dim // 2] / dim
        freqs = 1.0 / (theta**exponents)
        angles = np.outer(ids[:, axis].astype(np.float32).astype(np.float64), freqs)
        cos_parts.append(np.repeat(np.cos(angles), 2, axis=1).astype(np.float32))
        sin_parts.append(np.repeat(np.sin(angles), 2, axis=1).astype(np.float32))
    return np.concatenate(cos_parts, axis=1), np.concatenate(sin_parts, axis=1)


def interleaved_rotation(head_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Index and sign vectors with ``x[idx] * sign`` = ``(-x1, x0, -x3, x2, ...)``."""

    index = np.arange(head_dim, dtype=np.int32).reshape(-1, 2)[:, ::-1].reshape(-1)
    sign = np.tile(np.array([-1.0, 1.0], dtype=np.float32), head_dim // 2)
    return index.copy(), sign


def rotate_half_rotation(head_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Index and sign vectors with ``x[idx] * sign`` = ``cat(-x[d/2:], x[:d/2])``."""

    half = head_dim // 2
    index = np.concatenate([np.arange(half, head_dim), np.arange(0, half)]).astype(np.int32)
    sign = np.concatenate([-np.ones(half), np.ones(half)]).astype(np.float32)
    return index, sign


def text_encoder_tables(
    max_positions: int, head_dim: int, theta: float
) -> tuple[np.ndarray, np.ndarray]:
    """Qwen3 rotate-half ``(cos, sin)`` tables ``[max_positions, head_dim]`` (float32).

    Mirrors Transformers: float32 inverse frequencies, float32 position x
    frequency products, ``cat(freqs, freqs)``.
    """

    exponents = np.arange(0, head_dim, 2, dtype=np.int64).astype(np.float32) / np.float32(head_dim)
    inv_freq = (np.float32(1.0) / (np.float32(theta) ** exponents)).astype(np.float32)
    positions = np.arange(max_positions, dtype=np.float32)
    freqs = (positions[:, None] * inv_freq[None, :]).astype(np.float32)
    emb = np.concatenate([freqs, freqs], axis=1).astype(np.float64)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)
