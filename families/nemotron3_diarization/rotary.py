# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The rotary position tables the encoder engine carries as constants."""

from __future__ import annotations

import numpy as np


def inverse_frequencies(head_dim: int, theta: float) -> np.ndarray:
    """``1 / theta ** (2i / head_dim)`` in float32, the reference's own rounding."""
    exponents = np.arange(0, head_dim, 2, dtype=np.float32) / np.float32(head_dim)
    return np.float32(1.0) / np.power(np.float32(theta), exponents)


def rotary_tables(head_dim: int, theta: float, positions: int) -> tuple[np.ndarray, np.ndarray]:
    """Float32 cos and sin ``[positions, head_dim]`` for positions ``0 .. positions - 1``.

    The angles are float32 products, as the reference forms them; the cosine
    and sine of each float32 angle are rounded once. Positions restart at zero
    every encoder step, so the table covers one step, never the recording.
    """
    angles = np.arange(positions, dtype=np.float32)[:, None] * inverse_frequencies(
        head_dim, theta)[None, :]
    angles = np.concatenate([angles, angles], axis=1).astype(np.float64)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)
