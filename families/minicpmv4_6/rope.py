# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RoPE tables of the decoder (NumPy only, so the builder checks them before any engine)."""

from __future__ import annotations

import numpy as np


def rope_positions(max_cache_length: int) -> int:
    """Rows of the RoPE tables: the cache plus the current token."""
    return max_cache_length + 1


def rope_tables(config, positions: int) -> tuple[np.ndarray, np.ndarray]:
    """Half-dimension cos/sin tables of ``config.rotary_dim`` and ``config.rope_theta``, in the reference's FP32.

    The reference computes ``inv_freq = 1 / theta ** (arange(0, dim, 2) / dim)``
    and ``angle = position * inv_freq`` in FP32 before taking cos and sin, so
    the tables round the same intermediates.
    """
    dim = config.rotary_dim
    exponents = np.arange(0, dim, 2, dtype=np.float32) / np.float32(dim)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        inv_freq = (np.float32(1.0) / np.power(np.float32(config.rope_theta), exponents)).astype(np.float32)
        angles = np.arange(positions, dtype=np.float32)[:, None] * inv_freq[None, :]
    angles = angles.astype(np.float32)
    # A small rope_theta makes the frequencies, and then the angles of late positions, overflow FP32.
    if not (np.isfinite(inv_freq).all() and np.isfinite(angles).all()):
        raise ValueError(f"MiniCPM-V 4.6 rope_theta {config.rope_theta} overflows the FP32 RoPE angles "
                         f"of {positions} positions")
    return np.cos(angles.astype(np.float64)).astype(np.float32), np.sin(angles.astype(np.float64)).astype(np.float32)
