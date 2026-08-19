# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Llama 3.1 RoPE scaling tests for Minitron."""

from __future__ import annotations

import numpy as np

from tensorrt_model_connect.models.llama import graph_ops


LLAMA3_SCALING = {
    "factor": 8.0,
    "high_freq_factor": 4.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "llama3",
}


def _llama3_inverse_frequencies(head_dim: int, rope_theta: float) -> np.ndarray:
    inverse = 1.0 / (
        rope_theta
        ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    )
    wavelength = 2.0 * np.pi / inverse
    low_wavelength = 8192.0
    high_wavelength = 8192.0 / 4.0
    scaled = np.where(wavelength > low_wavelength, inverse / 8.0, inverse)
    smooth = (8192.0 / wavelength - 1.0) / 3.0
    interpolated = (1.0 - smooth) * scaled / 8.0 + smooth * scaled
    medium = (wavelength >= high_wavelength) & (wavelength <= low_wavelength)
    return np.where(medium, interpolated, scaled)


def test_half_dim_table_matches_llama3_reference_formula() -> None:
    positions = 32
    head_dim = 128
    rope_theta = 500000.0
    inverse = _llama3_inverse_frequencies(head_dim, rope_theta)
    angles = np.outer(np.arange(positions, dtype=np.float64), inverse)

    cosine = graph_ops.make_rope_table_half_dim(
        positions,
        head_dim,
        rope_theta,
        True,
        rope_scaling=LLAMA3_SCALING,
    )
    sine = graph_ops.make_rope_table_half_dim(
        positions,
        head_dim,
        rope_theta,
        False,
        rope_scaling=LLAMA3_SCALING,
    )

    np.testing.assert_allclose(cosine, np.cos(angles), atol=1e-7)
    np.testing.assert_allclose(sine, np.sin(angles), atol=1e-7)
