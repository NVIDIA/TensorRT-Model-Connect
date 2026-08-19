# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerical coverage for Qwen's family-owned YaRN RoPE cache."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="Qwen graph operations require TensorRT")

from tensorrt_model_connect.families.qwen import graph_ops


def _reference(
    *,
    max_cache_length: int,
    head_dim: int,
    rope_theta: float,
    cosine: bool,
    scaling_factor: float,
    original_max_position_embeddings: int,
    beta_fast: float,
    beta_slow: float,
) -> np.ndarray:
    half = head_dim // 2
    freq_extra = rope_theta ** -(
        np.arange(0, head_dim, 2, dtype=np.float64) / head_dim
    )
    freq_inter = freq_extra / scaling_factor

    def correction_dim(num_rotations: float) -> float:
        return head_dim * np.log(
            original_max_position_embeddings / (num_rotations * 2 * np.pi)
        ) / (2 * np.log(rope_theta))

    low = max(int(np.floor(correction_dim(beta_fast))), 0)
    high = min(int(np.ceil(correction_dim(beta_slow))), half - 1)
    ramp = np.clip(
        (np.arange(half, dtype=np.float64) - low) / max(high - low, 1),
        0.0,
        1.0,
    )
    inv_freq = freq_inter * ramp + freq_extra * (1.0 - ramp)
    angles = np.outer(np.arange(max_cache_length, dtype=np.float64), inv_freq)
    values = np.cos(angles) if cosine else np.sin(angles)
    return values.astype(np.float32)


@pytest.mark.parametrize("cosine", [True, False])
@pytest.mark.parametrize("scaling_factor", [1.0, 2.0, 4.0])
def test_yarn_rope_table_half_dim_matches_numpy_reference(
    cosine: bool,
    scaling_factor: float,
) -> None:
    kwargs = {
        "max_cache_length": 16,
        "head_dim": 64,
        "rope_theta": 10000.0,
        "cosine": cosine,
        "scaling_factor": scaling_factor,
        "original_max_position_embeddings": 4096,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
    }

    actual = graph_ops.make_yarn_rope_table_half_dim(**kwargs)
    expected = _reference(**kwargs)

    assert actual.shape == (16, 32)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, atol=1e-6)
