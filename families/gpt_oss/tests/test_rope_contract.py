# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for GPT-OSS YaRN RoPE resolution."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")

from families.gpt_oss import graph_ops  # noqa: E402
from families.gpt_oss.config import ModelConfig  # noqa: E402
from families.gpt_oss.utils import make_rope_half_tables, resolve_rope_parameters  # noqa: E402


GPT_OSS_ROPE_SCALING = {
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "factor": 32.0,
    "original_max_position_embeddings": 4096,
    "rope_type": "yarn",
    "truncate": False,
}


def _hf_yarn_reference(
    head_dim: int,
    base: float,
    factor: float,
    original_max: int,
    beta_fast: float,
    beta_slow: float,
    truncate: bool,
) -> tuple[np.ndarray, float]:
    def correction_dim(num_rotations: float) -> float:
        return (head_dim * math.log(original_max / (num_rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    low = correction_dim(beta_fast)
    high = correction_dim(beta_slow)
    if truncate:
        low = math.floor(low)
        high = math.ceil(high)
    low = max(low, 0)
    high = min(high, head_dim - 1)
    if low == high:
        high += 0.001

    position_frequencies = base ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    inverse_extra = 1.0 / position_frequencies
    inverse_interpolated = 1.0 / (factor * position_frequencies)
    ramp = np.clip(
        (np.arange(head_dim // 2, dtype=np.float64) - low) / (high - low),
        0.0,
        1.0,
    )
    extrapolation = 1.0 - ramp
    inverse = inverse_interpolated * (1.0 - extrapolation) + inverse_extra * extrapolation
    attention_factor = 0.1 * math.log(factor) + 1.0 if factor > 1.0 else 1.0
    return inverse, attention_factor


def test_resolve_rope_parameters_accepts_rope_scaling_key() -> None:
    config = ModelConfig.create_tiny("gpt_oss", rope_scaling=GPT_OSS_ROPE_SCALING)
    assert resolve_rope_parameters(config) == GPT_OSS_ROPE_SCALING


def test_resolve_rope_parameters_prefers_rope_parameters_key() -> None:
    parameters = {"rope_type": "yarn", "factor": 8.0}
    config = ModelConfig.create_tiny(
        "gpt_oss",
        rope_parameters=parameters,
        rope_scaling=GPT_OSS_ROPE_SCALING,
    )
    assert resolve_rope_parameters(config) == parameters


def test_resolve_rope_parameters_defaults_to_empty() -> None:
    assert resolve_rope_parameters(ModelConfig.create_tiny("gpt_oss")) == {}


def test_make_rope_half_tables_applies_yarn_from_rope_scaling() -> None:
    config = ModelConfig.create_tiny(
        "gpt_oss",
        rope_theta=150000.0,
        rope_scaling=dict(GPT_OSS_ROPE_SCALING),
    )
    head_dim, window = 64, 16

    cosine, sine = make_rope_half_tables(config, window, head_dim)
    default_cosine = graph_ops.make_rope_table_half_dim(window, head_dim, 150000.0, True)
    assert not np.allclose(cosine, default_cosine)

    inverse, attention_factor = _hf_yarn_reference(
        head_dim,
        150000.0,
        32.0,
        4096,
        32.0,
        1.0,
        truncate=False,
    )
    positions = np.arange(window, dtype=np.float64)[:, None]
    np.testing.assert_allclose(
        cosine,
        np.cos(positions * inverse[None, :]) * attention_factor,
        rtol=0,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        sine,
        np.sin(positions * inverse[None, :]) * attention_factor,
        rtol=0,
        atol=1e-5,
    )


def test_yarn_table_truncate_flag_changes_correction_range() -> None:
    options = {
        "scaling_factor": 32.0,
        "original_max_position_embeddings": 4096,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
    }
    without_truncation = graph_ops.make_yarn_rope_table_half_dim(
        16,
        64,
        150000.0,
        True,
        truncate=False,
        **options,
    )
    with_truncation = graph_ops.make_yarn_rope_table_half_dim(
        16,
        64,
        150000.0,
        True,
        truncate=True,
        **options,
    )
    assert not np.allclose(without_truncation, with_truncation)


def test_yarn_table_explicit_attention_factor() -> None:
    options = {
        "scaling_factor": 32.0,
        "original_max_position_embeddings": 4096,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "truncate": False,
    }
    default_factor = graph_ops.make_yarn_rope_table_half_dim(
        4,
        64,
        150000.0,
        True,
        **options,
    )
    unit_factor = graph_ops.make_yarn_rope_table_half_dim(
        4,
        64,
        150000.0,
        True,
        attention_factor=1.0,
        **options,
    )
    expected_factor = 0.1 * math.log(32.0) + 1.0
    np.testing.assert_allclose(default_factor, unit_factor * expected_factor, rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        default_factor[0],
        np.full(32, expected_factor),
        rtol=0,
        atol=1e-6,
    )
