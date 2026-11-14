# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""YaRN tables and the position-dependent query scale of ministral3 (CPU)."""

import math

import numpy as np

from families.ministral3 import rope
from families.ministral3.config import RopeConfig

YARN = RopeConfig(
    rope_type="yarn",
    theta=1_000_000.0,
    factor=16.0,
    original_max_position_embeddings=16384,
    beta_fast=32.0,
    beta_slow=1.0,
    mscale=1.0,
    mscale_all_dim=1.0,
    llama_4_scaling_beta=0.1,
)


def yarn_float64(config: RopeConfig, dim: int) -> np.ndarray:
    """The YaRN formula of the Transformers reference, evaluated in float64."""
    pos_freqs = config.theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim)

    def correction(rotations):
        return (
            dim
            * math.log(config.original_max_position_embeddings / (rotations * 2 * math.pi))
            / (2 * math.log(config.theta))
        )

    low = max(math.floor(correction(config.beta_fast)), 0)
    high = min(math.ceil(correction(config.beta_slow)), dim - 1)
    ramp = np.clip((np.arange(dim // 2) - low) / (high - low), 0, 1)
    extrapolation_factor = 1 - ramp
    return (1 / (config.factor * pos_freqs)) * (1 - extrapolation_factor) + (
        1 / pos_freqs
    ) * extrapolation_factor


def test_yarn_inverse_frequencies_follow_the_reference_formula():
    inv_freq = rope.inverse_frequencies(YARN, 128)
    assert inv_freq.dtype == np.float32 and inv_freq.shape == (64,)
    np.testing.assert_allclose(inv_freq, yarn_float64(YARN, 128), rtol=2e-7)
    assert inv_freq[0] == 1.0
    # Low-frequency pairs are fully interpolated by the factor, high ones untouched.
    plain = rope.inverse_frequencies(RopeConfig(rope_type="default", theta=1_000_000.0), 128)
    np.testing.assert_allclose(inv_freq[-1], plain[-1] / 16.0, rtol=1e-6)
    np.testing.assert_array_equal(inv_freq[:8], plain[:8])


def test_attention_factor_and_tables():
    assert rope.attention_factor(YARN) == 1.0
    assert math.isclose(
        rope.attention_factor(
            RopeConfig(rope_type="yarn", theta=1e4, factor=4.0, original_max_position_embeddings=8)
        ),
        0.1 * math.log(4.0) + 1.0,
    )
    cos, sin = rope.rotary_tables(YARN, 128, 300)
    assert cos.shape == sin.shape == (300, 64) and cos.dtype == np.float32
    np.testing.assert_allclose(cos * cos + sin * sin, 1.0, atol=1e-6)
    angle = np.float32(299.0) * rope.inverse_frequencies(YARN, 128)[5]
    assert math.isclose(cos[299, 5], math.cos(float(angle)), abs_tol=1e-7)


def test_query_scale_starts_after_the_original_context():
    scales = rope.query_scales(YARN, 40000)
    assert (scales[:16384] == 1.0).all()
    np.testing.assert_allclose(scales[16384:32768], 1.0 + 0.1 * math.log(2.0), rtol=1e-7)
    np.testing.assert_allclose(scales[32768:], 1.0 + 0.1 * math.log(3.0), rtol=1e-7)
    assert not rope.needs_query_scale(YARN, 16384)
    assert rope.needs_query_scale(YARN, 16385)
