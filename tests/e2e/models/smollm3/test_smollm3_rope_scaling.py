# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SmolLM3 YaRN RoPE scaling tests.

SmolLM3-3B ships with ``rope_scaling: null`` and a 65536-token window. The
model card extends that to 128k by setting ``max_position_embeddings`` to
131072 and adding a YaRN block, which is the configuration these tests pin.

The reference below is the exact float64 form of Hugging Face's
``_compute_yarn_parameters``. One detail is easy to miss and is asserted
separately: upstream folds ``attention_factor`` (``0.1 * ln(factor) + 1``) into
cos/sin inside the rotary embedding rather than applying it to attention
scores, so a table built without it is uniformly off by that factor.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorrt_model_connect.families.smollm3 import graph_ops


YARN_SCALING = {
    "rope_type": "yarn",
    "factor": 2.0,
    "original_max_position_embeddings": 65536,
}
HEAD_DIM = 128
ROPE_THETA = 5000000.0


def _yarn_reference(head_dim: int, rope_theta: float, factor: float,
                    original_context: float) -> tuple[np.ndarray, float]:
    """Exact float64 YaRN inverse frequencies and attention factor."""
    extrapolation = rope_theta ** (
        -np.arange(0, head_dim, 2, dtype=np.float64) / head_dim
    )
    half = head_dim // 2

    def correction_dim(rotations: float) -> float:
        return (
            head_dim
            * np.log(original_context / (rotations * 2.0 * np.pi))
            / (2.0 * np.log(rope_theta))
        )

    low = max(int(np.floor(correction_dim(32.0))), 0)
    high = min(int(np.ceil(correction_dim(1.0))), half - 1)
    ramp = np.clip(
        (np.arange(half, dtype=np.float64) - low) / max(high - low, 1), 0.0, 1.0
    )
    inverse = extrapolation / factor * ramp + extrapolation * (1.0 - ramp)
    return inverse, 0.1 * np.log(factor) + 1.0


def test_half_dim_table_matches_yarn_reference_formula() -> None:
    positions = 32
    inverse, attention_factor = _yarn_reference(
        HEAD_DIM, ROPE_THETA, 2.0, 65536.0
    )
    angles = np.outer(np.arange(positions, dtype=np.float64), inverse)

    cosine = graph_ops.make_rope_table_half_dim(
        positions, HEAD_DIM, ROPE_THETA, True, rope_scaling=YARN_SCALING
    )
    sine = graph_ops.make_rope_table_half_dim(
        positions, HEAD_DIM, ROPE_THETA, False, rope_scaling=YARN_SCALING
    )

    np.testing.assert_allclose(
        cosine, np.cos(angles) * attention_factor, atol=1e-7
    )
    np.testing.assert_allclose(
        sine, np.sin(angles) * attention_factor, atol=1e-7
    )


def test_attention_factor_is_folded_into_the_table() -> None:
    # Omitting it leaves every entry short by ~6.9% for factor=2.0, which is
    # four orders of magnitude above float32 noise.
    inverse, attention_factor = _yarn_reference(
        HEAD_DIM, ROPE_THETA, 2.0, 65536.0
    )
    assert attention_factor == pytest.approx(1.0693147180559945)

    cosine = graph_ops.make_rope_table_half_dim(
        8, HEAD_DIM, ROPE_THETA, True, rope_scaling=YARN_SCALING
    )
    unscaled = np.cos(np.outer(np.arange(8, dtype=np.float64), inverse))
    assert np.abs(cosine - unscaled).max() > 1e-3
    np.testing.assert_allclose(cosine, unscaled * attention_factor, atol=1e-7)


def test_explicit_attention_factor_overrides_the_derived_one() -> None:
    scaling = dict(YARN_SCALING, attention_factor=1.0)
    inverse, _ = _yarn_reference(HEAD_DIM, ROPE_THETA, 2.0, 65536.0)
    cosine = graph_ops.make_rope_table_half_dim(
        8, HEAD_DIM, ROPE_THETA, True, rope_scaling=scaling
    )
    expected = np.cos(np.outer(np.arange(8, dtype=np.float64), inverse))
    np.testing.assert_allclose(cosine, expected, atol=1e-7)


@pytest.mark.parametrize(
    "override",
    [
        {"factor": 0.0},
        {"original_max_position_embeddings": 0},
        {"beta_fast": 1.0, "beta_slow": 32.0},
    ],
)
def test_malformed_yarn_scaling_is_rejected(override) -> None:
    with pytest.raises(ValueError):
        graph_ops.make_rope_table_half_dim(
            8, HEAD_DIM, ROPE_THETA, True,
            rope_scaling=dict(YARN_SCALING, **override),
        )


def test_unscaled_rope_is_unaffected_by_yarn_support() -> None:
    positions = 16
    inverse = ROPE_THETA ** (
        -np.arange(0, HEAD_DIM, 2, dtype=np.float64) / HEAD_DIM
    )
    angles = np.outer(np.arange(positions, dtype=np.float64), inverse)
    cosine = graph_ops.make_rope_table_half_dim(
        positions, HEAD_DIM, ROPE_THETA, True
    )
    np.testing.assert_allclose(cosine, np.cos(angles), atol=1e-7)
