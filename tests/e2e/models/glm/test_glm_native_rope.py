# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GLM-owned TensorRT proof for partial interleaved active-position RoPE."""

from __future__ import annotations

import numpy as np
import pytest

from tests.builder.conftest import requires_trt

trt = pytest.importorskip("tensorrt")

from tests.builder.test_graph_ops_native_attn import _run_strongly_typed  # noqa: E402
from tensorrt_model_connect.families.glm import graph_ops  # noqa: E402


def _interleaved_partial_rope_reference(
    x: np.ndarray,
    positions: np.ndarray,
    inv_freq: np.ndarray,
    *,
    num_heads: int,
    head_dim: int,
) -> np.ndarray:
    rows = x.reshape(len(positions), num_heads, head_dim).copy()
    rotary_dim = inv_freq.size * 2
    angles = positions.astype(np.float32)[:, None] * inv_freq[None, :]
    cos = np.cos(angles).astype(np.float32)[:, None, :]
    sin = np.sin(angles).astype(np.float32)[:, None, :]
    even = rows[..., :rotary_dim:2].copy()
    odd = rows[..., 1:rotary_dim:2].copy()
    rows[..., :rotary_dim:2] = even * cos - odd * sin
    rows[..., 1:rotary_dim:2] = odd * cos + even * sin
    return rows.reshape(x.shape)


@requires_trt
def test_partial_interleaved_active_rope_matches_hf_glm_at_128k() -> None:
    num_heads = 2
    head_dim = 128
    partial_rotary_factor = 0.5
    rotary_dim = int(head_dim * partial_rotary_factor)
    positions = np.array([0, 3, 65535, 131071], dtype=np.int32)
    rng = np.random.default_rng(20260728)
    x = rng.standard_normal((len(positions), num_heads * head_dim)).astype(np.float32)
    inv_freq = graph_ops.make_native_active_rope_inv_freq(
        head_dim,
        10000.0,
        partial_rotary_factor,
    )

    def build(network, trt_inputs):
        cos_active, sin_active = graph_ops.add_active_rope_cache(
            network,
            trt_inputs["position_id"],
            inv_freq,
            trt.float32,
        )
        output = graph_ops.add_apply_active_rope(
            network,
            trt_inputs["x"],
            num_heads,
            head_dim,
            cos_active,
            sin_active,
            rotary_dim,
            interleaved=True,
        )
        return {"output": output, "cos": cos_active, "sin": sin_active}

    result = _run_strongly_typed(
        build,
        {"x": x, "position_id": positions},
    )
    reference = _interleaved_partial_rope_reference(
        x,
        positions,
        inv_freq,
        num_heads=num_heads,
        head_dim=head_dim,
    )

    assert result["cos"].shape == (1, len(positions), rotary_dim // 2)
    assert result["sin"].shape == (1, len(positions), rotary_dim // 2)
    np.testing.assert_allclose(result["output"], reference, atol=1e-5)
