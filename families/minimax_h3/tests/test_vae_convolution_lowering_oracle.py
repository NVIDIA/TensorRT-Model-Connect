# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host diagnostic guards, not trained-model or GPU qualification."""

import numpy as np
import pytest

from families.minimax_h3.tests.vae_convolution_lowering_oracle import (
    CASES,
    VARIANTS,
    ideal_two_round,
    voxel_rows,
)


def test_convolution_lowering_scope_is_explicit_and_unique():
    assert len(CASES) == len(set(CASES)) == 37
    assert sum(case.startswith("inspect-") for case in CASES) == 12
    assert sum(case.startswith("framework-") for case in CASES) == 6
    assert sum(any(case.startswith(v + "-") for v in VARIANTS) for case in CASES) == 18


def test_convolution_voxel_rows_preserve_actual_channel_and_spatial_order():
    values = np.arange(2 * 24 * 3 * 2 * 4).reshape(2, 24, 3, 2, 4)
    rows = voxel_rows(values)
    assert rows.shape == (2, 24, 24)
    for batch in range(2):
        for index, (t, y, x) in enumerate(np.ndindex(3, 2, 4)):
            np.testing.assert_array_equal(rows[batch, index], values[batch, :, t, y, x])


def test_ideal_diagnostic_uses_half_operands_and_separate_bias_rounding():
    values = (np.arange(96).reshape(1, 24, 1, 2, 2) / 19).astype(np.float16)
    weight = np.eye(24, dtype=np.float16).reshape(24, 24, 1, 1, 1)
    bias = np.linspace(-0.01, 0.01, 24).astype(np.float16)
    pre, post = ideal_two_round(values, weight, bias)
    np.testing.assert_array_equal(pre, voxel_rows(values))
    np.testing.assert_array_equal(post, (voxel_rows(values) + bias).astype(np.float16))
    assert pre.dtype == post.dtype == np.float16
    with pytest.raises(AssertionError):
        ideal_two_round(values.astype(np.float32), weight, bias)
