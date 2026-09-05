# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerical boundaries retained by the slim MoGe output graph."""

from __future__ import annotations

import numpy as np


def test_slim_graph_retains_legacy_mask_rounding_and_ieee_finite_edges() -> None:
    tiny_positive = np.nextafter(np.float16(0.0), np.float16(1.0))
    logits = np.asarray([-np.inf, -0.0, tiny_positive, np.inf, np.nan], dtype=np.float16)
    with np.errstate(over="ignore", invalid="ignore"):
        probabilities = np.asarray(
            1.0 / (1.0 + np.exp(-logits.astype(np.float32))),
            dtype=np.float16,
        )
    legacy_selected = np.isfinite(probabilities) & (probabilities > np.float16(0.5))
    ordered_probability_selected = probabilities > np.float16(0.5)
    logit_selected = logits > np.float16(0.0)

    np.testing.assert_array_equal(
        legacy_selected,
        np.asarray([False, False, False, True, False]),
    )
    np.testing.assert_array_equal(ordered_probability_selected, legacy_selected)
    np.testing.assert_array_equal(
        logit_selected,
        np.asarray([False, False, True, True, False]),
    )
    assert probabilities[2] == np.float16(0.5)

    values = np.asarray(
        [-np.finfo(np.float16).max, np.finfo(np.float16).max, -np.inf, np.inf, np.nan],
        dtype=np.float16,
    )
    with np.errstate(invalid="ignore"):
        ordered_finite = np.abs(values) < np.float16(np.inf)
    np.testing.assert_array_equal(ordered_finite, np.isfinite(values))


def test_fp16_valid_output_uses_exact_zero_and_one_bit_patterns() -> None:
    valid = np.asarray([False, True], dtype=np.bool_).astype(np.float16)
    np.testing.assert_array_equal(
        valid.view(np.uint16),
        np.asarray([0x0000, 0x3C00], np.uint16),
    )


def test_slim_sample_gather_preserves_the_legacy_fp16_cast_boundary() -> None:
    height, width = 67, 83
    affine_nchw = np.arange(3 * height * width, dtype=np.float32).reshape(1, 3, height, width)
    affine_nchw = np.asarray(affine_nchw / 257.0, dtype=np.float16)
    rows = np.asarray([index * height // 64 for index in range(64)])
    columns = np.asarray([index * width // 64 for index in range(64)])

    legacy_nhwc = np.transpose(affine_nchw, (0, 2, 3, 1)).astype(np.float32)
    legacy_samples = legacy_nhwc[:, rows, :, :][:, :, columns, :]
    sampled_nchw = affine_nchw[:, :, rows, :][:, :, :, columns]
    slim_samples = np.transpose(sampled_nchw, (0, 2, 3, 1)).astype(np.float32)
    np.testing.assert_array_equal(slim_samples, legacy_samples)

    legacy_depth = legacy_nhwc[..., 2]
    slim_depth = affine_nchw[:, 2, :, :].astype(np.float32)
    np.testing.assert_array_equal(slim_depth, legacy_depth)
