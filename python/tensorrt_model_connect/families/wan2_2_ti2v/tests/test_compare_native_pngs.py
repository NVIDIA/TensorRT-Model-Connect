# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accuracy-gate tests for the native Wan2.2 qualification tool."""

from __future__ import annotations

import math

import numpy as np

from tensorrt_model_connect.families.wan2_2_ti2v.reference.compare_native_pngs import (
    DEFAULT_MAXIMUM_RMSE_UINT8,
    DEFAULT_MINIMUM_COSINE,
    _accuracy_failures,
    _cosine_from_sums,
)


def test_accuracy_gate_accepts_the_declared_boundary() -> None:
    assert DEFAULT_MINIMUM_COSINE == 0.998
    assert DEFAULT_MAXIMUM_RMSE_UINT8 == 1.0
    assert not _accuracy_failures(
        cosine=DEFAULT_MINIMUM_COSINE,
        minimum_frame_cosine=DEFAULT_MINIMUM_COSINE,
        rmse=DEFAULT_MAXIMUM_RMSE_UINT8,
        maximum_frame_rmse=DEFAULT_MAXIMUM_RMSE_UINT8,
        min_cosine=DEFAULT_MINIMUM_COSINE,
        min_frame_cosine=DEFAULT_MINIMUM_COSINE,
        max_rmse=DEFAULT_MAXIMUM_RMSE_UINT8,
    )


def test_accuracy_gate_rejects_video_or_frame_regressions() -> None:
    assert _accuracy_failures(
        cosine=0.997,
        minimum_frame_cosine=0.996,
        rmse=0.0,
        maximum_frame_rmse=0.0,
        min_cosine=DEFAULT_MINIMUM_COSINE,
        min_frame_cosine=DEFAULT_MINIMUM_COSINE,
        max_rmse=DEFAULT_MAXIMUM_RMSE_UINT8,
    ) == [
        "cosine_uint8=0.997000000000 < 0.998000000000",
        "minimum_frame_cosine_uint8=0.996000000000 < 0.998000000000",
    ]


def test_accuracy_gate_rejects_brightness_scaling_despite_perfect_cosine() -> None:
    reference = np.array([32.0, 64.0, 96.0], dtype=np.float64)
    brightness_scaled = reference * 1.5
    cosine = float(
        np.dot(reference, brightness_scaled)
        / (np.linalg.norm(reference) * np.linalg.norm(brightness_scaled))
    )
    rmse = float(np.sqrt(np.mean(np.square(brightness_scaled - reference))))

    assert math.isclose(cosine, 1.0)
    assert _accuracy_failures(
        cosine=cosine,
        minimum_frame_cosine=cosine,
        rmse=rmse,
        maximum_frame_rmse=rmse,
        min_cosine=DEFAULT_MINIMUM_COSINE,
        min_frame_cosine=DEFAULT_MINIMUM_COSINE,
        max_rmse=DEFAULT_MAXIMUM_RMSE_UINT8,
    ) == [
        f"rmse_uint8={rmse:.12f} > 1.000000000000",
        f"maximum_frame_rmse_uint8={rmse:.12f} > 1.000000000000",
    ]


def test_accuracy_gate_fails_closed_for_non_finite_metrics() -> None:
    assert _accuracy_failures(
        cosine=float("nan"),
        minimum_frame_cosine=float("inf"),
        rmse=float("nan"),
        maximum_frame_rmse=float("inf"),
        min_cosine=DEFAULT_MINIMUM_COSINE,
        min_frame_cosine=DEFAULT_MINIMUM_COSINE,
        max_rmse=DEFAULT_MAXIMUM_RMSE_UINT8,
    ) == [
        "cosine_uint8 is not finite: nan",
        "minimum_frame_cosine_uint8 is not finite: inf",
        "rmse_uint8 is not finite: nan",
        "maximum_frame_rmse_uint8 is not finite: inf",
    ]


def test_cosine_zero_vector_semantics_fail_one_sided_mismatch() -> None:
    assert _cosine_from_sums(0, 0, 0) == 1.0
    assert _cosine_from_sums(0, 0, 1) == 0.0
    assert _cosine_from_sums(0, 1, 0) == 0.0
    assert _accuracy_failures(
        cosine=0.0,
        minimum_frame_cosine=0.0,
        rmse=0.5,
        maximum_frame_rmse=0.5,
        min_cosine=DEFAULT_MINIMUM_COSINE,
        min_frame_cosine=DEFAULT_MINIMUM_COSINE,
        max_rmse=DEFAULT_MAXIMUM_RMSE_UINT8,
    ) == [
        "cosine_uint8=0.000000000000 < 0.998000000000",
        "minimum_frame_cosine_uint8=0.000000000000 < 0.998000000000",
    ]
