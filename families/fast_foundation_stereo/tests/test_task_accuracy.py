# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from families.fast_foundation_stereo.tests.accuracy import (
    aggregate_task_accuracy,
    scene_statistics,
)


def test_scene_statistics_use_only_valid_nonoccluded_pixels() -> None:
    truth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    valid = np.array([[True, True], [False, True]])
    reference = truth + np.array([[0.0, 1.0], [0.0, 3.0]], dtype=np.float32)
    candidate = truth + np.array([[0.0, 2.0], [0.0, 4.0]], dtype=np.float32)

    result = scene_statistics(candidate, reference, truth, valid)

    assert result["valid_nonocc_pixels"] == 3
    assert result["candidate_nonocc_abs_error_sum_px"] == 6
    assert result["reference_nonocc_abs_error_sum_px"] == 4
    assert result["candidate_nonocc_bad2_pixel_count"] == 1
    assert result["reference_nonocc_bad2_pixel_count"] == 1


def test_aggregate_is_pixel_weighted_and_reference_relative() -> None:
    cases = [
        {
            "valid_nonocc_pixels": 100,
            "candidate_nonocc_abs_error_sum_px": 60,
            "reference_nonocc_abs_error_sum_px": 20,
            "candidate_nonocc_bad2_pixel_count": 4,
            "reference_nonocc_bad2_pixel_count": 1,
        },
        {
            "valid_nonocc_pixels": 10,
            "candidate_nonocc_abs_error_sum_px": 20,
            "reference_nonocc_abs_error_sum_px": 10,
            "candidate_nonocc_bad2_pixel_count": 1,
            "reference_nonocc_bad2_pixel_count": 1,
        },
    ]

    result = aggregate_task_accuracy(
        cases,
        epe_allowance_px=0.5,
        bp2_allowance_fraction=0.03,
    )

    assert result["candidate_nonocc_epe_passed"] is True
    assert result["candidate_nonocc_bp2_passed"] is True
    assert result["valid_nonocc_pixels"] == 110
    assert result["candidate_nonocc_epe_px"] == 80 / 110
    assert result["reference_nonocc_epe_px"] == 30 / 110
    assert result["candidate_nonocc_bp2_fraction"] == 5 / 110
    assert result["reference_nonocc_bp2_fraction"] == 2 / 110


def test_aggregate_fails_when_candidate_consumes_more_than_approved_budget() -> None:
    case = {
        "valid_nonocc_pixels": 100,
        "candidate_nonocc_abs_error_sum_px": 80,
        "reference_nonocc_abs_error_sum_px": 20,
        "candidate_nonocc_bad2_pixel_count": 8,
        "reference_nonocc_bad2_pixel_count": 1,
    }

    result = aggregate_task_accuracy(
        [case],
        epe_allowance_px=0.5,
        bp2_allowance_fraction=0.03,
    )

    assert result["candidate_nonocc_epe_passed"] is False
    assert result["candidate_nonocc_bp2_passed"] is False
