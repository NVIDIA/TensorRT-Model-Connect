# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from tests.e2e.models.fast_foundation_stereo.e2e_plugins.comparator import comparator
from tests.e2e_harness.contracts import StageOutput, StageSpec, StageStatus, ThresholdProfile


def _threshold() -> ThresholdProfile:
    return ThresholdProfile(
        task_strategy="stereo_disparity",
        metrics={
            "finite_fraction": 1.0,
            "nonnegative_fraction": 1.0,
            "global_cosine": 0.999,
            "mean_abs_error": 0.5,
            "bad_2px_fraction": 0.03,
        },
    )


def _metric_case(sample_id: str, metrics: dict[str, float]) -> dict:
    return {
        "sample_id": sample_id,
        "metrics": {name: {"value": value} for name, value in metrics.items()},
    }


def test_comparator_records_per_scene_nonoccluded_task_statistics() -> None:
    truth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    valid = np.array([[True, True], [False, True]])
    reference = truth + np.array([[0.0, 1.0], [0.0, 3.0]], dtype=np.float32)
    candidate = truth + np.array([[0.0, 2.0], [0.0, 4.0]], dtype=np.float32)

    result = comparator.compare(
        StageOutput(
            stage_name="full_inference",
            data={
                "disparity": candidate,
                "ground_truth_disparity": truth,
                "valid_nonocc_mask": valid,
            },
        ),
        StageOutput(
            stage_name="full_inference",
            data={"disparity": reference, "expected_shape": [2, 2]},
        ),
        _threshold(),
        StageSpec(name="full_inference", required=True),
    )

    assert result.status == StageStatus.FAILED.value  # Full-map parity deliberately fails.
    assert result.metrics["valid_nonocc_pixels"].value == 3
    assert result.metrics["candidate_nonocc_abs_error_sum_px"].value == 6
    assert result.metrics["reference_nonocc_abs_error_sum_px"].value == 4
    assert result.metrics["candidate_nonocc_bad2_pixel_count"].value == 1
    assert result.metrics["reference_nonocc_bad2_pixel_count"].value == 1


def test_aggregate_is_pixel_weighted_and_reference_relative() -> None:
    cases = [
        _metric_case(
            "large",
            {
                "valid_nonocc_pixels": 100,
                "candidate_nonocc_abs_error_sum_px": 60,
                "reference_nonocc_abs_error_sum_px": 20,
                "candidate_nonocc_bad2_pixel_count": 4,
                "reference_nonocc_bad2_pixel_count": 1,
            },
        ),
        _metric_case(
            "small",
            {
                "valid_nonocc_pixels": 10,
                "candidate_nonocc_abs_error_sum_px": 20,
                "reference_nonocc_abs_error_sum_px": 10,
                "candidate_nonocc_bad2_pixel_count": 1,
                "reference_nonocc_bad2_pixel_count": 1,
            },
        ),
    ]

    result = comparator.aggregate(
        cases,
        {
            "candidate_nonocc_epe_max_reference_plus_px": 0.5,
            "candidate_nonocc_bp2_max_reference_plus_fraction": 0.03,
        },
    )

    assert result["passed"] is True
    assert result["task_accuracy"]["valid_nonocc_pixels"] == 110
    assert result["task_accuracy"]["candidate_nonocc_epe_px"] == 80 / 110
    assert result["task_accuracy"]["reference_nonocc_epe_px"] == 30 / 110
    assert result["task_accuracy"]["candidate_nonocc_bp2_fraction"] == 5 / 110
    assert result["task_accuracy"]["reference_nonocc_bp2_fraction"] == 2 / 110


def test_aggregate_fails_when_candidate_consumes_more_than_approved_budget() -> None:
    case = _metric_case(
        "scene",
        {
            "valid_nonocc_pixels": 100,
            "candidate_nonocc_abs_error_sum_px": 80,
            "reference_nonocc_abs_error_sum_px": 20,
            "candidate_nonocc_bad2_pixel_count": 8,
            "reference_nonocc_bad2_pixel_count": 1,
        },
    )

    result = comparator.aggregate(
        [case],
        {
            "candidate_nonocc_epe_max_reference_plus_px": 0.5,
            "candidate_nonocc_bp2_max_reference_plus_fraction": 0.03,
        },
    )

    assert result["passed"] is False
    assert result["gate_results"] == {
        "candidate_nonocc_epe": False,
        "candidate_nonocc_bp2": False,
    }
