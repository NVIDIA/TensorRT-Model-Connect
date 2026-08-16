# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fast Foundation Stereo numeric disparity comparator."""

from __future__ import annotations

import numpy as np

from .contracts import (
    CompareResult,
    MetricResult,
    StageOutput,
    StageSpec,
    StageStatus,
    ThresholdProfile,
)


class StereoDisparityComparator:
    @property
    def task_strategy(self) -> str:
        return "stereo_disparity"

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        disparity = trt.data.get("disparity")
        if disparity is None:
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                message="Native runtime produced no disparity tensor",
            )
        actual = np.asarray(disparity, dtype=np.float32)
        reference = ref.data.get("disparity")
        if reference is None:
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                message="Official PyTorch reference produced no disparity tensor",
            )
        expected = np.asarray(reference, dtype=np.float32)
        expected_shape = tuple(ref.data.get("expected_shape", (700, 700)))
        shape_passed = actual.shape == expected.shape == expected_shape
        finite_fraction = float(np.isfinite(actual).mean())
        nonnegative_fraction = float((actual >= 0).mean())
        values_are_finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
        if shape_passed and values_are_finite:
            actual64 = actual.astype(np.float64, copy=False).reshape(-1)
            expected64 = expected.astype(np.float64, copy=False).reshape(-1)
            denominator = np.linalg.norm(actual64) * np.linalg.norm(expected64)
            cosine = (
                float(np.dot(actual64, expected64) / denominator)
                if denominator
                else float(np.array_equal(actual64, expected64))
            )
            absolute_error = np.abs(actual64 - expected64)
            mean_abs_error = float(np.mean(absolute_error))
            bad_2px_fraction = float(np.mean(absolute_error > 2.0))
        else:
            cosine = 0.0
            mean_abs_error = float("inf")
            bad_2px_fraction = 1.0
        finite_threshold = threshold.metrics.get("finite_fraction", 1.0)
        nonnegative_threshold = threshold.metrics.get("nonnegative_fraction", 1.0)
        cosine_threshold = threshold.metrics.get("global_cosine", 0.999)
        mean_abs_error_threshold = threshold.metrics.get("mean_abs_error", 0.5)
        bad_2px_threshold = threshold.metrics.get("bad_2px_fraction", 0.02)
        metrics = {
            "shape": MetricResult(
                value=1.0 if shape_passed else 0.0,
                threshold=1.0,
                operator="==",
                passed=shape_passed,
            ),
            "finite_fraction": MetricResult(
                value=finite_fraction,
                threshold=finite_threshold,
                operator=">=",
                passed=finite_fraction >= finite_threshold,
            ),
            "nonnegative_fraction": MetricResult(
                value=nonnegative_fraction,
                threshold=nonnegative_threshold,
                operator=">=",
                passed=nonnegative_fraction >= nonnegative_threshold,
            ),
            "global_cosine": MetricResult(
                value=cosine,
                threshold=cosine_threshold,
                operator=">=",
                passed=cosine >= cosine_threshold,
            ),
            "mean_abs_error": MetricResult(
                value=mean_abs_error,
                threshold=mean_abs_error_threshold,
                operator="<=",
                passed=mean_abs_error <= mean_abs_error_threshold,
            ),
            "bad_2px_fraction": MetricResult(
                value=bad_2px_fraction,
                threshold=bad_2px_threshold,
                operator="<=",
                passed=bad_2px_fraction <= bad_2px_threshold,
            ),
        }
        passed = all(metric.passed for metric in metrics.values())
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if passed else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule="shape, disparity invariants, cosine, EPE, and bad-2px must pass",
            message=(
                f"disparity shape={list(actual.shape)}, global cosine={cosine:.9f}, "
                f"mean absolute error={mean_abs_error:.6f}, "
                f"bad-2px fraction={bad_2px_fraction:.6f}"
            ),
        )


comparator = StereoDisparityComparator()
