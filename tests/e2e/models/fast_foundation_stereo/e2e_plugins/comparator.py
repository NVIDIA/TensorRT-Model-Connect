# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fast Foundation Stereo numeric disparity comparator."""

from __future__ import annotations

import numpy as np

from tests.e2e_harness.contracts import (
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
        bad_2px_threshold = threshold.metrics.get("bad_2px_fraction", 0.03)
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
        ground_truth = trt.data.get("ground_truth_disparity")
        valid_mask = trt.data.get("valid_nonocc_mask")
        if ground_truth is not None or valid_mask is not None:
            if ground_truth is None or valid_mask is None:
                return CompareResult(
                    stage_name=stage.name,
                    status=StageStatus.ERROR.value,
                    message="Task-accuracy output has incomplete ground-truth evidence",
                )
            truth = np.asarray(ground_truth, dtype=np.float32)
            valid = np.asarray(valid_mask, dtype=bool)
            if truth.shape != actual.shape or valid.shape != actual.shape or not valid.any():
                return CompareResult(
                    stage_name=stage.name,
                    status=StageStatus.ERROR.value,
                    message="Task-accuracy ground truth/mask does not match disparity output",
                )
            if not np.isfinite(truth[valid]).all():
                return CompareResult(
                    stage_name=stage.name,
                    status=StageStatus.ERROR.value,
                    message="Task-accuracy ground truth is non-finite on valid pixels",
                )
            candidate_error = np.abs(actual[valid].astype(np.float64) - truth[valid])
            reference_error = np.abs(expected[valid].astype(np.float64) - truth[valid])
            valid_count = int(valid.sum())
            task_values = {
                "valid_nonocc_pixels": float(valid_count),
                "candidate_nonocc_abs_error_sum_px": float(candidate_error.sum()),
                "reference_nonocc_abs_error_sum_px": float(reference_error.sum()),
                "candidate_nonocc_bad2_pixel_count": float((candidate_error > 2.0).sum()),
                "reference_nonocc_bad2_pixel_count": float((reference_error > 2.0).sum()),
                "candidate_nonocc_epe_px": float(candidate_error.mean()),
                "reference_nonocc_epe_px": float(reference_error.mean()),
                "candidate_nonocc_bp2_fraction": float(np.mean(candidate_error > 2.0)),
                "reference_nonocc_bp2_fraction": float(np.mean(reference_error > 2.0)),
            }
            metrics.update(
                {
                    name: MetricResult(
                        value=value,
                        threshold=None,
                        operator="informational",
                        passed=True,
                        note="Per-scene task statistic; gates are pixel-weighted over all scenes",
                    )
                    for name, value in task_values.items()
                }
            )
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

    def aggregate(self, cases: list[dict], gates: dict) -> dict:
        """Apply the approved pixel-weighted task gates across prepared scenes."""
        epe_allowance_key = "candidate_nonocc_epe_max_reference_plus_px"
        bp2_allowance_key = "candidate_nonocc_bp2_max_reference_plus_fraction"
        requested = epe_allowance_key in gates or bp2_allowance_key in gates
        if not requested:
            return {"evaluated": False, "passed": True}
        if epe_allowance_key not in gates or bp2_allowance_key not in gates:
            return {
                "evaluated": True,
                "passed": False,
                "gate_failures": ["task-accuracy workload must configure both EPE and BP-2 gates"],
            }

        required = (
            "valid_nonocc_pixels",
            "candidate_nonocc_abs_error_sum_px",
            "reference_nonocc_abs_error_sum_px",
            "candidate_nonocc_bad2_pixel_count",
            "reference_nonocc_bad2_pixel_count",
        )
        missing = [
            str(case.get("sample_id", ""))
            for case in cases
            if any(name not in case.get("metrics", {}) for name in required)
        ]
        if missing:
            return {
                "evaluated": True,
                "passed": False,
                "gate_failures": [
                    "task-accuracy sufficient statistics are missing for: " + ", ".join(missing)
                ],
            }

        totals = {
            name: sum(float(case["metrics"][name]["value"]) for case in cases)
            for name in required
        }
        valid_pixels = totals["valid_nonocc_pixels"]
        if valid_pixels <= 0:
            return {
                "evaluated": True,
                "passed": False,
                "gate_failures": ["task-accuracy aggregate has no valid non-occluded pixels"],
            }
        task_accuracy = {
            "valid_nonocc_pixels": int(valid_pixels),
            "candidate_nonocc_epe_px": (
                totals["candidate_nonocc_abs_error_sum_px"] / valid_pixels
            ),
            "reference_nonocc_epe_px": (
                totals["reference_nonocc_abs_error_sum_px"] / valid_pixels
            ),
            "candidate_nonocc_bp2_fraction": (
                totals["candidate_nonocc_bad2_pixel_count"] / valid_pixels
            ),
            "reference_nonocc_bp2_fraction": (
                totals["reference_nonocc_bad2_pixel_count"] / valid_pixels
            ),
        }
        epe_limit = task_accuracy["reference_nonocc_epe_px"] + float(
            gates[epe_allowance_key]
        )
        bp2_limit = task_accuracy["reference_nonocc_bp2_fraction"] + float(
            gates[bp2_allowance_key]
        )
        epe_passed = task_accuracy["candidate_nonocc_epe_px"] <= epe_limit
        bp2_passed = task_accuracy["candidate_nonocc_bp2_fraction"] <= bp2_limit
        failures = []
        if not epe_passed:
            failures.append(
                "candidate pixel-weighted non-occluded EPE exceeds reference plus allowance"
            )
        if not bp2_passed:
            failures.append(
                "candidate pixel-weighted non-occluded BP-2 exceeds reference plus allowance"
            )
        return {
            "evaluated": True,
            "passed": epe_passed and bp2_passed,
            "task_accuracy": task_accuracy,
            "gates": {
                epe_allowance_key: float(gates[epe_allowance_key]),
                bp2_allowance_key: float(gates[bp2_allowance_key]),
                "candidate_nonocc_epe_px_max": epe_limit,
                "candidate_nonocc_bp2_fraction_max": bp2_limit,
            },
            "gate_results": {
                "candidate_nonocc_epe": epe_passed,
                "candidate_nonocc_bp2": bp2_passed,
            },
            "gate_failures": failures,
        }


comparator = StereoDisparityComparator()
