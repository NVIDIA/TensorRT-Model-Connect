# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pose parity, ranking, rigidity, and tracking-rate qualification gates."""

from __future__ import annotations

import numpy as np

from tests.e2e_harness.contracts import CompareResult, MetricResult, StageOutput, StageSpec, StageStatus, ThresholdProfile

_OPERATORS = {
    "pose_max_abs_error": "<=", "score_max_abs_error": "<=",
    "ranking_agreement": ">=", "rigid_pose_fraction": ">=",
    "tracking_throughput_hz": ">=", "tracking_latency_p95_ms": "<=",
    "tracking_jitter_ms": "<=", "startup_ms": "<=", "gpu_memory_delta_mib": "<=",
}


def _rigid_fraction(poses: np.ndarray) -> float:
    valid = []
    for pose in poses:
        rotation = pose[:3, :3].astype(np.float64)
        valid.append(
            np.isfinite(pose).all()
            and np.max(np.abs(rotation @ rotation.T - np.eye(3))) <= 2.0e-3
            and abs(np.linalg.det(rotation) - 1.0) <= 6.0e-3
            and np.max(np.abs(pose[3] - np.array([0, 0, 0, 1], dtype=np.float32))) <= 2.0e-3
        )
    return float(np.mean(valid))


def _ranking_agreement(scores: np.ndarray, reference_scores: np.ndarray) -> float:
    candidate_order = np.argsort(-scores, kind="stable")
    reference_order = np.argsort(-reference_scores, kind="stable")
    return float(int(np.array_equal(candidate_order, reference_order)))


class FoundationPoseComparator:
    @property
    def task_strategy(self) -> str:
        return "pose_hypothesis_refinement"

    def compare(self, trt: StageOutput, ref: StageOutput, threshold: ThresholdProfile,
                stage: StageSpec) -> CompareResult:
        try:
            poses = np.asarray(trt.data["refined_poses"], dtype=np.float32)
            reference_poses = np.asarray(ref.data["refined_poses"], dtype=np.float32)
            scores = np.asarray(trt.data["scores"], dtype=np.float32)
            reference_scores = np.asarray(ref.data["scores"], dtype=np.float32)
            summary = trt.data["summary"]
            count = summary["num_hypotheses"]
            if type(count) is not int or not 1 <= count <= 252:
                raise ValueError("qualified hypothesis count must be in [1, 252]")
            if poses.shape != (count, 4, 4) or reference_poses.shape != poses.shape:
                raise ValueError(f"qualified pose arrays must have shape ({count}, 4, 4)")
            if scores.shape != (count,) or reference_scores.shape != scores.shape:
                raise ValueError(f"qualified score arrays must have shape ({count},)")
            if not all(name in threshold.metrics for name in _OPERATORS):
                raise ValueError("FoundationPose threshold profile is incomplete")
            numeric = {
                "pose_max_abs_error": float(np.max(np.abs(poses - reference_poses))),
                "score_max_abs_error": float(np.max(np.abs(scores - reference_scores))),
                "ranking_agreement": _ranking_agreement(scores, reference_scores),
                "rigid_pose_fraction": _rigid_fraction(poses),
                **{name: float(summary[name]) for name in (
                    "tracking_throughput_hz", "tracking_latency_p95_ms", "tracking_jitter_ms",
                    "startup_ms", "gpu_memory_delta_mib")},
            }
        except (KeyError, TypeError, ValueError) as error:
            return CompareResult(stage_name=stage.name, status=StageStatus.ERROR.value,
                                 message=f"Invalid FoundationPose result contract: {error}")
        metrics = {}
        for name, operator in _OPERATORS.items():
            value = numeric[name]
            limit = float(threshold.metrics[name])
            passed = np.isfinite(value) and (value <= limit if operator == "<=" else value >= limit)
            metrics[name] = MetricResult(value=value, threshold=limit, operator=operator, passed=bool(passed))
        passed = all(metric.passed for metric in metrics.values())
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if passed else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule="all pose parity, score, ranking, rigidity, and 10 Hz tracking gates must pass",
            message="FoundationPose qualification passed" if passed else "FoundationPose qualification failed",
        )


comparator = FoundationPoseComparator()
