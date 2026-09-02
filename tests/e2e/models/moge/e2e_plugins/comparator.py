# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measured FP32 parity comparator for native MoGe geometry."""

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

_OPERATORS = {
    "mask_iou": ">=",
    "depth_absrel_mean": "<=",
    "depth_rel_l2": "<=",
    "points_rel_l2": "<=",
    "points_cosine": ">=",
    "intrinsics_max_relative_error": "<=",
    "point_depth_consistency": "<=",
}


def _geometry(data: dict, label: str) -> tuple[np.ndarray, ...]:
    missing = [name for name in ("points", "depth", "mask", "intrinsics") if name not in data]
    if missing:
        raise ValueError(f"{label} output is missing {missing}")
    if int(data.get("num_tokens", 0)) != 1800:
        raise ValueError(f"{label} output does not use the fixed 1800-token contract")

    points = np.asarray(data["points"], dtype=np.float32)
    depth = np.asarray(data["depth"], dtype=np.float32)
    mask = np.asarray(data["mask"])
    intrinsics = np.asarray(data["intrinsics"], dtype=np.float32)
    if depth.ndim != 2 or points.shape != (*depth.shape, 3) or mask.shape != depth.shape:
        raise ValueError(f"{label} geometry shapes are inconsistent")
    if not mask.size or not np.isin(mask, (0, 1)).all() or not np.any(mask):
        raise ValueError(f"{label} mask is not a non-empty binary validity map")

    valid = mask.astype(bool, copy=False)
    if (
        not np.isfinite(points[valid]).all()
        or not np.isfinite(depth[valid]).all()
        or not np.all(depth[valid] > 0.0)
    ):
        raise ValueError(f"{label} valid geometry is not finite and positive")
    invalid = ~valid
    if np.any(invalid) and (
        not np.isposinf(points[invalid]).all() or not np.isposinf(depth[invalid]).all()
    ):
        raise ValueError(f"{label} invalid geometry does not use positive infinity")

    expected_last_row = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    if (
        intrinsics.shape != (3, 3)
        or not np.isfinite(intrinsics).all()
        or intrinsics[0, 0] <= 0.0
        or intrinsics[1, 1] <= 0.0
        or not np.array_equal(intrinsics[2], expected_last_row)
        or intrinsics[0, 1] != 0.0
        or intrinsics[1, 0] != 0.0
        or intrinsics[0, 2] != 0.5
        or intrinsics[1, 2] != 0.5
    ):
        raise ValueError(f"{label} intrinsics violate the normalized camera contract")
    return points, depth, mask, intrinsics


def _numeric_metrics(
    trt_geometry: tuple[np.ndarray, ...],
    ref_geometry: tuple[np.ndarray, ...],
) -> dict[str, float]:
    trt_points, trt_depth, trt_mask, trt_intrinsics = trt_geometry
    ref_points, ref_depth, ref_mask, ref_intrinsics = ref_geometry
    trt_valid = trt_mask.astype(bool, copy=False)
    ref_valid = ref_mask.astype(bool, copy=False)
    common = trt_valid & ref_valid
    union = trt_valid | ref_valid
    if not np.any(common):
        return {
            "mask_iou": 0.0,
            "depth_absrel_mean": np.inf,
            "depth_rel_l2": np.inf,
            "points_rel_l2": np.inf,
            "points_cosine": -np.inf,
            "intrinsics_max_relative_error": np.inf,
            "point_depth_consistency": np.inf,
        }

    trt_depth_valid = trt_depth[common].astype(np.float64)
    ref_depth_valid = ref_depth[common].astype(np.float64)
    depth_delta = trt_depth_valid - ref_depth_valid
    trt_points_valid = trt_points[common].astype(np.float64)
    ref_points_valid = ref_points[common].astype(np.float64)
    point_delta = trt_points_valid - ref_points_valid
    point_cosine_denominator = np.maximum(
        np.linalg.norm(trt_points_valid, axis=-1)
        * np.linalg.norm(ref_points_valid, axis=-1),
        1.0e-12,
    )
    reference_intrinsics = ref_intrinsics.astype(np.float64)
    intrinsics_delta = trt_intrinsics.astype(np.float64) - reference_intrinsics
    nonzero = np.abs(reference_intrinsics) > 1.0e-12
    return {
        "mask_iou": float(common.sum() / union.sum()),
        "depth_absrel_mean": float(
            np.mean(np.abs(depth_delta) / np.maximum(np.abs(ref_depth_valid), 1.0e-12))
        ),
        "depth_rel_l2": float(
            np.linalg.norm(depth_delta) / max(float(np.linalg.norm(ref_depth_valid)), 1.0e-12)
        ),
        "points_rel_l2": float(
            np.linalg.norm(point_delta)
            / max(float(np.linalg.norm(ref_points_valid)), 1.0e-12)
        ),
        "points_cosine": float(
            np.mean(
                np.sum(trt_points_valid * ref_points_valid, axis=-1)
                / point_cosine_denominator
            )
        ),
        "intrinsics_max_relative_error": float(
            np.max(
                np.abs(intrinsics_delta[nonzero])
                / np.abs(reference_intrinsics[nonzero])
            )
        ),
        "point_depth_consistency": float(
            np.max(np.abs(trt_points[..., 2][trt_valid] - trt_depth[trt_valid]))
        ),
    }


def _metric(value: float, threshold: float, operator: str) -> MetricResult:
    passed = np.isfinite(value) and (
        value <= threshold if operator == "<=" else value >= threshold
    )
    return MetricResult(value=value, threshold=threshold, operator=operator, passed=passed)


class MonocularGeometryComparator:
    @property
    def task_strategy(self) -> str:
        return "monocular_geometry"

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        try:
            trt_geometry = _geometry(trt.data, "TRT")
            ref_geometry = _geometry(ref.data, "reference")
            if trt_geometry[1].shape != ref_geometry[1].shape:
                raise ValueError("TRT and reference geometry shapes differ")
            missing_thresholds = set(_OPERATORS) - set(threshold.metrics)
            if missing_thresholds:
                raise ValueError(
                    f"MoGe numerical thresholds are missing {sorted(missing_thresholds)}"
                )
        except (TypeError, ValueError) as error:
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                message=f"Invalid MoGe geometry contract: {error}",
            )

        numeric = _numeric_metrics(trt_geometry, ref_geometry)
        metrics = {
            name: _metric(numeric[name], float(threshold.metrics[name]), operator)
            for name, operator in _OPERATORS.items()
        }
        passed = all(metric.passed for metric in metrics.values())
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if passed else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule="all geometry contracts and numerical parity thresholds must pass",
            message=(
                "MoGe native FP32 geometry parity passed"
                if passed
                else "MoGe native FP32 geometry parity failed"
            ),
        )


comparator = MonocularGeometryComparator()
