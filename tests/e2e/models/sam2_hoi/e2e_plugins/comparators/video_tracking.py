# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict five-frame SAM2 HOI tracking comparator."""

from __future__ import annotations

from pathlib import Path

from tensorrt_model_connect.families.sam2_hoi.validation import (
    AccuracyThresholds,
    compare_npz,
)

from ..contracts import (
    CompareResult,
    MetricResult,
    StageOutput,
    StageSpec,
    StageStatus,
    ThresholdProfile,
)


_REQUIRED_THRESHOLDS = {
    "detection_score_max_abs",
    "detection_box_max_abs_pixels",
    "detection_box_min_iou",
    "mask_min_iou",
    "mask_min_dice",
    "mask_min_pixel_agreement",
    "exact_object_ids",
    "exact_det_labels",
    "exact_interaction_pairs",
    "required_frame_count",
}


def _output_npz(output: StageOutput, side: str) -> Path:
    value = output.data.get("output_npz")
    if not isinstance(value, str) or not value:
        raise ValueError(f"SAM2 HOI {side} output is missing output_npz")
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"SAM2 HOI {side} output NPZ does not exist: {path}")
    return path


def _thresholds(profile: ThresholdProfile) -> AccuracyThresholds:
    missing = sorted(_REQUIRED_THRESHOLDS - set(profile.metrics))
    if missing:
        raise ValueError("SAM2 HOI threshold sidecar is incomplete; missing " + ", ".join(missing))
    if profile.metrics["exact_object_ids"] != 1.0:
        raise ValueError("SAM2 HOI object IDs must use exact-match threshold 1.0")
    if profile.metrics["exact_det_labels"] != 1.0:
        raise ValueError("SAM2 HOI detection labels must use exact-match threshold 1.0")
    if profile.metrics["exact_interaction_pairs"] != 1.0:
        raise ValueError("SAM2 HOI interaction pairs must use exact-match threshold 1.0")
    if profile.metrics["required_frame_count"] != 5.0:
        raise ValueError("SAM2 HOI comparison must require all five frames")
    return AccuracyThresholds(
        detection_score_max_abs=profile.metrics["detection_score_max_abs"],
        detection_box_max_abs_pixels=profile.metrics["detection_box_max_abs_pixels"],
        detection_box_min_iou=profile.metrics["detection_box_min_iou"],
        mask_min_iou=profile.metrics["mask_min_iou"],
        mask_min_dice=profile.metrics["mask_min_dice"],
        mask_min_pixel_agreement=profile.metrics["mask_min_pixel_agreement"],
    )


class HoiVideoTrackingComparator:
    """Apply the family accuracy contract without fallback tolerances."""

    @property
    def task_strategy(self) -> str:
        return "hoi_video_tracking"

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        try:
            gates = _thresholds(threshold)
            report = compare_npz(
                _output_npz(ref, "reference"),
                _output_npz(trt, "TensorRT"),
                thresholds=gates,
            )
        except Exception as error:
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                message=f"SAM2 HOI comparison error: {error}",
            )

        rows = report["frames"]
        exact_fields = ("object_ids", "det_labels", "interaction_pairs")
        exact_values = {
            field: 1.0 if all(bool(row["gates"][field]) for row in rows) else 0.0
            for field in exact_fields
        }
        values = {
            "detection_score_max_abs": max(float(row["detection_score_max_abs"]) for row in rows),
            "detection_box_max_abs_pixels": max(
                float(row["detection_box_max_abs_pixels"]) for row in rows
            ),
            "detection_box_min_iou": min(float(row["detection_box_min_iou"]) for row in rows),
            "mask_min_iou": min(float(row["min_iou"]) for row in rows),
            "mask_min_dice": min(float(row["min_dice"]) for row in rows),
            "mask_min_pixel_agreement": min(float(row["min_pixel_agreement"]) for row in rows),
            "exact_object_ids": exact_values["object_ids"],
            "exact_det_labels": exact_values["det_labels"],
            "exact_interaction_pairs": exact_values["interaction_pairs"],
            "required_frame_count": float(len(rows)),
        }
        operators = {
            "detection_score_max_abs": "<=",
            "detection_box_max_abs_pixels": "<=",
            "detection_box_min_iou": ">=",
            "mask_min_iou": ">=",
            "mask_min_dice": ">=",
            "mask_min_pixel_agreement": ">=",
            "exact_object_ids": "==",
            "exact_det_labels": "==",
            "exact_interaction_pairs": "==",
            "required_frame_count": "==",
        }
        metrics: dict[str, MetricResult] = {}
        for name, value in values.items():
            limit = threshold.metrics[name]
            operator = operators[name]
            if operator == "<=":
                passed = value <= limit
            elif operator == ">=":
                passed = value >= limit
            else:
                passed = value == limit
            metrics[name] = MetricResult(
                value=value,
                threshold=limit,
                operator=operator,
                passed=passed,
            )

        passed = report["status"] == "pass" and all(metric.passed for metric in metrics.values())
        failures = report.get("failures", [])
        return CompareResult(
            stage_name=stage.name,
            status=(StageStatus.PASSED.value if passed else StageStatus.FAILED.value),
            metrics=metrics,
            composite_rule=(
                "all five frames AND exact IDs/labels/interaction pairs AND score/box/mask gates"
            ),
            message=(
                "SAM2 HOI five-frame contract verified"
                if passed
                else "SAM2 HOI parity failed: " + "; ".join(str(item) for item in failures)
            ),
        )


plugin = HoiVideoTrackingComparator()
