# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gate SAM2 five-frame tracking accuracy from the public-C-ABI receipt."""

from __future__ import annotations

import math

from tests.e2e_harness.contracts import (
    CompareResult,
    MetricResult,
    StageOutput,
    StageSpec,
    StageStatus,
    ThresholdProfile,
)

_PUBLIC_CORE_VARIANT = "public_sam2_1_small_with_synthetic_bbox_v1"
_PLAN_SECTIONS = [
    "engine_plan",
    "sam2_prompt_engine_plan",
    "sam2_recurrent_h1_engine_plan",
    "sam2_recurrent_h2_engine_plan",
    "sam2_recurrent_h3_engine_plan",
    "sam2_recurrent_h4_engine_plan",
]
_SYNTHETIC_BBOX = [136.0, 160.0, 952.0, 1120.0]
_FRAME_PIXELS = 1280 * 1088
_MIN_MASK_PIXELS = _FRAME_PIXELS // 1000


def _number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{label} must be finite")
    return result


class Sam2AccuracyComparator:
    @property
    def task_strategy(self) -> str:
        return "prompted_segmentation"

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        if ref.data.get("_invariant_only") is True:
            return self._compare_runtime_invariants(trt, stage)
        if trt.data.get("schema_version") != 1:
            raise RuntimeError("SAM2 public E2E receipt schema version drifted")
        if trt.data.get("golden_manifest_sha256") != ref.data.get("golden_manifest_sha256"):
            raise RuntimeError("SAM2 public E2E did not bind the provisioned golden evidence")

        accuracy = trt.data.get("accuracy", {})
        frame_iou = accuracy.get("frame_mask_iou", [])
        if not isinstance(frame_iou, list) or len(frame_iou) != 5:
            raise RuntimeError("SAM2 public E2E must report five frame IoUs")
        values = {
            "minimum_frame_mask_iou": min(_number(value, "frame_mask_iou") for value in frame_iou),
            "minimum_macro_mask_iou": _number(accuracy.get("macro_mask_iou"), "macro_mask_iou"),
            "minimum_global_mask_iou": _number(accuracy.get("global_mask_iou"), "global_mask_iou"),
            "minimum_bbox_iou": _number(accuracy.get("bbox_iou"), "bbox_iou"),
            "maximum_bbox_coordinate_error": _number(
                accuracy.get("bbox_max_coordinate_error"), "bbox_max_coordinate_error"
            ),
            "maximum_bbox_score_error": _number(
                accuracy.get("bbox_score_error"), "bbox_score_error"
            ),
            "label_exact": 1.0 if accuracy.get("label_exact") is True else 0.0,
        }
        metrics = {}
        for name, value in values.items():
            limit = threshold.metrics[name]
            maximum = name.startswith("maximum_")
            passed = value <= limit if maximum else value >= limit
            metrics[name] = MetricResult(
                value=value,
                threshold=limit,
                operator="<=" if maximum else ">=",
                passed=passed,
            )
        passed = all(metric.passed for metric in metrics.values())
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if passed else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule="all SAM2 semantic gates must pass",
            message=f"SAM2 five-frame public C ABI: {'PASS' if passed else 'FAIL'}",
        )

    @staticmethod
    def _compare_runtime_invariants(trt: StageOutput, stage: StageSpec) -> CompareResult:
        data = trt.data.get("runtime_invariants", {})
        counts = data.get("mask_foreground_pixels", [])
        hashes = data.get("mask_sha256", [])
        valid_counts = (
            isinstance(counts, list)
            and len(counts) == 5
            and all(
                isinstance(value, int)
                and _MIN_MASK_PIXELS <= value <= _FRAME_PIXELS - _MIN_MASK_PIXELS
                for value in counts
            )
        )
        valid_hashes = (
            isinstance(hashes, list)
            and len(hashes) == 5
            and all(
                isinstance(value, str)
                and len(value) == 64
                and all(character in "0123456789abcdef" for character in value)
                for value in hashes
            )
        )
        checks = {
            "schema_version": trt.data.get("schema_version") == 2,
            "six_plan_bundle": data.get("plan_sections") == _PLAN_SECTIONS,
            "public_core_variant": data.get("checkpoint_variant") == _PUBLIC_CORE_VARIANT,
            "same_session_repeat_exact": data.get("same_session_repeat_exact") is True,
            "synthetic_bbox_exact": data.get("bbox_xyxy") == _SYNTHETIC_BBOX,
            "synthetic_detector_score_exact": data.get("detector_score") == 1.0,
            "synthetic_detector_label_exact": data.get("label") == 1,
            "binary_masks": data.get("binary_masks") is True,
            "five_mask_counts": valid_counts,
            "five_mask_receipts": valid_hashes,
            "temporally_distinct_masks": valid_hashes and len(set(hashes)) > 1,
        }
        metrics = {
            name: MetricResult(
                value=1.0 if passed else 0.0,
                threshold=1.0,
                operator="==",
                passed=passed,
            )
            for name, passed in checks.items()
        }
        passed = all(checks.values())
        failed = sorted(name for name, check_passed in checks.items() if not check_passed)
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if passed else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule=(
                "pinned public core variant AND exact six-plan bundle AND same-session exact "
                "repeat AND fixed synthetic detector contract AND five finite binary masks"
            ),
            message=(
                "SAM2 secretless six-plan runtime smoke: "
                f"{'PASS' if passed else 'FAIL (' + ', '.join(failed) + ')'}"
            ),
        )


comparator = Sam2AccuracyComparator()
