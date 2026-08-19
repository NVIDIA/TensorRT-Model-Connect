# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""locateanything model-owned E2E comparator plugins."""

from __future__ import annotations

from .contracts import (
    CompareResult,
    MetricResult,
    StageOutput,
    StageSpec,
    StageStatus,
    ThresholdProfile,
)
from .comparators.vision_language import VisionLanguageComparator
from tensorrt_model_connect.models.locateanything.task_contract import (
    matched_box_iou,
    matched_point_distance,
    parse_localizations,
)


class LocateanythingVisionLanguageGenerationComparator(VisionLanguageComparator):
    """locateanything local comparator for vision_language_generation."""

    def _compare_generation(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        """Require the grounded answer and its box, not only shared words."""
        base = super()._compare_generation(trt, ref, threshold, stage)
        metrics = dict(base.metrics)

        trt_text = (trt.text or trt.data.get("generated_text") or "").strip()
        ref_text = (ref.text or ref.data.get("generated_text") or "").strip()
        trt_localizations = parse_localizations(trt_text, require_reference=True)
        ref_localizations = parse_localizations(ref_text, require_reference=True)

        metrics["localization_markup_present"] = MetricResult(
            value=1.0 if trt_localizations is not None else 0.0,
            threshold=1.0,
            operator="==",
            passed=trt_localizations is not None,
            note="TRT output must contain <ref> text and only valid box-or-point groups",
        )
        metrics["reference_localization_markup_present"] = MetricResult(
            value=1.0 if ref_localizations is not None else 0.0,
            threshold=1.0,
            operator="==",
            passed=ref_localizations is not None,
            note="reference output must contain <ref> text and only valid box-or-point groups",
        )

        type_match = (
            trt_localizations is not None
            and ref_localizations is not None
            and trt_localizations[0].kind == ref_localizations[0].kind
        )
        metrics["localization_type_match"] = MetricResult(
            value=1.0 if type_match else 0.0,
            threshold=1.0,
            operator="==",
            passed=type_match,
            note="TRT and reference localization outputs must both be boxes or both be points",
        )

        count_match = (
            type_match
            and trt_localizations is not None
            and ref_localizations is not None
            and len(trt_localizations) == len(ref_localizations)
        )
        count_metric = MetricResult(
            value=1.0 if count_match else 0.0,
            threshold=1.0,
            operator="==",
            passed=count_match,
            note="TRT and reference must contain the same number of localizations",
        )
        metrics["localization_count_match"] = count_metric
        # Keep the original metric name for existing box dashboards and thresholds.
        metrics["localization_box_count_match"] = count_metric

        geometry_metric = "localization_box_iou"
        if type_match and ref_localizations is not None and ref_localizations[0].kind == "point":
            distance = (
                matched_point_distance(trt_localizations or (), ref_localizations)
                if count_match
                else float("inf")
            )
            distance_threshold = threshold.metrics.get("localization_point_distance", 10.0)
            geometry_metric = "localization_point_distance"
            metrics[geometry_metric] = MetricResult(
                value=distance,
                threshold=distance_threshold,
                operator="<=",
                passed=distance <= distance_threshold,
            )
        else:
            iou = (
                matched_box_iou(trt_localizations or (), ref_localizations or ())
                if count_match
                else 0.0
            )
            iou_threshold = threshold.metrics.get("localization_box_iou", 0.9)
            metrics[geometry_metric] = MetricResult(
                value=iou,
                threshold=iou_threshold,
                operator=">=",
                passed=iou >= iou_threshold,
            )

        non_empty_ok = metrics["non_empty_output"].passed
        ned_metric = metrics.get("normalized_text_edit_distance")
        if ned_metric is None:
            ned_metric = MetricResult(
                value=1.0,
                threshold=threshold.metrics.get("normalized_text_edit_distance"),
                operator="<=",
                passed=False,
                note="text parity is unavailable for empty output",
            )
            metrics["normalized_text_edit_distance"] = ned_metric
        ned_ok = ned_metric.passed
        localization_ok = all(
            metrics[name].passed
            for name in (
                "localization_markup_present",
                "reference_localization_markup_present",
                "localization_type_match",
                "localization_count_match",
                geometry_metric,
            )
        )
        passed = non_empty_ok and ned_ok and localization_ok
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if passed else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule=(
                "non_empty_output AND normalized_text_edit_distance AND "
                "valid TRT/reference localization markup AND matching output type/count AND "
                f"{geometry_metric}"
            ),
            message=f"LocateAnything grounded generation compare: {'PASS' if passed else 'FAIL'}",
        )


comparator = LocateanythingVisionLanguageGenerationComparator()
