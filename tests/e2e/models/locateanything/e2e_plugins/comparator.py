# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""locateanything model-owned E2E comparator plugins."""

from __future__ import annotations

import re

from .contracts import (
    CompareResult,
    MetricResult,
    StageOutput,
    StageSpec,
    StageStatus,
    ThresholdProfile,
)
from .comparators.vision_language import VisionLanguageComparator


_REF_RE = re.compile(r"<ref>.+?</ref>", re.DOTALL)
_BOX_RE = re.compile(r"<box>((?:<[0-9]{1,4}>){4})</box>")
_COORD_RE = re.compile(r"<([0-9]{1,4})>")


def _localization_boxes(text: str) -> list[tuple[float, float, float, float]] | None:
    """Parse every valid LocateAnything box in the model's 0..1000 space."""
    matches = list(_BOX_RE.finditer(text))
    if (
        _REF_RE.search(text) is None
        or not matches
        or text.count("<box>") != len(matches)
        or text.count("</box>") != len(matches)
    ):
        return None

    boxes: list[tuple[float, float, float, float]] = []
    for match in matches:
        values = [int(value) for value in _COORD_RE.findall(match.group(1))]
        if len(values) != 4 or any(value < 0 or value > 1000 for value in values):
            return None
        x1, y1, x2, y2 = values
        if x2 <= x1 or y2 <= y1:
            return None
        boxes.append((float(x1), float(y1), float(x2), float(y2)))
    return boxes


def _box_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


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
        trt_boxes = _localization_boxes(trt_text)
        ref_boxes = _localization_boxes(ref_text)

        metrics["localization_markup_present"] = MetricResult(
            value=1.0 if trt_boxes is not None else 0.0,
            threshold=1.0,
            operator="==",
            passed=trt_boxes is not None,
            note="TRT output must contain <ref> text and only valid <box> groups",
        )
        metrics["reference_localization_markup_present"] = MetricResult(
            value=1.0 if ref_boxes is not None else 0.0,
            threshold=1.0,
            operator="==",
            passed=ref_boxes is not None,
            note="reference output must contain <ref> text and only valid <box> groups",
        )

        box_count_match = (
            trt_boxes is not None and ref_boxes is not None and len(trt_boxes) == len(ref_boxes)
        )
        metrics["localization_box_count_match"] = MetricResult(
            value=1.0 if box_count_match else 0.0,
            threshold=1.0,
            operator="==",
            passed=box_count_match,
            note="TRT and reference must contain the same number of boxes",
        )

        iou = (
            min(_box_iou(trt_box, ref_box) for trt_box, ref_box in zip(trt_boxes, ref_boxes))
            if box_count_match and trt_boxes and ref_boxes
            else 0.0
        )
        iou_threshold = threshold.metrics.get("localization_box_iou", 0.9)
        metrics["localization_box_iou"] = MetricResult(
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
                "localization_box_count_match",
                "localization_box_iou",
            )
        )
        passed = non_empty_ok and ned_ok and localization_ok
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if passed else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule=(
                "non_empty_output AND normalized_text_edit_distance AND "
                "valid TRT/reference localization markup AND matching box count AND "
                "localization_box_iou"
            ),
            message=f"LocateAnything grounded generation compare: {'PASS' if passed else 'FAIL'}",
        )


comparator = LocateanythingVisionLanguageGenerationComparator()
