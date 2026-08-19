# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from tests.e2e_harness.contracts import StageOutput, StageSpec, StageStatus, ThresholdProfile
from tensorrt_model_connect.models.locateanything.tests.e2e_plugins.comparator import (
    LocateanythingVisionLanguageGenerationComparator,
)


def _compare(trt_text: str, ref_text: str):
    comparator = LocateanythingVisionLanguageGenerationComparator()
    return comparator.compare(
        StageOutput(stage_name="full_generation", text=trt_text),
        StageOutput(stage_name="full_generation", text=ref_text),
        ThresholdProfile(
            task_strategy="vision_language_generation",
            metrics={
                "normalized_text_edit_distance": 0.5,
                "token_agreement_rate": 0.3,
                "localization_box_iou": 0.9,
                "localization_point_distance": 10.0,
            },
        ),
        StageSpec(name="full_generation"),
    )


def test_grounding_comparator_rejects_nightly_output_without_box() -> None:
    result = _compare(
        "red vehicle in this image",
        "<ref>red vehicle in this image</ref><box><304><267><828><708></box>",
    )

    assert result.status == StageStatus.FAILED.value
    assert result.metrics["token_agreement_rate"].passed
    assert not result.metrics["localization_markup_present"].passed


def test_grounding_comparator_accepts_matching_grounded_output() -> None:
    output = "<ref>red vehicle in this image</ref><box><304><267><828><708></box>"
    result = _compare(output, output)

    assert result.status == StageStatus.PASSED.value
    assert result.metrics["localization_box_iou"].value == 1.0


def test_grounding_comparator_rejects_wrong_box() -> None:
    result = _compare(
        "<ref>red vehicle in this image</ref><box><0><0><100><100></box>",
        "<ref>red vehicle in this image</ref><box><304><267><828><708></box>",
    )

    assert result.status == StageStatus.FAILED.value
    assert not result.metrics["localization_box_iou"].passed


def test_grounding_comparator_rejects_missing_second_box() -> None:
    result = _compare(
        "<ref>vehicles</ref><box><304><267><828><708></box>",
        ("<ref>vehicles</ref><box><304><267><828><708></box><box><20><30><100><120></box>"),
    )

    assert result.status == StageStatus.FAILED.value
    assert not result.metrics["localization_box_count_match"].passed


def test_grounding_comparator_matches_multiple_boxes_without_ordering() -> None:
    result = _compare(
        "<ref>vehicles</ref><box><300><400><500><600></box><box><10><20><110><220></box>",
        "<ref>vehicles</ref><box><10><20><110><220></box><box><300><400><500><600></box>",
    )

    assert result.status == StageStatus.PASSED.value
    assert result.metrics["localization_box_iou"].value == 1.0


def test_grounding_comparator_reports_empty_output_as_failure() -> None:
    result = _compare(
        "",
        "<ref>white vehicle</ref><box><304><267><828><708></box>",
    )

    assert result.status == StageStatus.FAILED.value
    assert not result.metrics["non_empty_output"].passed
    assert not result.metrics["normalized_text_edit_distance"].passed


def test_grounding_comparator_accepts_matching_point_output() -> None:
    result = _compare(
        "<ref>search button</ref><box><504><252></box>",
        "<ref>search button</ref><box><500><250></box>",
    )

    assert result.status == StageStatus.PASSED.value
    assert result.metrics["localization_type_match"].passed
    assert result.metrics["localization_point_distance"].value < 5.0


def test_grounding_comparator_rejects_distant_point() -> None:
    result = _compare(
        "<ref>search button</ref><box><700><600></box>",
        "<ref>search button</ref><box><500><250></box>",
    )

    assert result.status == StageStatus.FAILED.value
    assert not result.metrics["localization_point_distance"].passed


def test_grounding_comparator_rejects_box_for_point_reference() -> None:
    result = _compare(
        "<ref>search button</ref><box><450><200><550><300></box>",
        "<ref>search button</ref><box><500><250></box>",
    )

    assert result.status == StageStatus.FAILED.value
    assert not result.metrics["localization_type_match"].passed


def test_grounding_comparator_rejects_mixed_box_and_point_output() -> None:
    result = _compare(
        "<ref>things</ref><box><500><250></box><box><10><20><100><200></box>",
        "<ref>things</ref><box><500><250></box>",
    )

    assert result.status == StageStatus.FAILED.value
    assert not result.metrics["localization_markup_present"].passed
