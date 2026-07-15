# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from tests.e2e_harness.contracts import StageOutput, StageSpec, StageStatus, ThresholdProfile
from tests.e2e.models.locateanything.e2e_plugins.comparator import (
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


def test_grounding_comparator_reports_empty_output_as_failure() -> None:
    result = _compare(
        "",
        "<ref>white vehicle</ref><box><304><267><828><708></box>",
    )

    assert result.status == StageStatus.FAILED.value
    assert not result.metrics["non_empty_output"].passed
    assert not result.metrics["normalized_text_edit_distance"].passed
