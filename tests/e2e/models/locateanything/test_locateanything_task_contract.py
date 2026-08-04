# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from tensorrt_model_connect.families.locateanything.task_contract import (
    box_iou,
    detect_text_prompt,
    detection_prompt,
    ground_gui_prompt,
    ground_multi_prompt,
    ground_single_prompt,
    ground_text_prompt,
    matched_box_iou,
    matched_point_distance,
    parse_localizations,
    point_distance,
    point_prompt,
)


def test_official_task_prompt_templates() -> None:
    assert detection_prompt(["person", "car"]) == (
        "Locate all the instances that matches the following description: person</c>car."
    )
    assert ground_single_prompt("white vehicle") == (
        "Locate a single instance that matches the following description: white vehicle."
    )
    assert ground_multi_prompt("red shirts") == (
        "Locate all the instances that match the following description: red shirts."
    )
    assert ground_text_prompt("NVIDIA") == "Please locate the text referred as NVIDIA."
    assert detect_text_prompt() == "Detect all the text in box format."
    assert ground_gui_prompt("search button") == (
        "Locate the region that matches the following description: search button."
    )
    assert ground_gui_prompt("search button", output_type="point") == (
        "Point to: search button."
    )
    assert point_prompt("traffic light") == "Point to: traffic light."


def test_task_prompts_reject_empty_inputs_and_unknown_output_type() -> None:
    with pytest.raises(ValueError, match="at least one category"):
        detection_prompt([])
    with pytest.raises(ValueError, match="must not be empty"):
        point_prompt(" ")
    with pytest.raises(ValueError, match="box.*point"):
        ground_gui_prompt("button", output_type="pixel")  # type: ignore[arg-type]


def test_parse_localizations_accepts_boxes_and_points() -> None:
    boxes = parse_localizations(
        "<ref>cars</ref><box><10><20><110><220></box><box><300><400><500><600></box>",
        require_reference=True,
    )
    points = parse_localizations(
        "<ref>button</ref><box><505><250></box>",
        require_reference=True,
    )

    assert boxes is not None and [item.kind for item in boxes] == ["box", "box"]
    assert points is not None and points[0].kind == "point"
    assert box_iou(boxes[0], boxes[0]) == 1.0
    assert point_distance(points[0], points[0]) == 0.0


def test_localization_set_matching_is_order_independent() -> None:
    boxes = parse_localizations(
        "<box><10><20><110><220></box><box><300><400><500><600></box>"
    )
    reversed_boxes = parse_localizations(
        "<box><300><400><500><600></box><box><10><20><110><220></box>"
    )
    points = parse_localizations("<box><10><20></box><box><300><400></box>")
    reversed_points = parse_localizations("<box><300><400></box><box><10><20></box>")

    assert boxes is not None and reversed_boxes is not None
    assert points is not None and reversed_points is not None
    assert matched_box_iou(boxes, reversed_boxes) == 1.0
    assert matched_point_distance(points, reversed_points) == 0.0


@pytest.mark.parametrize(
    "answer",
    [
        "<box><10><20><110><220></box>",
        "<ref>bad box</ref><box><110><20><10><220></box>",
        "<ref>mixed</ref><box><10><20></box><box><10><20><30><40></box>",
        "<ref>out of range</ref><box><1001><20></box>",
        "<ref>malformed</ref><box><10><20><30></box>",
    ],
)
def test_parse_localizations_rejects_invalid_contract(answer: str) -> None:
    assert parse_localizations(answer, require_reference=True) is None
