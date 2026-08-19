# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SAM3-owned prompted segmentation harness tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tensorrt_model_connect.models.sam3.tests.e2e_plugins.comparators.segmentation import (
    PromptedSegmentationComparator,
)
from tensorrt_model_connect.models.sam3.tests.e2e_plugins.contract import Sam3SegmentationPlugin
from tensorrt_model_connect.models.sam3.tests.e2e_plugins.runners.segmentation import _load_mask_outputs
from tests.e2e_harness.contracts import E2ECase, StageOutput, StageSpec, ThresholdProfile
from tests.e2e_harness.manifest_loader import load_manifest
from tests.e2e_harness.registry import (
    activate_model_plugins,
    get_contract_plugin,
    reset as reset_e2e_registry,
)


def test_comparator_checks_boxes_and_scores() -> None:
    mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    trt = StageOutput(
        stage_name="full_inference",
        data={
            "masks": [mask],
            "mask_scores": [0.92],
            "boxes": [[10.0, 20.0, 30.0, 40.0]],
        },
    )
    ref = StageOutput(
        stage_name="full_inference",
        data={
            "masks": [mask],
            "mask_scores": [0.90],
            "boxes": [[10.0, 20.0, 30.0, 40.0]],
        },
    )
    threshold = ThresholdProfile(
        task_strategy="prompted_segmentation",
        metrics={
            "iou_per_prompt": 0.7,
            "box_iou_mean": 0.95,
            "score_abs_error_mean": 0.05,
        },
    )

    result = PromptedSegmentationComparator().compare(
        trt, ref, threshold, StageSpec(name="full_inference")
    )

    assert result.status == "passed"
    assert result.metrics["box_iou_mean"].passed
    assert result.metrics["score_abs_error_mean"].passed


def test_comparator_fails_missing_boxes_and_scores() -> None:
    mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    trt = StageOutput(stage_name="full_inference", data={"masks": [mask]})
    ref = StageOutput(
        stage_name="full_inference",
        data={
            "masks": [mask],
            "mask_scores": [0.90],
            "boxes": [[10.0, 20.0, 30.0, 40.0]],
        },
    )
    threshold = ThresholdProfile(
        task_strategy="prompted_segmentation",
        metrics={
            "iou_per_prompt": 0.7,
            "box_iou_mean": 0.95,
            "score_abs_error_mean": 0.05,
        },
    )

    result = PromptedSegmentationComparator().compare(
        trt, ref, threshold, StageSpec(name="full_inference")
    )

    assert result.status == "failed"
    assert not result.metrics["box_count_consistency"].passed
    assert not result.metrics["score_count_consistency"].passed


def test_contract_plugin_requires_boxes_and_scores() -> None:
    mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    trt = StageOutput(
        stage_name="full_inference",
        data={
            "masks": [mask],
            "mask_scores": [0.92],
            "boxes": [[10.0, 20.0, 30.0, 40.0]],
        },
    )
    ref = StageOutput(
        stage_name="full_inference",
        data={
            "masks": [mask],
            "mask_scores": [0.90],
            "boxes": [[10.0, 20.0, 30.0, 40.0]],
        },
    )
    case = E2ECase(
        name="sam3",
        hf_id="facebook/sam3",
        family="sam3",
        runtime_strategy="sam3_prompted_segmentation",
        task_strategy="prompted_segmentation",
        reference_family="prompted_segmentation_sam3",
    )
    threshold = ThresholdProfile(
        task_strategy="prompted_segmentation",
        metrics={
            "num_masks_consistency": 1.0,
            "iou_per_prompt": 0.7,
            "box_iou_mean": 0.95,
            "score_abs_error_mean": 0.05,
        },
    )

    result = Sam3SegmentationPlugin().verify(trt, ref, case, threshold)

    assert result.status == "passed"
    assert result.metrics["box_iou_mean"].passed
    assert result.metrics["score_abs_error_mean"].passed


def test_contract_plugin_errors_when_boxes_missing() -> None:
    mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    trt = StageOutput(stage_name="full_inference", data={"masks": [mask]})
    ref = StageOutput(
        stage_name="full_inference",
        data={
            "masks": [mask],
            "mask_scores": [0.90],
            "boxes": [[10.0, 20.0, 30.0, 40.0]],
        },
    )
    case = E2ECase(
        name="sam3",
        hf_id="facebook/sam3",
        family="sam3",
        runtime_strategy="sam3_prompted_segmentation",
        task_strategy="prompted_segmentation",
        reference_family="prompted_segmentation_sam3",
    )
    threshold = ThresholdProfile(
        task_strategy="prompted_segmentation",
        metrics={"iou_per_prompt": 0.7},
    )

    result = Sam3SegmentationPlugin().verify(trt, ref, case, threshold)

    assert result.status == "error"
    assert "boxes" in result.message


def test_model_plugins_register_model_owned_contract() -> None:
    model_dir = Path(__file__).resolve().parent
    try:
        activate_model_plugins(model_dir)
        plugin = get_contract_plugin("prompted_segmentation_sam3")
        assert type(plugin).__module__.startswith("tensorrt_model_connect.models.sam3.tests.e2e_plugins.")
    finally:
        reset_e2e_registry()


def test_runner_loads_boxes_and_scores(tmp_path) -> None:
    output_dir = tmp_path / "masks"
    output_dir.mkdir()
    np.save(output_dir / "mask_000.npy", np.array([[1, 0], [0, 1]], dtype=np.uint8))
    (output_dir / "score_000.txt").write_text("0.92\n", encoding="utf-8")
    (output_dir / "box_000.txt").write_text("10 20 30 40\n", encoding="utf-8")

    masks, scores, boxes = _load_mask_outputs(str(output_dir), "")

    assert len(masks) == 1
    assert scores == [0.92]
    assert boxes == [[10.0, 20.0, 30.0, 40.0]]


def test_manifest_loader_keeps_text_prompt_contract(tmp_path) -> None:
    manifest_path = tmp_path / "sam3.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "sam3",
                "hf_id": "facebook/sam3",
                "bundle": "sam3.bundle",
                "family": "sam3",
                "runtime_strategy": "sam3_prompted_segmentation",
                "task_strategy": "prompted_segmentation",
                "testcases": [
                    {
                        "name": "sam3",
                        "test_type": "prompted_segmentation",
                        "reference_family": "prompted_segmentation_sam3",
                        "user_contract": "prompted_mask",
                        "inputs": {
                            "image": "data/test_img.jpeg",
                            "prompt": "car",
                        },
                        "threshold_overrides": {
                            "box_iou_mean": 0.95,
                            "score_abs_error_mean": 0.05,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    case = load_manifest(manifest_path)

    assert case.reference_family == "prompted_segmentation_sam3"
    assert case.user_contract == "prompted_mask"
    assert case.inputs["prompt"] == "car"
    assert case.threshold_overrides["box_iou_mean"] == 0.95
    assert case.threshold_overrides["score_abs_error_mean"] == 0.05
    assert case.inputs["image"] == "data/test_img.jpeg"


def test_sam3_manifest_maps_model_local_image_asset() -> None:
    manifest_path = Path(__file__).resolve().parent / "manifests" / "sam3.json"

    case = load_manifest(manifest_path)

    image_path = Path(case.inputs["image"])
    assert image_path.is_absolute()
    assert image_path == Path(__file__).resolve().parent / "data" / "test_img.jpeg"
    assert image_path.is_file()
    assert case.inputs["prompt"] == "car"
