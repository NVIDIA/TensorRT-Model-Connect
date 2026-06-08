"""Tests for prompted segmentation E2E harness comparator.

Trace: ARCH-PIP-SEG-001, UD-SEG-HARNESS
Intent: Validate PromptedSegmentationComparator mask loading, IoU computation, and threshold gating
Preconditions: Synthetic mask arrays and threshold profiles are available in temp directories
Postconditions: Comparator correctly loads reference masks from .npy and computes per-mask IoU metrics
"""

from __future__ import annotations

import json

import numpy as np

from tests.e2e_harness.comparators.segmentation import PromptedSegmentationComparator
from tests.e2e_harness.contracts import E2ECase, StageOutput, StageSpec, ThresholdProfile
from tests.e2e_harness.plugins.segmentation import SegmentationPlugin
from tests.e2e_harness.manifest_loader import load_manifest
from tests.e2e_harness.runners.segmentation import _load_mask_outputs


def test_prompted_segmentation_comparator_loads_reference_masks_from_npy(tmp_path) -> None:
    masks_path = tmp_path / "hf_sam_masks.npy"
    np.save(
        masks_path,
        np.array(
            [
                [[1, 0], [0, 1]],
                [[0, 1], [1, 0]],
                [[1, 1], [0, 0]],
            ],
            dtype=np.uint8,
        ),
    )

    trt = StageOutput(
        stage_name="full_inference",
        data={
            "masks": [
                np.array([[1, 0], [0, 1]], dtype=np.uint8),
                np.array([[0, 1], [1, 0]], dtype=np.uint8),
                np.array([[1, 1], [0, 0]], dtype=np.uint8),
            ],
            "mask_scores": [0.9, 0.8, 0.7],
        },
    )
    ref = StageOutput(
        stage_name="full_inference",
        data={
            "masks_path": str(masks_path),
            "iou_scores": [0.9, 0.8, 0.7],
        },
    )
    threshold = ThresholdProfile(
        task_strategy="prompted_segmentation",
        metrics={
            "num_masks_consistency": 1.0,
            "iou_per_prompt": 0.7,
        },
    )

    result = PromptedSegmentationComparator().compare(
        trt, ref, threshold, StageSpec(name="full_inference")
    )

    assert result.status == "passed"
    assert result.metrics["num_masks_consistency"].passed
    assert result.metrics["iou_per_prompt"].passed


def test_prompted_segmentation_comparator_checks_sam3_boxes_and_scores() -> None:
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


def test_prompted_segmentation_comparator_fails_missing_sam3_boxes_and_scores() -> None:
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


def test_prompted_segmentation_contract_plugin_verifies_masks_from_npy(tmp_path) -> None:
    masks_path = tmp_path / "hf_sam_masks.npy"
    np.save(
        masks_path,
        np.array(
            [
                [[1, 0], [0, 1]],
                [[0, 1], [1, 0]],
                [[1, 1], [0, 0]],
            ],
            dtype=np.uint8,
        ),
    )

    trt = StageOutput(
        stage_name="full_inference",
        data={
            "masks": [
                np.array([[1, 0], [0, 1]], dtype=np.uint8),
                np.array([[0, 1], [1, 0]], dtype=np.uint8),
                np.array([[1, 1], [0, 0]], dtype=np.uint8),
            ],
        },
    )
    ref = StageOutput(
        stage_name="full_inference",
        data={"masks_path": str(masks_path)},
    )
    case = E2ECase(
        name="sam-vit-base",
        hf_id="facebook/sam-vit-base",
        family="sam",
        runtime_strategy="prompted_segmentation",
        task_strategy="prompted_segmentation",
        reference_family="prompted_segmentation_sam",
    )
    threshold = ThresholdProfile(
        task_strategy="prompted_segmentation",
        metrics={
            "num_masks_consistency": 1.0,
            "iou_per_prompt": 0.7,
        },
    )

    result = SegmentationPlugin().verify(trt, ref, case, threshold)

    assert result.status == "passed"
    assert result.metrics["num_masks_consistency"].passed
    assert result.metrics["iou_per_prompt"].passed


def test_prompted_segmentation_contract_plugin_requires_sam3_boxes_and_scores() -> None:
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
        runtime_strategy="prompted_segmentation",
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

    result = SegmentationPlugin().verify(trt, ref, case, threshold)

    assert result.status == "passed"
    assert result.metrics["box_iou_mean"].passed
    assert result.metrics["score_abs_error_mean"].passed


def test_prompted_segmentation_contract_plugin_errors_when_sam3_boxes_missing() -> None:
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
        runtime_strategy="prompted_segmentation",
        task_strategy="prompted_segmentation",
        reference_family="prompted_segmentation_sam3",
    )
    threshold = ThresholdProfile(
        task_strategy="prompted_segmentation",
        metrics={"iou_per_prompt": 0.7},
    )

    result = SegmentationPlugin().verify(trt, ref, case, threshold)

    assert result.status == "error"
    assert "boxes" in result.message


def test_prompted_segmentation_contract_plugin_fails_bad_trt_masks(tmp_path) -> None:
    masks_path = tmp_path / "hf_sam_masks.npy"
    np.save(
        masks_path,
        np.array(
            [
                [[1, 1], [0, 0]],
                [[0, 0], [1, 1]],
                [[1, 0], [1, 0]],
            ],
            dtype=np.uint8,
        ),
    )

    trt = StageOutput(
        stage_name="full_inference",
        data={
            "masks": [
                np.array([[0, 0], [1, 1]], dtype=np.uint8),
                np.array([[1, 1], [0, 0]], dtype=np.uint8),
                np.array([[0, 1], [0, 1]], dtype=np.uint8),
            ],
        },
    )
    ref = StageOutput(
        stage_name="full_inference",
        data={"masks_path": str(masks_path)},
    )
    case = E2ECase(
        name="sam-vit-base",
        hf_id="facebook/sam-vit-base",
        family="sam",
        runtime_strategy="prompted_segmentation",
        task_strategy="prompted_segmentation",
        reference_family="prompted_segmentation_sam",
    )
    threshold = ThresholdProfile(
        task_strategy="prompted_segmentation",
        metrics={
            "num_masks_consistency": 1.0,
            "iou_per_prompt": 0.7,
        },
    )

    result = SegmentationPlugin().verify(trt, ref, case, threshold)

    assert result.status == "failed"
    assert result.metrics["num_masks_consistency"].passed
    assert not result.metrics["iou_per_prompt"].passed


def test_prompted_segmentation_runner_loads_boxes_and_scores(tmp_path) -> None:
    output_dir = tmp_path / "masks"
    output_dir.mkdir()
    np.save(output_dir / "mask_000.npy", np.array([[1, 0], [0, 1]], dtype=np.uint8))
    (output_dir / "score_000.txt").write_text("0.92\n", encoding="utf-8")
    (output_dir / "box_000.txt").write_text("10 20 30 40\n", encoding="utf-8")

    masks, scores, boxes = _load_mask_outputs(str(output_dir), "")

    assert len(masks) == 1
    assert scores == [0.92]
    assert boxes == [[10.0, 20.0, 30.0, 40.0]]


def test_manifest_loader_promotes_num_expected_masks_into_inputs(tmp_path) -> None:
    manifest_path = tmp_path / "sam.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "sam-vit-base",
                "hf_id": "facebook/sam-vit-base",
                "bundle": "sam-vit-base.trtfb",
                "family": "sam",
                "runtime_strategy": "prompted_segmentation",
                "test_type": "prompted_segmentation",
                "test_image": "data/test_img.jpeg",
                "point_x": 0.5,
                "point_y": 0.5,
                "num_expected_masks": 3,
            }
        ),
        encoding="utf-8",
    )

    case = load_manifest(manifest_path)

    assert case.inputs["num_expected_masks"] == 3
    assert case.threshold_overrides["num_expected_masks"] == 3


def test_manifest_loader_keeps_sam3_text_prompt_contract(tmp_path) -> None:
    manifest_path = tmp_path / "sam3.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "sam3",
                "hf_id": "facebook/sam3",
                "bundle": "sam3.trtfb",
                "family": "sam3",
                "runtime_strategy": "prompted_segmentation",
                "test_type": "prompted_segmentation",
                "reference_family": "prompted_segmentation_sam3",
                "inputs": {
                    "image_url": "http://images.cocodataset.org/val2017/000000077595.jpg",
                    "prompt": "ear",
                },
                "threshold_overrides": {
                    "box_iou_mean": 0.95,
                    "score_abs_error_mean": 0.05,
                },
            }
        ),
        encoding="utf-8",
    )

    case = load_manifest(manifest_path)

    assert case.reference_family == "prompted_segmentation_sam3"
    assert case.user_contract == "prompted_mask"
    assert case.inputs["prompt"] == "ear"
    assert case.threshold_overrides["box_iou_mean"] == 0.95
    assert case.threshold_overrides["score_abs_error_mean"] == 0.05
    assert "000000077595.jpg" in case.inputs["image_url"]
