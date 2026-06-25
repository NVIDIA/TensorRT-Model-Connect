"""SAM-owned prompted segmentation harness tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tests.e2e.models.sam.e2e_plugins.comparators.segmentation import (
    PromptedSegmentationComparator,
)
from tests.e2e.models.sam.e2e_plugins.contract import SamSegmentationPlugin
from tests.e2e_harness.contracts import E2ECase, StageOutput, StageSpec, ThresholdProfile
from tests.e2e_harness.manifest_loader import load_manifest
from tests.e2e_harness.registry import (
    activate_model_plugins,
    get_contract_plugin,
    reset as reset_e2e_registry,
)


def test_comparator_loads_reference_masks_from_npy(tmp_path) -> None:
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


def test_contract_plugin_verifies_masks_from_npy(tmp_path) -> None:
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
        runtime_strategy="sam_prompted_segmentation",
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

    result = SamSegmentationPlugin().verify(trt, ref, case, threshold)

    assert result.status == "passed"
    assert result.metrics["num_masks_consistency"].passed
    assert result.metrics["iou_per_prompt"].passed


def test_model_plugins_register_model_owned_contract() -> None:
    model_dir = Path(__file__).resolve().parent
    try:
        activate_model_plugins(model_dir)
        plugin = get_contract_plugin("prompted_segmentation_sam")
        assert type(plugin).__module__.startswith(
            "tests.e2e.models.sam.e2e_plugins."
        )
    finally:
        reset_e2e_registry()


def test_contract_plugin_fails_bad_trt_masks(tmp_path) -> None:
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
        runtime_strategy="sam_prompted_segmentation",
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

    result = SamSegmentationPlugin().verify(trt, ref, case, threshold)

    assert result.status == "failed"
    assert result.metrics["num_masks_consistency"].passed
    assert not result.metrics["iou_per_prompt"].passed


def test_manifest_loader_promotes_num_expected_masks_into_inputs(tmp_path) -> None:
    manifest_path = tmp_path / "sam.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "sam-vit-base",
                "hf_id": "facebook/sam-vit-base",
                "bundle": "sam-vit-base.trtfb",
                "family": "sam",
                "runtime_strategy": "sam_prompted_segmentation",
                "test_type": "prompted_segmentation",
                "test_image": "data/test_img.jpeg",
                "point_x": 0.5,
                "point_y": 0.5,
                "num_expected_masks": 3,
                "input_fields": [
                    {"input": "point_x", "manifest": "point_x"},
                    {"input": "point_y", "manifest": "point_y"},
                    {"input": "num_expected_masks", "manifest": "num_expected_masks"},
                ],
            }
        ),
        encoding="utf-8",
    )

    case = load_manifest(manifest_path)

    assert case.inputs["point_x"] == 0.5
    assert case.inputs["point_y"] == 0.5
    assert case.inputs["num_expected_masks"] == 3
    assert case.threshold_overrides["num_expected_masks"] == 3
