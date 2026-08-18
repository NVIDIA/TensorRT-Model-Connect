# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from tensorrt_model_connect.families.sam2_hoi.validation import (
    ACCURACY_CONTRACT_ID,
    EXACT_DECISION_FIELDS,
    compare_outputs,
)


def _outputs() -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for frame in range(5):
        prefix = f"frame_{frame:06d}"
        masks = np.zeros((2, 1, 20, 20), dtype=np.uint8)
        masks[0, 0, 2:12, 3:13] = 1
        masks[1, 0, 8:18, 7:17] = 1
        result[f"{prefix}_object_ids"] = np.array([0, 1], dtype=np.int64)
        result[f"{prefix}_binary_masks"] = masks
        result[f"{prefix}_det_bboxes"] = np.array(
            [[10.0, 20.0, 110.0, 220.0], [30.0, 40.0, 230.0, 340.0]],
            dtype=np.float32,
        )
        result[f"{prefix}_det_labels"] = np.array([1, 2], dtype=np.int64)
        result[f"{prefix}_det_scores"] = np.array([0.49, 0.39], dtype=np.float32)
        result[f"{prefix}_interaction_pairs"] = np.array([[0, 1]], dtype=np.int64)
    return result


def test_exact_outputs_pass() -> None:
    reference = _outputs()
    result = compare_outputs(reference, {key: value.copy() for key, value in reference.items()})
    assert result["status"] == "pass"
    assert result["schema_version"] == 2
    assert result["accuracy_contract"] == ACCURACY_CONTRACT_ID
    assert result["decision_contract"] == {
        "scope": "observable_full_chain_topology",
        "exact_fields": list(EXACT_DECISION_FIELDS),
    }
    assert not result["failures"]


def test_user_reviewed_point_ninety_nine_mask_boundary_passes() -> None:
    reference = _outputs()
    candidate = {key: value.copy() for key, value in reference.items()}
    for frame in range(5):
        prefix = f"frame_{frame:06d}"
        reference_mask = np.zeros((2, 1, 100, 100), dtype=np.uint8)
        reference_mask[:, 0, :10, :10] = 1
        candidate_mask = reference_mask.copy()
        candidate_mask[:, 0, 0, 0] = 0
        reference[f"{prefix}_binary_masks"] = reference_mask
        candidate[f"{prefix}_binary_masks"] = candidate_mask

    result = compare_outputs(reference, candidate)

    assert result["status"] == "pass"
    assert min(float(row["min_iou"]) for row in result["frames"]) == pytest.approx(0.99)


def test_topology_difference_fails() -> None:
    reference = _outputs()
    candidate = {key: value.copy() for key, value in reference.items()}
    candidate["frame_000003_interaction_pairs"] = np.array([[1, 0]], dtype=np.int64)
    result = compare_outputs(reference, candidate)
    assert result["status"] == "fail"
    assert "frame_000003_interaction_pairs differs" in result["failures"]


def test_detection_regression_fails() -> None:
    reference = _outputs()
    candidate = {key: value.copy() for key, value in reference.items()}
    candidate["frame_000001_det_scores"][0] += 0.02
    candidate["frame_000001_det_bboxes"][0, 0] += 5.0
    result = compare_outputs(reference, candidate)
    assert result["status"] == "fail"
    assert "frame_000001 failed detection_score" in result["failures"]
    assert "frame_000001 failed detection_box_abs" in result["failures"]


def test_mask_regression_fails() -> None:
    reference = _outputs()
    candidate = {key: value.copy() for key, value in reference.items()}
    candidate["frame_000004_binary_masks"][0, 0, 2:4, 3:13] = 0
    result = compare_outputs(reference, candidate)
    assert result["status"] == "fail"
    assert "frame_000004 failed mask_iou" in result["failures"]
    assert "frame_000004 failed mask_dice" in result["failures"]


def test_missing_frame_is_rejected() -> None:
    reference = _outputs()
    candidate = {
        key: value.copy() for key, value in reference.items() if not key.startswith("frame_000004")
    }
    with pytest.raises(ValueError, match="output keys differ"):
        compare_outputs(reference, candidate)
