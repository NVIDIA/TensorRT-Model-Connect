# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from families.sam3.tests.test_e2e import _assert_parity, _box_iou, _thresholds


def _payloads():
    masks = np.array(
        [
            [[1, 0], [0, 1]],
            [[0, 1], [1, 0]],
        ],
        dtype=np.uint8,
    )
    actual = {
        "masks": masks.reshape(-1).tolist(),
        "iou_scores": [0.91, 0.72],
        "boxes": [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]],
        "num_masks": 2,
    }
    expected = {
        "masks": masks,
        "iou_scores": [0.89, 0.73],
        "boxes": [[0.1, 0.1, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]],
    }
    return actual, expected


def test_sam3_restores_every_active_oracle_gate() -> None:
    thresholds = _thresholds("sam3")
    assert thresholds == {
        "iou_per_prompt": 0.7,
        "num_masks_consistency": 1.0,
        "box_iou_mean": 0.95,
        "score_abs_error_mean": 0.05,
    }

    actual, expected = _payloads()
    _assert_parity(
        actual,
        expected,
        {"task": "text_prompted_segmentation"},
        {},
        thresholds,
    )


def test_sam3_box_iou_rejects_disjoint_boxes() -> None:
    assert _box_iou([0, 0, 1, 1], [2, 2, 3, 3]) == 0.0


@pytest.mark.parametrize("missing_from", ["actual", "expected"])
def test_sam3_mask_count_consistency_is_gated(missing_from: str) -> None:
    actual, expected = _payloads()
    if missing_from == "actual":
        actual["num_masks"] = 1
    else:
        expected["masks"] = expected["masks"][:1]

    with pytest.raises(AssertionError):
        _assert_parity(
            actual,
            expected,
            {"task": "text_prompted_segmentation"},
            {},
            _thresholds("sam3"),
        )


def test_sam3_score_error_is_gated() -> None:
    actual, expected = _payloads()
    expected["iou_scores"] = [0.70, 0.51]
    with pytest.raises(AssertionError):
        _assert_parity(
            actual,
            expected,
            {"task": "text_prompted_segmentation"},
            {},
            _thresholds("sam3"),
        )


def test_sam3_box_iou_is_gated() -> None:
    actual, expected = _payloads()
    expected["boxes"][0] = [40.0, 40.0, 50.0, 50.0]
    with pytest.raises(AssertionError):
        _assert_parity(
            actual,
            expected,
            {"task": "text_prompted_segmentation"},
            {},
            _thresholds("sam3"),
        )
