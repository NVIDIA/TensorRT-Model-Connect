# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The one-to-one reference keeps overlapping detections and filters scores."""

import torch

from families.yolov10.tests.reference.model import detections


def test_end_to_end_head_does_not_suppress_overlapping_boxes():
    raw = torch.tensor([[[1, 2, 3, 4, 0.9, 0], [1, 2, 3, 4, 0.8, 0], [1, 2, 3, 4, 0.1, 1]]])
    torch.testing.assert_close(detections(raw), raw[0, :2])
    assert detections(raw, threshold=1.0).shape == (0, 6)


def _snapshot_case():
    from families.yolov10.tests.reference.e2e import _check_snapshot

    raw = torch.tensor(
        [
            [
                [1, 2, 3, 4, 0.9, 0],
                [10, 20, 30, 40, 0.00001450042, 1],
                [50, 60, 70, 80, 0.00001450031, 2],
            ]
        ]
    )
    predictions = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
    expected = {
        "shape": [3, 6],
        "prediction_shape": [1, 6, 4],
        "prediction_values": predictions.flatten().tolist(),
        "detections": raw[0, :1].tolist(),
    }
    return _check_snapshot, raw, predictions, expected


def test_snapshot_keeps_cell_identity_when_low_confidence_ranks_swap():
    check, raw, predictions, expected = _snapshot_case()
    check(raw[:, [0, 2, 1]], expected, predictions)


def test_snapshot_still_rejects_changed_cell_values():
    import pytest

    check, raw, predictions, expected = _snapshot_case()
    predictions[0, 0, 0] += 1
    with pytest.raises(AssertionError):
        check(raw, expected, predictions)


def test_snapshot_still_rejects_changed_accepted_detections():
    import pytest

    for column in (0, 4, 5):
        check, raw, predictions, expected = _snapshot_case()
        raw[0, 0, column] += 0.01 if column == 4 else 1
        with pytest.raises(AssertionError):
            check(raw, expected, predictions)
    check, raw, predictions, expected = _snapshot_case()
    raw[0, 1, 4] = 0.3
    with pytest.raises(AssertionError):
        check(raw, expected, predictions)
