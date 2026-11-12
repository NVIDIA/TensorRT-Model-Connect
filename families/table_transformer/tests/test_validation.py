# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for the direct E2E's output and input identity checks."""

import pytest

from families.table_transformer.tests.test_e2e import _assert_parity, _model_dir


@pytest.mark.parametrize("shape", [(1, 1), (8, 12), (12, 9)])
def test_cli_dimensions_must_match_input(shape):
    actual = {
        "boxes": [1, 1, 7, 7],
        "scores": [0.9],
        "classes": [0],
        "image_height": shape[0],
        "image_width": shape[1],
    }
    expected = [{"box": [1, 1, 7, 7], "score": 0.9, "label": 0}]
    with pytest.raises(AssertionError, match="image dimensions"):
        _assert_parity(actual, expected, (12, 8))


def test_matching_dimensions_retain_detection_parity():
    actual = {
        "boxes": [1, 1, 7, 7],
        "scores": [0.9],
        "classes": [0],
        "image_height": 12,
        "image_width": 8,
    }
    expected = [{"box": [1, 1, 7, 7], "score": 0.9, "label": 0}]
    assert _assert_parity(actual, expected, (12, 8))["min_iou"] == 1.0


def test_explicit_model_path_is_resolved(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    link = tmp_path / "alias"
    link.symlink_to(checkpoint, target_is_directory=True)
    monkeypatch.setenv("TRTMC_TABLE_TRANSFORMER_MODEL_DIR", str(link))
    assert _model_dir({}) == checkpoint.resolve()
