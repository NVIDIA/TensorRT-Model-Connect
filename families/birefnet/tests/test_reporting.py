# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check model-owned display copies without a GPU or a changed comparator."""

import numpy as np
from PIL import Image

from tools import e2e_evidence
from tools.e2e_report import render_case
from families.birefnet.tests.reporting import record_mask_views


def test_paired_views_preserve_values_and_render_before_details(tmp_path):
    left = np.array([[0, 1], [1, 0]], dtype=np.float32)
    right = np.array([[0, 1], [0, 0]], dtype=np.float32)
    saved_left, saved_right = left.copy(), right.copy()
    recorder = e2e_evidence.Evidence(tmp_path / "evidence", family="birefnet", case="views",
                                    source_revision="a" * 40, roots=(tmp_path,))
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        e2e_evidence.record_evidence("native", {"values": left})
        e2e_evidence.record_evidence("reference", {"values": right})
        record_mask_views({"mask": left.ravel().tolist(), "height": 2, "width": 2}, right, tmp_path / "views")
    finally:
        e2e_evidence._ACTIVE.reset(token)
    np.testing.assert_array_equal(left, saved_left)
    np.testing.assert_array_equal(right, saved_right)
    views = recorder.data["views"]
    assert [view["title"] for view in views[:2]] == ["Sample 0 / Native", "Sample 0 / Reference"]
    images = [np.asarray(Image.open(recorder.directory / view["image"]["artifact"])) for view in views[:2]]
    np.testing.assert_array_equal(images[0], left * 255)
    np.testing.assert_array_equal(images[1], right * 255)
    report = render_case(recorder.data, recorder.directory)
    top = report.split('<details class="case-details">', 1)[0]
    assert 'data-media-paired="true"' in top
    assert top.count("data:image/png;base64,") == 2


def test_preview_failure_does_not_replace_the_model_result(tmp_path, monkeypatch):
    recorder = e2e_evidence.Evidence(tmp_path / "evidence", family="birefnet", case="views",
                                    source_revision="a" * 40, roots=(tmp_path,))
    def fail(*args, **kwargs):
        raise OSError("preview disk unavailable")
    monkeypatch.setattr(Image.Image, "save", fail)
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        record_mask_views({"mask": [0, 1, 1, 0], "height": 2, "width": 2}, np.zeros((2, 2)), tmp_path / "views")
    finally:
        e2e_evidence._ACTIVE.reset(token)
    assert "preview disk unavailable" in recorder.data["views"][-1]["caption"]
