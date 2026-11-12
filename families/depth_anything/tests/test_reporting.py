# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check model-owned display copies without a GPU or a changed comparator."""

import numpy as np
from PIL import Image

from tools import e2e_evidence
from tools.e2e_report import render_case
from families.depth_anything.tests.reporting import record_depth_views


def test_paired_views_preserve_values_and_render_before_details(tmp_path):
    left = np.array([[0, 1], [1, 0]], dtype=np.float32)
    right = np.array([[0, 1], [0, 0]], dtype=np.float32)
    saved_left, saved_right = left.copy(), right.copy()
    recorder = e2e_evidence.Evidence(tmp_path / "evidence", family="depth_anything", case="views",
                                    source_revision="a" * 40, roots=(tmp_path,))
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        e2e_evidence.record_evidence("native", {"values": left})
        e2e_evidence.record_evidence("reference", {"values": right})
        record_depth_views(left, right, 2, 2, tmp_path / "views")
    finally:
        e2e_evidence._ACTIVE.reset(token)
    np.testing.assert_array_equal(left, saved_left)
    np.testing.assert_array_equal(right, saved_right)
    views = recorder.data["views"]
    assert [view["title"] for view in views[:2]] == ["Sample 0 / Native", "Sample 0 / Reference"]
    images = [np.asarray(Image.open(recorder.directory / view["image"]["artifact"])) for view in views[:2]]
    np.testing.assert_array_equal(images[0][0, 0], images[1][0, 0])
    np.testing.assert_array_equal(images[0][0, 1], images[1][0, 1])
    assert views[0]["caption"] == views[1]["caption"]
    report = render_case(recorder.data, recorder.directory)
    top = report.split('<details class="case-details">', 1)[0]
    assert 'data-media-paired="true"' in top
    assert top.count("data:image/png;base64,") == 2


def test_preview_failure_does_not_replace_the_model_result(tmp_path, monkeypatch):
    recorder = e2e_evidence.Evidence(tmp_path / "evidence", family="depth_anything", case="views",
                                    source_revision="a" * 40, roots=(tmp_path,))
    def fail(*args, **kwargs):
        raise OSError("preview disk unavailable")
    monkeypatch.setattr(Image.Image, "save", fail)
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        record_depth_views(np.zeros((2, 2)), np.ones((2, 2)), 2, 2, tmp_path / "views")
    finally:
        e2e_evidence._ACTIVE.reset(token)
    assert "preview disk unavailable" in recorder.data["views"][-1]["caption"]
