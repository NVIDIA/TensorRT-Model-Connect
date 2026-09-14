# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failed parity retains overlays and disagreement at the family's mask boundary."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from families.sam3.tests import test_e2e as e2e
from families.sam3.tests.reporting import record_mask_views
from tools import e2e_evidence


def test_mask_views_survive_real_comparator_failure(monkeypatch, tmp_path: Path):
    name = next(iter(e2e.CASES))
    _, manifest, original_case = e2e.CASES[name]
    source = tmp_path / "input.png"
    Image.new("RGB", (2, 2), (20, 30, 40)).save(source)
    case = {**original_case, "test_image": str(source)}
    monkeypatch.setitem(e2e.CASES, name, (tmp_path, manifest, case))
    actual = {"masks": [0.0, 1.0, 0.0, 1.0], "num_masks": 1, "height": 2, "width": 2}
    expected = {"masks": np.array([[[0.0, 0.0], [0.0, 1.0]]])}
    before = np.asarray(actual["masks"]).copy()
    monkeypatch.setattr(e2e, "_model_dir", lambda manifest: tmp_path)
    monkeypatch.setattr(e2e, "_runtime", lambda manifest: (tmp_path / "trtmc", tmp_path))
    monkeypatch.setattr(e2e, "_build", lambda *args: None)
    monkeypatch.setattr(e2e, "_native", lambda *args: actual)
    monkeypatch.setattr(e2e, "_official_reference", lambda *args: expected)
    monkeypatch.setattr(
        e2e,
        "_thresholds",
        lambda case: {"iou_per_prompt": 1.0, "mean_mask_iou": 1.0, "num_masks_consistency": 1.0},
    )
    recorder = e2e_evidence.Evidence(
        tmp_path / "evidence", family="sam3", case=name, source_revision="a" * 40, roots=(tmp_path,)
    )
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        with pytest.raises(AssertionError):
            e2e.test_official_checkpoint_e2e(name, tmp_path)
    finally:
        e2e_evidence._ACTIVE.reset(token)
    assert recorder.data["failure_stage"] == "compare"
    assert len(recorder.data["views"]) <= 6
    views = {view["title"]: view for view in recorder.data["views"]}
    view = views["Mask 0 disagreement"]
    with Image.open(recorder.directory / view["image"]["artifact"]) as image:
        pixels = np.asarray(image)
    np.testing.assert_array_equal(pixels[0, 0], [0, 0, 0])
    np.testing.assert_array_equal(pixels[0, 1], [255, 70, 70])
    np.testing.assert_array_equal(actual["masks"], before)
    assert "reference" in recorder.data


def test_mask_views_are_opt_in(tmp_path: Path):
    record_mask_views({}, {}, tmp_path / "absent.png", tmp_path / "views")
    assert not (tmp_path / "views").exists()


def test_invalid_mask_diagnostic_does_not_raise(tmp_path: Path):
    recorder = e2e_evidence.Evidence(
        tmp_path / "evidence",
        family="sam3",
        case="bad-mask",
        source_revision="a" * 40,
        roots=(tmp_path,),
    )
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        record_mask_views({}, {}, tmp_path / "absent.png", tmp_path / "views")
    finally:
        e2e_evidence._ACTIVE.reset(token)
    assert all("unavailable" in view["caption"] for view in recorder.data["views"])
