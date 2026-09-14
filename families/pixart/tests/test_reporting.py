# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paired previews use the same existing frame indices and remain bounded."""

from pathlib import Path

import numpy as np
from PIL import Image

from families.pixart.tests import reporting
from tools import e2e_evidence


def test_paired_previews_keep_matched_indices_and_pixels(tmp_path: Path):
    native_dir = tmp_path / "native"
    native_dir.mkdir()
    frames = [np.full((8, 8, 3), index / 9, dtype=np.float32) for index in range(10)]
    paths = []
    for index, values in enumerate(frames):
        path = native_dir / f"frame-{index:04d}.png"
        Image.fromarray(np.rint(values * 255).astype(np.uint8)).save(path)
        paths.append(path)
    actual = {"artifact": str(native_dir)}
    expected = {"images": frames}
    recorder = e2e_evidence.Evidence(tmp_path / "evidence", family="pixart", case="preview-probe",
                                   source_revision="a" * 40, roots=(tmp_path,))
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        reporting.record_report_views(actual, expected, tmp_path / "views")
        snapshot = reporting.reference_snapshot(expected)
    finally:
        e2e_evidence._ACTIVE.reset(token)
    views = recorder.data["views"]
    assert len(views) == 12
    assert snapshot["frame_count"] == 10
    assert snapshot["sampled_indices"] == [0, 2, 4, 5, 7, 9]
    for index in range(0, len(views), 2):
        native, reference = views[index:index + 2]
        assert native["title"].split(" / ")[0] == reference["title"].split(" / ")[0]
        with Image.open(recorder.directory / native["image"]["artifact"]) as left:
            with Image.open(recorder.directory / reference["image"]["artifact"]) as right:
                np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    assert len(reporting._indices(1000)) == 6


def test_disabled_previews_do_not_touch_outputs(tmp_path: Path):
    expected = {"images": [object()]}
    token = e2e_evidence._ACTIVE.set(None)
    try:
        reporting.record_report_views({}, expected, tmp_path / "views")
        assert reporting.reference_snapshot(expected) is expected
    finally:
        e2e_evidence._ACTIVE.reset(token)
    assert not (tmp_path / "views").exists()


def test_failed_l0_comparison_retains_both_image_previews(monkeypatch, tmp_path: Path):
    import pytest
    from families.pixart.tests import test_e2e as e2e

    name = next(name for name in e2e.CASES if name.endswith("-l0"))
    native = tmp_path / "native.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(native)
    reference = np.ones((8, 8, 3), dtype=np.float32)
    monkeypatch.setattr(e2e, "_model_dir", lambda manifest: tmp_path)
    monkeypatch.setattr(e2e, "_runtime", lambda manifest: (tmp_path / "trtmc", tmp_path))
    monkeypatch.setattr(e2e, "_build", lambda *args: None)
    monkeypatch.setattr(e2e, "_initial_latents", lambda *args: np.zeros((1, 1, 1, 1), dtype=np.float32))
    monkeypatch.setattr(e2e, "_native", lambda *args: {"artifact": str(native)})
    monkeypatch.setattr(e2e, "_official_reference", lambda *args: {"images": [reference]})
    monkeypatch.setattr(e2e, "_thresholds", lambda name: {})

    def fail(*args):
        raise AssertionError("original pixel comparison failed")

    monkeypatch.setattr(e2e, "_assert_contract", fail)
    recorder = e2e_evidence.Evidence(tmp_path / "evidence", family="pixart", case=name,
                                   source_revision="a" * 40, roots=(tmp_path,))
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        with pytest.raises(AssertionError, match="original pixel comparison failed"):
            e2e.test_official_checkpoint_e2e(name, tmp_path)
    finally:
        e2e_evidence._ACTIVE.reset(token)
    assert recorder.data["failure_stage"] == "compare"
    assert [view["title"] for view in recorder.data["views"]] == ["Frame 0 / native", "Frame 0 / reference"]
    assert all((recorder.directory / view["image"]["artifact"]).is_file() for view in recorder.data["views"])
    np.testing.assert_array_equal(reference, np.ones((8, 8, 3), dtype=np.float32))
