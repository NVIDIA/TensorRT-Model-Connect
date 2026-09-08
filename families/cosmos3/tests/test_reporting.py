# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native-only Cosmos previews must not claim a reference comparison."""

from pathlib import Path

from PIL import Image

from families.cosmos3.tests.reporting import record_native_preview
from tools import e2e_evidence


def test_native_only_frames_are_visible_and_bounded(tmp_path: Path):
    frames = tmp_path / "native"
    frames.mkdir()
    for index in range(12):
        Image.new("RGB", (4, 4), color=(index, 0, 0)).save(frames / f"frame-{index:04d}.png")
    recorder = e2e_evidence.Evidence(tmp_path / "evidence", family="cosmos3", case="preview-probe",
                                   source_revision="a" * 40, roots=(tmp_path,))
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        record_native_preview(frames)
    finally:
        e2e_evidence._ACTIVE.reset(token)
    assert len(recorder.data["views"]) == 6
    assert all(view["title"].startswith("Native frame") for view in recorder.data["views"])
    assert "reference" not in recorder.data
