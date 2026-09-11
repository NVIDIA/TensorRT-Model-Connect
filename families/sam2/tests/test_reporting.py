# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The operational receipt gate retains native-only mask views on failure."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from families.sam2.tests import operational_oracle, reporting, test_e2e as e2e
from tools import e2e_evidence


def test_native_masks_survive_contract_failure(monkeypatch, tmp_path: Path):
    name = next(iter(e2e.CASES))
    masks_path = tmp_path / "native-masks.u8"
    masks = np.memmap(
        masks_path,
        mode="w+",
        dtype=np.uint8,
        shape=(reporting._FRAME_COUNT, reporting._HEIGHT, reporting._WIDTH),
    )
    masks[:] = 0
    masks[0] = 1
    masks.flush()
    del masks
    monkeypatch.setattr(e2e, "_model_dir", lambda manifest: tmp_path)
    monkeypatch.setattr(e2e, "_runtime", lambda manifest: tmp_path)
    monkeypatch.setattr(e2e, "_build", lambda *args: None)
    monkeypatch.setattr(e2e, "_operational_receipt", lambda *args: {"binary_masks": True})

    def fail(*args):
        raise AssertionError("original bundle contract failed")

    monkeypatch.setattr(operational_oracle, "assert_bundle_contract", fail)
    recorder = e2e_evidence.Evidence(
        tmp_path / "evidence", family="sam2", case=name, source_revision="a" * 40, roots=(tmp_path,)
    )
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        with pytest.raises(AssertionError, match="original bundle contract failed"):
            e2e.test_public_core_invariant_e2e(name, tmp_path)
    finally:
        e2e_evidence._ACTIVE.reset(token)
    assert recorder.data["failure_stage"] == "compare"
    assert recorder.data["reference"]["mode"] == "contract_only"
    assert len(recorder.data["views"]) == 5
    for index, view in enumerate(recorder.data["views"]):
        assert "no reference masks" in view["caption"]
        with Image.open(recorder.directory / view["image"]["artifact"]) as image:
            assert np.all(np.asarray(image) == (255 if index == 0 else 0))
    assert "masks" in recorder.data["native_artifacts"]


def test_layout_matches_native_probe_and_rendering_is_opt_in(tmp_path: Path):
    source = Path(__file__).with_name("cpp") / "operational_probe.cpp"
    code = source.read_text()
    assert f"kHeight = {reporting._HEIGHT};" in code
    assert f"kWidth = {reporting._WIDTH};" in code
    assert f"kFrameCount = {reporting._FRAME_COUNT};" in code
    reporting.record_mask_views(tmp_path / "missing.u8", tmp_path / "views")
    assert not (tmp_path / "views").exists()


def test_malformed_mask_file_records_unavailable(tmp_path: Path):
    masks = tmp_path / "native-masks.u8"
    masks.write_bytes(b"\x01")
    recorder = e2e_evidence.Evidence(
        tmp_path / "evidence",
        family="sam2",
        case="bad-layout",
        source_revision="a" * 40,
        roots=(tmp_path,),
    )
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        reporting.record_mask_views(masks, tmp_path / "views")
    finally:
        e2e_evidence._ACTIVE.reset(token)
    assert "unavailable" in recorder.data["views"][0]["caption"]
    assert not (tmp_path / "views").exists()
