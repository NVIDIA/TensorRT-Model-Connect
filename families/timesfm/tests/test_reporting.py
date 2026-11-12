# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the last forecast's window, mask and available sides together."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from families.timesfm.tests import etth1, test_e2e as e2e
from tools import e2e_evidence
from tools.e2e_report import render_case


def _forecast(values):
    return {
        "values": values, "shape": [2, 1], "task": "series_to_point_forecast", "kind": "point",
        "axes": ["horizon", "channel"], "horizon_steps": [1, 2], "channel_names": [], "channel_units": [],
    }


@pytest.mark.parametrize("fail_reference", [False, True])
def test_masked_request_records_window_zero_without_previous_reference(tmp_path, monkeypatch, fail_reference):
    windows = [{"past_values": [float(i + 1), float(i + 2)], "frequency": 0} for i in range(10)]
    original = copy.deepcopy(windows)
    monkeypatch.setattr(etth1, "windows", lambda: windows)
    case = {"name": etth1.CASE}
    manifest = {"name": "report", "family": "timesfm", "task": "series_to_point_forecast",
                "bundle": "model.bundle", "hf_id": "example/model"}
    monkeypatch.setattr(e2e, "CASES", {etth1.CASE: (tmp_path / "manifest.json", manifest, case)})
    monkeypatch.setattr(e2e, "_model_dir", lambda *args: tmp_path)
    monkeypatch.setattr(e2e, "_runtime", lambda *args: (tmp_path / "binary", tmp_path))
    monkeypatch.setattr(e2e, "_build", lambda *args: None)
    calls, sdk_calls = [], []
    recorder = e2e_evidence.Evidence(tmp_path / "evidence", family="timesfm", case="report",
                                    source_revision="a" * 40, roots=(tmp_path,))

    def native(binary, runtime, bundle, model, config, current, directory):
        assert recorder.data["native"] is recorder.data["reference"] is None
        assert recorder.data["views"] == []
        calls.append(("native", copy.deepcopy(current["inputs"])))
        return _forecast(current["inputs"]["past_values"])

    def reference(model, config, current, directory):
        calls.append(("reference", copy.deepcopy(current["inputs"])))
        assert recorder.data["reference"] is None
        if current["inputs"]["frequency"] == 1:
            assert recorder.data["inputs"]["window_index"] == 0
            assert recorder.data["inputs"]["variant"] == "masked_frequency_one"
            if fail_reference:
                raise RuntimeError("masked reference failed")
        return {"values": np.asarray([current["inputs"]["past_values"]])}

    monkeypatch.setattr(e2e, "_native", native)
    monkeypatch.setattr(e2e, "_official_reference", reference)
    monkeypatch.setattr(e2e, "_assert_sdk_consumers", lambda *args: sdk_calls.append(args))
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        if fail_reference:
            with pytest.raises(RuntimeError, match="masked reference failed"):
                e2e.test_official_checkpoint_e2e(etth1.CASE, tmp_path)
        else:
            e2e.test_official_checkpoint_e2e(etth1.CASE, tmp_path)
    finally:
        e2e_evidence._ACTIVE.reset(token)
    assert windows == original
    assert len(calls) == 22 and len(sdk_calls) == (1 if fail_reference else 2)
    assert recorder.data["inputs"]["window_index"] == 0
    assert recorder.data["inputs"]["case"]["inputs"]["past_values"] == windows[0]["past_values"]
    assert recorder.data["native"]["values"] == windows[0]["past_values"]
    if fail_reference:
        assert recorder.data["reference"] is None
    top = render_case(recorder.data, recorder.directory).split('<details class="case-details">', 1)[0]
    assert "Recorded window 0" in top and "Recorded window 9" not in top


def test_failed_second_sdk_cannot_display_the_first_sdk_output(tmp_path, monkeypatch):
    for language in ("c", "cpp"):
        (tmp_path / f"test_timesfm_sdk_{language}").touch()
    monkeypatch.setattr(e2e, "_required_path", lambda *args: tmp_path)
    expected = {"values": np.asarray([[1.0, 2.0]])}
    recorder = e2e_evidence.Evidence(tmp_path / "evidence", family="timesfm", case="report",
                                    source_revision="a" * 40, roots=(tmp_path,))

    def run(command, **kwargs):
        if command[0].endswith("_cpp"):
            raise RuntimeError("C++ consumer failed")
        return SimpleNamespace(stdout=json.dumps(_forecast([1.0, 2.0])))

    monkeypatch.setattr(e2e.subprocess, "run", run)
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        with pytest.raises(RuntimeError, match=r"C\+\+ consumer failed"):
            e2e._assert_sdk_consumers(tmp_path, tmp_path / "model.bundle",
                                      {"tensor_parallel_size": 1, "task": "series_to_point_forecast"},
                                      {"inputs": {"frequency": 1}}, tmp_path, expected, etth1.GATES)
    finally:
        e2e_evidence._ACTIVE.reset(token)
    assert recorder.data["inputs"]["sdk_language"] == "cpp"
    assert recorder.data["native"] is None and recorder.data["views"] == []
    assert any(row["name"] == "native" and row["value"] for row in recorder.data["observations"])
