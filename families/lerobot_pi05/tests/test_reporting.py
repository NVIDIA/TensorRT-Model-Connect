# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU plumbing controls keep action/reference precision attribution exact."""

import base64
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from families.lerobot_pi05.tests import test_e2e as e2e
from tools import e2e_evidence as evidence


def _run_control(monkeypatch, tmp_path, scenario):
    name = "pi05-libero-base-bf16-ep0000-f000"
    manifest, case = deepcopy(e2e.CASES[name])
    frame = tmp_path / "frame"
    frame.mkdir()
    instruction = "Primary recorded instruction"
    (frame / "observation.json").write_text(json.dumps({"state": [0.] * 8, "task": instruction}))
    png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=")
    for image in ("image.png", "wrist_image.png"):
        (frame / image).write_bytes(png)
    actions = np.ones((e2e.CHUNK, e2e.ACTION_DIM), dtype="<f4")
    case["inputs"] = {"frame": "frame", "noise": "noise.f32",
                      "reference_fp32": "reference_fp32.f32", "reference_bf16": "reference_bf16.f32"}
    actions.tofile(tmp_path / "noise.f32")
    for key in ("reference_fp32", "reference_bf16"):
        values = actions + 1 if scenario == key + "_mismatch" else actions
        values.tofile(tmp_path / case["inputs"][key])
    monkeypatch.setattr(e2e, "TEST_ROOT", tmp_path)
    monkeypatch.setitem(e2e.CASES, name, (manifest, case))
    monkeypatch.setattr(e2e, "_qualification_binary", lambda: (tmp_path / "binary", tmp_path))
    summary = {"rows": e2e.CHUNK, "columns": e2e.ACTION_DIM, "wall_ms": [1.], "load_ms": 1.,
               "rejections": [{"case": key, "code": "invalid_argument", "message": value}
                              for key, value in e2e.EXPECTED_REJECTIONS.items()]}
    commands, calls = [], []

    def run(command, **kwargs):
        commands.append(command)
        actions.tofile(command[command.index("--output") + 1])
        return SimpleNamespace(returncode=0, args=command, stdout=json.dumps(summary), stderr="")

    original_reference = e2e._reference

    def read(selected, key):
        calls.append(key)
        if scenario == key + "_unavailable":
            raise FileNotFoundError(key)
        return original_reference(selected, key)

    monkeypatch.setattr(e2e.subprocess, "run", run)
    monkeypatch.setattr(e2e, "_reference", read)
    bundle = tmp_path / "model.bundle"
    bundle.write_bytes(b"CPU plumbing fixture")
    recorder = evidence.Evidence(tmp_path / "report", family="lerobot_pi05", case=name,
                                 source_revision="a" * 40, roots=(tmp_path,))
    token = evidence._ACTIVE.set(recorder)
    error = None
    try:
        try:
            e2e.test_pi05_bundle_matches_the_lerobot_reference_e2e(name, lambda _: bundle, tmp_path)
        except (AssertionError, FileNotFoundError) as caught:
            error = caught
    finally:
        evidence._ACTIVE.reset(token)
    return recorder, error, calls, commands


@pytest.mark.parametrize("scenario", ["success", "reference_fp32_unavailable", "reference_bf16_unavailable",
                                     "reference_fp32_mismatch", "reference_bf16_mismatch"])
def test_primary_comparison_keeps_the_reference_for_the_enforced_gate(monkeypatch, tmp_path, scenario):
    recorder, error, calls, commands = _run_control(monkeypatch, tmp_path, scenario)
    data = recorder.data
    assert data["inputs"]["prompt"] == "Primary recorded instruction"
    assert data["native"]["values"]["preview"][0] == 1.
    assert len(commands) == 1
    assert commands[0][commands[0].index("--repeat") + 1] == "20"
    if scenario == "success":
        assert error is None
        assert calls == ["reference_fp32", "reference_bf16"]
        assert data["reference"]["name"] == "reference_fp32"
        assert data["reference"]["values"]["preview"][0] == 1.
    elif scenario.endswith("_unavailable"):
        assert isinstance(error, FileNotFoundError)
        assert data["reference"] == {}
        assert calls[-1] == scenario.removesuffix("_unavailable")
    else:
        assert isinstance(error, AssertionError)
        assert calls == ["reference_fp32", "reference_bf16"]
        assert data["reference"]["name"] == scenario.removesuffix("_mismatch")
        assert data["reference"]["values"]["preview"][0] == 2.


def test_reference_read_does_not_change_default_evidence(monkeypatch, tmp_path):
    values = np.ones((e2e.CHUNK, e2e.ACTION_DIM), dtype="<f4")
    values.tofile(tmp_path / "reference.f32")
    monkeypatch.setattr(e2e, "TEST_ROOT", tmp_path)
    recorder = evidence.Evidence(tmp_path / "report", family="lerobot_pi05", case="read-contract", source_revision="a" * 40)
    recorder.data["reference"] = {"chosen": "fp32"}
    before = deepcopy(recorder.data)
    token = evidence._ACTIVE.set(recorder)
    try:
        actual = e2e._reference({"inputs": {"reference_bf16": "reference.f32"}}, "reference_bf16")
    finally:
        evidence._ACTIVE.reset(token)
    np.testing.assert_array_equal(actual, values)
    assert {key: value for key, value in recorder.data.items() if key != "checks"} == {
        key: value for key, value in before.items() if key != "checks"
    }
    assert [check["expression"] for check in recorder.data["checks"]] == ["path.exists()"]
