# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LocateAnything-owned tests for object-detection runner CLI behavior."""

from __future__ import annotations

import subprocess

from tests.e2e.models.locateanything.e2e_plugins.runners import object_detection
from tests.e2e_harness.contracts import E2ECase, RunContext, StageSpec


def test_runner_uses_detection_alias_flags(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "img.jpg"
    image_path.write_text("img", encoding="utf-8")
    case = E2ECase(
        name="locateanything-case",
        hf_id="dummy/model",
        family="locateanything",
        runtime_strategy="locateanything_vision_language",
        task_strategy="object_detection",
        bundle="locateanything-case.bundle",
        inputs={"image": str(image_path), "score_threshold": 0.42},
    )
    binary_path = tmp_path / "trtmc"
    binary_path.write_text("", encoding="utf-8")
    ctx = RunContext(
        case=case,
        artifacts_dir=str(tmp_path),
        binary_path=str(binary_path),
        engine_dir=str(tmp_path),
    )

    captured: dict = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(object_detection.subprocess, "run", _fake_run)

    out = object_detection.ObjectDetectionRunner().run_stage(
        case, StageSpec(name="full_inference"), ctx)

    cmd = captured["cmd"]
    assert "--output-json" in cmd
    assert "--score-threshold" in cmd
    assert out.metadata["command"] == cmd
