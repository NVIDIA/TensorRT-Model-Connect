# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SAM-owned E2E runner to CLI alignment tests."""

from __future__ import annotations

import subprocess

from tensorrt_model_connect.models.sam.tests.e2e_plugins import runner as sam_runner
from tensorrt_model_connect.models.sam.tests.e2e_plugins.runners import segmentation
from tests.e2e_harness.contracts import E2ECase, RunContext, StageSpec


def _make_case(inputs: dict | None = None, **overrides) -> E2ECase:
    defaults = dict(
        name="sam-case",
        hf_id="dummy/sam",
        family="sam",
        runtime_strategy="sam_prompted_segmentation",
        task_strategy="prompted_segmentation",
        bundle="sam-case.bundle",
        inputs=inputs or {},
    )
    defaults.update(overrides)
    return E2ECase(**defaults)


def _make_ctx(case: E2ECase, tmp_path) -> RunContext:
    binary_path = tmp_path / "trtmc"
    binary_path.write_text("", encoding="utf-8")
    return RunContext(
        case=case,
        artifacts_dir=str(tmp_path),
        binary_path=str(binary_path),
        engine_dir=str(tmp_path),
    )


def test_prompted_segmentation_runner_uses_segment_prompted_cli(monkeypatch, tmp_path):
    image_path = tmp_path / "img.jpg"
    image_path.write_text("img", encoding="utf-8")
    case = _make_case(
        inputs={"image": str(image_path), "point_x": 0.25, "point_y": 0.75},
    )
    ctx = _make_ctx(case, tmp_path)

    captured: dict = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(segmentation.subprocess, "run", _fake_run)

    out = sam_runner.SamPromptedSegmentationRunner().run_stage(
        case, StageSpec(name="full_inference"), ctx)

    cmd = captured["cmd"]
    assert cmd[1] == "segment-prompted"
    assert "--output" in cmd
    assert "--point-x" in cmd
    assert "--point-y" in cmd
    assert "--output-dir" not in cmd
    assert "--point" not in cmd
    assert out.metadata["command"] == cmd
