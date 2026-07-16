# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SAM3-owned E2E runner to CLI alignment tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

from tests.e2e.models.sam3.e2e_plugins import runner as sam3_runner
from tests.e2e.models.sam3.e2e_plugins.runners import segmentation
from tests.e2e_harness import orchestrator
from tests.e2e_harness.contracts import E2ECase, RunContext, StageSpec
from tests.e2e_harness.manifest_loader import load_manifest


def _make_case(inputs: dict | None = None, **overrides) -> E2ECase:
    defaults = dict(
        name="sam3-case",
        hf_id="dummy/sam3",
        family="sam3",
        runtime_strategy="sam3_prompted_segmentation",
        task_strategy="prompted_segmentation",
        bundle="sam3-case.trtfb",
        reference_family="prompted_segmentation_sam3",
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


def test_prompted_segmentation_runner_uses_text_prompt_cli(monkeypatch, tmp_path):
    image_path = tmp_path / "img.jpg"
    image_path.write_text("img", encoding="utf-8")
    case = _make_case(inputs={"image": str(image_path), "prompt": "ear"})
    ctx = _make_ctx(case, tmp_path)

    captured: dict = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(
            cmd,
            1,
            stdout="",
            stderr="Bundle missing sam3 vision_encoder",
        )

    monkeypatch.setattr(segmentation.subprocess, "run", _fake_run)

    out = sam3_runner.Sam3PromptedSegmentationRunner().run_stage(
        case, StageSpec(name="full_inference"), ctx
    )

    cmd = captured["cmd"]
    assert cmd[1] == "segment-prompted"
    assert "--prompt" in cmd
    assert cmd[cmd.index("--prompt") + 1] == "ear"
    assert "--point-x" not in cmd
    assert "--point-y" not in cmd
    assert out.metadata["returncode"] == 1
    assert "sam3 vision_encoder" in out.metadata["stderr"]


def test_sam3_manifest_build_uses_default_cli_precision(monkeypatch, tmp_path):
    manifest_path = Path(__file__).resolve().parent / "manifests" / "sam3.json"
    case = load_manifest(manifest_path)
    ctx = _make_ctx(case, tmp_path)
    captured: dict[str, list[str]] = {}

    def _fake_run(cmd, **_kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.delenv("TRTMC_ENGINE_BUILD_GUARD_DIR", raising=False)
    monkeypatch.setattr(orchestrator.subprocess, "run", _fake_run)

    bundle, _elapsed, error, _build_info = orchestrator._resolve_bundle(case, ctx)

    assert case.metadata["precision"] == "fp32"
    assert bundle == str(Path(ctx.engine_dir) / case.bundle)
    assert error == ""
    assert captured["cmd"][3:6] == ["build", "facebook/sam3", "-o"]
    assert "--precision" not in captured["cmd"]
