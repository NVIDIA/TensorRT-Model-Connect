# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-Omni-owned tests for omni runner CLI behavior."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.e2e.models.qwen3_omni.e2e_plugins.runners import omni
from tests.e2e_harness.contracts import E2ECase, RunContext, StageSpec
from tests.e2e_harness.manifest_loader import load_model_manifest


def _make_case(inputs: dict | None = None, **overrides) -> E2ECase:
    defaults = dict(
        name="qwen3-omni-case",
        hf_id="dummy/model",
        family="qwen3_omni",
        runtime_strategy="qwen3_omni_multimodal",
        task_strategy="omni_multimodal",
        bundle="qwen3-omni-case.trtfb",
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


def test_thinker_stage_drops_unsupported_stage_flag(monkeypatch, tmp_path) -> None:
    case = _make_case(inputs={"prompt": "hello", "max_new_tokens": 7})
    ctx = _make_ctx(case, tmp_path)

    captured: dict = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["timeout"] = kwargs["timeout"]
        return subprocess.CompletedProcess(cmd, 0, stdout="hello back", stderr="")

    monkeypatch.setattr(omni.subprocess, "run", _fake_run)

    out = omni.OmniMultimodalRunner().run_stage(case, StageSpec(name="thinker_decode"), ctx)

    cmd = captured["cmd"]
    assert cmd[1] == "run"
    assert "--stage" not in cmd
    assert captured["timeout"] == 600
    assert out.metadata["cli_stage_supported"] is False
    assert out.metadata["entrypoint"] == "run"


def test_vision_stage_maps_to_embed_without_stage_flag(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "img.jpg"
    image_path.write_text("img", encoding="utf-8")
    case = _make_case(inputs={"image": str(image_path), "prompt": "caption me"})
    ctx = _make_ctx(case, tmp_path)

    captured: dict = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(
            cmd, 0, stdout='{"embedding": [0.1, 0.2], "dim": 2}\n', stderr=""
        )

    monkeypatch.setattr(omni.subprocess, "run", _fake_run)

    out = omni.OmniMultimodalRunner().run_stage(case, StageSpec(name="vision_encode"), ctx)

    cmd = captured["cmd"]
    assert cmd[1] == "embed"
    assert "--stage" not in cmd
    assert out.metadata["entrypoint"] == "embed"
    assert out.data["embedding"] == [0.1, 0.2]


def test_qwen3_omni_manifest_owns_extended_runtime_budget() -> None:
    manifest_path = Path(__file__).parent / "manifests" / "qwen3-omni-30b-a3b-instruct.json"
    model = load_model_manifest(manifest_path)

    assert model.testcases[0].inputs["runtime_timeout_s"] == 900


def test_omni_runner_uses_model_owned_runtime_budget(monkeypatch, tmp_path) -> None:
    case = _make_case(
        inputs={
            "prompt": "hello",
            "max_new_tokens": 7,
            "runtime_timeout_s": 900,
        }
    )
    ctx = _make_ctx(case, tmp_path)
    captured: dict = {}

    def _fake_run(cmd, **kwargs):
        captured["timeout"] = kwargs["timeout"]
        return subprocess.CompletedProcess(cmd, 0, stdout="hello back", stderr="")

    monkeypatch.setattr(omni.subprocess, "run", _fake_run)

    omni.OmniMultimodalRunner().run_stage(case, StageSpec(name="thinker_decode"), ctx)

    assert captured["timeout"] == 900


def test_talker_runner_captures_thinker_text(monkeypatch, tmp_path) -> None:
    case = _make_case(inputs={"prompt": "hello", "max_new_tokens": 16})
    ctx = _make_ctx(case, tmp_path)

    def _fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="Generated 37845 audio samples\n",
            stderr="[trtmc] Omni Thinker text: Hello from Qwen-Omni!\n",
        )

    monkeypatch.setattr(omni.subprocess, "run", _fake_run)

    out = omni.OmniMultimodalRunner().run_stage(case, StageSpec(name="talker_decode"), ctx)

    assert out.data["thinker_text"] == "Hello from Qwen-Omni!"


def test_omni_timeout_preserves_partial_stderr(monkeypatch, tmp_path) -> None:
    case = _make_case(
        inputs={
            "prompt": "hello",
            "runtime_timeout_s": 900,
        }
    )
    ctx = _make_ctx(case, tmp_path)

    def _timeout(cmd, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd,
            kwargs["timeout"],
            output=b"",
            stderr=b'[trtmc.load_timing] label="omni thinker" still loading\n',
        )

    monkeypatch.setattr(omni.subprocess, "run", _timeout)

    with pytest.raises(RuntimeError, match="model-owned runtime budget of 900s"):
        omni.OmniMultimodalRunner().run_stage(case, StageSpec(name="thinker_decode"), ctx)

    timeout_log = tmp_path / case.name / "omni_thinker_decode_timeout_stderr.log"
    assert "still loading" in timeout_log.read_text(encoding="utf-8")


def test_talker_rejects_simple_waveform_fallback(monkeypatch, tmp_path) -> None:
    case = _make_case(inputs={"prompt": "hello", "max_new_tokens": 16})
    ctx = _make_ctx(case, tmp_path)

    def _fallback(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="",
            stderr=("[trtmc] Omni: no Code2Wav engine, generating simple waveform\n"),
        )

    monkeypatch.setattr(omni.subprocess, "run", _fallback)

    with pytest.raises(RuntimeError, match="Code2Wav engine is missing"):
        omni.OmniMultimodalRunner().run_stage(case, StageSpec(name="talker_decode"), ctx)


def test_composite_runner_uses_run_without_stage_flag(monkeypatch, tmp_path) -> None:
    case = _make_case(
        inputs={"prompt": "hello", "max_new_tokens": 4},
        runtime_strategy="composite_pipeline",
        task_strategy="composite_pipeline",
    )
    ctx = _make_ctx(case, tmp_path)

    captured: dict = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(omni.subprocess, "run", _fake_run)

    out = omni.CompositePipelineRunner().run_stage(case, StageSpec(name="end_to_end"), ctx)

    cmd = captured["cmd"]
    assert cmd[1] == "run"
    assert "--stage" not in cmd
    assert out.metadata["entrypoint"] == "run"
