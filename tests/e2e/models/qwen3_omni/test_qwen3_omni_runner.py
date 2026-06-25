"""Qwen3-Omni-owned tests for omni runner CLI behavior."""

from __future__ import annotations

import subprocess

from tests.e2e.models.qwen3_omni.e2e_plugins.runners import omni
from tests.e2e_harness.contracts import E2ECase, RunContext, StageSpec


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
        return subprocess.CompletedProcess(cmd, 0, stdout="hello back", stderr="")

    monkeypatch.setattr(omni.subprocess, "run", _fake_run)

    out = omni.OmniMultimodalRunner().run_stage(
        case, StageSpec(name="thinker_decode"), ctx)

    cmd = captured["cmd"]
    assert cmd[1] == "run"
    assert "--stage" not in cmd
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
            cmd, 0, stdout='{"embedding": [0.1, 0.2], "dim": 2}\n', stderr="")

    monkeypatch.setattr(omni.subprocess, "run", _fake_run)

    out = omni.OmniMultimodalRunner().run_stage(
        case, StageSpec(name="vision_encode"), ctx)

    cmd = captured["cmd"]
    assert cmd[1] == "embed"
    assert "--stage" not in cmd
    assert out.metadata["entrypoint"] == "embed"
    assert out.data["embedding"] == [0.1, 0.2]


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

    out = omni.CompositePipelineRunner().run_stage(
        case, StageSpec(name="end_to_end"), ctx)

    cmd = captured["cmd"]
    assert cmd[1] == "run"
    assert "--stage" not in cmd
    assert out.metadata["entrypoint"] == "run"
