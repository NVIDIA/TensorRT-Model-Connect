"""Magpie-owned E2E runner to CLI alignment tests."""

from __future__ import annotations

import subprocess

from tests.e2e.models.magpie_tts.e2e_plugins.runners import audio_speech
from tests.e2e_harness.contracts import E2ECase, RunContext, StageSpec


def _make_case(inputs: dict | None = None, **overrides) -> E2ECase:
    defaults = dict(
        name="magpie-case",
        hf_id="dummy/magpie",
        family="magpie_tts",
        runtime_strategy="text_to_audio_magpie",
        task_strategy="text_to_audio",
        bundle="magpie-case.trtfb",
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


def test_audio_runner_maps_runtime_config_to_set_flags(monkeypatch, tmp_path):
    case = _make_case(
        inputs={"prompt": "hello", "max_new_tokens": 12},
        metadata={
            "runtime_config": {
                "audio_magpie": {
                    "cfg_scale": 2.5,
                    "temperature": 0.6,
                    "seed": 42,
                }
            }
        },
    )
    ctx = _make_ctx(case, tmp_path)

    captured: dict = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env", {})
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(audio_speech.subprocess, "run", _fake_run)

    out = audio_speech.TextToAudioRunner().run_stage(
        case, StageSpec(name="generate"), ctx)

    cmd = captured["cmd"]
    assert "--set" in cmd
    assert "audio_magpie.cfg_scale=2.5" in cmd
    assert "audio_magpie.temperature=0.6" in cmd
    assert "audio_magpie.seed=42" in cmd
    assert "TRTMC_MAGPIE_SEED" not in captured["env"]
    assert out.metadata["command"] == cmd
