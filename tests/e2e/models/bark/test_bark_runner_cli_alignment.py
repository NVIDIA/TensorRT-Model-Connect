# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bark-owned E2E runner to CLI alignment tests."""

from __future__ import annotations

import subprocess

from tests.e2e.models.bark.e2e_plugins.runners import audio_speech
from tests.e2e_harness.contracts import E2ECase, RunContext, StageSpec


def _make_case(inputs: dict | None = None, **overrides) -> E2ECase:
    defaults = dict(
        name="bark-case",
        hf_id="dummy/bark",
        family="bark",
        runtime_strategy="text_to_audio_bark",
        task_strategy="text_to_audio",
        bundle="bark-case.trtfb",
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


def test_distributed_audio_runner_wraps_mpirun_once(monkeypatch, tmp_path):
    case = _make_case(
        inputs={"prompt": "hello"},
        metadata={
            "distributed_runtime": {
                "enabled": True,
                "launcher": "mpirun",
                "world_size": 4,
            },
        },
        determinism={"seed": 42},
    )
    ctx = _make_ctx(case, tmp_path)

    captured: dict = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(audio_speech.subprocess, "run", _fake_run)

    out = audio_speech.TextToAudioRunner().run_stage(
        case, StageSpec(name="generate"), ctx)

    cmd = captured["cmd"]
    assert cmd[:4] == ["mpirun", "--tag-output", "-np", "4"]
    assert cmd.count("mpirun") == 1
    assert "trtmc_rank_audio" in cmd
    assert "audio_bark.seed=42" in cmd
    assert out.metadata["command"] == cmd
