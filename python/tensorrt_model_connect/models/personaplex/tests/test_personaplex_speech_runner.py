# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the PersonaPlex speech runtime command."""

from __future__ import annotations

import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tensorrt_model_connect.models.personaplex.tests.e2e_plugins.runners import audio_speech
from tests.e2e_harness.contracts import E2ECase, RunContext, StageSpec


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.array([1024, -1024, 512, -512], dtype="<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24_000)
        wav.writeframes(samples.tobytes())


def test_runner_uses_total_frame_budget_without_adding_tail_frames(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audio_path = tmp_path / "input.wav"
    _write_wav(audio_path)
    case = E2ECase(
        name="personaplex-runner-test",
        hf_id="nvidia/personaplex-7b-v1",
        family="personaplex",
        runtime_strategy="personaplex_speech_to_speech",
        task_strategy="speech_to_speech",
        bundle="personaplex.bundle",
        inputs={
            "audio": str(audio_path),
            "speech_test_max_frames": 100,
        },
    )
    ctx = RunContext(
        case=case,
        binary_path="/runtime/trtmc",
        engine_dir=str(tmp_path / "engines"),
        artifacts_dir=str(tmp_path / "artifacts"),
        model_plugin_dir="/runtime/models",
    )
    seen_command: list[str] = []

    def fake_run(command, **kwargs):
        seen_command.extend(command)
        output_path = Path(command[command.index("--audio-out") + 1])
        _write_wav(output_path)
        return SimpleNamespace(
            returncode=0,
            stdout="generated",
            stderr=(
                "[speech] Output frame 0: 1 2 3 4 5 6 7 8\n"
                "[speech] Output frame 1: 9 10 11 12 13 14 15 16\n"
            ),
        )

    monkeypatch.setattr(audio_speech.subprocess, "run", fake_run)

    output = audio_speech.SpeechToSpeechRunner().run_stage(
        case, StageSpec(name="full_generation"), ctx
    )

    assert seen_command[seen_command.index("--max-new-tokens") + 1] == "100"
    assert seen_command[seen_command.index("--model-plugin-dir") + 1] == "/runtime/models"
    assert "--tail-frames" not in seen_command
    assert output.data["num_frames"] == 2
    assert output.data["wav_exists"] is True
    assert Path(output.data["wav_path"]) == (
        tmp_path / "artifacts" / case.name / "trt_speech_out.wav"
    )
    assert Path(output.data["wav_path"]).is_file()
