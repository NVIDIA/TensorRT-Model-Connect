# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the PersonaPlex speech runtime command."""

from __future__ import annotations

import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.e2e.models.personaplex.e2e_plugins.runners import audio_speech
from tests.e2e_harness.contracts import E2ECase, RunContext, StageOutput, StageSpec


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
        bundle="personaplex.trtfb",
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
                "[speech] Input frame 0: 101 102 103 104 105 106 107 108\n"
                "[speech] Input frame 1: 109 110 111 112 113 114 115 116\n"
                "[speech] Output text frame 0: 201\n"
                "[speech] Output text frame 1: 202\n"
                "[speech] Output frame 0: 1 2 3 4 5 6 7 8\n"
                "[speech] Output frame 1: 9 10 11 12 13 14 15 16\n"
            ),
        )

    monkeypatch.setattr(audio_speech.subprocess, "run", fake_run)

    output = audio_speech.SpeechToSpeechRunner().run_stage(
        case, StageSpec(name="full_generation"), ctx
    )

    assert seen_command[seen_command.index("--max-new-tokens") + 1] == "100"
    assert seen_command[seen_command.index("--model-plugin-dir") + 1] == (
        "/runtime/models"
    )
    assert "--tail-frames" not in seen_command
    assert output.data["num_frames"] == 2
    np.testing.assert_array_equal(
        output.data["input_codec_tokens"],
        np.arange(101, 117, dtype=np.int32).reshape(2, 8),
    )
    np.testing.assert_array_equal(
        output.data["output_text_tokens"],
        np.array([201, 202], dtype=np.int32),
    )
    assert output.data["wav_exists"] is True
    assert Path(output.data["wav_path"]).is_file()


def test_runner_replays_reference_trace_without_replacing_free_generation(
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
        bundle="personaplex.trtfb",
        inputs={"audio": str(audio_path), "speech_test_max_frames": 2},
    )
    reference = StageOutput(
        stage_name="full_generation",
        data={
            "reference_text_tokens": np.array([101, 102], dtype=np.int32),
            "reference_tokens": np.arange(201, 217, dtype=np.int32).reshape(2, 8),
        },
    )
    ctx = RunContext(
        case=case,
        binary_path="/runtime/trtmc",
        engine_dir=str(tmp_path / "engines"),
        artifacts_dir=str(tmp_path / "artifacts"),
        model_plugin_dir="/runtime/models",
        reference_output=reference,
    )
    commands: list[list[str]] = []
    teacher_file_contents = ""

    def fake_run(command, **kwargs):
        nonlocal teacher_file_contents
        commands.append(command)
        output_path = Path(command[command.index("--audio-out") + 1])
        _write_wav(output_path)
        if "--speech-teacher-tokens" not in command:
            return SimpleNamespace(
                returncode=0,
                stdout="generated",
                stderr=(
                    "[speech] Output text frame 0: 999\n"
                    "[speech] Output text frame 1: 998\n"
                    "[speech] Output frame 0: 1 2 3 4 5 6 7 8\n"
                    "[speech] Output frame 1: 9 10 11 12 13 14 15 16\n"
                ),
            )
        teacher_path = Path(command[command.index("--speech-teacher-tokens") + 1])
        teacher_file_contents = teacher_path.read_text(encoding="utf-8")
        return SimpleNamespace(
            returncode=0,
            stdout="teacher replay",
            stderr=(
                "[speech.teacher] frame 0 text_target=101 text_pred=101 "
                "audio_target=201,202,203,204,205,206,207,208 "
                "audio_pred=201,202,203,204,205,206,207,208\n"
                "[speech.teacher] frame 1 text_target=102 text_pred=999 "
                "audio_target=209,210,211,212,213,214,215,216 "
                "audio_pred=209,210,999,212,213,214,215,216\n"
            ),
        )

    monkeypatch.setattr(audio_speech.subprocess, "run", fake_run)

    output = audio_speech.SpeechToSpeechRunner().run_stage(
        case, StageSpec(name="full_generation"), ctx
    )

    assert len(commands) == 2
    assert "--speech-teacher-tokens" not in commands[0]
    assert "--speech-teacher-tokens" in commands[1]
    assert all(
        command[command.index("--model-plugin-dir") + 1] == "/runtime/models"
        for command in commands
    )
    assert Path(commands[0][commands[0].index("--audio-out") + 1]) == (
        tmp_path
        / "artifacts"
        / "personaplex-runner-test"
        / "trt_speech_out.wav"
    )
    assert Path(commands[1][commands[1].index("--audio-out") + 1]) == (
        tmp_path
        / "artifacts"
        / "personaplex-runner-test"
        / "teacher_speech_out.wav"
    )
    assert teacher_file_contents == (
        "101 201 202 203 204 205 206 207 208\n102 209 210 211 212 213 214 215 216\n"
    )
    teacher_command = output.metadata["teacher_forced_command"]
    teacher_tokens_path = Path(
        teacher_command[teacher_command.index("--speech-teacher-tokens") + 1]
    )
    assert teacher_tokens_path == (
        tmp_path
        / "artifacts"
        / "personaplex-runner-test"
        / "teacher_tokens.txt"
    )
    assert teacher_tokens_path.read_text(encoding="utf-8") == teacher_file_contents
    np.testing.assert_array_equal(
        output.data["teacher_text_predicted_tokens"],
        np.array([101, 999], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        output.data["teacher_audio_predicted_tokens"],
        np.array(
            [
                [201, 202, 203, 204, 205, 206, 207, 208],
                [209, 210, 999, 212, 213, 214, 215, 216],
            ],
            dtype=np.int32,
        ),
    )
    np.testing.assert_array_equal(
        output.data["output_text_tokens"],
        np.array([999, 998], dtype=np.int32),
    )


def test_runner_can_skip_teacher_replay_for_behavior_benchmark(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audio_path = tmp_path / "input.wav"
    _write_wav(audio_path)
    case = E2ECase(
        name="personaplex-behavior-test",
        hf_id="nvidia/personaplex-7b-v1",
        family="personaplex",
        runtime_strategy="personaplex_speech_to_speech",
        task_strategy="speech_to_speech",
        bundle="personaplex.trtfb",
        inputs={
            "audio": str(audio_path),
            "speech_test_max_frames": 2,
            "disable_teacher_replay": True,
        },
    )
    reference = StageOutput(
        stage_name="full_generation",
        data={
            "reference_text_tokens": np.array([101, 102], dtype=np.int32),
            "reference_tokens": np.arange(201, 217, dtype=np.int32).reshape(2, 8),
        },
    )
    ctx = RunContext(
        case=case,
        binary_path="/runtime/trtmc",
        engine_dir=str(tmp_path / "engines"),
        artifacts_dir=str(tmp_path / "artifacts"),
        reference_output=reference,
    )
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        _write_wav(Path(command[command.index("--audio-out") + 1]))
        return SimpleNamespace(returncode=0, stdout="generated", stderr="")

    monkeypatch.setattr(audio_speech.subprocess, "run", fake_run)

    output = audio_speech.SpeechToSpeechRunner().run_stage(
        case, StageSpec(name="full_generation"), ctx
    )

    assert len(commands) == 1
    assert "--speech-teacher-tokens" not in commands[0]
    assert output.metadata["teacher_forced_command"] is None
