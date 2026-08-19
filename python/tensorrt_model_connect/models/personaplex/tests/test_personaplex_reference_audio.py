# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for PersonaPlex reference-audio materialization."""

from __future__ import annotations

import json
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tensorrt_model_connect.models.personaplex.tests.e2e_plugins.references import torch_reference
from tests.e2e_harness.contracts import E2ECase, RunContext, StageSpec
from tests.e2e_harness.manifest_loader import get_case_by_name


def _case(tokens_path: Path) -> E2ECase:
    return E2ECase(
        name="personaplex-reference-test",
        hf_id="nvidia/personaplex-7b-v1",
        family="personaplex",
        runtime_strategy="personaplex_speech_to_speech",
        task_strategy="speech_to_speech",
        reference_backend="torch_reference",
        inputs={"speech_reference_tokens": str(tokens_path)},
    )


def _write_test_wav(path: Path, *, sample_rate: int = 24_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.array([0, 1024, -1024, 512], dtype="<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())


def test_manifest_exposes_model_owned_reference_tokens() -> None:
    model_dir = Path(__file__).resolve().parent
    case = get_case_by_name("personaplex-7b-l0", model_dir)

    assert case is not None
    assert case.inputs["speech_reference_tokens"] == str(
        model_dir / "data" / "personaplex_recording_official_tokens_greedy.npy"
    )
    assert Path(case.inputs["speech_reference_tokens"]).is_file()
    assert case.inputs["speech_test_max_frames"] == 100


def test_reference_tokens_are_decoded_to_case_local_wav(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tokens_path = tmp_path / "tokens.npy"
    np.save(tokens_path, np.arange(24, dtype=np.int32).reshape(3, 8))
    case = _case(tokens_path)
    ctx = RunContext(
        case=case,
        artifacts_dir=str(tmp_path / "artifacts"),
        hf_python="/base/python",
        reference_python="/reference/python",
    )
    seen_command: list[str] = []

    def fake_run(command, **kwargs):
        seen_command.extend(command)
        _write_test_wav(Path(command[-1]))
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({
                "sample_rate": 24_000,
                "num_samples": 4,
                "duration_s": 4 / 24_000,
                "rms": 0.03125,
                "codec_model": "kyutai/mimi",
                "codec_backend": "transformers.MimiModel",
            }),
            stderr="codec diagnostic",
        )

    monkeypatch.setattr(torch_reference.subprocess, "run", fake_run)

    output = torch_reference.TorchReference().run_stage(
        case, StageSpec(name="full_generation"), ctx)

    expected_wav = (
        tmp_path / "artifacts" / case.name / "reference_speech.wav"
    )
    assert seen_command[0] == "/reference/python"
    assert seen_command[-2:] == [str(tokens_path), str(expected_wav)]
    assert output.data["reference_tokens"].shape == (3, 8)
    assert output.data["token_shape"] == [3, 8]
    assert output.data["wav_path"] == str(expected_wav)
    assert output.data["wav_exists"] is True
    assert output.data["sample_rate"] == 24_000
    assert output.data["num_samples"] == 4
    assert output.data["codec_model"] == "kyutai/mimi"
    assert expected_wav.is_file()
    assert output.metadata["codec_command"] == seen_command
    assert output.metadata["codec_stderr_log"] == str(
        tmp_path
        / "artifacts"
        / case.name
        / "personaplex_mimi_reference_stderr.log"
    )


def test_reference_codec_failure_is_gating_and_diagnostic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tokens_path = tmp_path / "tokens.npy"
    np.save(tokens_path, np.zeros((2, 8), dtype=np.int32))
    case = _case(tokens_path)
    ctx = RunContext(
        case=case,
        artifacts_dir=str(tmp_path / "artifacts"),
        reference_python="/reference/python",
    )

    monkeypatch.setattr(
        torch_reference.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=17,
            stdout="",
            stderr="Mimi weights are not cached",
        ),
    )

    with pytest.raises(RuntimeError, match=(
        r"PersonaPlex reference WAV generation failed .*rc=17.*"
        r"Mimi weights are not cached"
    )):
        torch_reference.TorchReference().run_stage(
            case, StageSpec(name="full_generation"), ctx)

    stderr_log = (
        tmp_path
        / "artifacts"
        / case.name
        / "personaplex_mimi_reference_stderr.log"
    )
    assert stderr_log.read_text(encoding="utf-8") == "Mimi weights are not cached"


def test_reference_rejects_invalid_wav_format(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tokens_path = tmp_path / "tokens.npy"
    np.save(tokens_path, np.zeros((2, 8), dtype=np.int32))
    case = _case(tokens_path)
    ctx = RunContext(
        case=case,
        artifacts_dir=str(tmp_path / "artifacts"),
        reference_python="/reference/python",
    )

    def fake_run(command, **kwargs):
        _write_test_wav(Path(command[-1]), sample_rate=16_000)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"rms": 0.1}),
            stderr="",
        )

    monkeypatch.setattr(torch_reference.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match=r"sample_rate=16000"):
        torch_reference.TorchReference().run_stage(
            case, StageSpec(name="full_generation"), ctx)
