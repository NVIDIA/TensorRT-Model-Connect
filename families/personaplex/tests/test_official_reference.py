# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from . import official_reference
from .personaplex_audio_compat import sphn


TEST_ROOT = Path(__file__).resolve().parent
SOURCE_REVISION = "3428dfd95309a7f3c84fd93259ded0f810d1ff91"


def _write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24_000)
        output.writeframes(b"\0\0" * 320)


def test_reference_source_is_pinned_and_required(monkeypatch, tmp_path: Path) -> None:
    source = json.loads((TEST_ROOT / "reference-source.json").read_text())
    assert source == {"repository": "NVIDIA/personaplex", "revision": SOURCE_REVISION}
    monkeypatch.delenv(official_reference.SOURCE_ENVIRONMENT, raising=False)
    with pytest.raises(RuntimeError, match=official_reference.SOURCE_ENVIRONMENT):
        official_reference._source()
    monkeypatch.setenv(official_reference.SOURCE_ENVIRONMENT, str(tmp_path))
    with pytest.raises(RuntimeError, match=official_reference.SOURCE_ENTRYPOINT):
        official_reference._source()


def test_audio_compat_reads_the_checked_in_float_wav() -> None:
    audio, sample_rate = sphn.read(str(TEST_ROOT / "data/Recording.wav"))
    assert sample_rate == 24_000
    assert audio.shape == (1, 99_840)
    assert audio.dtype == np.float32
    same = sphn.resample(audio, src_sample_rate=24_000, dst_sample_rate=24_000)
    assert np.array_equal(same, audio)
    with pytest.raises(RuntimeError, match="24 kHz"):
        sphn.resample(audio, src_sample_rate=24_000, dst_sample_rate=16_000)


def test_generate_runs_the_official_source_and_requires_live_outputs(
    monkeypatch, tmp_path: Path
) -> None:
    source = tmp_path / "source"
    entrypoint = source / official_reference.SOURCE_ENTRYPOINT
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("# pinned official entrypoint\n", encoding="utf-8")
    monkeypatch.setenv(official_reference.SOURCE_ENVIRONMENT, str(source))
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    input_wav = tmp_path / "input.wav"
    _write_wav(input_wav)
    captured = {}

    def run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        tokens = Path(command[command.index("--tokens-output") + 1])
        audio = Path(command[command.index("--audio-output") + 1])
        np.save(tokens, np.zeros((5, 8), dtype=np.int32), allow_pickle=False)
        _write_wav(audio)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(official_reference.subprocess, "run", run)
    result = official_reference.generate(
        model_dir,
        input_wav,
        tmp_path / "output",
        max_frames=5,
        precision="bf16",
        timeout_s=30,
    )

    command = captured["command"]
    assert command[:2] == [sys.executable, str(Path(official_reference.__file__).resolve())]
    assert command[command.index("--official-repo") + 1] == str(source)
    assert command[command.index("--model-dir") + 1] == str(model_dir)
    assert command[command.index("--max-frames") + 1] == "5"
    assert command[command.index("--precision") + 1] == "bf16"
    assert captured["kwargs"]["timeout"] == 30
    assert captured["kwargs"]["env"]["HF_HUB_OFFLINE"] == "1"
    assert (
        captured["kwargs"]["env"]["PYTHONPATH"]
        .split(os.pathsep)[0]
        .endswith("personaplex_audio_compat")
    )
    assert result["speech_tokens"].shape == (5, 8)
    assert result["sample_rate"] == 24_000
    assert Path(result["audio"]).is_file()
