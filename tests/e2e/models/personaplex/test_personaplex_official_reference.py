# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path

import numpy as np

from tests.e2e.models.personaplex.e2e_plugins.references import (
    official_personaplex,
)
from tests.e2e_harness.contracts import RunContext, StageSpec
from tests.e2e_harness.manifest_loader import get_case_by_name


MODEL_DIR = Path(__file__).resolve().parent


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24_000)
        output.writeframes(b"\0\0" * 320)


def test_full_manifest_selects_live_official_reference() -> None:
    case = get_case_by_name("personaplex-7b", MODEL_DIR)

    assert case is not None
    assert case.reference_backend == "personaplex_official"
    assert case.oracle_level == "L1_external_reference"


def test_official_reference_runs_same_audio_and_frame_budget(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "personaplex"
    entrypoint = source / "moshi/moshi/offline.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("# pinned source\n", encoding="utf-8")
    monkeypatch.setenv("PERSONAPLEX_OFFICIAL_REPO", str(source))
    input_wav = tmp_path / "input.wav"
    _write_wav(input_wav)
    captured: list[str] = []

    def run(command, **_kwargs):
        captured.extend(command)
        token_path = Path(command[command.index("--tokens-output") + 1])
        audio_path = Path(command[command.index("--audio-output") + 1])
        metadata_path = Path(command[command.index("--metadata-output") + 1])
        token_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(token_path, np.zeros((5, 8), dtype=np.int32))
        _write_wav(audio_path)
        metadata_path.write_text(
            json.dumps(
                {
                    "model_id": "nvidia/personaplex-7b-v1",
                    "resolved_revision": "a" * 40,
                    "reference_source_revision": (
                        official_personaplex.REFERENCE_SOURCE_REVISION
                    ),
                    "decoding": "greedy",
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(official_personaplex.subprocess, "run", run)
    case = get_case_by_name("personaplex-7b", MODEL_DIR)
    assert case is not None
    case.inputs["audio"] = str(input_wav)
    case.inputs["speech_test_max_frames"] = 5
    output = official_personaplex.OfficialPersonaPlexReference().run_stage(
        case,
        StageSpec(name="full_generation"),
        RunContext(
            case=case,
            artifacts_dir=str(tmp_path / "artifacts"),
            reference_python="/profiles/personaplex/bin/python",
        ),
    )

    assert captured[0] == "/profiles/personaplex/bin/python"
    assert captured[1].endswith("/personaplex/official_reference.py")
    assert captured[captured.index("--input-wav") + 1] == str(input_wav)
    assert captured[captured.index("--max-frames") + 1] == "5"
    assert captured[captured.index("--official-repo") + 1] == str(source)
    assert output.data["reference_tokens"].shape == (5, 8)
    assert output.data["num_frames"] == 5
    assert output.data["sample_rate"] == 24_000
    assert output.metadata["command"] == captured
