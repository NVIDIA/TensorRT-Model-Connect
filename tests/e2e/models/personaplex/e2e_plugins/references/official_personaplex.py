# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned official PersonaPlex greedy reference backend."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

from .. import _case_artifact_dir, save_full_stderr
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec


REFERENCE_SOURCE_REVISION = "3428dfd95309a7f3c84fd93259ded0f810d1ff91"
REFERENCE_SAMPLE_RATE = 24_000
_AUDIO_COMPAT = Path(__file__).with_name("personaplex_audio_compat")


def _reference_environment(
    base: dict[str, str] | None = None,
) -> dict[str, str]:
    """Expose the narrow sphn-compatible WAV reader to official Moshi."""
    environment = dict(os.environ if base is None else base)
    existing = environment.get("PYTHONPATH", "").strip()
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(_AUDIO_COMPAT), existing)
        if value
    )
    return environment


def _reference_source() -> Path:
    value = os.environ.get("PERSONAPLEX_OFFICIAL_REPO", "").strip()
    if not value:
        raise RuntimeError(
            "PersonaPlex reference requires PERSONAPLEX_OFFICIAL_REPO; "
            "trtmc-validate prepares this pinned source automatically"
        )
    source = Path(value).resolve()
    if not (source / "moshi" / "moshi" / "offline.py").is_file():
        raise RuntimeError(f"Invalid PersonaPlex reference source: {source}")
    return source


def _audio_input(case: E2ECase) -> Path:
    value = (
        case.inputs.get("audio")
        or case.inputs.get("audio_path")
        or case.metadata.get("test_input_audio")
        or ""
    )
    path = Path(str(value)).resolve()
    if not path.is_file():
        raise RuntimeError(f"PersonaPlex reference audio does not exist: {path}")
    return path


class OfficialPersonaPlexReference:
    """Execute the official Moshi/PersonaPlex model on the selected WAV."""

    @property
    def backend_name(self) -> str:
        return "personaplex_official"

    def run_stage(
        self,
        case: E2ECase,
        stage: StageSpec,
        ctx: RunContext,
    ) -> StageOutput:
        if (
            case.task_strategy != "speech_to_speech"
            or stage.name != "full_generation"
        ):
            return StageOutput(
                stage_name=stage.name,
                data={
                    "error": "PersonaPlex official reference only supports "
                    "speech_to_speech/full_generation"
                },
            )

        model_dir = Path(__file__).resolve().parents[2]
        artifact_dir = Path(
            _case_artifact_dir(
                ctx.artifacts_dir or tempfile.gettempdir(),
                case.name,
            )
        )
        tokens_path = artifact_dir / "official_tokens.npy"
        audio_path = artifact_dir / "official_speech.wav"
        metadata_path = artifact_dir / "official_metadata.json"
        max_frames = int(
            case.inputs.get(
                "speech_test_max_frames",
                case.metadata.get("speech_test_max_frames", 50),
            )
        )
        environment = _reference_environment()
        command = [
            "env",
            f"PYTHONPATH={environment['PYTHONPATH']}",
            ctx.reference_python_path() or sys.executable,
            str(model_dir / "official_reference.py"),
            "--official-repo",
            str(_reference_source()),
            "--source-revision",
            REFERENCE_SOURCE_REVISION,
            "--model",
            case.hf_id,
            "--input-wav",
            str(_audio_input(case)),
            "--max-frames",
            str(max_frames),
            "--tokens-output",
            str(tokens_path),
            "--audio-output",
            str(audio_path),
            "--metadata-output",
            str(metadata_path),
            "--local-files-only",
        ]
        if case.hf_revision:
            command[command.index("--input-wav"):command.index("--input-wav")] = [
                "--revision",
                case.hf_revision,
            ]
        started = time.monotonic()
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        elapsed = time.monotonic() - started
        stderr, stderr_log = save_full_stderr(
            result.stderr or "",
            ctx.artifacts_dir or tempfile.gettempdir(),
            "personaplex_official_reference",
            case.name,
        )
        if result.returncode != 0:
            detail = stderr.strip() or (result.stdout or "").strip()
            raise RuntimeError(
                "PersonaPlex official reference failed "
                f"(rc={result.returncode}): {detail or 'no subprocess output'}"
            )

        import numpy as np

        tokens = np.load(tokens_path, allow_pickle=False)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with wave.open(str(audio_path), "rb") as wav:
            sample_rate = wav.getframerate()
            num_samples = wav.getnframes()
        if (
            tokens.ndim != 2
            or tokens.shape[0] < 1
            or tokens.shape[1] != 8
            or sample_rate != REFERENCE_SAMPLE_RATE
            or num_samples < 1
        ):
            raise RuntimeError(
                "PersonaPlex official reference emitted invalid artifacts: "
                f"tokens={tokens.shape}, sample_rate={sample_rate}, "
                f"num_samples={num_samples}"
            )
        return StageOutput(
            stage_name=stage.name,
            data={
                "reference_tokens": tokens,
                "num_frames": int(tokens.shape[0]),
                "token_shape": list(tokens.shape),
                "wav_path": str(audio_path),
                "wav_exists": True,
                "sample_rate": sample_rate,
                "num_samples": num_samples,
                "duration_s": num_samples / sample_rate,
            },
            timing_s=elapsed,
            metadata={
                "backend": self.backend_name,
                "command": command,
                "returncode": result.returncode,
                "stderr": stderr,
                "stderr_log": stderr_log,
                **metadata,
            },
        )


plugin = OfficialPersonaPlexReference()
