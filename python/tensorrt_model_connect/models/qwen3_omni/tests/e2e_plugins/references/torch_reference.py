# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live official-HF reference for Qwen3-Omni text-to-audio."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

from .. import _case_artifact_dir, save_full_stderr
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec


REFERENCE_SAMPLE_RATE = 24_000
DEFAULT_SPEAKER = "Ethan"
DEFAULT_SEED = 42
DEFAULT_TALKER_MAX_NEW_TOKENS = 32


class TorchReference:
    """Run Qwen3-Omni through its official Transformers API."""

    @property
    def backend_name(self) -> str:
        return "torch_reference"

    def run_stage(
        self,
        case: E2ECase,
        stage: StageSpec,
        ctx: RunContext,
    ) -> StageOutput:
        if (
            case.task_strategy != "omni_multimodal"
            or stage.name != "talker_decode"
        ):
            return StageOutput(
                stage_name=stage.name,
                data={
                    "error": "Qwen3-Omni HF audio reference only supports "
                    "omni_multimodal/talker_decode"
                },
            )

        prompt = str(case.inputs.get("prompt", "") or "").strip()
        if not prompt:
            raise RuntimeError("Qwen3-Omni reference requires a prompt")
        artifact_dir = Path(
            _case_artifact_dir(
                ctx.artifacts_dir or tempfile.gettempdir(),
                case.name,
            )
        )
        audio_path = artifact_dir / "hf_reference.wav"
        metadata_path = artifact_dir / "hf_reference.json"
        model_dir = Path(__file__).resolve().parents[2]
        command = [
            ctx.reference_python_path() or sys.executable,
            str(model_dir / "official_hf_audio.py"),
            "--model",
            case.hf_id,
            "--prompt",
            prompt,
            "--speaker",
            str(case.metadata.get("reference_speaker", DEFAULT_SPEAKER)),
            "--seed",
            str(
                case.inputs.get(
                    "seed",
                    case.determinism.get("seed", DEFAULT_SEED),
                )
            ),
            "--thinker-max-new-tokens",
            str(case.inputs.get("max_new_tokens", 16)),
            "--talker-max-new-tokens",
            str(
                case.metadata.get(
                    "reference_talker_max_new_tokens",
                    DEFAULT_TALKER_MAX_NEW_TOKENS,
                )
            ),
            "--audio-output",
            str(audio_path),
            "--metadata-output",
            str(metadata_path),
            "--local-files-only",
        ]
        if case.hf_revision:
            prompt_index = command.index("--prompt")
            command[prompt_index:prompt_index] = [
                "--revision",
                case.hf_revision,
            ]
        started = time.monotonic()
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        elapsed = time.monotonic() - started
        stderr, stderr_log = save_full_stderr(
            result.stderr or "",
            ctx.artifacts_dir or tempfile.gettempdir(),
            "qwen3_omni_official_hf_reference",
            case.name,
        )
        if result.returncode != 0:
            detail = stderr.strip() or (result.stdout or "").strip()
            raise RuntimeError(
                "Qwen3-Omni official HF reference failed "
                f"(rc={result.returncode}): {detail or 'no subprocess output'}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with wave.open(str(audio_path), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            num_samples = wav.getnframes()
        if (
            channels != 1
            or sample_width != 2
            or sample_rate != REFERENCE_SAMPLE_RATE
            or num_samples < 1
        ):
            raise RuntimeError(
                "Qwen3-Omni HF reference emitted invalid audio: "
                f"channels={channels}, sample_width={sample_width}, "
                f"sample_rate={sample_rate}, num_samples={num_samples}"
            )
        data = {
            "_invariant_only": True,
            "wav_path": str(audio_path),
            "wav_exists": True,
            "sample_rate": sample_rate,
            "num_samples": num_samples,
            "duration_s": num_samples / sample_rate,
            "decoded_text": str(metadata.get("decoded_text", "") or ""),
            **metadata,
        }
        return StageOutput(
            stage_name=stage.name,
            data=data,
            text=data["decoded_text"],
            timing_s=elapsed,
            metadata={
                "backend": self.backend_name,
                "source": "official_hf_live_reference",
                "comparison_mode": "waveform_cosine_and_invariants",
                "command": command,
                "returncode": result.returncode,
                "stderr": stderr,
                "stderr_log": stderr_log,
                **metadata,
            },
        )


plugin = TorchReference()
