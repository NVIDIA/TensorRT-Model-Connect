# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned Hugging Face Parakeet TDT speech-to-text reference."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

from .. import _case_artifact_dir, save_full_stderr
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec


PROJECT_DIR = Path(__file__).resolve().parents[6]
E2E_DIR = PROJECT_DIR / "tests" / "e2e"
MODEL_DIR = Path(__file__).resolve().parents[2]


def _resolve_audio_path(value: str) -> str:
    path = Path(value)
    if path.is_file():
        return str(path.resolve())
    for base in (MODEL_DIR, E2E_DIR, PROJECT_DIR):
        candidate = base / path
        if candidate.is_file():
            return str(candidate.resolve())
    return value


class HfTransformersReference:
    @property
    def backend_name(self) -> str:
        return "hf_transformers"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        if stage.name not in {"full_generation", "full_inference"}:
            raise ValueError(f"Unknown Parakeet TDT reference stage: {stage.name!r}")
        if case.task_strategy != "speech_to_text":
            raise ValueError(
                f"Parakeet TDT reference only supports speech_to_text, got "
                f"{case.task_strategy!r}"
            )

        artifact_root = ctx.artifacts_dir or tempfile.gettempdir()
        output_dir = (
            _case_artifact_dir(artifact_root, case.name)
            if ctx.artifacts_dir
            else artifact_root
        )
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = str(Path(output_dir) / "hf_parakeet_tdt.json")
        audio_path = _resolve_audio_path(str(case.inputs.get("audio", "")))
        revision = str(case.hf_revision or "")
        if not revision:
            raise ValueError("Parakeet TDT HF reference requires an immutable hf_revision")

        script = textwrap.dedent(
            f"""\
            import json
            import math
            import numpy as np
            import scipy.io.wavfile as wav
            from scipy.signal import resample_poly
            import torch
            from transformers import pipeline

            model_id = {case.hf_id!r}
            revision = {revision!r}
            audio_path = {audio_path!r}
            output_path = {output_path!r}

            sample_rate, audio = wav.read(audio_path)
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            else:
                audio = audio.astype(np.float32)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            asr = pipeline(
                "automatic-speech-recognition",
                model=model_id,
                revision=revision,
                torch_dtype=torch.float32,
                device=0 if torch.cuda.is_available() else -1,
            )
            target_sample_rate = int(asr.feature_extractor.sampling_rate)
            if sample_rate != target_sample_rate:
                divisor = math.gcd(int(sample_rate), target_sample_rate)
                audio = resample_poly(
                    audio,
                    target_sample_rate // divisor,
                    int(sample_rate) // divisor,
                ).astype(np.float32)
                sample_rate = target_sample_rate
            result = asr({{"array": audio, "sampling_rate": int(sample_rate)}})
            text = str(result.get("text", "")).strip()
            if not text:
                raise RuntimeError("pinned HF Parakeet TDT reference returned empty text")
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump({{"text": text}}, handle)
            """
        )
        python = ctx.reference_python_path() or sys.executable
        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path
        started = time.monotonic()
        try:
            result = subprocess.run(
                [python, "-c", script],
                capture_output=True,
                text=True,
                timeout=1800,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            stderr, log_path = save_full_stderr(
                str(exc.stderr or ""), ctx.artifacts_dir or "",
                "hf_parakeet_tdt", case.name
            )
            suffix = f" (full stderr: {log_path})" if log_path else ""
            raise RuntimeError(
                f"HF Parakeet TDT reference timed out: {stderr}{suffix}"
            ) from exc
        elapsed = time.monotonic() - started
        if result.returncode != 0:
            stderr, log_path = save_full_stderr(
                result.stderr or "", ctx.artifacts_dir or "",
                "hf_parakeet_tdt", case.name
            )
            suffix = f" (full stderr: {log_path})" if log_path else ""
            raise RuntimeError(
                f"HF Parakeet TDT reference failed (rc={result.returncode}): "
                f"{stderr}{suffix}"
            )
        data = json.loads(Path(output_path).read_text(encoding="utf-8"))
        return StageOutput(
            stage_name=stage.name,
            data=data,
            text=str(data["text"]),
            timing_s=elapsed,
            metadata={
                "model_id": case.hf_id,
                "revision": revision,
                "returncode": result.returncode,
            },
        )


plugin = HfTransformersReference()
