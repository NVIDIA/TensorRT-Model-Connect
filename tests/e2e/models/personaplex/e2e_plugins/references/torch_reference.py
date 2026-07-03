# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PersonaPlex-owned torch reference backend."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import wave
from pathlib import Path

from .. import _case_artifact_dir, save_full_stderr
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec


PROJECT_DIR = Path(__file__).resolve().parents[6]
E2E_DIR = PROJECT_DIR / "tests" / "e2e"
MIMI_MODEL_ID = "kyutai/mimi"
REFERENCE_SAMPLE_RATE = 24_000


_MIMI_DECODE_SCRIPT = textwrap.dedent(
    """\
    import json
    import sys
    import wave

    import numpy as np
    import torch
    from transformers import MimiModel

    tokens_path, wav_path = sys.argv[1:3]
    tokens = np.load(tokens_path)
    if tokens.ndim != 2 or tokens.shape[0] == 0 or tokens.shape[1] == 0:
        raise ValueError(
            f"expected non-empty [frames, codebooks] tokens, got {tokens.shape}"
        )

    model = MimiModel.from_pretrained("kyutai/mimi", local_files_only=True).eval()
    sample_rate = int(getattr(model.config, "sampling_rate", 0) or 0)
    if sample_rate != 24000:
        raise ValueError(f"kyutai/mimi sample rate must be 24000 Hz, got {sample_rate}")

    # PersonaPlex stores one row per generated frame. Mimi expects
    # [batch, codebook, frame].
    codes = torch.from_numpy(tokens.T.copy()).to(dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        decoded = model.decode(codes, return_dict=True).audio_values
    audio = decoded.detach().cpu().to(dtype=torch.float32).reshape(-1).numpy()
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("Mimi produced empty or non-finite reference audio")

    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = np.rint(clipped * 32767.0).astype("<i2")
    with wave.open(wav_path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16.tobytes())

    print(json.dumps({
        "sample_rate": sample_rate,
        "num_samples": int(audio.size),
        "duration_s": float(audio.size / sample_rate),
        "rms": float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))),
        "codec_model": "kyutai/mimi",
        "codec_backend": "transformers.MimiModel",
    }))
    """
)


class TorchReference:
    """Load PersonaPlex reference token snapshots."""

    @property
    def backend_name(self) -> str:
        return "torch_reference"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        if case.task_strategy != "speech_to_speech":
            return StageOutput(
                stage_name=stage.name,
                data={"error": f"Unsupported PersonaPlex task_strategy: {case.task_strategy}"},
            )
        return self._run_reference_tokens(case, stage, ctx)

    def _run_reference_tokens(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        ref_tokens_path = case.inputs.get(
            "speech_reference_tokens",
            case.metadata.get("speech_reference_tokens", ""),
        )
        if not ref_tokens_path:
            return StageOutput(
                stage_name=stage.name,
                data={"error": "No speech_reference_tokens path in manifest"},
            )

        if not os.path.isabs(ref_tokens_path):
            project_relative = PROJECT_DIR / ref_tokens_path
            ref_tokens_path = str(
                project_relative
                if project_relative.exists()
                else E2E_DIR / ref_tokens_path
            )

        if not os.path.exists(ref_tokens_path):
            return StageOutput(
                stage_name=stage.name,
                data={"error": f"Reference tokens file not found: {ref_tokens_path}"},
            )

        try:
            import numpy as np

            ref_tokens = np.load(ref_tokens_path)
        except Exception as exc:
            return StageOutput(
                stage_name=stage.name,
                data={"error": f"Failed to load reference tokens: {exc}"},
            )

        model_dir = Path(_case_artifact_dir(
            ctx.artifacts_dir or tempfile.gettempdir(), case.name))
        wav_path = model_dir / "reference_speech.wav"
        python = ctx.reference_python_path() or sys.executable
        command = [
            python,
            "-c",
            _MIMI_DECODE_SCRIPT,
            ref_tokens_path,
            str(wav_path),
        ]

        started = time.monotonic()
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(
                f"PersonaPlex reference WAV generation could not run with "
                f"{python!r}: {exc}"
            ) from exc
        elapsed = time.monotonic() - started

        stderr_truncated, stderr_log = save_full_stderr(
            result.stderr or "",
            ctx.artifacts_dir or tempfile.gettempdir(),
            "personaplex_mimi_reference",
            case.name,
        )
        if result.returncode != 0:
            detail = stderr_truncated.strip() or (result.stdout or "").strip()
            raise RuntimeError(
                "PersonaPlex reference WAV generation failed "
                f"(rc={result.returncode}): {detail or 'no subprocess output'}"
            )

        try:
            generated = json.loads((result.stdout or "").strip())
        except (json.JSONDecodeError, TypeError) as exc:
            raise RuntimeError(
                "PersonaPlex reference WAV generation emitted invalid metadata: "
                f"{(result.stdout or '').strip()!r}"
            ) from exc

        try:
            with wave.open(str(wav_path), "rb") as wav:
                channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
                sample_rate = wav.getframerate()
                num_samples = wav.getnframes()
        except (OSError, EOFError, wave.Error) as exc:
            raise RuntimeError(
                f"PersonaPlex Mimi reference did not produce a valid WAV at "
                f"{wav_path}: {exc}"
            ) from exc
        if channels != 1 or sample_width != 2 or sample_rate != REFERENCE_SAMPLE_RATE:
            raise RuntimeError(
                "PersonaPlex Mimi reference WAV has an invalid format: "
                f"channels={channels}, sample_width={sample_width}, "
                f"sample_rate={sample_rate}"
            )
        if num_samples <= 0:
            raise RuntimeError("PersonaPlex Mimi reference WAV contains no samples")

        generated.update({
            "wav_path": str(wav_path),
            "wav_exists": True,
            "sample_rate": sample_rate,
            "num_samples": num_samples,
            "duration_s": num_samples / sample_rate,
        })

        return StageOutput(
            stage_name=stage.name,
            data={
                "reference_tokens": ref_tokens,
                "num_frames": ref_tokens.shape[0] if ref_tokens.ndim >= 1 else 0,
                "token_shape": list(ref_tokens.shape),
                "source_path": ref_tokens_path,
                **generated,
            },
            timing_s=elapsed,
            metadata={
                "backend": "torch_reference",
                "codec_command": command,
                "codec_returncode": result.returncode,
                "codec_stderr": stderr_truncated,
                "codec_stderr_log": stderr_log,
            },
        )


plugin = TorchReference()
