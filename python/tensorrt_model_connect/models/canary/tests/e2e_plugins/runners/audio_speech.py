# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audio and speech strategy runners.

Provides TRT inference runners for three audio/speech task strategies:
- speech_to_text: Whisper-style transcription (audio in, text out)
- text_to_audio: audio generation (text in, audio out)
- speech_to_speech: PersonaPlex-style speech transformation (audio in, audio out)

All GPU work runs in subprocesses for memory isolation. The registry
auto-discovers ONE plugin per module, so this module registers via
explicit calls in the module footer rather than a single ``plugin``.
"""

from __future__ import annotations

import logging
import os
import re
import struct
import subprocess
import time
from pathlib import Path

from .. import save_full_stderr, _case_artifact_dir
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec

logger = logging.getLogger(__name__)

_MODEL_TEST_DIR = Path(__file__).resolve().parents[2]


def _find_trt_lib_dir() -> str:
    """Find TRT library directory from the Python tensorrt_libs package."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("tensorrt_libs")
        if spec and spec.submodule_search_locations:
            return spec.submodule_search_locations[0]
    except ImportError:
        pass
    return ""


def _build_ld_library_path(ctx: RunContext) -> str:
    """Build LD_LIBRARY_PATH from context or auto-detect."""
    if ctx.ld_library_path:
        return ctx.ld_library_path
    trt_lib = _find_trt_lib_dir()
    parts = []
    if trt_lib:
        parts.append(trt_lib)
    parts.append("/usr/local/cuda/lib64")
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if existing:
        parts.append(existing)
    return ":".join(parts)


def _resolve_bundle_path(case: E2ECase, ctx: RunContext) -> str:
    """Resolve the full path to the .bundle artifact."""
    bundle_name = case.bundle or case.inputs.get("bundle", "")
    if not bundle_name:
        bundle_name = f"{case.name}.bundle"
    if os.path.isabs(bundle_name):
        return bundle_name
    return os.path.join(ctx.engine_dir, bundle_name)


def _distributed_runtime_config(case: E2ECase) -> dict:
    config = case.metadata.get("distributed_runtime", {})
    return config if isinstance(config, dict) and config.get("enabled") else {}


def _wrap_distributed_command(cmd: list[str], case: E2ECase) -> list[str]:
    config = _distributed_runtime_config(case)
    if not config:
        return cmd
    launcher = str(config.get("launcher", "mpirun") or "mpirun")
    world_size = int(config.get("world_size", config.get("tp_size", 2)) or 2)
    launcher_args = config.get("launcher_args")
    if isinstance(launcher_args, list):
        return [launcher] + [str(arg) for arg in launcher_args] + cmd
    return [launcher, "--tag-output", "-np", str(world_size)] + cmd


def _strip_mpirun_tags(text: str) -> str:
    lines = []
    for line in text.splitlines():
        lines.append(re.sub(r"^\[[^\]]+\]<std(?:out|err)>:\s?", "", line))
    return "\n".join(lines)


def _untag_ranked_mpirun_line(line: str) -> tuple[int | None, str]:
    match = re.match(r"^\[([^\]]+)\]<std(?:out|err)>:\s?(.*)$", line)
    if not match:
        return None, line
    rank_text = match.group(1).split(",")[-1]
    try:
        return int(rank_text), match.group(2)
    except ValueError:
        return None, match.group(2)


def _read_wav_rms(path: str) -> float:
    """Read a WAV file and return its RMS energy."""
    import numpy as np
    with open(path, "rb") as f:
        riff = f.read(4)
        if riff != b"RIFF":
            return 0.0
        f.read(4)  # chunk size
        f.read(4)  # WAVE

        data_bytes = b""
        audio_format = 1
        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                break
            chunk_size = struct.unpack("<I", f.read(4))[0]
            if chunk_id == b"fmt ":
                fmt_data = f.read(chunk_size)
                audio_format = struct.unpack("<H", fmt_data[0:2])[0]
            elif chunk_id == b"data":
                data_bytes = f.read(chunk_size)
            else:
                f.read(chunk_size)

    if not data_bytes:
        return 0.0

    if audio_format == 3:  # IEEE float32
        samples = np.frombuffer(data_bytes, dtype=np.float32)
    elif audio_format == 1:  # PCM int16
        samples = np.frombuffer(data_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        return 0.0

    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples ** 2)))


# ---------------------------------------------------------------------------
# SpeechToTextRunner
# ---------------------------------------------------------------------------


class SpeechToTextRunner:
    """TRT strategy runner for speech-to-text (Whisper-style) transcription."""

    @property
    def strategy_name(self) -> str:
        return "canary_speech_to_text"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        if stage.name in ("generate", "transcribe", "end_to_end", "full_generation"):
            return self._run_transcribe(case, stage, ctx)
        return StageOutput(
            stage_name=stage.name,
            data={"error": f"Unknown speech_to_text stage: {stage.name}"},
        )

    def _run_transcribe(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run C++ binary with audio input to produce transcript."""
        bundle_path = _resolve_bundle_path(case, ctx)
        binary = ctx.binary_path
        ld_path = _build_ld_library_path(ctx)

        # Resolve audio input path
        audio_input = (case.inputs.get("audio") or case.inputs.get("audio_path")
                       or case.metadata.get("test_input_audio", ""))
        if audio_input:
            audio_path = Path(audio_input)
            if not audio_path.is_absolute():
                audio_path = _MODEL_TEST_DIR / audio_path
            if not audio_path.is_file():
                raise FileNotFoundError(
                    f"Model-owned audio asset not found: {audio_path}"
                )
            audio_input = str(audio_path)

        max_new_tokens = case.inputs.get("max_new_tokens", 100)

        cmd = [
            binary, "transcribe", bundle_path,
            "--audio", audio_input,
            "--max-new-tokens", str(max_new_tokens),
        ]
        runtime_cli_python = ctx.runtime_cli_hf_python()
        if runtime_cli_python:
            cmd.extend(["--hf-python", runtime_cli_python])

        env = {**os.environ, "LD_LIBRARY_PATH": ld_path}
        cmd = _wrap_distributed_command(cmd, case)

        t0 = time.monotonic()
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, env=env)
        elapsed = time.monotonic() - t0

        # Parse output: expect transcript text on stdout.
        # Strip special tokens like <|notimestamp|>, <|endoftext|>, etc.
        clean_stdout = _strip_mpirun_tags(result.stdout)
        transcript_lines = [
            re.sub(r'<\|[^|]+\|>', '', line).strip()
            for line in clean_stdout.splitlines()
        ]
        transcript = next((line for line in transcript_lines if line), "")

        # Try to extract token IDs if the binary outputs them
        token_ids = []
        for line in _strip_mpirun_tags(result.stderr).splitlines():
            if line.startswith("tokens:"):
                try:
                    token_ids = [int(t) for t in line.split(":", 1)[1].strip().split()]
                except (ValueError, IndexError):
                    pass

        # Persist transcript for human inspection
        if ctx.artifacts_dir and transcript:
            art_dir = Path(_case_artifact_dir(ctx.artifacts_dir, case.name))
            txt_path = art_dir / "trt_transcript.txt"
            txt_path.write_text(transcript, encoding="utf-8")

        stderr_truncated, stderr_log = save_full_stderr(
            result.stderr or "", ctx.artifacts_dir or "",
            "speech_to_text", case.name)
        stt_data: dict = {
            "returncode": result.returncode,
            "transcript": transcript,
            "token_ids": token_ids,
            "stderr": stderr_truncated,
        }
        if stderr_log:
            stt_data["stderr_log"] = stderr_log

        return StageOutput(
            stage_name=stage.name,
            data=stt_data,
            text=transcript,
            timing_s=elapsed,
            metadata={"command": cmd},
        )


plugin = SpeechToTextRunner()
