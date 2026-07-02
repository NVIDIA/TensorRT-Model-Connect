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
import shutil
import struct
import subprocess
import tempfile
import time
from pathlib import Path

from .. import save_full_stderr, _case_artifact_dir
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec
from ..runtime_config import runtime_config_set_tokens

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[6]


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
    """Resolve the full path to the .trtfb bundle."""
    bundle_name = case.bundle or case.inputs.get("bundle", "")
    if not bundle_name:
        bundle_name = f"{case.name}.trtfb"
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
# TextToAudioRunner
# ---------------------------------------------------------------------------


class TextToAudioRunner:
    """TRT strategy runner for text-to-audio generation."""

    @property
    def strategy_name(self) -> str:
        return "text_to_audio"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        if stage.name in ("generate", "generate_audio", "end_to_end", "full_generation"):
            return self._run_generate_audio(case, stage, ctx)
        if stage.name == "smoke_test":
            return self._run_smoke_test(case, stage, ctx)
        return StageOutput(
            stage_name=stage.name,
            data={"error": f"Unknown text_to_audio stage: {stage.name}"},
        )

    def _run_generate_audio(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run C++ binary to generate audio from text prompt."""
        bundle_path = _resolve_bundle_path(case, ctx)
        binary = ctx.binary_path
        prompt = case.inputs.get("prompt", "Hello, this is a test.")
        ld_path = _build_ld_library_path(ctx)

        with tempfile.TemporaryDirectory(prefix="trtmc_audio_") as tmpdir:
            distributed_runtime = _distributed_runtime_config(case)
            output_root = os.path.join(tmpdir, "rank_outputs")
            wav_path = (
                os.path.join(output_root, "rank_0", "output.wav")
                if distributed_runtime else os.path.join(tmpdir, "output.wav")
            )

            cmd = [
                binary, "generate-audio", bundle_path,
                "--prompt", prompt,
            ]
            if not distributed_runtime:
                cmd.extend(["--output", wav_path])
            runtime_cli_python = ctx.runtime_cli_hf_python()
            if runtime_cli_python:
                cmd.extend(["--hf-python", runtime_cli_python])

            max_tokens = case.inputs.get("max_new_tokens", 0)
            if max_tokens > 0:
                cmd.extend(["--max-new-tokens", str(max_tokens)])

            env = {**os.environ, "LD_LIBRARY_PATH": ld_path}
            runtime_tokens = runtime_config_set_tokens(case)
            for token in runtime_tokens:
                cmd.extend(["--set", token])
            if distributed_runtime:
                wrapper = (
                    'rank="${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${PMIX_RANK:-${RANK:-0}}}}"; '
                    'out="$1/rank_${rank}"; mkdir -p "$out"; shift; '
                    'exec "$@" --output "$out/output.wav"'
                )
                cmd = ["bash", "-lc", wrapper, "trtmc_rank_audio", output_root] + cmd

            cmd = _wrap_distributed_command(cmd, case)

            t0 = time.monotonic()
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600, env=env)
            elapsed = time.monotonic() - t0

            stderr_truncated, stderr_log = save_full_stderr(
                result.stderr or "", ctx.artifacts_dir or "",
                "text_to_audio", case.name)
            data: dict = {
                "returncode": result.returncode,
                "stdout": _strip_mpirun_tags(result.stdout),
                "stderr": _strip_mpirun_tags(stderr_truncated),
            }
            if stderr_log:
                data["stderr_log"] = stderr_log

            if os.path.exists(wav_path):
                rms = _read_wav_rms(wav_path)
                data["wav_path"] = wav_path
                data["rms"] = rms
                data["wav_exists"] = True

                # Read WAV duration (handles both float32 and int16 formats)
                try:
                    with open(wav_path, "rb") as f:
                        f.read(4)  # RIFF
                        f.read(4)  # size
                        f.read(4)  # WAVE
                        sample_rate = 24000
                        data_size = 0
                        bits_per_sample = 32
                        while True:
                            chunk_id = f.read(4)
                            if len(chunk_id) < 4:
                                break
                            chunk_size = struct.unpack("<I", f.read(4))[0]
                            if chunk_id == b"fmt ":
                                fmt_data = f.read(chunk_size)
                                sample_rate = struct.unpack("<I", fmt_data[4:8])[0]
                                if len(fmt_data) >= 16:
                                    bits_per_sample = struct.unpack("<H", fmt_data[14:16])[0]
                            elif chunk_id == b"data":
                                data_size = chunk_size
                                f.read(chunk_size)
                            else:
                                f.read(chunk_size)
                        bytes_per_sample = max(bits_per_sample // 8, 1)
                        num_samples = data_size // bytes_per_sample
                        data["duration_s"] = num_samples / sample_rate if sample_rate else 0
                        data["sample_rate"] = sample_rate
                except Exception as e:
                    data["duration_error"] = str(e)

                # Persist WAV to artifacts_dir — also update wav_path so
                # comparators can access it after the tempdir is cleaned up.
                if ctx.artifacts_dir:
                    art_dir = Path(_case_artifact_dir(ctx.artifacts_dir, case.name))
                    dst = str(art_dir / "trt_audio.wav")
                    shutil.copy2(wav_path, dst)
                    data["wav_path"] = dst
            else:
                data["wav_exists"] = False

            # Persist input prompt for traceability
            if ctx.artifacts_dir:
                art_dir = Path(_case_artifact_dir(ctx.artifacts_dir, case.name))
                prompt_file = art_dir / "input_prompt.txt"
                prompt_file.write_text(prompt, encoding="utf-8")

            return StageOutput(
                stage_name=stage.name,
                data=data,
                timing_s=elapsed,
                metadata={"command": cmd},
            )

    def _run_smoke_test(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Quick smoke test: generate audio and check RMS is above threshold."""
        output = self._run_generate_audio(case, stage, ctx)
        rms = output.data.get("rms", 0.0)
        min_rms = case.inputs.get("min_rms", 0.001)
        output.data["smoke_passed"] = rms >= min_rms
        return output


plugin = TextToAudioRunner()
