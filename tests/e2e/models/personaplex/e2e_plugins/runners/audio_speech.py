"""Audio and speech strategy runners.

Provides TRT inference runners for three audio/speech task strategies:
- speech_to_text: Whisper-style transcription (audio in, text out)
- text_to_audio: audio generation (text in, audio out)
- personaplex_speech_to_speech: PersonaPlex-style speech transformation (audio in, audio out)

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
# SpeechToSpeechRunner
# ---------------------------------------------------------------------------


class SpeechToSpeechRunner:
    """TRT strategy runner for speech-to-speech (PersonaPlex-style) pipelines."""

    @property
    def strategy_name(self) -> str:
        return "personaplex_speech_to_speech"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        if stage.name in ("generate", "end_to_end", "full_generation"):
            return self._run_personaplex_speech_to_speech(case, stage, ctx)
        return StageOutput(
            stage_name=stage.name,
            data={"error": f"Unknown personaplex_speech_to_speech stage: {stage.name}"},
        )

    def _run_personaplex_speech_to_speech(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run C++ binary with audio input to produce speech output."""
        bundle_path = _resolve_bundle_path(case, ctx)
        binary = ctx.binary_path
        ld_path = _build_ld_library_path(ctx)

        # Resolve audio input path
        audio_input = (case.inputs.get("audio") or case.inputs.get("audio_path")
                       or case.metadata.get("test_input_audio", ""))
        if audio_input and not os.path.isabs(audio_input):
            e2e_dir = Path(__file__).resolve().parents[2] / "e2e"
            audio_input = str(e2e_dir / audio_input)

        max_frames = case.inputs.get(
            "speech_test_max_frames",
            case.metadata.get("speech_test_max_frames", 50),
        )

        with tempfile.TemporaryDirectory(prefix="trtmc_s2s_") as tmpdir:
            distributed_runtime = _distributed_runtime_config(case)
            output_root = os.path.join(tmpdir, "rank_outputs")
            wav_path = (
                os.path.join(output_root, "rank_0", "output.wav")
                if distributed_runtime else os.path.join(tmpdir, "output.wav")
            )
            tokens_path = (
                os.path.join(output_root, "rank_0", "output_tokens.npy")
                if distributed_runtime else os.path.join(tmpdir, "output_tokens.npy")
            )

            cmd = [
                binary, "speak", bundle_path,
                "--audio-in", audio_input,
                "--tail-frames", str(max_frames),
            ]
            if distributed_runtime:
                wrapper = (
                    'rank="${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${PMIX_RANK:-${RANK:-0}}}}"; '
                    'out="$1/rank_${rank}"; mkdir -p "$out"; shift; '
                    'exec "$@" --audio-out "$out/output.wav"'
                )
                cmd = ["bash", "-lc", wrapper, "trtmc_rank_speech", output_root] + cmd
            else:
                cmd.extend(["--audio-out", wav_path])
            cmd = _wrap_distributed_command(cmd, case)

            env = {
                **os.environ,
                "LD_LIBRARY_PATH": ld_path,
            }

            t0 = time.monotonic()
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600, env=env)
            elapsed = time.monotonic() - t0

            stderr_truncated, stderr_log = save_full_stderr(
                result.stderr or "", ctx.artifacts_dir or "",
                "personaplex_speech_to_speech", case.name)
            data: dict = {
                "returncode": result.returncode,
                "stdout": _strip_mpirun_tags(result.stdout),
                "stderr": _strip_mpirun_tags(stderr_truncated),
            }
            if stderr_log:
                data["stderr_log"] = stderr_log

            # Load output tokens if dumped
            if os.path.exists(tokens_path):
                try:
                    import numpy as np
                    tokens = np.load(tokens_path)
                    data["output_tokens"] = tokens
                    data["num_frames"] = tokens.shape[0] if tokens.ndim >= 1 else 0
                except Exception as e:
                    data["token_load_error"] = str(e)

            # Parse tokens from stderr if not dumped via env var
            if "output_tokens" not in data:
                self._parse_tokens_from_stderr(result.stderr, data)

            # Check output audio
            if os.path.exists(wav_path):
                rms = _read_wav_rms(wav_path)
                data["wav_path"] = wav_path
                data["rms"] = rms
                data["wav_exists"] = True

                # Persist WAV to artifacts_dir — update wav_path so
                # comparators can access it after the tempdir is cleaned up.
                if ctx.artifacts_dir:
                    art_dir = Path(_case_artifact_dir(ctx.artifacts_dir, case.name))
                    dst = str(art_dir / "trt_speech_out.wav")
                    shutil.copy2(wav_path, dst)
                    data["wav_path"] = dst
            else:
                data["wav_exists"] = False

            # Record input audio path for traceability
            if audio_input:
                data["input_audio_path"] = audio_input

            return StageOutput(
                stage_name=stage.name,
                data=data,
                timing_s=elapsed,
                metadata={"command": cmd},
            )

    @staticmethod
    def _parse_tokens_from_stderr(stderr: str, data: dict) -> None:
        """Try to parse frame tokens from C++ stderr output."""
        import numpy as np
        frames = []
        for raw_line in (stderr or "").splitlines():
            rank, line = _untag_ranked_mpirun_line(raw_line)
            if rank not in (None, 0):
                continue
            if line.startswith("frame["):
                try:
                    # format: frame[N]: depth=X audio=Y,Z,...
                    parts = line.split(":", 1)[1].strip()
                    token_strs = []
                    for kv in parts.split():
                        if "=" in kv:
                            val = kv.split("=", 1)[1]
                            for t in val.split(","):
                                token_strs.append(int(t))
                    if token_strs:
                        frames.append(token_strs)
                except (ValueError, IndexError):
                    pass
                continue

            match = re.search(r"\[speech\]\s+Output frame\s+\d+:\s*(.*)$", line)
            if match:
                try:
                    token_strs = [int(t) for t in match.group(1).split()]
                    if token_strs:
                        frames.append(token_strs)
                except ValueError:
                    pass
        if frames:
            data["output_tokens"] = np.array(frames, dtype=np.int32)
            data["num_frames"] = len(frames)


plugin = SpeechToSpeechRunner()
