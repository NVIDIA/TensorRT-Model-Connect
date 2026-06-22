"""Audio and speech strategy runners.

Provides TRT inference runners for three audio/speech task strategies:
- speech_to_text: Whisper-style transcription (audio in, text out)
- text_to_audio: Bark-style audio generation (text in, audio out)
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
from ..runtime_config import runtime_config_get, runtime_config_set_tokens

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
# SpeechToTextRunner
# ---------------------------------------------------------------------------


class SpeechToTextRunner:
    """TRT strategy runner for speech-to-text (Whisper-style) transcription."""

    @property
    def strategy_name(self) -> str:
        return "speech_to_text"

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
        if audio_input and not os.path.isabs(audio_input):
            # Resolve relative to project's tests/e2e/ directory
            e2e_dir = Path(__file__).resolve().parents[2] / "e2e"
            audio_input = str(e2e_dir / audio_input)

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


# ---------------------------------------------------------------------------
# TextToAudioRunner
# ---------------------------------------------------------------------------


class TextToAudioRunner:
    """TRT strategy runner for text-to-audio (Bark-style) generation."""

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
            # Keep Bark TRT sampling reproducible in CI unless explicitly overridden.
            bark_seed = runtime_config_get(case, "audio_bark.seed")
            if case.family == "bark" and bark_seed is None:
                seed = case.determinism.get("seed")
                if seed is not None:
                    bark_seed = int(seed)
                    cmd.extend(["--set", f"audio_bark.seed={int(seed)}"])
            # Dump intermediate tokens for diversity/degeneration checks.
            bark_dump_prefix = (
                os.path.join(output_root, "rank_0", "bark_dump")
                if distributed_runtime else os.path.join(tmpdir, "bark_dump")
            )
            if distributed_runtime:
                wrapper = (
                    'rank="${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${PMIX_RANK:-${RANK:-0}}}}"; '
                    'out="$1/rank_${rank}"; mkdir -p "$out"; shift; '
                    'exec "$@" --output "$out/output.wav"'
                )
                if case.family == "bark":
                    wrapper += ' --set "audio_bark.dump_path=$out/bark_dump"'
                cmd = ["bash", "-lc", wrapper, "trtmc_rank_audio", output_root] + cmd
            elif case.family == "bark":
                cmd.extend(["--set", f"audio_bark.dump_path={bark_dump_prefix}"])

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
            if case.family == "bark" and bark_seed is not None:
                data["trt_seed"] = str(bark_seed)
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

            # Capture Bark token dump files for diversity and golden-token checks.
            if case.family == "bark":
                for suffix, key in [(".sem_tokens", "sem_tokens_path"),
                                    (".coarse_tokens", "coarse_tokens_path")]:
                    dump_file = bark_dump_prefix + suffix
                    if os.path.exists(dump_file):
                        if ctx.artifacts_dir:
                            art_dir = Path(_case_artifact_dir(ctx.artifacts_dir, case.name))
                            dst = str(art_dir / suffix.lstrip("."))
                            shutil.copy2(dump_file, dst)
                            data[key] = dst
                        else:
                            data[key] = dump_file

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


# ---------------------------------------------------------------------------
# SpeechToSpeechRunner
# ---------------------------------------------------------------------------


class SpeechToSpeechRunner:
    """TRT strategy runner for speech-to-speech (PersonaPlex-style) pipelines."""

    @property
    def strategy_name(self) -> str:
        return "speech_to_speech"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        if stage.name in ("generate", "end_to_end", "full_generation"):
            return self._run_speech_to_speech(case, stage, ctx)
        return StageOutput(
            stage_name=stage.name,
            data={"error": f"Unknown speech_to_speech stage: {stage.name}"},
        )

    def _run_speech_to_speech(
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
                "speech_to_speech", case.name)
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


# ---------------------------------------------------------------------------
# Plugin registration
#
# The registry discovers ONE plugin per module. Since we have three runners,
# we register SpeechToTextRunner as the primary plugin and manually register
# the others from this module.
# ---------------------------------------------------------------------------


# Primary plugin for auto-discovery
plugin = SpeechToTextRunner()

# Additional runners registered explicitly at import time
_text_to_audio_runner = TextToAudioRunner()
_speech_to_speech_runner = SpeechToSpeechRunner()


def _register_extra_runners() -> None:
    """Register additional runners that share this module."""
    try:
        from ..registry import register_runner
        register_runner(_text_to_audio_runner)
        register_runner(_speech_to_speech_runner)
    except ImportError:
        pass


_register_extra_runners()
