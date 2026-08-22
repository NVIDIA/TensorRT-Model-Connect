# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native CLI runner for the public Nemotron VoiceChat model-card sample."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from tests.e2e_harness.contracts import E2ECase, RunContext, StageOutput, StageSpec

_SPEECH_REPO_ENV = "NEMOTRON_VOICECHAT_SPEECH_REPO"
_MODEL_CARD_STAGE = "model_card_general_conversation"
_LIFECYCLE_STAGE = "native_full_duplex_lifecycle"
_LIFECYCLE_TARGET = "test_nemotron_voicechat_native_lifecycle"
_LIFECYCLE_SCHEMA_VERSION = 3
_REALTIME_STAGE = "realtime_websocket_interop"
_REALTIME_TARGET = "trtmc_realtime_worker"
_REALTIME_SCHEMA_VERSION = 1
_REALTIME_PROBE = Path(__file__).resolve().parents[1] / "realtime_transport_probe.py"
_NEW_RUNTIME_MARKER = "backend=trt_new_runtime"
_GENERATED_RE = re.compile(r"^Generated\s+(\d+)\s+audio samples\s+->\s+(.+)$")
_AGENT_TEXT_RE = re.compile(r"^Agent text:\s*(.+)$")


def _bundle_path(case: E2ECase, ctx: RunContext) -> Path:
    bundle = Path(case.bundle or f"{case.name}.bundle")
    return bundle if bundle.is_absolute() else Path(ctx.engine_dir) / bundle


def _source_path_from_input(case: E2ECase, key: str) -> Path:
    root = os.environ.get(_SPEECH_REPO_ENV, "")
    if not root:
        raise RuntimeError(f"{_SPEECH_REPO_ENV} must point at the pinned public Speech checkout")
    relative = case.inputs.get(key, "")
    if not isinstance(relative, str) or not relative:
        raise RuntimeError(f"VoiceChat manifest is missing {key}")
    source = Path(root) / relative
    if not source.is_file():
        raise RuntimeError(f"pinned VoiceChat source sample is unavailable: {source}")
    return source


def _source_path(case: E2ECase) -> Path:
    return _source_path_from_input(case, "speech_source_relative_path")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wav_stats(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    if len(payload) < 12 or payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        raise RuntimeError(f"invalid RIFF/WAVE output: {path}")

    fmt = None
    audio = None
    offset = 12
    while offset + 8 <= len(payload):
        name = payload[offset : offset + 4]
        size = struct.unpack_from("<I", payload, offset + 4)[0]
        start = offset + 8
        end = start + size
        if end > len(payload):
            raise RuntimeError(f"truncated WAV chunk {name!r}: {path}")
        if name == b"fmt ":
            fmt = payload[start:end]
        elif name == b"data":
            audio = payload[start:end]
        offset = end + (size & 1)
    if fmt is None or len(fmt) < 16 or audio is None:
        raise RuntimeError(f"WAV is missing fmt or data chunk: {path}")

    encoding_id, channels, sample_rate, _, block_align, bits = struct.unpack_from("<HHIIHH", fmt)
    sample_width = bits // 8
    if channels < 1 or block_align != channels * sample_width or not audio:
        raise RuntimeError(f"WAV has invalid channel or block alignment: {path}")
    if len(audio) % block_align:
        raise RuntimeError(f"WAV data is not frame aligned: {path}")

    encoding = {(1, 16): "pcm_s16le", (3, 32): "ieee_float32le"}.get(
        (encoding_id, bits), "unsupported"
    )
    if encoding == "pcm_s16le":
        values = (sample[0] / 32768.0 for sample in struct.iter_unpack("<h", audio))
    elif encoding == "ieee_float32le":
        values = (sample[0] for sample in struct.iter_unpack("<f", audio))
    else:
        raise RuntimeError(f"unsupported WAV encoding format={encoding_id}, bits={bits}: {path}")

    count = 0
    square_sum = 0.0
    peak = 0.0
    all_finite = True
    for value in values:
        count += 1
        all_finite = all_finite and math.isfinite(value)
        square_sum += value * value
        peak = max(peak, abs(value))
    num_frames = len(audio) // block_align
    if count != num_frames * channels:
        raise RuntimeError(f"WAV sample accounting failed: {path}")
    return {
        "encoding": encoding,
        "channels": channels,
        "sample_rate": sample_rate,
        "num_samples": num_frames,
        "sample_width": sample_width,
        "all_finite": all_finite,
        "rms": math.sqrt(square_sum / count),
        "peak": peak,
    }


def _timeout(case: E2ECase, name: str, default: int) -> int:
    value = case.inputs.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 3600:
        raise ValueError(f"{name} must be an integer in [1, 3600], got {value!r}")
    return value


def _run(command: list[str], env: dict[str, str], timeout_s: int) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(
            f"VoiceChat native command could not run: {command!r}: {error}"
        ) from error


def _load_lifecycle_receipt(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("VoiceChat lifecycle probe did not write its receipt")
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"VoiceChat lifecycle receipt is invalid: {error}") from error
    if not isinstance(receipt, dict):
        raise RuntimeError("VoiceChat lifecycle receipt must be a JSON object")
    if receipt.get("schema_version") != _LIFECYCLE_SCHEMA_VERSION:
        raise RuntimeError(
            "VoiceChat lifecycle receipt schema mismatch: "
            f"expected {_LIFECYCLE_SCHEMA_VERSION}, got {receipt.get('schema_version')!r}"
        )
    if isinstance(receipt.get("error"), str) and receipt["error"]:
        raise RuntimeError(f"VoiceChat lifecycle probe failed: {receipt['error']}")
    return receipt


def _load_realtime_receipt(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("VoiceChat realtime probe did not write its receipt")
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"VoiceChat realtime receipt is invalid: {error}") from error
    if not isinstance(receipt, dict):
        raise RuntimeError("VoiceChat realtime receipt must be a JSON object")
    if receipt.get("schema_version") != _REALTIME_SCHEMA_VERSION:
        raise RuntimeError(
            "VoiceChat realtime receipt schema mismatch: "
            f"expected {_REALTIME_SCHEMA_VERSION}, got {receipt.get('schema_version')!r}"
        )
    return receipt


class VoiceChatModelCardRunner:
    @property
    def strategy_name(self) -> str:
        return "speech_to_speech"

    def run_stage(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        if stage.name == _LIFECYCLE_STAGE:
            return self._run_lifecycle_stage(case, stage, ctx)
        if stage.name == _REALTIME_STAGE:
            return self._run_realtime_stage(case, stage, ctx)
        if stage.name != _MODEL_CARD_STAGE:
            return StageOutput(
                stage_name=stage.name,
                data={"error": f"unsupported VoiceChat stage: {stage.name}"},
            )
        if not ctx.binary_path:
            raise RuntimeError("VoiceChat E2E requires the native trtmc binary")

        source = _source_path(case)
        source_sha = _sha256(source)
        expected_sha = str(case.metadata.get("speech_source_sha256", ""))
        if source_sha != expected_sha:
            raise RuntimeError(
                f"VoiceChat source sample SHA256 mismatch: expected {expected_sha}, got {source_sha}"
            )
        source_stats = _wav_stats(source)

        base = (
            Path(ctx.artifacts_dir)
            if ctx.artifacts_dir
            else Path(tempfile.mkdtemp(prefix="trtmc_voicechat_e2e_"))
        )
        artifact_dir = base / case.name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        output = artifact_dir / "model_card_sample_general_output.wav"
        bundle = _bundle_path(case, ctx)
        tail_frames = int(case.inputs.get("tail_frames", 0))

        speak_command = [
            ctx.binary_path,
            "speak",
            str(bundle),
            "--audio-in",
            str(source),
            "--audio-out",
            str(output),
            "--tail-frames",
            str(tail_frames),
            "--seed",
            str(case.inputs.get("seed", 0)),
        ]
        if ctx.model_plugin_dir:
            speak_command.extend(["--model-plugin-dir", ctx.model_plugin_dir])
        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path

        started = time.monotonic()
        speak = _run(speak_command, env, _timeout(case, "runtime_timeout_s", 1800))
        speak_elapsed = time.monotonic() - started
        if speak.returncode != 0:
            raise RuntimeError(
                f"native trtmc speak failed (rc={speak.returncode}): {speak.stderr[-2000:]}"
            )
        if not output.is_file():
            raise RuntimeError("native trtmc speak did not write its WAV artifact")
        output_stats = _wav_stats(output)

        generated_match = None
        agent_text_matches: list[str] = []
        for line in (speak.stdout or "").splitlines():
            match = _GENERATED_RE.fullmatch(line.strip())
            if match:
                generated_match = match
            agent_text_match = _AGENT_TEXT_RE.fullmatch(line.strip())
            if agent_text_match:
                agent_text_matches.append(agent_text_match.group(1).strip())
        generated_count = int(generated_match.group(1)) if generated_match else -1
        agent_text = agent_text_matches[0] if len(agent_text_matches) == 1 else ""

        transcribe_command = [
            ctx.binary_path,
            "transcribe",
            str(bundle),
            "--audio",
            str(output),
            "--max-new-tokens",
            str(case.inputs.get("max_new_tokens", 256)),
        ]
        if ctx.model_plugin_dir:
            transcribe_command.extend(["--model-plugin-dir", ctx.model_plugin_dir])
        started = time.monotonic()
        transcribe = _run(transcribe_command, env, _timeout(case, "transcribe_timeout_s", 1800))
        transcribe_elapsed = time.monotonic() - started
        if transcribe.returncode != 0:
            raise RuntimeError(
                "native trtmc transcribe over the generated VoiceChat WAV failed "
                f"(rc={transcribe.returncode}): {transcribe.stderr[-2000:]}"
            )
        # cmd_transcribe writes one transcript followed by a newline to stdout.
        transcript_lines = [
            line.strip() for line in (transcribe.stdout or "").splitlines() if line.strip()
        ]
        transcript = transcript_lines[0] if len(transcript_lines) == 1 else ""

        return StageOutput(
            stage_name=stage.name,
            text=transcript,
            timing_s=speak_elapsed + transcribe_elapsed,
            data={
                "source_path": str(source),
                "source_sha256": source_sha,
                "source_stats": source_stats,
                "wav_path": str(output),
                "output_stats": output_stats,
                "generated_count": generated_count,
                "tail_frames": tail_frames,
                "agent_text": agent_text,
                "agent_text_line_count": len(agent_text_matches),
                "transcript": transcript,
                "transcript_line_count": len(transcript_lines),
            },
            metadata={
                "speak": {
                    "command": speak_command,
                    "returncode": speak.returncode,
                    "stdout": speak.stdout or "",
                    "stderr": speak.stderr or "",
                },
                "transcribe": {
                    "command": transcribe_command,
                    "returncode": transcribe.returncode,
                    "stdout": transcribe.stdout or "",
                    "stderr": transcribe.stderr or "",
                },
            },
        )

    def _run_realtime_stage(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        if not ctx.binary_path:
            raise RuntimeError("VoiceChat realtime E2E requires the native trtmc binary")
        binary = Path(ctx.binary_path).resolve()
        build_dir = binary.parent
        if not (build_dir / "CMakeCache.txt").is_file():
            raise RuntimeError(
                "VoiceChat realtime E2E requires a configured source build beside the "
                f"selected trtmc binary: {build_dir}"
            )
        if not _REALTIME_PROBE.is_file():
            raise RuntimeError("VoiceChat realtime E2E probe source is unavailable")

        source = _source_path(case)
        source_sha = _sha256(source)
        expected_sha = str(case.metadata.get("speech_source_sha256", ""))
        if source_sha != expected_sha:
            raise RuntimeError(
                f"VoiceChat source sample SHA256 mismatch: expected {expected_sha}, got {source_sha}"
            )
        function_source = _source_path_from_input(case, "function_speech_source_relative_path")
        function_source_sha = _sha256(function_source)
        expected_function_sha = str(case.metadata.get("function_speech_source_sha256", ""))
        if function_source_sha != expected_function_sha:
            raise RuntimeError(
                "VoiceChat function-call sample SHA256 mismatch: "
                f"expected {expected_function_sha}, got {function_source_sha}"
            )
        bundle = _bundle_path(case, ctx)
        if not bundle.is_file():
            raise RuntimeError(f"VoiceChat realtime bundle is unavailable: {bundle}")
        if not ctx.model_plugin_dir:
            raise RuntimeError("VoiceChat realtime E2E requires a model plugin directory")

        base = (
            Path(ctx.artifacts_dir)
            if ctx.artifacts_dir
            else Path(tempfile.mkdtemp(prefix="trtmc_voicechat_realtime_"))
        )
        artifact_dir = base / case.name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        output_wav = artifact_dir / "realtime_websocket_output.wav"
        receipt_path = artifact_dir / "realtime_websocket_interop.json"

        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path
        project_root = str(Path(__file__).resolve().parents[5])
        python_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{project_root}/python:{project_root}"
            if not python_path
            else f"{project_root}/python:{project_root}:{python_path}"
        )
        build_command = ["cmake", "--build", str(build_dir), "--target", _REALTIME_TARGET]
        build = _run(
            build_command,
            env,
            _timeout(case, "realtime_build_timeout_s", 600),
        )
        if build.returncode != 0:
            raise RuntimeError(
                "VoiceChat realtime worker target failed to build "
                f"(rc={build.returncode}): {build.stderr[-2000:]}"
            )
        worker = build_dir / _REALTIME_TARGET
        if not worker.is_file():
            raise RuntimeError(f"VoiceChat realtime target did not produce {worker}")

        probe_command = [
            sys.executable,
            str(_REALTIME_PROBE),
            "--worker",
            str(worker),
            "--bundle",
            str(bundle),
            "--backend-dir",
            str(build_dir),
            "--model-plugin-dir",
            str(Path(ctx.model_plugin_dir)),
            "--general-audio",
            str(source),
            "--function-audio",
            str(function_source),
            "--output-wav",
            str(output_wav),
            "--receipt",
            str(receipt_path),
            "--event-timeout-s",
            str(_timeout(case, "realtime_event_timeout_s", 180)),
        ]
        started = time.monotonic()
        completed = _run(
            probe_command,
            env,
            _timeout(case, "realtime_runtime_timeout_s", 1800),
        )
        elapsed = time.monotonic() - started
        if completed.returncode not in (0, 1):
            raise RuntimeError(
                "VoiceChat realtime probe could not complete "
                f"(rc={completed.returncode}): {completed.stderr[-2000:]}"
            )
        receipt = _load_realtime_receipt(receipt_path)
        if not output_wav.is_file():
            raise RuntimeError("VoiceChat realtime probe did not write its WAV artifact")

        return StageOutput(
            stage_name=stage.name,
            timing_s=elapsed,
            data={
                "receipt": receipt,
                "receipt_path": str(receipt_path),
                "wav_path": str(output_wav),
                "source_sha256": source_sha,
                "function_source_sha256": function_source_sha,
                "probe_returncode": completed.returncode,
            },
            metadata={
                "worker_build": {
                    "command": build_command,
                    "returncode": build.returncode,
                    "stdout": build.stdout or "",
                    "stderr": build.stderr or "",
                },
                "probe": {
                    "command": probe_command,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout or "",
                    "stderr": completed.stderr or "",
                },
            },
        )

    def _run_lifecycle_stage(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        if not ctx.binary_path:
            raise RuntimeError("VoiceChat lifecycle E2E requires the native trtmc binary")
        binary = Path(ctx.binary_path).resolve()
        build_dir = binary.parent
        if not (build_dir / "CMakeCache.txt").is_file():
            raise RuntimeError(
                "VoiceChat lifecycle E2E requires a configured source build beside the "
                f"selected trtmc binary: {build_dir}"
            )

        source = _source_path(case)
        source_sha = _sha256(source)
        expected_sha = str(case.metadata.get("speech_source_sha256", ""))
        if source_sha != expected_sha:
            raise RuntimeError(
                f"VoiceChat source sample SHA256 mismatch: expected {expected_sha}, got {source_sha}"
            )
        function_source = _source_path_from_input(case, "function_speech_source_relative_path")
        function_source_sha = _sha256(function_source)
        expected_function_sha = str(case.metadata.get("function_speech_source_sha256", ""))
        if function_source_sha != expected_function_sha:
            raise RuntimeError(
                "VoiceChat function-call sample SHA256 mismatch: "
                f"expected {expected_function_sha}, got {function_source_sha}"
            )
        bundle = _bundle_path(case, ctx)
        if not bundle.is_file():
            raise RuntimeError(f"VoiceChat lifecycle bundle is unavailable: {bundle}")
        if not ctx.model_plugin_dir:
            raise RuntimeError(
                "VoiceChat lifecycle E2E requires an explicit model plugin directory"
            )

        base = (
            Path(ctx.artifacts_dir)
            if ctx.artifacts_dir
            else Path(tempfile.mkdtemp(prefix="trtmc_voicechat_lifecycle_"))
        )
        artifact_dir = base / case.name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        output_wav = artifact_dir / "native_full_duplex_chunked.wav"
        receipt_path = artifact_dir / "native_full_duplex_lifecycle.json"

        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path
        build_command = [
            "cmake",
            "--build",
            str(build_dir),
            "--target",
            _LIFECYCLE_TARGET,
        ]
        build = _run(
            build_command,
            env,
            _timeout(case, "lifecycle_build_timeout_s", 600),
        )
        if build.returncode != 0:
            raise RuntimeError(
                "VoiceChat lifecycle probe target failed to build "
                f"(rc={build.returncode}): {build.stderr[-2000:]}"
            )

        probe = build_dir / _LIFECYCLE_TARGET
        if not probe.is_file():
            raise RuntimeError(f"VoiceChat lifecycle target did not produce {probe}")
        probe_command = [
            str(probe),
            str(bundle),
            str(source),
            str(build_dir),
            str(Path(ctx.model_plugin_dir)),
            str(output_wav),
            str(receipt_path),
        ]
        started = time.monotonic()
        completed = _run(
            probe_command,
            env,
            _timeout(case, "lifecycle_runtime_timeout_s", 1800),
        )
        elapsed = time.monotonic() - started
        if completed.returncode not in (0, 1):
            raise RuntimeError(
                "VoiceChat lifecycle probe could not complete "
                f"(rc={completed.returncode}): {completed.stderr[-2000:]}"
            )
        if _NEW_RUNTIME_MARKER not in (completed.stderr or ""):
            raise RuntimeError("VoiceChat lifecycle probe did not confirm the native runtime path")
        receipt = _load_lifecycle_receipt(receipt_path)
        if not output_wav.is_file():
            raise RuntimeError("VoiceChat lifecycle probe did not write its chunked WAV artifact")

        return StageOutput(
            stage_name=stage.name,
            timing_s=elapsed,
            data={
                "receipt": receipt,
                "receipt_path": str(receipt_path),
                "wav_path": str(output_wav),
                "source_sha256": source_sha,
                "function_source_sha256": function_source_sha,
                "probe_returncode": completed.returncode,
                "runtime_path_confirmed": True,
            },
            metadata={
                "probe_build": {
                    "command": build_command,
                    "returncode": build.returncode,
                    "stdout": build.stdout or "",
                    "stderr": build.stderr or "",
                },
                "probe": {
                    "command": probe_command,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout or "",
                    "stderr": completed.stderr or "",
                },
            },
        )


runner = VoiceChatModelCardRunner()
