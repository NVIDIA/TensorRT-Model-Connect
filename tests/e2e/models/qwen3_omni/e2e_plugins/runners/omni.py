# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Omni-multimodal and composite pipeline strategy runners.

OmniMultimodalRunner handles multi-branch models (thinker, vision, audio,
talker, code2wav) with stage-by-stage execution.

CompositePipelineRunner handles generic composite pipelines following
the stage graph from the manifest.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from typing import Any

from .. import _case_artifact_dir, save_full_stderr
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec

logger = logging.getLogger(__name__)

_DEFAULT_RUNTIME_TIMEOUT_S = 600
_MAX_RUNTIME_TIMEOUT_S = 3600
_SIMPLE_WAVEFORM_FALLBACK = "no Code2Wav engine, generating simple waveform"
_THINKER_TEXT_PREFIX = "[trtmc] Omni Thinker text: "


def _runtime_timeout_s(case: E2ECase) -> int:
    """Return the model-owned runtime budget for this testcase."""
    value = case.inputs.get("runtime_timeout_s", _DEFAULT_RUNTIME_TIMEOUT_S)
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 1
        or value > _MAX_RUNTIME_TIMEOUT_S
    ):
        raise ValueError(
            "runtime_timeout_s must be an integer from 1 to "
            f"{_MAX_RUNTIME_TIMEOUT_S}; got {value!r}"
        )
    return value


def _timeout_stream(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


class OmniMultimodalRunner:
    """Execute TRT omni-multimodal inference via the C++ binary.

    Multi-branch model with separate stages: thinker text decoding,
    vision encoding, audio encoding, talker decoding, and code2wav.
    """

    @property
    def strategy_name(self) -> str:
        return "qwen3_omni_multimodal"

    def run_stage(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        stage_name = stage.name

        cmd, stage_meta = _build_omni_command(case, stage, ctx)
        if cmd is None:
            return StageOutput(
                stage_name=stage_name,
                metadata={
                    "error": stage_meta.get("error", "Unsupported stage"),
                    "skipped": True,
                    **stage_meta,
                },
            )

        if ctx.model_plugin_dir:
            cmd.extend(["--model-plugin-dir", ctx.model_plugin_dir])

        runtime_cli_python = ctx.runtime_cli_hf_python()
        if runtime_cli_python:
            cmd.extend(["--hf-python", runtime_cli_python])

        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path

        logger.info("Running omni stage %s: %s", stage_name, " ".join(cmd))
        timeout_s = _runtime_timeout_s(case)
        t0 = time.monotonic()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = time.monotonic() - t0
            stderr = _timeout_stream(exc.stderr)
            if not stderr:
                stderr = "No partial stderr was captured before timeout.\n"
            truncated, log_path = save_full_stderr(
                stderr,
                ctx.artifacts_dir or "",
                f"omni_{stage_name}_timeout",
                case.name,
            )
            msg = (
                f"Omni stage {stage_name} exceeded its model-owned runtime "
                f"budget of {timeout_s}s after {elapsed:.1f}s; command: "
                f"{' '.join(cmd)}"
            )
            if truncated:
                msg += f"; partial stderr: {truncated.rstrip()}"
            if log_path:
                msg += f" (full partial stderr: {log_path})"
            raise RuntimeError(msg) from exc
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            truncated, log_path = save_full_stderr(
                result.stderr, ctx.artifacts_dir or "", f"omni_{stage_name}", case.name
            )
            msg = f"Omni stage {stage_name} failed (rc={result.returncode}): {truncated}"
            if log_path:
                msg += f" (full stderr: {log_path})"
            raise RuntimeError(msg)

        if stage_name == "talker_decode" and _SIMPLE_WAVEFORM_FALLBACK in (result.stderr or ""):
            raise RuntimeError(
                "Qwen3-Omni talker_decode used the synthetic simple-waveform "
                "fallback because the Code2Wav engine is missing"
            )

        data = _parse_stage_output(result.stdout.strip(), stage_name)
        thinker_text = _parse_thinker_text(result.stderr or "")
        if thinker_text:
            data["thinker_text"] = thinker_text

        return StageOutput(
            stage_name=stage_name,
            data=data,
            text=data.get("text"),
            timing_s=elapsed,
            metadata={
                "command": cmd,
                "returncode": result.returncode,
                "stdout": result.stdout or "",
                "stderr": result.stderr or "",
                **stage_meta,
            },
        )


class CompositePipelineRunner:
    """Execute a generic composite pipeline following stage graph from manifest.

    The CLI has no stage selector, so stages are mapped to supported
    entrypoints (currently ``run``) with explicit metadata.
    """

    @property
    def strategy_name(self) -> str:
        return "composite_pipeline"

    def run_stage(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        stage_name = stage.name

        cmd, stage_meta = _build_composite_command(case, stage, ctx)

        runtime_cli_python = ctx.runtime_cli_hf_python()
        if runtime_cli_python:
            cmd.extend(["--hf-python", runtime_cli_python])

        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path

        logger.info("Running composite stage %s: %s", stage_name, " ".join(cmd))
        t0 = time.monotonic()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
        )
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            truncated, log_path = save_full_stderr(
                result.stderr, ctx.artifacts_dir or "", f"composite_{stage_name}", case.name
            )
            msg = f"Composite stage {stage_name} failed (rc={result.returncode}): {truncated}"
            if log_path:
                msg += f" (full stderr: {log_path})"
            raise RuntimeError(msg)

        data = _parse_stage_output(result.stdout.strip(), stage_name)

        return StageOutput(
            stage_name=stage_name,
            data=data,
            text=data.get("text"),
            timing_s=elapsed,
            metadata={
                "command": cmd,
                "returncode": result.returncode,
                "stdout": result.stdout or "",
                "stderr": result.stderr or "",
                **stage_meta,
            },
        )


def _resolve_bundle_path(case: E2ECase, ctx: RunContext) -> str:
    bundle = case.bundle or f"{case.name}.trtfb"
    if os.path.isabs(bundle):
        return bundle
    return os.path.join(ctx.engine_dir, bundle)


def _build_omni_command(
    case: E2ECase, stage: StageSpec, ctx: RunContext
) -> tuple[list[str] | None, dict[str, Any]]:
    stage_name = stage.name
    bundle_path = _resolve_bundle_path(case, ctx)
    prompt = case.inputs.get("prompt", "")
    image = case.inputs.get("image")
    audio = case.inputs.get("audio") or case.inputs.get("audio_path")
    max_new_tokens = _safe_int(case.inputs.get("max_new_tokens", 30), default=30)

    stage_meta: dict[str, Any] = {
        "requested_stage": stage_name,
        "cli_stage_supported": False,
        "stage_resolution": "mapped_without_stage_flag",
    }

    if stage_name == "vision_encode":
        if not image:
            stage_meta.update(
                {
                    "error": "vision_encode requires image input",
                    "stage_resolution": "unsupported_missing_image",
                }
            )
            return None, stage_meta
        cmd = [ctx.binary_path, "embed", bundle_path, "--image", str(image)]
        if prompt:
            cmd.extend(["--prompt", str(prompt)])
        stage_meta["entrypoint"] = "embed"
        return cmd, stage_meta

    if stage_name == "audio_encode":
        if not audio:
            stage_meta.update(
                {
                    "error": "audio_encode requires audio input",
                    "stage_resolution": "unsupported_missing_audio",
                }
            )
            return None, stage_meta
        cmd = [ctx.binary_path, "transcribe", bundle_path, "--audio", str(audio)]
        if max_new_tokens > 0:
            cmd.extend(["--max-new-tokens", str(max_new_tokens)])
        stage_meta["entrypoint"] = "transcribe"
        stage_meta["note"] = "audio_encode mapped to transcribe (no embedding CLI)"
        return cmd, stage_meta

    if stage_name == "talker_decode" and prompt:
        out_audio = os.path.join(
            _case_artifact_dir(ctx.artifacts_dir or "/tmp/claude", case.name),
            "talker_decode.wav",
        )
        cmd = [
            ctx.binary_path,
            "generate-audio",
            bundle_path,
            "--prompt",
            str(prompt),
            "--output",
            out_audio,
        ]
        if max_new_tokens > 0:
            cmd.extend(["--max-new-tokens", str(max_new_tokens)])
        stage_meta["entrypoint"] = "generate-audio"
        stage_meta["audio_output_path"] = out_audio
        return cmd, stage_meta

    if stage_name == "talker_decode" and audio:
        out_audio = os.path.join(
            _case_artifact_dir(ctx.artifacts_dir or "/tmp/claude", case.name),
            "talker_decode.wav",
        )
        cmd = [
            ctx.binary_path,
            "speak",
            bundle_path,
            "--audio-in",
            str(audio),
            "--audio-out",
            out_audio,
        ]
        if max_new_tokens > 0:
            cmd.extend(["--max-new-tokens", str(max_new_tokens)])
        stage_meta["entrypoint"] = "speak"
        stage_meta["audio_output_path"] = out_audio
        return cmd, stage_meta

    cmd = [ctx.binary_path, "run", bundle_path]
    if prompt:
        cmd.extend(["--prompt", str(prompt)])
    if image and stage_name == "end_to_end":
        cmd.extend(["--image", str(image)])
    if max_new_tokens > 0 and stage_name in ("thinker_decode", "talker_decode", "end_to_end"):
        cmd.extend(["--max-new-tokens", str(max_new_tokens)])
    stage_meta["entrypoint"] = "run"
    if stage_name not in ("thinker_decode", "talker_decode", "end_to_end"):
        stage_meta["stage_resolution"] = "fallback_run_no_direct_entrypoint"
    return cmd, stage_meta


def _build_composite_command(
    case: E2ECase, stage: StageSpec, ctx: RunContext
) -> tuple[list[str], dict[str, Any]]:
    bundle_path = _resolve_bundle_path(case, ctx)
    prompt = case.inputs.get("prompt", "")
    image = case.inputs.get("image")
    max_new_tokens = _safe_int(case.inputs.get("max_new_tokens", 30), default=30)

    cmd = [ctx.binary_path, "run", bundle_path]
    if prompt:
        cmd.extend(["--prompt", str(prompt)])
    if image:
        cmd.extend(["--image", str(image)])
    if max_new_tokens > 0:
        cmd.extend(["--max-new-tokens", str(max_new_tokens)])

    stage_meta: dict[str, Any] = {
        "requested_stage": stage.name,
        "entrypoint": "run",
        "cli_stage_supported": False,
        "stage_resolution": "mapped_without_stage_flag",
    }
    if stage.name != "end_to_end":
        stage_meta["stage_resolution"] = "fallback_run_no_direct_entrypoint"
    return cmd, stage_meta


def _safe_int(raw: object, default: int) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _parse_stage_output(stdout: str, stage_name: str) -> dict[str, Any]:
    """Parse stage output from C++ binary stdout."""
    # Try JSON
    try:
        data = json.loads(stdout)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    # For text-producing stages, treat stdout as generated text
    if stage_name in ("thinker_decode", "talker_decode", "end_to_end"):
        return {"text": stdout}

    # For encoding stages, try to parse as embedding
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            values = [float(x) for x in line.split()]
            if values:
                return {"embedding": values}
        except ValueError:
            continue

    return {"raw_output": stdout}


def _parse_thinker_text(stderr: str) -> str:
    """Extract the final Thinker response reported by the C++ runtime."""
    for line in reversed(stderr.splitlines()):
        line = line.strip()
        if line.startswith(_THINKER_TEXT_PREFIX):
            return line[len(_THINKER_TEXT_PREFIX) :].strip()
    return ""


# Primary plugin for auto-discovery
plugin = OmniMultimodalRunner()
