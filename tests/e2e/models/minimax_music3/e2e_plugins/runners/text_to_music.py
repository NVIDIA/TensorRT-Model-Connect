# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runner for the native MiniMax-Music3 bundle.

The request the harness builds is text_to_audio's: a prompt, a token
budget and a seed, plus the family's runtime-config namespace. The lyrics
occupy the prompt because the contract scores a transcript against that field;
the music description travels as music_minimax_music3.caption.
"""

from __future__ import annotations

import os
import struct
import subprocess
import tempfile
import time

from .. import save_full_stderr
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec

RUNTIME_CONFIG_NAMESPACE = "music_minimax_music3"


def build_request(case: E2ECase) -> dict:
    """Return the generate-audio request for one case."""

    inputs = getattr(case, "inputs", {}) or {}
    runtime = (getattr(case, "metadata", {}) or {}).get("runtime_config", {})
    namespace = runtime.get(RUNTIME_CONFIG_NAMESPACE, {}) if runtime else {}
    request = {
        "batch_size": 1,
        "prompt": inputs.get("prompt", ""),
        "max_new_tokens": int(inputs.get("max_new_tokens", 0) or 0),
    }
    seed = inputs.get("seed", namespace.get("seed"))
    if seed is not None:
        request["seed"] = int(seed)
    if namespace:
        request["runtime_config"] = {RUNTIME_CONFIG_NAMESPACE: dict(namespace)}
    return request


class TextToMusicRunner:
    @property
    def strategy_name(self) -> str:
        """The task strategy, which is what the registry keys runners by.

        Not the runtime strategy: ``TaskStrategyRunner`` wants the task-side
        name, and a runner that offers ``runtime_strategy`` instead satisfies
        no protocol and is silently skipped.
        """

        return "text_to_audio"

    @property
    def runtime_strategy(self) -> str:
        """The engine-side strategy this family's bundle declares."""

        return "minimax_music3_text_to_music"

    def run_stage(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        if stage.name in ("generate", "generate_audio", "end_to_end", "full_generation",
                          "smoke_test"):
            return self._run_generate_audio(case, stage, ctx)
        return StageOutput(
            stage_name=stage.name,
            data={"error": f"Unknown text_to_audio stage: {stage.name}"},
        )

    def _run_generate_audio(self, case: E2ECase, stage: StageSpec,
                            ctx: RunContext) -> StageOutput:
        """Drive the C++ binary and report what came back."""

        request = build_request(case)
        bundle_path = _resolve_bundle_path(case, ctx)

        with tempfile.TemporaryDirectory(prefix="trtmc_music3_") as tmpdir:
            wav_path = os.path.join(tmpdir, "output.wav")
            cmd = [
                ctx.binary_path, "generate-audio", bundle_path,
                "--prompt", request["prompt"],
                "--output", wav_path,
            ]
            if request.get("max_new_tokens"):
                cmd.extend(["--max-new-tokens", str(request["max_new_tokens"])])

            # The caption and the seed travel as runtime-config assignments,
            # which is the only channel the CLI has for a namespace field.
            namespace = request.get("runtime_config", {}).get(RUNTIME_CONFIG_NAMESPACE, {})
            for key, value in sorted(namespace.items()):
                cmd.extend(["--set", f"{RUNTIME_CONFIG_NAMESPACE}.{key}={value}"])
            if "seed" in request and "seed" not in namespace:
                cmd.extend(["--set", f"{RUNTIME_CONFIG_NAMESPACE}.seed={request['seed']}"])

            runtime_cli_python = ctx.runtime_cli_hf_python()
            if runtime_cli_python:
                cmd.extend(["--hf-python", runtime_cli_python])

            started = time.monotonic()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            elapsed = time.monotonic() - started

            stderr_truncated, stderr_log = save_full_stderr(
                result.stderr or "", ctx.artifacts_dir or "", "text_to_audio", case.name)
            data: dict = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": stderr_truncated,
                "elapsed_s": elapsed,
                "wav_exists": False,
            }
            if stderr_log:
                data["stderr_log"] = stderr_log
            if os.path.exists(wav_path):
                # Outlive the temporary directory: the contract transcribes
                # this after run_stage returns, and a path into a directory
                # that no longer exists fails the lyric check for the wrong
                # reason.
                kept = os.path.join(
                    ctx.artifacts_dir or tempfile.gettempdir(),
                    f"{case.name}_generated.wav",
                )
                os.makedirs(os.path.dirname(kept), exist_ok=True)
                os.replace(wav_path, kept)
                wav_path = kept
            data.update(_describe_wav(wav_path))
            return StageOutput(stage_name=stage.name, data=data)


def _resolve_bundle_path(case: E2ECase, ctx: RunContext) -> str:
    """Return the bundle the harness built for this case.

    The manifest names it; the harness says which directory it lands in. An
    earlier revision guessed at a ctx.bundle_path that does not exist, which
    would have failed the moment the harness ran this runner rather than the
    CLI being driven by hand.
    """

    name = case.bundle or case.inputs.get("bundle", "") or f"{case.name}.bundle"
    if os.path.isabs(name):
        return name
    return os.path.join(ctx.engine_dir, name)


def _describe_wav(path: str) -> dict:
    """Return what the contract scores: existence, RMS, duration, channels.

    The header is read directly rather than through a wave module because this
    model writes IEEE float32, which the standard library's reader rejects.
    """

    import math

    if not os.path.exists(path):
        return {"wav_exists": False}

    with open(path, "rb") as handle:
        raw = handle.read()
    if len(raw) < 44 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        return {"wav_exists": True, "wav_valid": False}

    audio_format, channels, sample_rate = struct.unpack("<HHI", raw[20:28])
    bits_per_sample = struct.unpack("<H", raw[34:36])[0]
    payload = raw[44:]
    if audio_format == 3 and bits_per_sample == 32:
        count = len(payload) // 4
        samples = struct.unpack(f"<{count}f", payload[: count * 4])
    elif bits_per_sample == 16:
        count = len(payload) // 2
        samples = [value / 32768.0 for value in
                   struct.unpack(f"<{count}h", payload[: count * 2])]
    else:
        return {"wav_exists": True, "wav_valid": False}

    frames = len(samples) // max(channels, 1)
    total = math.fsum(value * value for value in samples)
    return {
        "wav_exists": True,
        "wav_valid": True,
        "wav_path": path,
        "channels": channels,
        "sample_rate": sample_rate,
        "num_frames": frames,
        "duration_s": frames / sample_rate if sample_rate else 0.0,
        "rms": math.sqrt(total / len(samples)) if samples else 0.0,
    }


plugin = TextToMusicRunner()
