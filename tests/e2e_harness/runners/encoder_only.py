"""Encoder-only NLP strategy runner — TRT inference for encoder-only models.

Runs the C++ binary for encoder-only forward pass and
captures hidden states / CLS embedding.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time

from .. import save_full_stderr
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec
from .text_generation import (
    _distributed_runtime_config,
    _ensure_distributed_runtime_env,
    _extract_rank_zero_stdout,
    _maybe_start_gpu_memory_sampler,
    _strip_mpi_stream_tags,
    _wrap_distributed_command,
)

logger = logging.getLogger(__name__)


class EncoderOnlyRunner:
    """Execute TRT encoder-only inference via the C++ binary."""

    @property
    def strategy_name(self) -> str:
        return "encoder_only_nlp"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        bundle_path = os.path.join(ctx.engine_dir, case.bundle)
        prompt = case.inputs.get("prompt", "")

        # encoder-only models use 'encode' to get hidden states / CLS embedding
        cmd = [
            ctx.binary_path, "encode", bundle_path,
            "--prompt", prompt,
        ]

        runtime_cli_python = ctx.runtime_cli_hf_python()
        if runtime_cli_python:
            cmd.extend(["--hf-python", runtime_cli_python])

        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path

        distributed_runtime = _distributed_runtime_config(case)
        if distributed_runtime:
            _ensure_distributed_runtime_env(case, ctx, env)
            extra_env = distributed_runtime.get("env", {})
            if isinstance(extra_env, dict):
                env.update({str(k): str(v) for k, v in extra_env.items()})
            cmd = _wrap_distributed_command(cmd, case, env)

        logger.info("Running encoder-only: %s", " ".join(cmd))
        t0 = time.monotonic()
        memory_sampler = _maybe_start_gpu_memory_sampler(
            distributed_runtime, ctx, case, env)
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, env=env, timeout=600,
            )
        finally:
            memory_meta = memory_sampler.stop() if memory_sampler is not None else None
        elapsed = time.monotonic() - t0
        stdout_for_parse = (
            _extract_rank_zero_stdout(result.stdout)
            if distributed_runtime else result.stdout.strip()
        )
        stderr_for_display = (
            _strip_mpi_stream_tags(result.stderr)
            if distributed_runtime else result.stderr
        )

        if result.returncode != 0:
            truncated, log_path = save_full_stderr(
                result.stderr, ctx.artifacts_dir or "",
                "encoder_only", case.name)
            msg = (f"Encoder-only inference failed (rc={result.returncode}): "
                   f"{truncated}")
            if log_path:
                msg += f" (full stderr: {log_path})"
            raise RuntimeError(msg)

        data = _parse_encoder_output(stdout_for_parse)

        metadata = {
            "command": cmd,
            "returncode": result.returncode,
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
        }
        if distributed_runtime:
            metadata["distributed_runtime"] = distributed_runtime
            metadata["rank_zero_stdout"] = stdout_for_parse
            metadata["stderr_without_mpi_tags"] = stderr_for_display
        if memory_meta is not None:
            metadata["gpu_memory"] = memory_meta

        return StageOutput(
            stage_name=stage.name,
            data=data,
            timing_s=elapsed,
            metadata=metadata,
        )


def _parse_encoder_output(stdout: str) -> dict:
    """Parse encoder-only output (hidden states / CLS embedding).

    trtmc encode outputs:
        Hidden states shape: [512, 768]
        [CLS] embedding (first 8 dims): -0.0522 0.0800 ...
    """
    # Try JSON first
    try:
        data = json.loads(stdout)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    # Parse "trtmc encode" text format
    cls_embedding = []
    for line in stdout.splitlines():
        line = line.strip()
        # "[CLS] embedding (first N dims): 0.1 0.2 0.3 ..."
        if line.startswith("[CLS] embedding"):
            colon_idx = line.find(":")
            if colon_idx >= 0:
                values_str = line[colon_idx + 1:].replace("...", "").strip()
                try:
                    cls_embedding = [float(x) for x in values_str.split() if x]
                except ValueError:
                    pass

    if cls_embedding:
        return {"cls_embedding": cls_embedding}

    # Fall back: try last line as whitespace-separated floats
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            values = [float(x) for x in line.split()]
            if values:
                return {"cls_embedding": values}
        except ValueError:
            continue

    return {"raw_output": stdout}


plugin = EncoderOnlyRunner()
