"""Embedding strategy runner — TRT inference for embedding models.

Runs the C++ binary with ``trtmc embed`` to produce L2-normalized vectors.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time

from .. import _case_artifact_dir, save_full_stderr
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec
from ._runtime_common import (
    _distributed_runtime_config,
    _ensure_distributed_runtime_env,
    _extract_rank_zero_stdout,
    _maybe_start_gpu_memory_sampler,
    _strip_mpi_stream_tags,
    _wrap_distributed_command,
)

logger = logging.getLogger(__name__)


class EmbeddingRunner:
    """Execute TRT embedding inference via the C++ binary."""

    @property
    def strategy_name(self) -> str:
        return "embedding"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        bundle_path = os.path.join(ctx.engine_dir, case.bundle)
        prompt = case.inputs.get("prompt", "")
        image_path = case.inputs.get("image", "")

        cmd = [ctx.binary_path, "embed", bundle_path]

        if prompt:
            cmd.extend(["--prompt", prompt])
        if image_path:
            cmd.extend(["--image", image_path])

        runtime_cli_python = ctx.runtime_cli_hf_python()
        if runtime_cli_python:
            cmd.extend(["--hf-python", runtime_cli_python])

        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path
        distributed_runtime = _distributed_runtime_config(case)
        embedding_stdout_path = ""
        if distributed_runtime:
            _ensure_distributed_runtime_env(case, ctx, env)
            extra_env = distributed_runtime.get("env", {})
            if isinstance(extra_env, dict):
                env.update({str(k): str(v) for k, v in extra_env.items()})
            if ctx.artifacts_dir:
                artifact_dir = _case_artifact_dir(ctx.artifacts_dir, case.name)
                embedding_stdout_path = os.path.join(
                    artifact_dir, "embedding_rank0_stdout.json")
                try:
                    os.unlink(embedding_stdout_path)
                except FileNotFoundError:
                    pass
                env["TRTMC_EMBEDDING_STDOUT"] = embedding_stdout_path
                cmd = _wrap_rank0_stdout_capture(cmd)
                export_env = distributed_runtime.get("export_env", [])
                if isinstance(export_env, list):
                    export_names = [str(name) for name in export_env]
                    if "TRTMC_EMBEDDING_STDOUT" not in export_names:
                        distributed_runtime = dict(distributed_runtime)
                        distributed_runtime["export_env"] = (
                            export_names + ["TRTMC_EMBEDDING_STDOUT"])
            cmd = _wrap_distributed_command(cmd, case, env)

        logger.info("Running embedding: %s", " ".join(cmd))
        t0 = time.monotonic()
        memory_sampler = _maybe_start_gpu_memory_sampler(
            distributed_runtime, ctx, case, env)
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, env=env, timeout=600,
            )
        finally:
            memory_meta = (
                memory_sampler.stop() if memory_sampler is not None else None)
        elapsed = time.monotonic() - t0
        parse_stdout = (
            _extract_rank_zero_stdout(result.stdout)
            if distributed_runtime else result.stdout.strip())
        if embedding_stdout_path and os.path.isfile(embedding_stdout_path):
            with open(embedding_stdout_path, encoding="utf-8") as f:
                parse_stdout = f.read().strip()
        parse_stderr = (
            _strip_mpi_stream_tags(result.stderr)
            if distributed_runtime else result.stderr)

        if result.returncode != 0:
            truncated, log_path = save_full_stderr(
                parse_stderr, ctx.artifacts_dir or "",
                "embedding", case.name)
            msg = (f"Embedding inference failed (rc={result.returncode}): "
                   f"{truncated}")
            if log_path:
                msg += f" (full stderr: {log_path})"
            raise RuntimeError(msg)

        # Parse embedding vector from stdout (JSON array or whitespace-separated floats)
        embedding = _parse_embedding(parse_stdout)

        metadata = {
            "command": cmd,
            "returncode": result.returncode,
            "stdout": result.stdout or "",
            "stderr": parse_stderr or "",
        }
        if distributed_runtime:
            metadata["distributed_runtime"] = distributed_runtime
        if memory_meta is not None:
            metadata["gpu_memory"] = memory_meta

        return StageOutput(
            stage_name=stage.name,
            data={"embedding": embedding},
            timing_s=elapsed,
            metadata=metadata,
        )


def _parse_embedding(stdout: str) -> list[float]:
    """Parse embedding vector from C++ binary output."""
    # Try JSON array first
    try:
        data = json.loads(stdout)
        if isinstance(data, list):
            return [float(x) for x in data]
        if isinstance(data, dict) and "embedding" in data:
            return [float(x) for x in data["embedding"]]
    except (json.JSONDecodeError, ValueError):
        pass

    # OpenMPI --tag-output can split long rank-0 JSON lines into tagged
    # chunks. The distributed stdout extractor rejoins those chunks with
    # newlines, which may land inside JSON numbers.
    try:
        compact = "".join(stdout.splitlines())
        data = json.loads(compact)
        if isinstance(data, list):
            return [float(x) for x in data]
        if isinstance(data, dict) and "embedding" in data:
            return [float(x) for x in data["embedding"]]
    except (json.JSONDecodeError, ValueError):
        pass

    # Try whitespace-separated floats (last line may be the embedding)
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            values = [float(x) for x in line.replace(",", " ").split()]
            if len(values) > 1:
                return values
        except ValueError:
            continue

    raise ValueError(f"Could not parse embedding from output: {stdout[:500]}")


def _wrap_rank0_stdout_capture(cmd: list[str]) -> list[str]:
    script = (
        'rank="${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${PMIX_RANK:-${RANK:-0}}}}"; '
        'if [ "$rank" = "0" ] && [ -n "${TRTMC_EMBEDDING_STDOUT:-}" ]; then '
        'exec "$@" > "$TRTMC_EMBEDDING_STDOUT"; '
        'else exec "$@" >/dev/null; fi'
    )
    return ["bash", "-lc", script, "trtmc-embed-rank-wrapper"] + cmd


plugin = EmbeddingRunner()
