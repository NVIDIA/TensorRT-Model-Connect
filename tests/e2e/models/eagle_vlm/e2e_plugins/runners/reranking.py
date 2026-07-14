# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reranking strategy runner — TRT inference for reranking models.

Runs the C++ binary with ``trtmc rerank`` to produce relevance scores for one
or more query/document pairs. The binary prints ``Relevance score: <float>``
on stdout; this runner parses each score and returns them in manifest order.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from typing import Any

from .. import save_full_stderr
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

_SCORE_RE = re.compile(r"Relevance score:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


class RerankingRunner:
    """Execute TRT reranking inference via the C++ binary."""

    @property
    def strategy_name(self) -> str:
        return "reranking"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        bundle_path = os.path.join(ctx.engine_dir, case.bundle)
        prompt = case.inputs.get("prompt", "")
        documents = _documents_from_inputs(case.inputs)

        if not prompt or not documents:
            raise ValueError(
                "Reranking requires 'prompt' and at least one document in "
                f"manifest inputs (case={case.name!r})"
            )

        runtime_cli_python = ctx.runtime_cli_hf_python()

        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path
        distributed_runtime = _distributed_runtime_config(case)
        extra_env = distributed_runtime.get("env", {}) if distributed_runtime else {}
        if distributed_runtime and isinstance(extra_env, dict):
            env.update({str(k): str(v) for k, v in extra_env.items()})

        scores: list[float] = []
        commands: list[list[str]] = []
        stdout_by_document: list[str] = []
        stderr_by_document: list[str] = []
        command_metadata: list[dict[str, Any]] = []
        t0 = time.monotonic()

        for index, document in enumerate(documents):
            cmd = [
                ctx.binary_path, "rerank", bundle_path,
                "--prompt", prompt,
                "--document", document,
            ]
            if runtime_cli_python:
                cmd.extend(["--hf-python", runtime_cli_python])
            if ctx.model_plugin_dir:
                cmd.extend(["--model-plugin-dir", ctx.model_plugin_dir])
            run_env = dict(env)
            if distributed_runtime:
                _ensure_distributed_runtime_env(
                    case, ctx, run_env, rendezvous_suffix=f"-doc{index}")
                cmd = _wrap_distributed_command(cmd, case, run_env)

            logger.info("Running reranking document %d: %s", index, " ".join(cmd))
            memory_sampler = _maybe_start_gpu_memory_sampler(
                distributed_runtime, ctx, case, run_env)
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, env=run_env, timeout=600,
                )
            finally:
                memory_meta = (
                    memory_sampler.stop() if memory_sampler is not None else None)
            parse_stdout = (
                _extract_rank_zero_stdout(result.stdout)
                if distributed_runtime else result.stdout)
            parse_stderr = (
                _strip_mpi_stream_tags(result.stderr)
                if distributed_runtime else result.stderr)
            commands.append(cmd)
            stdout_by_document.append(parse_stdout or "")
            stderr_by_document.append(parse_stderr or "")
            doc_meta: dict[str, Any] = {
                "command": cmd,
                "returncode": result.returncode,
                "stdout": result.stdout or "",
                "stderr": parse_stderr or "",
            }
            if distributed_runtime:
                doc_meta["distributed_runtime"] = distributed_runtime
            if memory_meta is not None:
                doc_meta["gpu_memory"] = memory_meta
            command_metadata.append(doc_meta)

            if result.returncode != 0:
                truncated, log_path = save_full_stderr(
                    parse_stderr, ctx.artifacts_dir or "",
                    f"reranking_doc_{index}", case.name)
                msg = (
                    f"Reranking inference failed for document {index} "
                    f"(rc={result.returncode}): {truncated}"
                )
                if log_path:
                    msg += f" (full stderr: {log_path})"
                raise RuntimeError(msg)

            scores.append(_parse_score(parse_stdout))

        elapsed = time.monotonic() - t0
        metadata: dict[str, Any] = {
            "commands": commands,
            "stdout_by_document": stdout_by_document,
            "stderr_by_document": stderr_by_document,
            "document_count": len(documents),
        }
        for index, doc_meta in enumerate(command_metadata):
            metadata[f"document_{index}"] = doc_meta
        if len(commands) == 1:
            metadata.update(command_metadata[0])
        if distributed_runtime:
            metadata["distributed_runtime"] = distributed_runtime

        return StageOutput(
            stage_name=stage.name,
            data={"scores": scores, "documents": documents},
            timing_s=elapsed,
            metadata=metadata,
        )


def _parse_score(stdout: str) -> float:
    """Parse the single relevance score from ``trtmc rerank`` stdout."""
    match = _SCORE_RE.search(stdout)
    if match:
        return float(match.group(1))

    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return float(line)
        except ValueError:
            continue

    raise ValueError(f"Could not parse relevance score from output: {stdout[:500]}")


def _documents_from_inputs(inputs: dict[str, Any]) -> list[str]:
    documents = inputs.get("documents")
    if documents is not None:
        if not isinstance(documents, list):
            raise TypeError("Reranking 'documents' input must be a list")
        return [str(doc) for doc in documents if str(doc)]

    document = inputs.get("document", "")
    return [str(document)] if str(document) else []


plugin = RerankingRunner()
