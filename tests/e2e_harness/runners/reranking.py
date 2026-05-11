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

            logger.info("Running reranking document %d: %s", index, " ".join(cmd))
            result = subprocess.run(
                cmd, capture_output=True, text=True, env=env, timeout=600,
            )
            commands.append(cmd)
            stdout_by_document.append(result.stdout or "")
            stderr_by_document.append(result.stderr or "")
            command_metadata.append({
                "command": cmd,
                "returncode": result.returncode,
                "stdout": result.stdout or "",
                "stderr": result.stderr or "",
            })

            if result.returncode != 0:
                truncated, log_path = save_full_stderr(
                    result.stderr, ctx.artifacts_dir or "",
                    f"reranking_doc_{index}", case.name)
                msg = (
                    f"Reranking inference failed for document {index} "
                    f"(rc={result.returncode}): {truncated}"
                )
                if log_path:
                    msg += f" (full stderr: {log_path})"
                raise RuntimeError(msg)

            scores.append(_parse_score(result.stdout))

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
