# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image-classification strategy runner."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

from .. import save_full_stderr, _case_artifact_dir
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec

logger = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parents[7]
_MPI_TAGGED_STDOUT_RE = re.compile(
    r"^\[[^\]]+,(?P<rank>\d+)\]<stdout>:\s?(?P<text>.*)$")
_MPI_STREAM_TAG_RE = re.compile(r"\[[^\]]+,\d+\]<(?:stdout|stderr)>:\s?")


def _distributed_runtime_config(case: E2ECase) -> dict:
    config = case.metadata.get("distributed_runtime", {})
    return config if isinstance(config, dict) and config.get("enabled") else {}


def _extract_rank_zero_stdout(stdout: str) -> str:
    rank0_lines: list[str] = []
    saw_tagged = False
    for line in (stdout or "").splitlines():
        match = _MPI_TAGGED_STDOUT_RE.match(line)
        if match is None:
            continue
        saw_tagged = True
        if int(match.group("rank")) == 0:
            rank0_lines.append(match.group("text"))
    if saw_tagged:
        return "\n".join(rank0_lines).strip()
    return (stdout or "").strip()


def _strip_mpi_stream_tags(text: str) -> str:
    return _MPI_STREAM_TAG_RE.sub("", text or "")


def _ensure_distributed_runtime_env(
    case: E2ECase,
    ctx: RunContext,
    env: dict[str, str],
) -> None:
    if not _distributed_runtime_config(case) or env.get("TRTMC_NCCL_RENDEZVOUS"):
        return
    root = (
        Path(_case_artifact_dir(ctx.artifacts_dir, case.name))
        if ctx.artifacts_dir else Path(tempfile.gettempdir())
    )
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{case.name}.nccl_rendezvous.bin"
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    env["TRTMC_NCCL_RENDEZVOUS"] = str(path)


def _wrap_distributed_command(
    cmd: list[str],
    case: E2ECase,
    env: dict[str, str],
) -> list[str]:
    config = _distributed_runtime_config(case)
    if not config:
        return cmd
    launcher = str(config.get("launcher", "mpirun") or "mpirun")
    world_size = int(config.get("world_size", config.get("tp_size", 2)) or 2)
    launcher_args = config.get("launcher_args")
    if isinstance(launcher_args, list):
        prefix = [launcher] + [str(arg) for arg in launcher_args]
    else:
        prefix = [launcher, "--tag-output", "-np", str(world_size)]

    export_env = config.get("export_env", ["LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES"])
    if isinstance(export_env, list) and Path(launcher).name == "mpirun":
        export_names = [str(name) for name in export_env]
        if "TRTMC_NCCL_RENDEZVOUS" in env and "TRTMC_NCCL_RENDEZVOUS" not in export_names:
            export_names.append("TRTMC_NCCL_RENDEZVOUS")
        for name in export_names:
            if name in env:
                prefix.extend(["-x", name])

    return prefix + cmd


class ImageClassificationRunner:
    @property
    def strategy_name(self) -> str:
        return "image_classification"

    def run_stage(
        self,
        case: E2ECase,
        stage: StageSpec,
        ctx: RunContext,
    ) -> StageOutput:
        bundle_path = os.path.join(ctx.engine_dir, case.bundle)
        image_path = self._resolve_image_path(case, ctx)
        cmd = [
            ctx.binary_path,
            "classify",
            bundle_path,
            "--image",
            image_path or "",
        ]
        runtime_cli_python = ctx.runtime_cli_hf_python()
        if runtime_cli_python:
            cmd.extend(["--hf-python", runtime_cli_python])
        if ctx.model_plugin_dir:
            cmd.extend(["--model-plugin-dir", ctx.model_plugin_dir])

        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path
        _ensure_distributed_runtime_env(case, ctx, env)
        cmd = _wrap_distributed_command(cmd, case, env)

        logger.info("Running image classification: %s", " ".join(cmd))
        t0 = time.monotonic()
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=600)
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            truncated, log_path = save_full_stderr(
                result.stderr, ctx.artifacts_dir or "",
                "image_classification", case.name)
            msg = (
                f"Image classification failed (rc={result.returncode}): "
                f"{truncated}"
            )
            if log_path:
                msg += f" (full stderr: {log_path})"
            raise RuntimeError(msg)

        stdout = _extract_rank_zero_stdout(result.stdout)
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            data = {"raw_output": stdout}

        stderr_truncated, stderr_log = save_full_stderr(
            result.stderr or "", ctx.artifacts_dir or "",
            "image_classification", case.name)
        metadata = {
            "command": cmd,
            "returncode": result.returncode,
            "stdout": _strip_mpi_stream_tags(result.stdout or ""),
            "stderr": _strip_mpi_stream_tags(stderr_truncated),
        }
        if stderr_log:
            metadata["stderr_log"] = stderr_log

        return StageOutput(
            stage_name=stage.name,
            data=data,
            timing_s=elapsed,
            metadata=metadata,
        )

    def _resolve_image_path(self, case: E2ECase, ctx: RunContext) -> str | None:
        image = (
            case.inputs.get("image") or case.inputs.get("test_image")
            or case.inputs.get("image_path")
        )
        if not image:
            return None
        path = Path(image)
        if path.is_absolute():
            return str(path)
        for base in (ctx.engine_dir, str(PROJECT_DIR), str(PROJECT_DIR / "tests" / "e2e")):
            candidate = Path(base) / image
            if candidate.is_file():
                return str(candidate)
        return str(path)


plugin = ImageClassificationRunner()
