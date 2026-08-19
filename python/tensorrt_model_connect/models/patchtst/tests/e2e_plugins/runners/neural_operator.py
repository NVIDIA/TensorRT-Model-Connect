# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Neural operator strategy runner — TRT inference for neural operator models.

Runs the C++ binary with field inputs and captures output field arrays.
Neural operators (e.g. FourCastNet, SFNO) map physical fields to fields.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

from .. import _case_artifact_dir, save_full_stderr
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec

logger = logging.getLogger(__name__)

_MPI_STREAM_TAG_RE = re.compile(
    r"\[[^\]]+,(?P<rank>\d+)\]<(?P<stream>stdout|stderr)>:")


class NeuralOperatorRunner:
    """Execute TRT neural operator inference via the C++ binary."""

    @property
    def strategy_name(self) -> str:
        return "neural_operator"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        if stage.name != "full_inference":
            return StageOutput(
                stage_name=stage.name,
                metadata={"error": f"Unknown stage: {stage.name}"},
            )

        bundle_path = _resolve_bundle_path(case, ctx)
        solve_args, input_mode, input_error = _build_solve_input_args(case.inputs)
        if solve_args is None:
            return StageOutput(
                stage_name=stage.name,
                metadata={"error": input_error, "skipped": True},
            )

        cmd = [ctx.binary_path, "solve", bundle_path, *solve_args]
        output_path = case.inputs.get("output_field")

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

        logger.info("Running neural operator: %s", " ".join(cmd))
        t0 = time.monotonic()
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=600,
        )
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            truncated, log_path = save_full_stderr(
                result.stderr, ctx.artifacts_dir or "",
                "neural_operator", case.name)
            msg = (f"Neural operator inference failed (rc={result.returncode}): "
                   f"{truncated}")
            if log_path:
                msg += f" (full stderr: {log_path})"
            raise RuntimeError(msg)

        stdout = (
            _extract_rank_zero_stdout(result.stdout)
            if distributed_runtime else result.stdout.strip()
        )
        stderr = (
            _strip_mpi_stream_tags(result.stderr)
            if distributed_runtime else result.stderr
        )
        data = _parse_field_output(stdout, output_path)

        return StageOutput(
            stage_name=stage.name,
            data=data,
            timing_s=elapsed,
            metadata={
                "command": cmd,
                "returncode": result.returncode,
                "stdout": stdout or "",
                "stderr": stderr or "",
                "input_mode": input_mode,
                **({"distributed_runtime": distributed_runtime}
                   if distributed_runtime else {}),
            },
        )


def _resolve_bundle_path(case: E2ECase, ctx: RunContext) -> str:
    bundle = case.bundle or f"{case.name}.bundle"
    if os.path.isabs(bundle):
        return bundle
    return os.path.join(ctx.engine_dir, bundle)


def _normalize_numeric_values(raw: object) -> str:
    if raw is None:
        return ""
    if isinstance(raw, (list, tuple)):
        return ",".join(str(v) for v in raw)
    if isinstance(raw, (int, float)):
        return str(raw)
    if isinstance(raw, str):
        value = raw.strip()
        if value and os.path.isfile(value):
            try:
                with open(value, encoding="utf-8") as f:
                    payload = f.read().strip()
            except OSError:
                return value
            if not payload:
                return ""
            try:
                parsed = json.loads(payload)
                if isinstance(parsed, list):
                    return _normalize_numeric_values(parsed)
            except json.JSONDecodeError:
                pass
            return payload
        return value
    return str(raw).strip()


def _build_solve_input_args(inputs: dict) -> tuple[list[str] | None, str, str]:
    branch_input = inputs.get("branch_input")
    trunk_input = inputs.get("trunk_input")
    if branch_input is not None or trunk_input is not None:
        branch_csv = _normalize_numeric_values(branch_input)
        if not branch_csv:
            return None, "branch_trunk", "Neural operator branch_input must be non-empty"
        args = ["--branch-input", branch_csv]
        mode = "branch"
        if trunk_input is not None:
            trunk_csv = _normalize_numeric_values(trunk_input)
            if not trunk_csv:
                return None, "branch_trunk", "Neural operator trunk_input must be non-empty"
            args.extend(["--trunk-input", trunk_csv])
            mode = "branch_trunk"
        return args, mode, ""

    field_input = inputs.get("field_input")
    if field_input is None:
        field_input = inputs.get("input_field")
    if field_input is None:
        input_fields = inputs.get("input_fields")
        if isinstance(input_fields, list) and input_fields:
            field_input = input_fields[0]
    field_csv = _normalize_numeric_values(field_input)
    if not field_csv:
        return None, "field", (
            "Neural operator requires field_input (or legacy input_field/input_fields)"
        )
    return ["--field-input", field_csv], "field", ""


def _parse_field_output(stdout: str, output_path: str | None) -> dict:
    """Parse neural operator output fields."""
    data: dict = {}

    # If an output file was written, record its path
    if output_path and os.path.isfile(output_path):
        data["output_field_path"] = output_path

    # Try JSON from stdout
    try:
        parsed = json.loads(stdout)
        if isinstance(parsed, dict):
            data.update(parsed)
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue

        match = re.match(r"^Output \[(\d+)\]:\s*(.*)$", line)
        if match:
            values = match.group(2).strip()
            if values:
                data["output_dim"] = int(match.group(1))
                data["output_field"] = [
                    float(x) for x in values.split() if x.strip()
                ]
            continue

        match = re.match(r"^Output shape:\s*\[(\d+),\s*(\d+),\s*(\d+)\]$", line)
        if match:
            data["output_shape"] = [
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
            ]
            continue

        match = re.match(r"^First \d+ values:\s*(.*)$", line)
        if match:
            preview_str = match.group(1).replace("...", "").strip()
            if preview_str:
                data["output_field_preview"] = [
                    float(x) for x in preview_str.split() if x.strip()
                ]

    data["raw_output"] = stdout
    return data


def _distributed_runtime_config(case: E2ECase | None) -> dict:
    if case is None:
        return {}
    config = case.metadata.get("distributed_runtime", {})
    return config if isinstance(config, dict) and config.get("enabled") else {}


def _extract_rank_zero_stdout(stdout: str) -> str:
    text = stdout or ""
    tags = list(_MPI_STREAM_TAG_RE.finditer(text))
    if not tags:
        return text.strip()

    rank0_parts: list[str] = []
    for index, match in enumerate(tags):
        start = match.end()
        end = tags[index + 1].start() if index + 1 < len(tags) else len(text)
        if match.group("stream") == "stdout" and int(match.group("rank")) == 0:
            rank0_parts.append(text[start:end])
    return "".join(rank0_parts).strip()


def _strip_mpi_stream_tags(text: str) -> str:
    return _MPI_STREAM_TAG_RE.sub("", text or "")


def _safe_artifact_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name or "case")


def _ensure_distributed_runtime_env(
    case: E2ECase,
    ctx: RunContext,
    env: dict[str, str],
) -> None:
    if not _distributed_runtime_config(case):
        return
    if env.get("TRTMC_NCCL_RENDEZVOUS"):
        return

    safe_name = _safe_artifact_name(case.name)
    root = (
        Path(_case_artifact_dir(ctx.artifacts_dir, case.name))
        if ctx.artifacts_dir else Path(tempfile.gettempdir())
    )
    path = root / f"{safe_name}.nccl_rendezvous.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    env["TRTMC_NCCL_RENDEZVOUS"] = str(path)


def _wrap_distributed_command(
    cmd: list[str],
    case: E2ECase | None,
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


plugin = NeuralOperatorRunner()
