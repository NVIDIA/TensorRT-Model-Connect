# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute resolved cases and preserve reproducible evidence."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
from typing import Any, Iterable, Mapping
import uuid

from .metrics import reduce_metrics
from .report import write_html_report
from .telemetry import GpuTelemetry
from .types import BenchmarkError, ResolvedCase
from .worker import (
    discard_success_protocol_evidence,
    find_worker,
    run_worker,
    worker_backend_abi,
    worker_metadata,
)


class BenchmarkService:
    """Sequential benchmark runner; one process/load lifecycle per resolved case."""

    def __init__(self, worker: Path | None = None) -> None:
        self.worker = find_worker(worker)

    def run(
        self,
        cases: Iterable[ResolvedCase],
        output_dir: Path,
        *,
        bundle_preparation: Iterable[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        resolved_cases = tuple(cases)
        if not resolved_cases:
            raise BenchmarkError("benchmark contains no cases")
        output_dir = output_dir.expanduser().resolve()
        if output_dir.exists():
            raise BenchmarkError(f"output directory already exists: {output_dir}")
        try:
            output_dir.mkdir(parents=True)
        except OSError as exc:
            raise BenchmarkError(f"cannot create output directory {output_dir}: {exc}") from exc
        started = _now()
        timing_scopes = {case.measurement.timing_scope for case in resolved_cases}
        asset_loading_values = {
            case.measurement.asset_loading_included for case in resolved_cases
        }
        result: dict[str, Any] = {
            "schema_version": "trtmc.benchmark-run/v1",
            "run_id": str(uuid.uuid4()),
            "status": "running",
            "started_at": started,
            "measurement_policy": {
                "timing_scope": (
                    next(iter(timing_scopes)) if len(timing_scopes) == 1 else "per_case"
                ),
                "input_preparation_included": (
                    next(iter(timing_scopes)) == "public_pipeline_call_wall"
                    if len(timing_scopes) == 1
                    else "per_case"
                ),
                "asset_loading_included": (
                    next(iter(asset_loading_values))
                    if len(asset_loading_values) == 1
                    else "per_case"
                ),
                "load_excluded": True,
                "warmup_excluded": True,
                "task_quality_evaluated": False,
                "telemetry_in_timed_path": False,
            },
            "preparation": {
                "included_in_performance_metrics": False,
                "bundles": [dict(record) for record in bundle_preparation],
            },
            "environment": _environment(self.worker),
            "cells": [],
        }
        _write_json(output_dir / "result.json", result)
        for index, case in enumerate(resolved_cases, start=1):
            cell = self._run_case(index, case, output_dir)
            result["cells"].append(cell)
            _write_json(output_dir / "result.json", result)
        result["finished_at"] = _now()
        result["status"] = (
            "completed"
            if all(cell["status"] == "completed" for cell in result["cells"])
            else "failed"
        )
        _write_json(output_dir / "result.json", result)
        write_html_report(result, output_dir / "report.html")
        return result

    def _run_case(self, index: int, case: ResolvedCase, output_dir: Path) -> dict[str, Any]:
        directory_name = f"{index:03d}-{_slug(case.model.name)}-{_slug(case.name)}"
        case_dir = output_dir / directory_name
        case_dir.mkdir()
        _write_json(case_dir / "resolved-case.json", case.to_json())
        telemetry = GpuTelemetry(case.measurement.telemetry, case.measurement.telemetry_interval_ms)
        try:
            with telemetry:
                worker_result = run_worker(case, case_dir, self.worker)
            _write_jsonl(case_dir / "observations.jsonl", worker_result["observations"])
            metrics = reduce_metrics(
                case.operation,
                worker_result["observations"],
                request=case.request,
            )
            cell: dict[str, Any] = {
                "status": "completed",
                "name": case.name,
                "model": case.model.name,
                "operation": case.operation,
                "case_digest": case.digest,
                "artifact_dir": directory_name,
                "metrics": metrics,
                "output_summary": worker_result.get("output_summary", {}),
                "pipeline_type": worker_result.get("pipeline_type"),
                "load_ms": worker_result.get("load_ms"),
                "timing_scope": worker_result.get("timing_scope"),
                "asset_loading_included": worker_result.get(
                    "asset_loading_included"
                ),
            }
            discard_success_protocol_evidence(case_dir)
        except (BenchmarkError, OSError, subprocess.SubprocessError) as exc:
            cell = {
                "status": "failed",
                "name": case.name,
                "model": case.model.name,
                "operation": case.operation,
                "case_digest": case.digest,
                "artifact_dir": directory_name,
                "error": str(exc),
            }
        if case.measurement.telemetry != "off":
            _write_json(case_dir / "telemetry.json", telemetry.result())
        return cell


def default_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    return Path("trtmc-bench-results") / timestamp


def _environment(worker: Path) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[3]
    explicit_revision = os.environ.get("TRTMC_BENCH_SOURCE_REVISION")
    commit = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    environment = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "worker": str(worker),
        "runtime_backend_abi": worker_backend_abi(worker),
        "worker_build": worker_metadata(worker)["build"],
        "git_commit": commit,
        "source_revision": explicit_revision or commit,
    }
    environment.update(_gpu_environment())
    return environment


def _gpu_environment() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"gpus": [], "gpu_query_error": "nvidia-smi was not found"}
    command = [
        executable,
        "--query-gpu=index,name,uuid,driver_version",
        "--format=csv,noheader,nounits",
    ]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",", maxsplit=1)[0].strip()
    if visible:
        command.extend(["--id", visible])
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        gpus = []
        for row in csv.reader(io.StringIO(completed.stdout)):
            if len(row) != 4:
                continue
            index, name, gpu_uuid, driver = (value.strip() for value in row)
            gpus.append(
                {
                    "index": int(index),
                    "name": name,
                    "uuid": gpu_uuid,
                    "driver_version": driver,
                }
            )
        return {"gpus": gpus, "gpu_query_error": None}
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {"gpus": [], "gpu_query_error": str(exc)}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: Iterable[Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, sort_keys=True) + "\n")


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-") or "case"
