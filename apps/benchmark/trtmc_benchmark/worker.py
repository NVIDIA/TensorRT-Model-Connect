# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocol adapter for the native Task API benchmark worker."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .types import BenchmarkError, ResolvedCase


def find_worker(explicit: Path | None = None) -> Path:
    if explicit is not None:
        candidate = explicit.expanduser().resolve()
    else:
        import tensorrt_model_connect

        candidate = (
            Path(tensorrt_model_connect.__file__).resolve().parent
            / "bin/trtmc_benchmark_worker"
        )
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise BenchmarkError(
            f"benchmark worker does not exist or is not executable: {candidate}; use --worker"
        )
    return candidate


def run_worker(case: ResolvedCase, case_dir: Path, worker: Path) -> dict[str, Any]:
    request_path = case_dir / "worker-request.json"
    result_path = case_dir / "worker-result.json"
    log_path = case_dir / "worker.log"
    request_path.write_text(
        json.dumps(case.worker_request(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    timeout = int(os.environ.get("TRTMC_BENCH_WORKER_TIMEOUT_S", "7200"))
    with log_path.open("w", encoding="utf-8") as log:
        try:
            completed = subprocess.run(
                [str(worker), "--request", str(request_path), "--output", str(result_path)],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise BenchmarkError(f"worker timed out after {timeout}s; see {log_path}") from error
    if not result_path.is_file():
        raise BenchmarkError(
            f"worker exited {completed.returncode} without {result_path}; see {log_path}"
        )
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkError(f"invalid worker result {result_path}: {error}") from error
    if not isinstance(result, dict):
        raise BenchmarkError(f"worker result must be an object: {result_path}")
    if completed.returncode != 0 or result.get("status") != "completed":
        raise BenchmarkError(f"worker failed: {result.get('error', completed.returncode)}; see {log_path}")
    if result.get("schema_version") != "trtmc.benchmark-worker-result/v2":
        raise BenchmarkError("worker returned an unsupported result schema")
    if result.get("case_name") != case.name:
        raise BenchmarkError("worker result case_name does not match the request")
    if result.get("operation") != case.operation:
        raise BenchmarkError("worker result operation does not match the request")
    if result.get("timing_scope") != case.measurement.timing_scope:
        raise BenchmarkError("worker result timing_scope does not match the request")
    if result.get("asset_loading_included") is not case.measurement.asset_loading_included:
        raise BenchmarkError("worker result asset-loading policy does not match the request")
    observations = result.get("observations")
    if not isinstance(observations, list) or len(observations) != case.measurement.iterations:
        raise BenchmarkError("worker observation count does not match measurement.iterations")
    return result


def discard_success_protocol_evidence(case_dir: Path) -> None:
    for path in (case_dir / "worker-request.json", case_dir / "worker-result.json"):
        path.unlink(missing_ok=True)
