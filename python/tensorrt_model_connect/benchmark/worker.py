# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocol adapter for the native TRTMC benchmark worker."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

from .types import BenchmarkError, ResolvedCase


_BACKEND_ABI_PATTERN = re.compile(r"^libtrtmc_backend_trt_(\d+)_(\d+)\.so$")


def find_worker(explicit: Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    configured = os.environ.get("TRTMC_BENCH_WORKER")
    if configured:
        candidates.append(Path(configured).expanduser())
    repository = Path(__file__).resolve().parents[3]
    candidates.append(repository / "build/trtmc_benchmark_worker")
    candidates.append(repository / "build-make/trtmc_benchmark_worker")
    candidates.append(repository / "build-local/trtmc_benchmark_worker")
    candidates.append(Path(__file__).resolve().parents[1] / "bin/trtmc_benchmark_worker")
    discovered = shutil.which("trtmc_benchmark_worker")
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise BenchmarkError(f"cannot find executable trtmc_benchmark_worker; searched: {searched}")


def worker_backend_abi(worker: Path) -> str | None:
    """Return the unique standard TensorRT backend ABI beside a worker."""
    values = {
        f"{match.group(1)}.{match.group(2)}"
        for path in worker.expanduser().resolve().parent.glob("libtrtmc_backend_trt_*_*.so")
        if (match := _BACKEND_ABI_PATTERN.match(path.name)) is not None
    }
    if len(values) > 1:
        raise BenchmarkError(
            f"multiple TensorRT backend ABIs are installed beside {worker}: "
            f"{', '.join(sorted(values))}"
        )
    return next(iter(values), None)


def worker_metadata(worker: Path) -> dict[str, Any]:
    """Read immutable build provenance from a native benchmark worker."""
    try:
        completed = subprocess.run(
            [str(worker.expanduser().resolve()), "--metadata"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise BenchmarkError(f"cannot query benchmark worker metadata: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit code {completed.returncode}"
        raise BenchmarkError(f"cannot query benchmark worker metadata: {detail}")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"benchmark worker returned invalid metadata: {exc}") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != "trtmc.benchmark-worker-metadata/v1"
        or not isinstance(value.get("build"), dict)
    ):
        raise BenchmarkError("benchmark worker returned unsupported metadata")
    return value


def run_worker(case: ResolvedCase, case_dir: Path, worker: Path) -> dict[str, Any]:
    request_path = case_dir / "worker-request.json"
    result_path = case_dir / "worker-result.json"
    log_path = case_dir / "worker.log"
    request_path.write_text(
        json.dumps(case.worker_request(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            [str(worker), "--request", str(request_path), "--output", str(result_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if not result_path.is_file():
        raise BenchmarkError(
            f"worker exited {completed.returncode} without {result_path}; see {log_path}"
        )
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"invalid worker result {result_path}: {exc}") from exc
    if not isinstance(result, dict):
        raise BenchmarkError(f"worker result must be an object: {result_path}")
    if completed.returncode != 0 or result.get("status") != "completed":
        error = result.get("error", f"exit code {completed.returncode}")
        raise BenchmarkError(f"worker failed: {error}; see {log_path}")
    if result.get("case_digest") != case.digest:
        raise BenchmarkError("worker result case_digest does not match the resolved case")
    if result.get("operation") not in {None, case.operation}:
        raise BenchmarkError("worker result operation does not match the resolved case")
    if result.get("timing_scope") != case.measurement.timing_scope:
        raise BenchmarkError(
            "worker result timing_scope does not match measurement.timing_scope"
        )
    if (
        result.get("asset_loading_included")
        is not case.measurement.asset_loading_included
    ):
        raise BenchmarkError(
            "worker result asset_loading_included does not match measurement policy"
        )
    observations = result.get("observations")
    if not isinstance(observations, list) or len(observations) != case.measurement.iterations:
        raise BenchmarkError("worker observation count does not match measurement.iterations")
    return result


def discard_success_protocol_evidence(case_dir: Path) -> None:
    """Remove worker protocol intermediates after user evidence is durable."""
    for path in (case_dir / "worker-request.json", case_dir / "worker-result.json"):
        try:
            path.unlink()
        except OSError:
            pass
