# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal Performance qualification through installed trtmc-bench."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml

from .catalog import QualificationCase, QualificationError, load_benchmark
from .runtime import (
    RuntimeContext,
    benchmark_executable,
    reference_python,
    require_candidate,
    run_command,
    write_model_descriptor,
    write_result,
)


_COMPLETED_COMPARISONS = frozenset({"green", "yellow", "red"})


def _qualification_status(
    process_returncode: int,
    matrix: Mapping[str, Any],
    row: Mapping[str, Any],
) -> tuple[str, str | None]:
    row_error = row.get("error")
    if isinstance(row_error, str) and row_error:
        return "error", row_error
    row_status = row.get("status")
    matrix_status = matrix.get("status")
    if row_status == "contract-mismatch":
        return "failed", None
    if (
        row_status in _COMPLETED_COMPARISONS
        and matrix_status == "completed"
        and process_returncode == 0
    ):
        return "passed", None
    comparison = row.get("comparison")
    reason = comparison.get("reason") if isinstance(comparison, Mapping) else None
    if not isinstance(reason, str) or not reason:
        reason = (
            f"Performance matrix ended with row status {row_status!r}, "
            f"matrix status {matrix_status!r}, and exit code {process_returncode}"
        )
    return "error", reason


def run_performance(case: QualificationCase, context: RuntimeContext) -> dict[str, Any]:
    output = context.case_artifacts(case)
    output.mkdir(parents=True, exist_ok=True)
    definition = load_benchmark(context.repository, case)
    worker, runtime_root = require_candidate(context)
    configured = case.values
    request = configured.get("request")
    baseline = configured.get("reference")
    measurement = configured.get("measurement")
    reference_timing = definition.get("reference_timing")
    if not all(
        isinstance(value, Mapping)
        for value in (request, baseline, measurement, reference_timing)
    ):
        raise QualificationError(
            "Performance request, reference, measurement, and reference timing must be objects"
        )
    assert isinstance(request, Mapping)
    assert isinstance(baseline, Mapping)
    assert isinstance(measurement, Mapping)
    assert isinstance(reference_timing, Mapping)
    request = _resolve_family_assets(case, request)
    descriptor = write_model_descriptor(case, output, request)
    entry_id = f"qualification.{case.family}.{case.name}"
    suite = {
        "schema_version": "trtmc.perf-suite/v2",
        "name": case.id,
        "entries": [
            {
                "id": entry_id,
                "family": case.family,
                "operation": str(configured["operation"]),
                "model": case.model,
                "manifest": str(descriptor),
                "workload": {
                    "testcase": case.name,
                    "request": dict(request),
                },
                "measurement": {
                    "warmup": int(measurement.get("warmup", 5)),
                    "iterations": int(measurement.get("iterations", 10)),
                },
                "baseline": {**dict(baseline), **dict(reference_timing)},
                "equivalence_margin_percent": float(
                    configured.get("equivalence_margin_percent", 5.0)
                ),
            }
        ],
    }
    if definition.get("stability") != {
        "samples": 10,
        "max_half_median_change_percent": 5.0,
        "median_band_percent": 5.0,
        "minimum_samples_within_band": 8,
        "retries": 1,
    }:
        raise QualificationError("unsupported Performance stability definition")
    suite_path = output / "resolved-suite.yaml"
    suite_path.write_text(yaml.safe_dump(suite, sort_keys=False), encoding="utf-8")
    results_root = output / "matrix"
    references = {
        "elf_repo": "",
        "lance_repo": "",
        "lerobot_repo": "",
        "sana_repo": "",
        "sana_model": "",
        "personaplex_repo": "",
        "fast_foundation_stereo_model": "",
    }
    environment = {
        "schema_version": "trtmc.perf-environment/v2",
        "name": "qualification",
        "tools": {
            "trtmc_bench": str(benchmark_executable(case, context)),
            "trtmc_worker": str(worker),
            "hf_transformers_runner": str(
                context.repository / "apps/benchmark/performance/baselines/hf_transformers.py"
            ),
            "task_reference_runner": str(
                context.repository / "apps/benchmark/performance/baselines/task_reference.py"
            ),
            "reference_python": str(reference_python(case, context)),
        },
        "references": references,
        "storage": {
            "results_root": str(results_root),
            "scratch_root": str(output / "scratch"),
            "bundle_cache": str(context.bundle_cache),
            "bundle_roots": [str(path) for path in context.bundle_roots],
            "runtime_root": str(runtime_root),
            "bundle_retention": "retain",
        },
        "execution": {
            "local_files_only": os.environ.get("TRTMC_QUALIFICATION_LOCAL_FILES_ONLY") == "1",
            "hf_cache_mode": "shared",
            "hf_cache_retention": "retain",
            "timeout_seconds": int(configured.get("timeout_seconds", 7200)),
        },
    }
    environment_path = output / "resolved-environment.yaml"
    environment_path.write_text(yaml.safe_dump(environment, sort_keys=False), encoding="utf-8")
    command = [
        sys.executable,
        str(context.repository / "tools/perf_matrix.py"),
        "run",
        str(suite_path),
        "--environment",
        str(environment_path),
        "--entry",
        entry_id,
        "--allow-partial",
    ]
    if context.no_build:
        command.append("--no-build")
    if context.verbose:
        command.append("--verbose")
    completed = run_command(
        command,
        output,
        "performance",
        timeout=int(configured.get("timeout_seconds", 7200)) * 2,
        verbose=context.verbose,
    )
    run_directories = sorted(path.parent for path in results_root.glob("*/results.json"))
    if not run_directories:
        raise QualificationError(f"Performance produced no result; see {output}")
    run_directory = run_directories[-1]
    try:
        matrix = json.loads((run_directory / "results.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QualificationError(
            f"Performance produced an unreadable result; see {run_directory}"
        ) from error
    if not isinstance(matrix, Mapping):
        raise QualificationError(f"Performance produced an invalid result; see {run_directory}")
    rows = matrix.get("rows")
    row = (
        rows[0]
        if isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], Mapping)
        else {}
    )
    status, error = _qualification_status(completed.returncode, matrix, row)
    comparison_value = row.get("comparison")
    comparison = comparison_value if isinstance(comparison_value, Mapping) else {}
    result = {
        "schema_version": "trtmc.qualification-result/v1",
        "case": case.id,
        "kind": "performance",
        "status": status,
        "model": case.model,
        "benchmark": case.benchmark,
        "matrix_run": str(run_directory),
        "comparison_status": row.get("status") if isinstance(row, Mapping) else None,
        "metrics": {
            "candidate_p50_ms": comparison.get("candidate_p50_ms"),
            "reference_p50_ms": comparison.get("reference_p50_ms"),
            "reference_over_candidate_p50": comparison.get("reference_over_candidate_p50"),
        },
        "reference_attempts": row.get("reference_attempts", []) if isinstance(row, Mapping) else [],
    }
    if error is not None:
        result["error"] = error
    write_result(output, result)
    return result


def _resolve_family_assets(
    case: QualificationCase, request: Mapping[str, Any]
) -> dict[str, Any]:
    resolved = dict(request)
    family_root = case.source.parents[2].resolve()
    for key, value in request.items():
        if not key.endswith("_path") or not isinstance(value, str):
            continue
        path = Path(value)
        if path.is_absolute():
            continue
        path = (case.source.parent / path).resolve()
        if family_root not in path.parents or not path.is_file():
            raise QualificationError(
                f"Performance request asset {key!r} is unavailable inside {family_root}: {path}"
            )
        resolved[key] = str(path)
    return resolved
