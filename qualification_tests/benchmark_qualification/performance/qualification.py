# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal Performance qualification through installed trtmc-bench."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from qualification_tests.benchmark_qualification.performance import matrix as performance_matrix
from qualification_tests.benchmark_qualification.performance.runner import run_case

from ..catalog import QualificationCase, QualificationError, load_benchmark
from ..runtime import (
    RuntimeContext,
    reference_environment_options,
    reference_python,
    require_candidate,
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
        isinstance(value, Mapping) for value in (request, baseline, measurement, reference_timing)
    ):
        raise QualificationError(
            "Performance request, reference, measurement, and reference timing must be objects"
        )
    assert isinstance(request, Mapping)
    assert isinstance(baseline, Mapping)
    assert isinstance(measurement, Mapping)
    assert isinstance(reference_timing, Mapping)
    stability = definition.get("stability", {})
    required_samples = stability.get("samples") if isinstance(stability, Mapping) else None
    if (
        isinstance(required_samples, bool)
        or not isinstance(required_samples, int)
        or int(measurement.get("iterations", 10)) < required_samples
    ):
        raise QualificationError(
            "Performance measurement.iterations must satisfy the stability sample count"
        )
    request = _resolve_family_assets(case, request)
    baseline = _resolve_reference_assets(case, context, baseline)
    descriptor = write_model_descriptor(case, output, request, context=context)
    entry_id = f"qualification.{case.family}.{case.name}"
    spec = {
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
    if stability != {
        "samples": 10,
        "max_half_median_change_percent": 5.0,
        "median_band_percent": 5.0,
        "minimum_samples_within_band": 8,
        "retries": 1,
    }:
        raise QualificationError("unsupported Performance stability definition")
    results_root = output / "matrix"
    environment = performance_matrix.Environment(
        name="qualification",
        trtmc_bench=context.trtmc_bench,
        worker=worker,
        hf_runner=context.repository / "qualification_tests/benchmark_qualification/performance/references/hf_transformers.py",
        task_runner=context.repository / "qualification_tests/benchmark_qualification/performance/references/generic_reference.py",
        reference_python=reference_python(case, context),
        results_root=results_root,
        scratch_root=output / "scratch",
        bundle_cache=context.bundle_cache,
        bundle_roots=context.bundle_roots,
        runtime_root=runtime_root,
        bundle_retention="retain",
        local_files_only=os.environ.get("TRTMC_QUALIFICATION_LOCAL_FILES_ONLY") == "1",
        timeout_seconds=int(configured.get("timeout_seconds", 7200)),
        references={},
        hf_cache_mode="shared",
        hf_cache_retention="retain",
    )
    try:
        row = run_case(
            spec,
            environment,
            results_root,
            no_build=context.no_build,
            verbose=context.verbose,
        )
    except (OSError, performance_matrix.PerfMatrixError) as error:
        raise QualificationError(f"Performance execution failed: {error}") from error
    matrix_status = "completed" if row.get("status") in _COMPLETED_COMPARISONS else "failed"
    status, error = _qualification_status(
        0 if matrix_status == "completed" else 1,
        {"status": matrix_status},
        row,
    )
    comparison_value = row.get("comparison")
    comparison = comparison_value if isinstance(comparison_value, Mapping) else {}
    result = {
        "schema_version": "trtmc.qualification-result/v1",
        "case": case.id,
        "kind": "performance",
        "status": status,
        "model": case.model,
        "benchmark": case.benchmark,
        "matrix_run": str(results_root),
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


def _resolve_family_assets(case: QualificationCase, request: Mapping[str, Any]) -> dict[str, Any]:
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
        if key == "prompt_path":
            payload = (
                json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else None
            )
            prompt = (
                payload.get("prompt")
                if isinstance(payload, Mapping)
                else path.read_text(encoding="utf-8").strip()
            )
            if not isinstance(prompt, str) or not prompt:
                raise QualificationError(f"Performance prompt is unavailable: {path}")
            resolved.pop(key, None)
            resolved["prompt"] = prompt
        else:
            resolved[key] = str(path)
    return resolved


def _resolve_reference_assets(
    case: QualificationCase,
    context: RuntimeContext,
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = dict(reference)
    options = reference.get("adapter_options", {})
    if not isinstance(options, Mapping):
        raise QualificationError("Performance reference.adapter_options must be an object")
    resolved_options = dict(options)
    family_root = case.source.parents[2].resolve()
    for key, value in options.items():
        if not key.endswith("_path") or not isinstance(value, str):
            continue
        path = Path(value)
        if not path.is_absolute():
            path = (case.source.parent / path).resolve()
        if not path.is_relative_to(family_root) or not path.is_file():
            raise QualificationError(f"Performance reference asset {key!r} is unavailable: {path}")
        resolved_options[key] = str(path)
    resolved["adapter_options"] = reference_environment_options(case, context, resolved_options)
    return resolved
