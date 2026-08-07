# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize supported report schemas into stable cross-run observations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .domain import Observation, ReportHubError, stable_id


def normalize_report(source: str, payload: Mapping[str, Any]) -> list[Observation]:
    schema = str(payload.get("schema_version", ""))
    if schema == "trtmc.validation-report/v2" or isinstance(payload.get("results"), list):
        return _normalize_validation(source, payload)
    if schema == "trtmc.perf-matrix/v1" or isinstance(payload.get("cases"), list):
        return _normalize_performance(source, payload)
    if schema == "trtmc.benchmark-report/v1" or isinstance(payload.get("runs"), list):
        return _normalize_benchmark(source, payload)
    raise ReportHubError(f"unsupported report schema: {schema or 'missing'}")


def _normalize_validation(source: str, payload: Mapping[str, Any]) -> list[Observation]:
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise ReportHubError("validation report results must be a list")
    observations: list[Observation] = []
    for index, raw in enumerate(results):
        if not isinstance(raw, Mapping):
            continue
        model = _text(raw.get("model") or raw.get("name"), f"case-{index + 1}")
        workload = _text(
            raw.get("workload")
            or _nested(raw, "workload_contract", "testcase")
            or _nested(raw, "resolved_settings", "testcase")
            or raw.get("benchmark")
            or raw.get("task"),
            "unspecified",
        )
        comparison = raw.get("comparison") if isinstance(raw.get("comparison"), Mapping) else {}
        metric = (
            comparison.get("primary_metric")
            if isinstance(comparison.get("primary_metric"), Mapping)
            else {}
        )
        metric_value = metric.get("value", raw.get("value"))
        metric_name = _text(metric.get("name") or raw.get("metric"), "validation")
        observations.append(
            Observation(
                finding_id=stable_id("finding", source, model, workload, metric_name),
                model=model,
                workload=workload,
                family=_text(raw.get("family"), model.split("-")[0]),
                operation=_text(raw.get("operation") or raw.get("task_strategy"), "run"),
                status=_validation_status(raw),
                metric_name=metric_name,
                metric_value=_number(metric_value),
                details={
                    "task_type": _text(raw.get("task_type") or raw.get("taskType"), ""),
                    "failures": list(comparison.get("failures", []))
                    if isinstance(comparison.get("failures"), list)
                    else [],
                    "error": _attempt_error(raw),
                    "timestamp": raw.get("timestamp"),
                },
            )
        )
    return observations


def _normalize_performance(source: str, payload: Mapping[str, Any]) -> list[Observation]:
    cases = payload.get("cases", [])
    if not isinstance(cases, list):
        raise ReportHubError("performance report cases must be a list")
    observations: list[Observation] = []
    for index, raw in enumerate(cases):
        if not isinstance(raw, Mapping):
            continue
        model = _text(raw.get("model"), f"case-{index + 1}")
        workload = _text(
            _nested(raw, "workload_contract", "testcase")
            or _nested(raw, "resolved_settings", "testcase")
            or raw.get("id"),
            "unspecified",
        )
        comparison = raw.get("comparison") if isinstance(raw.get("comparison"), Mapping) else {}
        candidate_metrics = _nested(raw, "candidate", "metrics")
        candidate_metrics = candidate_metrics if isinstance(candidate_metrics, Mapping) else {}
        candidate_primary = (
            candidate_metrics.get("primary")
            if isinstance(candidate_metrics.get("primary"), Mapping)
            else {}
        )
        baseline_latency = _nested(raw, "baseline", "metrics", "latency_ms", "p50")
        candidate_latency = _nested(raw, "candidate", "metrics", "latency_ms", "p50")
        ratio = comparison.get("baseline_over_trtmc_p50")
        metric_name = "baseline_over_trtmc_p50" if _number(ratio) is not None else _text(
            candidate_primary.get("name"), "runtime_e2e_wall_ms.p50"
        )
        metric_value = ratio if _number(ratio) is not None else candidate_primary.get(
            "value", candidate_latency
        )
        observations.append(
            Observation(
                finding_id=stable_id("finding", source, model, workload, metric_name),
                model=model,
                workload=workload,
                family=_text(raw.get("family"), model.split("-")[0]),
                operation=_text(raw.get("operation") or raw.get("task_strategy"), "run"),
                status=_performance_status(raw.get("status")),
                metric_name=metric_name,
                metric_value=_number(metric_value),
                details={
                    "task_type": _text(raw.get("task_type"), ""),
                    "baseline_p50_ms": _number(baseline_latency),
                    "candidate_p50_ms": _number(candidate_latency),
                    "equivalence_margin_percent": _number(
                        comparison.get("equivalence_margin_percent")
                    ),
                    "error": _text(raw.get("error"), ""),
                },
            )
        )
    return observations


def _normalize_benchmark(source: str, payload: Mapping[str, Any]) -> list[Observation]:
    runs = payload.get("runs", [])
    if not isinstance(runs, list):
        raise ReportHubError("benchmark report runs must be a list")
    observations: list[Observation] = []
    for run in runs:
        if not isinstance(run, Mapping) or not isinstance(run.get("cells"), list):
            continue
        for index, cell in enumerate(run["cells"]):
            if not isinstance(cell, Mapping):
                continue
            model = _text(cell.get("model"), f"case-{index + 1}")
            workload = _text(cell.get("name") or cell.get("case"), "default")
            metrics = cell.get("metrics") if isinstance(cell.get("metrics"), Mapping) else {}
            primary = metrics.get("primary") if isinstance(metrics.get("primary"), Mapping) else {}
            latency = metrics.get("latency_ms") if isinstance(metrics.get("latency_ms"), Mapping) else {}
            metric_value = primary.get("value", latency.get("p50"))
            metric_name = _text(primary.get("name"), "runtime_e2e_wall_ms.p50")
            observations.append(
                Observation(
                    finding_id=stable_id("finding", source, model, workload, metric_name),
                    model=model,
                    workload=workload,
                    family=model.split("-")[0],
                    operation=_text(cell.get("operation"), "run"),
                    status=_benchmark_status(cell.get("status")),
                    metric_name=metric_name,
                    metric_value=_number(metric_value),
                    details={
                        "unit": _text(primary.get("unit"), "ms"),
                        "sample_count": metrics.get("sample_count"),
                        "source_run_id": run.get("run_id"),
                        "result_path": run.get("result_path"),
                    },
                )
            )
    return observations


def _validation_status(raw: Mapping[str, Any]) -> str:
    execution = str(_nested(raw, "execution", "status") or raw.get("execution_status") or "").lower()
    validation = str(
        _nested(raw, "validation", "status") or raw.get("validation_status") or raw.get("status") or ""
    ).lower()
    comparison = str(_nested(raw, "comparison", "status") or "").lower()
    if execution and execution not in {"completed", "passed", "success"}:
        return "error"
    if validation in {"failed", "failure", "red"} or comparison in {"disagreement", "failed", "red"}:
        return "failed"
    if validation in {"passed", "pass", "green", "success"} or comparison in {"agreement", "passed", "green"}:
        return "passed"
    if validation in {"skipped", "unsupported", "yellow", "not_compared"}:
        return "other"
    return "other"


def _benchmark_status(value: Any) -> str:
    status = str(value or "").lower()
    if status in {"completed", "passed", "success"}:
        return "passed"
    if status in {"failed", "failure"}:
        return "failed"
    if status in {"error", "aborted"}:
        return "error"
    return "other"


def _performance_status(value: Any) -> str:
    status = str(value or "").lower()
    if status in {"green", "passed", "completed", "success"}:
        return "passed"
    if status in {"red", "failed", "failure", "regression"}:
        return "failed"
    if status in {"error", "aborted"}:
        return "error"
    return "other"


def _nested(value: Mapping[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _text(value: Any, default: str) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else default


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _attempt_error(raw: Mapping[str, Any]) -> str:
    execution = raw.get("execution")
    attempts = execution.get("attempts") if isinstance(execution, Mapping) else None
    if isinstance(attempts, list) and attempts and isinstance(attempts[-1], Mapping):
        return _text(attempts[-1].get("error"), "")
    return _text(raw.get("error") or raw.get("message"), "")
