# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reduce raw worker observations into task-aware performance metrics."""

from __future__ import annotations

import math
from statistics import fmean
from typing import Any, Mapping, Sequence

from .operations import operation_for_name
from .types import BenchmarkError


def _numbers(observations: Sequence[Mapping[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for observation in observations:
        value = observation.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        converted = float(value)
        if math.isfinite(converted):
            values.append(converted)
    return values


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise BenchmarkError("cannot compute a percentile from no observations")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "mean": fmean(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def reduce_metrics(
    operation: str,
    observations: Sequence[Mapping[str, Any]],
    *,
    request: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return common latency plus operation-specific rate/stage metrics."""

    operation_spec = operation_for_name(operation)
    latency = _numbers(observations, "runtime_e2e_wall_ms")
    if not latency:
        raise BenchmarkError("worker returned no runtime_e2e_wall_ms observations")
    total_seconds = sum(latency) / 1000.0
    if total_seconds <= 0:
        raise BenchmarkError("worker returned a non-positive total measured duration")
    metrics: dict[str, Any] = {
        "sample_count": len(latency),
        "latency_ms": _summary(latency),
        "request_throughput_per_s": len(latency) / total_seconds,
        "primary": {
            "name": "runtime_e2e_wall_ms.p50",
            "value": _percentile(latency, 0.50),
            "unit": "ms",
        },
    }

    rate_metrics, per_item = operation_spec.metrics_for_request(request or {})
    for rate in rate_metrics:
        values = _numbers(observations, rate.observation_field)
        rate_value = sum(values) / total_seconds
        metrics[rate.result_name] = rate_value
        if rate.inverse_result_name is not None and rate_value > 0:
            metrics[rate.inverse_result_name] = 1.0 / rate_value

    stages = {}
    for field in operation_spec.stage_timings:
        values = _numbers(observations, field)
        if values:
            stages[field] = _summary(values)
    if stages:
        metrics["reported_stages_ms"] = stages

    if per_item is not None:
        counts = _numbers(observations, per_item.count_field)
        seconds_per_item = [
            duration_ms / count / 1000.0
            for duration_ms, count in zip(latency, counts, strict=False)
            if count > 0
        ]
        if seconds_per_item:
            metrics[per_item.result_name] = _percentile(seconds_per_item, 0.50)
    return metrics
