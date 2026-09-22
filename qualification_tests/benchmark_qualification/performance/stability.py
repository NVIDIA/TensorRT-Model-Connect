# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-independent timing stability policy."""

from __future__ import annotations

import math
import statistics
from typing import Any, Mapping, Sequence

from .types import PerfMatrixError


SAMPLE_COUNT = 10
MAX_HALF_MEDIAN_CHANGE_PERCENT = 5.0
MEDIAN_BAND_PERCENT = 5.0
MINIMUM_SAMPLES_IN_BAND = 8


def p50(value: Mapping[str, Any]) -> float:
    metrics = value.get("metrics", {})
    latency = metrics.get("latency_ms", {}) if isinstance(metrics, Mapping) else {}
    value_p50 = latency.get("p50") if isinstance(latency, Mapping) else None
    if (
        isinstance(value_p50, bool)
        or not isinstance(value_p50, (int, float))
        or not math.isfinite(float(value_p50))
    ):
        raise PerfMatrixError("measurement has no finite latency p50")
    return float(value_p50)


def timing_stability(values: Sequence[Any]) -> dict[str, Any]:
    if len(values) != SAMPLE_COUNT:
        return {
            "status": "not_evaluated",
            "sample_count": len(values),
            "reason": "requires_10_samples",
        }
    try:
        samples = [float(value) for value in values]
    except (TypeError, ValueError):
        return {
            "status": "not_evaluated",
            "sample_count": len(values),
            "reason": "invalid_samples",
        }
    if not all(math.isfinite(value) and value > 0.0 for value in samples):
        return {
            "status": "not_evaluated",
            "sample_count": len(samples),
            "reason": "invalid_samples",
        }
    middle = len(samples) // 2
    median_ms = statistics.median(samples)
    first_half_median_ms = statistics.median(samples[:middle])
    second_half_median_ms = statistics.median(samples[middle:])
    half_change_percent = (
        abs(second_half_median_ms - first_half_median_ms) / first_half_median_ms * 100.0
    )
    samples_within_band = sum(
        abs(sample - median_ms) / median_ms * 100.0 <= MEDIAN_BAND_PERCENT for sample in samples
    )
    stable = (
        half_change_percent <= MAX_HALF_MEDIAN_CHANGE_PERCENT
        and samples_within_band >= MINIMUM_SAMPLES_IN_BAND
    )
    return {
        "status": "stable" if stable else "unstable",
        "sample_count": len(samples),
        "median_ms": median_ms,
        "first_half_median_ms": first_half_median_ms,
        "second_half_median_ms": second_half_median_ms,
        "half_median_change_percent": half_change_percent,
        "samples_within_band": samples_within_band,
    }


def measurement_stability(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    reference_result = timing_stability(reference.get("samples_ms", []))
    candidate_result = timing_stability(candidate.get("samples_ms", []))
    statuses = {reference_result["status"], candidate_result["status"]}
    if statuses == {"stable"}:
        status = "stable"
    elif "unstable" in statuses:
        status = "unstable"
    else:
        status = "not_evaluated"
    return {
        "status": status,
        "policy": {
            "required_samples": SAMPLE_COUNT,
            "max_half_median_change_percent": MAX_HALF_MEDIAN_CHANGE_PERCENT,
            "median_band_percent": MEDIAN_BAND_PERCENT,
            "minimum_samples_within_band": MINIMUM_SAMPLES_IN_BAND,
        },
        "reference": reference_result,
        "candidate": candidate_result,
    }
