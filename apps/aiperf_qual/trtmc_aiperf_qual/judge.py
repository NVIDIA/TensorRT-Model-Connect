# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The model verdict (Acc and Perf) and the Perf lights from AIPerf exports."""

from __future__ import annotations

import json
import statistics
from typing import Any, Mapping, Sequence

METRIC = "trtmc_model_call_time"


def verdict(result: Mapping[str, Any], *, expected_suites: Sequence[str], expected_modes: int) -> dict[str, Any]:
    """Model-level outcome: Acc pass/fail/error (``n/a`` for a Perf-only model), one light per reference
    mode, and a category.

    ``expected_suites`` names every judged result the configuration requires (benchmarks, family cases,
    and the entries each supplementary check emits); a missing one is an error, whatever else ran.
    ``informational`` entries are reported, never judged.
    """
    accuracy = [item for item in result.get("accuracy", []) if not item.get("informational")]
    produced = {item.get("suite") for item in accuracy}
    if any(name not in produced for name in expected_suites):
        acc = "error"
    elif result.get("accuracy_source") == "none" and not accuracy:
        acc = "n/a"
    else:
        statuses = {item["status"] for item in accuracy}
        acc = ("pass" if statuses == {"pass"} else "error" if "error" in statuses else "fail" if "fail" in statuses else
               "not-comparable" if "not-comparable" in statuses else "inconclusive")
    session_state = any(item.get("isolated_check", {}).get("status") == "pass" for item in accuracy)
    lights = {item["reference_mode"]: item["light"] for item in result.get("performance_l1", [])}
    # "n/a": the native model could not run in that mode (for example torch.compile failing).
    measured = [value for value in lights.values() if value != "n/a"]
    perf = "error" if len(lights) < expected_modes or not measured or "error" in measured else (
        "green" if all(value == "green" for value in measured) else
        "red" if "red" in measured else "white" if "white" in measured else "yellow")
    if acc == "error" or perf == "error":
        category = "error"
    elif acc == "fail":
        category = "acc-session-state" if session_state else "acc-issue"
    elif acc == "not-comparable":
        category = "not-comparable"
    elif acc == "inconclusive":
        category = "acc-inconclusive"
    elif perf in ("red", "yellow"):
        category = "perf-issue"
    elif perf == "white":
        category = "perf-inconclusive"
    else:
        category = "pass"
    return {"acc": acc, "perf": perf, "lights": lights, "category": category}


# Two-sided 95% Student-t critical values by degrees of freedom (AIPerf's confidence method).
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


AGGREGATIONS = ("mean", "best")
# Run-to-run spread below this is timer and scheduling jitter, not instability (sub-ms models).
MIN_CI_MS = 0.05
# GPU utilization before a timed run (our servers idle) at or above this means another process shares it.
GPU_BUSY_PERCENT = 20


def across_runs(p50_values: Sequence[float | None], aggregation: str = "mean") -> dict[str, Any]:
    """Combine per-run p50 model-call times: their mean (default) or the best (fastest) run.

    The 95% CI half-width is always reported as a percent of the mean; it gates stability only
    for mean-aggregated sides.
    """
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {AGGREGATIONS}")
    values = [float(value) for value in p50_values if value is not None]
    if not values:
        return {"p50_ms": None, "ci_percent": None, "runs": 0, "per_run_p50_ms": [], "aggregation": aggregation}
    mean = statistics.fmean(values)
    ci = None
    if len(values) > 1 and mean:
        half = _T95.get(len(values) - 1, 1.96) * statistics.stdev(values) / len(values) ** 0.5
        ci = half / mean * 100
    center = min(values) if aggregation == "best" else mean
    return {"p50_ms": center, "ci_percent": ci, "runs": len(values), "per_run_p50_ms": values,
            "aggregation": aggregation}


def light(candidate_ms: float, reference_ms: float, margin_percent: float) -> str:
    """Same rule as the perf matrix: green faster by more than the margin, red slower by more."""
    ratio = candidate_ms / reference_ms
    if ratio < 1 - margin_percent / 100:
        return "green"
    if ratio > 1 + margin_percent / 100:
        return "red"
    return "yellow"


def judge_performance(candidate: Mapping[str, Any], reference: Mapping[str, Any], *, margin_percent: float,
                      max_ci_percent: float, outputs_match: bool, output_reason: str,
                      not_equivalent: str | None = None) -> dict[str, Any]:
    """Light of one reference mode. A side whose runs did not complete every request, or that exported
    no model-call time, is an ``error`` light: a successful subset is not a measurement."""
    result = {"candidate": dict(candidate), "reference": dict(reference), "margin_percent": margin_percent,
              "output_check": {"match": outputs_match, "reason": output_reason}}
    errors = [f"{side}: {stats['incomplete']}" for side, stats in (("candidate", candidate), ("reference", reference))
              if stats.get("incomplete")]
    errors += [f"{side} has no {METRIC}" for side, stats in (("candidate", candidate), ("reference", reference))
               if stats.get("p50_ms") is None]
    if errors:
        return {**result, "notes": [], "light": "error", "reasons": errors}
    reasons, notes = [], []
    if not_equivalent:  # the native reference times a different workload (configured per model)
        reasons.append(f"not the same workload: {not_equivalent}")
    if not outputs_match:
        reasons.append(f"output check failed: {output_reason}")
    wide = []
    for side, stats in (("candidate", candidate), ("reference", reference)):
        if (stats.get("aggregation", "mean") == "mean" and stats.get("ci_percent") is not None
              and stats["ci_percent"] > max_ci_percent
              and stats["ci_percent"] / 100 * stats["p50_ms"] > MIN_CI_MS):
            wide.append(f"{side} CI ±{stats['ci_percent']:.2f}% > {max_ci_percent}%")
    if wide and candidate.get("p50_ms") and reference.get("p50_ms"):
        # A wide CI only matters if it could change the light: evaluate both extremes.
        def bounds(stats: Mapping[str, Any]) -> tuple[float, float]:
            half = (stats.get("ci_percent") or 0.0) / 100 * stats["p50_ms"] if stats.get("aggregation", "mean") == "mean" else 0.0
            return stats["p50_ms"] - half, stats["p50_ms"] + half
        (c_low, c_high), (r_low, r_high) = bounds(candidate), bounds(reference)
        extremes = {light(c_low, r_high, margin_percent), light(c_high, r_low, margin_percent)}
        if len(extremes) == 1:
            notes += [f"{item} (light unchanged across the interval)" for item in wide]
        else:
            reasons += [f"{item}: {' or '.join(sorted(extremes))} within the interval" for item in wide]
    else:
        reasons += wide
    for side, stats in (("candidate", candidate), ("reference", reference)):
        if (stats.get("gpu_busy_percent") or 0) >= GPU_BUSY_PERCENT:
            reasons.append(f"GPU {stats['gpu_busy_percent']:.0f}% busy with other processes before the {side} timing")
    if reference.get("precision_fallback"):
        # The native model could not run at the candidate precision: a slower precision is no baseline.
        reasons.append(f"reference timed at {reference.get('precision')} (candidate precision failed: "
                       f"{str(reference['precision_fallback'])[:120]})")
    notes += [f"{side}: {stats['exit_note']}" for side, stats in (("candidate", candidate), ("reference", reference))
              if stats.get("exit_note")]
    result["notes"] = notes
    if candidate.get("p50_ms") and reference.get("p50_ms"):
        result["speedup"] = reference["p50_ms"] / candidate["p50_ms"]  # informative even when white
    if reasons:
        return {**result, "light": "white", "reasons": reasons}
    return {**result, "light": light(candidate["p50_ms"], reference["p50_ms"], margin_percent), "reasons": []}


def first_observation(raw_records: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    for record in raw_records:
        if record.get("status") == 200 and record.get("responses"):
            body = json.loads(record["responses"][-1]["text"])
            return body.get("trtmc_observation")
    return None


def median_client_latency(raw_records: Sequence[Mapping[str, Any]]) -> float | None:
    values = [(r["metadata"]["request_end_ns"] - r["metadata"]["request_start_ns"]) / 1e6
              for r in raw_records if r.get("status") == 200]
    return statistics.median(values) if values else None
