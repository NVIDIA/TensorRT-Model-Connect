# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Turn AIPerf exports into Acc gate results and Perf lights."""

from __future__ import annotations

import json
import math
import statistics
from typing import Any, Mapping, Sequence

METRIC = "trtmc_model_call_time"


def _corpus_wer(pairs: Sequence[tuple[str, str]]) -> float:
    from trtmc_aiperf_plugins.accuracy import _edit_distance, _words

    errors = sum(_edit_distance(_words(hypothesis), _words(label)) for hypothesis, label in pairs)
    words = sum(len(_words(label)) for _, label in pairs)
    return errors / words if words else 0.0


def label_metrics(records: Sequence[Mapping[str, Any]], labels_by_reference: Mapping[str, str]) -> dict[str, Any]:
    """Corpus WER of candidate and reference against dataset labels.

    Records are matched to labels through the reference (golden) text, which the grader echoes
    as ``expected``; unmatched records are counted and excluded.
    """
    candidate, reference, unmatched = [], [], 0
    for record in records:
        label = labels_by_reference.get(str(record.get("expected", "")).strip())
        if label is None:
            unmatched += 1
            continue
        candidate.append((str(record.get("actual", "")), label))
        reference.append((str(record.get("expected", "")), label))
    wer_candidate, wer_reference = _corpus_wer(candidate), _corpus_wer(reference)
    return {"candidate_wer_to_label": wer_candidate, "reference_wer_to_label": wer_reference,
            "wer_increase_from_reference": wer_candidate - wer_reference, "matched": len(candidate),
            "unmatched": unmatched}


def required_passes(gate: Mapping[str, Any], expected: int) -> int:
    """Samples that must pass: the gate as declared (the native noise floor never lowers it)."""
    required = math.ceil(float(gate.get("min_pass_rate", 1.0)) * expected - 1e-9)
    if "allowed_failures" in gate:
        required = max(required, expected - int(gate["allowed_failures"]))
    return required


def sample_index(conversation_id: Any) -> int | None:
    """Suite position of an AIPerf record (``session_000004`` is the fifth sample)."""
    tail = str(conversation_id or "").rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else None


def precision_sensitive(failed: Sequence[int], noise: Mapping[str, Any] | None) -> bool:
    """Every failing sample also fails for the native model at the candidate precision."""
    native = (noise or {}).get("failed_indices")
    return bool(failed) and native is not None and set(failed) <= set(native)


def judge_accuracy(records: Sequence[Mapping[str, Any]], gate: Mapping[str, Any], expected: int,
                   labels: Mapping[str, Any] | None = None, noise: Mapping[str, Any] | None = None,
                   sampled: bool = False) -> dict[str, Any]:
    """Pass when the candidate meets the declared gate. A failure is ``inconclusive`` (not an issue)
    only when every failing sample also fails for the native model at the candidate precision, or
    when the model always samples; the native noise floor is otherwise diagnostic."""
    passed = sum(bool(record.get("passed")) for record in records)
    failed = [index for index in (sample_index(r.get("conversation_id")) for r in records if not r.get("passed"))
              if index is not None]
    failures = [{"conversation_id": r.get("conversation_id"), "task": r.get("task"),
                 "explanation": r.get("explanation"), "actual": str(r.get("actual"))[:200],
                 "expected": str(r.get("expected"))[:200]} for r in records if not r.get("passed")]
    total = len(records)
    required = required_passes(gate, expected)
    ok = total == expected and passed >= required
    unparsed = sum(bool(record.get("unparsed")) for record in records)
    result = {"samples": total, "expected_samples": expected, "passed": passed, "pass_rate": passed / total if total
              else 0.0, "unparsed": unparsed, "required_passes": required, "gate": dict(gate),
              "failed_indices": failed, "failures": failures[:10]}
    if noise:
        result["noise_floor"] = {"passed": noise.get("passed"), "total": noise.get("total"),
                                 "precision": noise.get("precision")}
    if "max_wer_increase_from_reference" in gate:
        metrics = label_metrics(records, labels or {})
        result["label_metrics"] = metrics
        ok = (ok and metrics["unmatched"] == 0
              and metrics["wer_increase_from_reference"] <= float(gate["max_wer_increase_from_reference"]))
    # Outputs the grader cannot compare at all (different fields) say nothing about accuracy.
    status = "pass" if ok else ("not-comparable" if total and unparsed == total else "fail")
    return {"status": settle(status, failed, noise, sampled, result), **result}


def settle(status: str, failed: Sequence[int], noise: Mapping[str, Any] | None, sampled: bool,
           result: dict[str, Any]) -> str:
    """A failing suite becomes inconclusive when the mismatch cannot be attributed to TRTMC."""
    if status != "fail":
        return status
    if sampled:
        # The model always samples and TRTMC does not replay PyTorch's random stream.
        result["sampled"] = True
        return "inconclusive"
    if precision_sensitive(failed, noise):
        result["precision_sensitive"] = True
        return "inconclusive"
    return status


def verdict(result: Mapping[str, Any], *, expected_suites: int, expected_modes: int) -> dict[str, Any]:
    """Model-level outcome: Acc pass/fail/error, one light per reference mode, and a category."""
    accuracy = result.get("accuracy", [])
    if len(accuracy) < expected_suites:
        acc = "error"
    else:
        statuses = {item["status"] for item in accuracy}
        acc = ("pass" if statuses == {"pass"} else "error" if "error" in statuses else "fail" if "fail" in statuses else
               "not-comparable" if "not-comparable" in statuses else "inconclusive")
    session_state = any(item.get("isolated_check", {}).get("status") == "pass" for item in accuracy)
    lights = {item["reference_mode"]: item["light"] for item in result.get("performance_l1", [])}
    # "n/a": the native model could not run in that mode (for example torch.compile failing).
    measured = [value for value in lights.values() if value != "n/a"]
    perf = "error" if len(lights) < expected_modes or not measured else (
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
                      max_ci_percent: float, outputs_match: bool, output_reason: str) -> dict[str, Any]:
    reasons, notes = [], []
    if not outputs_match:
        reasons.append(f"output check failed: {output_reason}")
    wide = []
    for side, stats in (("candidate", candidate), ("reference", reference)):
        if stats.get("p50_ms") is None:
            reasons.append(f"{side} has no {METRIC}")
        elif (stats.get("aggregation", "mean") == "mean" and stats.get("ci_percent") is not None
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
    result = {"candidate": dict(candidate), "reference": dict(reference), "margin_percent": margin_percent,
              "output_check": {"match": outputs_match, "reason": output_reason}, "notes": notes}
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
