# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The model verdict (Acc and Perf) and the Perf lights from AIPerf exports."""

from __future__ import annotations

import json
import math
import statistics
from typing import Any, Mapping, Sequence

from .noninferiority import t_quantile

METRIC = "trtmc_model_call_time"


def light_key(item: Mapping[str, Any]) -> str:
    """One light per reference mode and timed request: ``eager``, or ``eager/<request>`` with several."""
    return f"{item['reference_mode']}/{item['request']}" if item.get("request") else item["reference_mode"]


QUALIFYING_MODE = "eager"  # torch.compile lights are opt-in reports outside the category (DESIGN.md 4.6)


def verdict(result: Mapping[str, Any], *, expected_suites: Sequence[str], expected_modes: int) -> dict[str, Any]:
    """Model-level outcome (DESIGN.md 4.7): Acc (``n/a`` for a Perf-only model), the lights, and a category.
    Only ``pass`` qualifies. ``expected_modes`` eager lights (one per timed request) decide Perf; a missing,
    unavailable (``n/a``), or ``error`` eager light is an error, and other reference modes are reported only.

    ``expected_suites`` names every judged result the configuration requires (benchmarks and the
    entries each supplementary check emits); a missing one is an error, whatever else ran.
    ``informational`` entries are reported, never judged. A model without a native path is
    ``not-covered``.
    """
    lights = {light_key(item): item["light"] for item in result.get("performance_l1", [])}
    if (result.get("reference") or {}).get("backend") == "unsupported":
        return {"acc": "n/a", "perf": "n/a", "lights": lights, "category": "not-covered"}
    accuracy = [item for item in result.get("accuracy", []) if not item.get("informational")]
    produced = {item.get("suite") for item in accuracy}
    if any(name not in produced for name in expected_suites):
        acc = "error"
    elif result.get("accuracy_source") == "none" and not accuracy:
        acc = "n/a"
    else:
        statuses = {item["status"] for item in accuracy}
        acc = ("pass" if statuses == {"pass"} else "error" if "error" in statuses else "fail" if "fail" in statuses else
               "inconclusive" if "inconclusive" in statuses else "not-comparable")
    qualifying = [item["light"] for item in result.get("performance_l1", [])
                  if item.get("reference_mode") == QUALIFYING_MODE]
    perf = "error" if len(qualifying) < max(expected_modes, 1) or {"error", "n/a"} & set(qualifying) else (
        "green" if all(value == "green" for value in qualifying) else
        "red" if "red" in qualifying else "yellow" if "yellow" in qualifying else "white")
    if acc == "error" or perf == "error":
        category = "error"
    elif acc == "fail":
        category = "acc-issue"
    elif acc == "inconclusive":
        category = "acc-inconclusive"
    elif acc == "not-comparable":
        category = "not-comparable"
    elif perf in ("red", "yellow"):
        category = "perf-issue"
    elif perf == "white":
        category = "perf-inconclusive"
    else:
        category = "pass"
    return {"acc": acc, "perf": perf, "lights": lights, "category": category}


AGGREGATIONS = ("mean", "best")
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
        half = t_quantile(0.975, len(values) - 1) * statistics.stdev(values) / len(values) ** 0.5
        ci = half / mean * 100
    center = min(values) if aggregation == "best" else mean
    return {"p50_ms": center, "ci_percent": ci, "runs": len(values), "per_run_p50_ms": values,
            "aggregation": aggregation}


def light(candidate_ms: float, reference_ms: float, margin_percent: float) -> str:
    """Point-estimate light (informational workload timings): green faster by more than the margin, red
    slower by more."""
    ratio = candidate_ms / reference_ms
    if ratio < 1 - margin_percent / 100:
        return "green"
    if ratio > 1 + margin_percent / 100:
        return "red"
    return "yellow"


def speedup_interval(candidate_runs: Sequence[float], reference_runs: Sequence[float]) -> tuple[float, float, float] | None:
    """Speedup (native / TRTMC) and its 90% two-sided interval (95% one-sided per bound): Welch's t
    interval of log(native) - log(TRTMC) over the per-run p50s. None with fewer than two runs a side."""
    if len(candidate_runs) < 2 or len(reference_runs) < 2:
        return None
    mine, theirs = [math.log(value) for value in candidate_runs], [math.log(value) for value in reference_runs]
    center = statistics.fmean(theirs) - statistics.fmean(mine)
    parts = [statistics.variance(values) / len(values) for values in (theirs, mine)]
    spread = sum(parts)
    if spread == 0.0:
        return math.exp(center), math.exp(center), math.exp(center)
    df = spread ** 2 / sum(part ** 2 / (len(values) - 1) for part, values in zip(parts, (theirs, mine)) if part)
    half = t_quantile(0.95, df) * math.sqrt(spread)  # Welch-Satterthwaite degrees of freedom, fractional
    return math.exp(center), math.exp(center - half), math.exp(center + half)


TEXT_OPERATIONS = ("generate", "translate", "transcribe")
MEDIA_OPERATIONS = ("generate_image",)
AUDIO_OPERATIONS = ("generate_audio", "speak")


def work_signature(operation: str, observation: Mapping[str, Any] | None) -> tuple | None:
    """What a response did (DESIGN.md 4.6), as (evidence, value) pairs from fields both backends report:
    for text (generation, translation, transcription) the generated token count and the generated text,
    two alternatives (equal counts: the same decode steps; equal texts: the same tokens, whichever way a
    backend counts the end-of-sequence token); for generated media ``media_digest`` frames / height / width;
    for generated speech the ``audio_digest`` length in 10 ms. Denoising steps are not in a response: both
    sides get the request's stated value (an unstated one is a configuration error). ``()`` where the input
    fixes the work; None when a response lacks the evidence its operation needs."""
    observation = observation or {}
    if operation in TEXT_OPERATIONS:
        tokens, text = observation.get("output_tokens"), observation.get("text")
        if tokens is None and text is None:
            return None
        return (("output_tokens", None if tokens is None else int(tokens)),
                ("text", None if text is None else str(text)))  # exact: whitespace is generated work too
    if operation in MEDIA_OPERATIONS:
        media = observation.get("media_digest") or {}
        geometry = tuple(media.get(key) for key in ("frames", "height", "width"))
        return None if None in geometry else (("frames_height_width", geometry),)
    if operation in AUDIO_OPERATIONS:
        seconds = (observation.get("audio_digest") or {}).get("seconds")
        return None if seconds is None else (("audio_10ms", round(float(seconds) * 100)),)
    return ()


def work_check(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> str | None:
    """Why the timed responses did not all do the same work, or None: every response of both sides carries
    its evidence, and on one of its kinds (``work_signature``) all responses of both sides agree."""
    missing = [f"{side}: {stats['work_missing']} responses report no work" for side, stats in
               (("TRTMC", candidate), ("native", reference)) if stats.get("work_missing")]
    missing += [f"{side}: no work evidence" for side, stats in (("TRTMC", candidate), ("native", reference))
                if not stats.get("work")]
    if missing:
        return "; ".join(missing)
    signatures = [dict(tuple(pair) for pair in signature) for stats in (candidate, reference)
                  for signature in stats["work"]]
    kinds = {kind for signature in signatures for kind in signature}
    if not kinds:  # the input fixes the work (every signature is empty)
        return None
    if any(len({repr(signature.get(kind)) for signature in signatures}) == 1
           and signatures[0].get(kind) is not None for kind in kinds):
        return None
    shown = {side: [signature for signature in stats.get("work") or []][:2]
             for side, stats in (("TRTMC", candidate), ("native", reference))}
    return f"work differs: TRTMC {shown['TRTMC']} vs native {shown['native']}"[:400]


def measurement_problems(stats: Mapping[str, Any], max_ci_percent: float) -> list[str]:
    """Why one side's timing is no valid measurement on its own, or []: a run incomplete, no model-call time, a
    response without work evidence, a busy GPU or one whose utilization was not measured before every run, or its
    runs spread beyond ``max_ci_percent`` (or too few for a CI)."""
    problems = [str(stats["incomplete"])] if stats.get("incomplete") else []
    if stats.get("p50_ms") is None:
        problems.append(f"no {METRIC}")
    if stats.get("work_missing") or not stats.get("work"):
        problems.append(f"{stats.get('work_missing') or 'all'} responses report no work")
    if stats.get("gpu_unmeasured_runs") or stats.get("gpu_busy_percent") is None:
        problems.append(f"GPU utilization not measured before {stats.get('gpu_unmeasured_runs') or 'any'} runs")
    elif stats["gpu_busy_percent"] >= GPU_BUSY_PERCENT:
        problems.append(f"GPU {stats['gpu_busy_percent']:.0f}% busy with other processes")
    if stats.get("ci_percent") is None:
        problems.append("fewer than two runs: no CI")
    elif stats["ci_percent"] > max_ci_percent:
        problems.append(f"CI ±{stats['ci_percent']:.2f}% > {max_ci_percent}%")
    return problems


def judge_performance(candidate: Mapping[str, Any], reference: Mapping[str, Any], *, margin_percent: float,
                      max_ci_percent: float, outputs_match: bool, output_reason: str,
                      not_equivalent: str | None = None, candidate_precision: str | None = None,
                      guard_percent: float = 0.0) -> dict[str, Any]:
    """Light of one reference mode (DESIGN.md 4.6). white: the comparison is invalid (work or outputs
    differ, the native model ran at another precision than ``candidate_precision``, a busy GPU) or a
    side's runs spread more than ``max_ci_percent``; otherwise green / red when the speedup interval lies
    beyond the margin widened by the guard (the server-instance effect the order check measured, which the
    runs of one instance do not show), yellow when it does not. A side whose runs did not complete every
    request, or that exported no model-call time, is an ``error`` light."""
    result = {"candidate": dict(candidate), "reference": dict(reference), "margin_percent": margin_percent,
              "guard_percent": guard_percent, "output_check": {"match": outputs_match, "reason": output_reason}}
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
    work = work_check(candidate, reference)
    if work:
        reasons.append(work)
    for side, stats in (("candidate", candidate), ("reference", reference)):
        if (stats.get("gpu_busy_percent") or 0) >= GPU_BUSY_PERCENT:
            reasons.append(f"GPU {stats['gpu_busy_percent']:.0f}% busy with other processes before the {side} timing")
        if stats.get("gpu_unmeasured_runs"):
            reasons.append(f"GPU utilization not measured before {stats['gpu_unmeasured_runs']} {side} runs")
    if reference.get("precision_fallback"):
        # The native model could not run at the candidate precision: a slower precision is no baseline.
        reasons.append(f"reference timed at {reference.get('precision')} (candidate precision failed: "
                       f"{str(reference['precision_fallback'])[:120]})")
    elif candidate_precision and reference.get("precision") and reference["precision"] != candidate_precision:
        reasons.append(f"reference timed at {reference['precision']}, TRTMC runs {candidate_precision}")
    notes += [f"{side}: {stats['exit_note']}" for side, stats in (("candidate", candidate), ("reference", reference))
              if stats.get("exit_note")]
    result["notes"] = notes
    result["speedup"] = reference["p50_ms"] / candidate["p50_ms"]  # informative even when white
    interval = speedup_interval(candidate.get("per_run_p50_ms") or [], reference.get("per_run_p50_ms") or [])
    if interval:
        result["speedup_interval90"] = [interval[1], interval[2]]
    if interval is None:
        reasons.append("fewer than two runs a side: no interval")
    # Unstable timing on either side invalidates the comparison, whatever the interval says (a side
    # aggregated by its best run, the opt-in torch.compile reference, is not held to it).
    reasons += [f"{side} CI ±{stats['ci_percent']:.2f}% > {max_ci_percent}%"
                for side, stats in (("TRTMC", candidate), ("native", reference))
                if stats.get("aggregation", "mean") == "mean" and stats.get("ci_percent") is not None
                and stats["ci_percent"] > max_ci_percent]
    if reasons:
        return {**result, "light": "white", "reasons": reasons}
    low, high = interval[1], interval[2]
    guard = 1 + guard_percent / 100
    faster, slower = (1 + margin_percent / 100) * guard, (1 - margin_percent / 100) / guard
    if low > faster:
        return {**result, "light": "green", "reasons": []}
    if high < slower:
        return {**result, "light": "red", "reasons": [f"TRTMC slower: speedup interval {low:.3f}..{high:.3f} "
                                                      f"below {slower:.3f}"]}
    return {**result, "light": "yellow", "reasons": [f"not faster by {margin_percent}% beyond the {guard_percent}% "
                                                     f"guard: speedup interval {low:.3f}..{high:.3f}, green above "
                                                     f"{faster:.3f}"]}


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
