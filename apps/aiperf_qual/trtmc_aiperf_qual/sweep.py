# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Perf L2: an AIPerf serving sweep over the OpenAI completions route (informational).

TRTMC and the native model (eager) each receive synthetic prompts of a fixed input/output length at
every concurrency level; the report compares request throughput and latency percentiles. The light
compares throughput at the highest level. trtmc-perf-serve runs one request at a time, so higher
concurrency measures queueing rather than batching; the sweep does not change the category.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .aiperf_runner import run_aiperf
from .config import Environment
from .judge import light

METRICS = {"request_throughput": "avg", "request_latency": ("p50", "p99"), "request_error_rate": "avg",
           "output_token_throughput": "avg"}


def lengths(l2: Mapping[str, Any], sequence_limit: int | None) -> tuple[int, int]:
    """(input, output) tokens that fit the bundle's sequence limit."""
    isl, osl = int(l2.get("isl", 96)), int(l2.get("osl", 32))
    if sequence_limit:
        osl = max(4, min(osl, sequence_limit // 4))
        isl = max(8, min(isl, sequence_limit - osl - 16))
    return isl, osl


def _level(environment: Environment, service: Mapping[str, Any], model: Mapping[str, Any], isl: int, osl: int,
           concurrency: int, requests: int, out: Path) -> dict[str, Any]:
    arguments = ["--endpoint-type", "completions", "--url", service["url"],
                 "--tokenizer", str(model["candidate"]["checkpoint"]), "--isl", str(isl),
                 "--synthetic-input-tokens-stddev", "0", "--osl", str(osl), "--output-tokens-stddev", "0",
                 "--extra-inputs", f"max_tokens:{osl}", "--concurrency", str(concurrency),
                 "--request-count", str(requests), "--warmup-request-count", str(max(1, concurrency))]
    if model["reference"].get("trust_remote_code"):
        arguments.append("--tokenizer-trust-remote-code")
    run = run_aiperf(environment, out, arguments)
    summary = run.summary
    level: dict[str, Any] = {"concurrency": concurrency, "aiperf_exit": run.exit_code}
    for name, fields in METRICS.items():
        for field in (fields if isinstance(fields, tuple) else (fields,)):
            value = (summary.get(name) or {}).get(field)
            if value is not None:
                level[f"{name}_{field}"] = value
    return level


def compare(candidate: list[Mapping[str, Any]], reference: list[Mapping[str, Any]],
            margin_percent: float) -> dict[str, Any]:
    """Light on request throughput at the highest concurrency both sides completed without errors."""
    reasons = []
    for side, levels in (("candidate", candidate), ("reference", reference)):
        if any(level.get("request_error_rate_avg") for level in levels):
            reasons.append(f"{side} had request errors")
        if not levels or "request_throughput_avg" not in levels[-1]:
            reasons.append(f"{side} has no throughput")
    if reasons:
        return {"light": "white", "reasons": reasons}
    top_candidate, top_reference = candidate[-1]["request_throughput_avg"], reference[-1]["request_throughput_avg"]
    # Throughput: higher is better, so compare the per-request times.
    return {"light": light(1 / top_candidate, 1 / top_reference, margin_percent), "reasons": [],
            "throughput_ratio": top_candidate / top_reference, "concurrency": candidate[-1]["concurrency"]}


def run(environment: Environment, model: Mapping[str, Any], l2: Mapping[str, Any], services: Mapping[str, Any],
        out: Path) -> dict[str, Any]:
    """``services``: {"candidate": trtmc service, "reference": native eager service}."""
    isl, osl = lengths(l2, model["candidate"].get("max_sequence_length"))
    levels = [int(value) for value in l2.get("concurrency", [1, 4])]
    requests = int(l2.get("requests", 32))
    sides = {side: [_level(environment, service, model, isl, osl, concurrency, requests,
                           out / f"l2-{side}-c{concurrency}") for concurrency in levels]
             for side, service in services.items()}
    return {"isl": isl, "osl": osl, "requests": requests, **sides,
            **compare(sides["candidate"], sides["reference"], float(l2.get("margin_percent", 5))),
            "note": "trtmc-perf-serve serializes requests: concurrency measures queueing, not batching"}
