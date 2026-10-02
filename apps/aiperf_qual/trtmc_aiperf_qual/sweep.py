# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Perf L2: AIPerf serving sweeps (informational; they never change the category).

Text generation: TRTMC and the native model (eager) each receive synthetic prompts of a fixed
input/output length over the OpenAI completions route at every concurrency level; the light compares
request throughput at the highest level. trtmc-perf-serve runs one request at a time, so higher
concurrency measures queueing rather than batching.

Image and video generation (``kind: media``): AIPerf's ``image_generation`` / ``video_generation``
endpoints send PartiPrompts at the catalog request (size, frames, seed) with the denoising steps at
half and at the catalog count. Each side runs alone on the GPU with ``--memory-probe``; the server
records give the model-call time and peak GPU memory, and the two step counts split the call into a
per-step (denoiser) and a fixed part (text encoders, VAE decode). The light compares the model-call
time at the catalog steps.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

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


MEDIA_ROUTES = ("/v1/images/generations", "/v1/videos")


def media_variants(request: Mapping[str, Any]) -> list[int | None]:
    """Denoising steps to measure: half and the catalog count (one level when the request has none)."""
    steps = int(request.get("num_steps") or 0)
    return sorted({max(1, steps // 2), steps}) if steps > 1 else [None]


def _median(values: Sequence[float]) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def _route_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [record for record in records if record.get("route") in MEDIA_ROUTES]


def media_stats(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Model-call p50 and peak GPU memory over the measured requests' server records."""
    timings = [record.get("timing") or {} for record in records]
    calls = [float(timing["model_call_ms"]) for timing in timings if timing.get("model_call_ms") is not None]
    memory = [float(timing["peak_memory_mb"]) for timing in timings if timing.get("peak_memory_mb") is not None]
    return {"measured": len(calls), "model_call_p50_ms": _median(calls),
            "peak_memory_mb": max(memory) if memory else None}


def _media_level(environment: Environment, service: Mapping[str, Any], endpoint: str, prompts: Path,
                 steps: int | None, requests: int, out: Path) -> dict[str, Any]:
    arguments = ["--endpoint-type", endpoint, "--url", service["url"], "--tokenizer", "builtin",
                 "--input-file", str(prompts), "--custom-dataset-type", "single_turn", "--concurrency", "1",
                 "--request-count", str(requests), "--warmup-request-count", "1"]
    if steps:
        arguments += ["--extra-inputs", f"num_inference_steps:{steps}"]
    before = len(_route_records(Path(service["records"])))
    run = run_aiperf(environment, out, arguments)
    measured = _route_records(Path(service["records"]))[before + 1:]  # the first one is AIPerf's warmup
    latency = run.summary.get("request_latency") or {}
    return {"steps": steps, "aiperf_exit": run.exit_code, "request_latency_p50": latency.get("p50"),
            "request_error_rate_avg": (run.summary.get("request_error_rate") or {}).get("avg"),
            **media_stats(measured)}


def decompose(levels: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Per-step and fixed model-call time from two step counts (linear in the steps)."""
    points = [(level["steps"], level["model_call_p50_ms"]) for level in levels
              if level.get("steps") and level.get("model_call_p50_ms") is not None
              and not level.get("request_error_rate_avg")]
    if len(points) < 2 or points[0][0] == points[-1][0]:
        return {}
    (low_steps, low_ms), (high_steps, high_ms) = points[0], points[-1]
    per_step = (high_ms - low_ms) / (high_steps - low_steps)
    return {"per_step_ms": per_step, "fixed_ms": high_ms - high_steps * per_step}


def compare_media(candidate: Sequence[Mapping[str, Any]], reference: Sequence[Mapping[str, Any]],
                  margin_percent: float) -> dict[str, Any]:
    """Light on the model-call p50 at the catalog steps (the last level)."""
    reasons, notes = [], []
    for side, levels in (("candidate", candidate), ("reference", reference)):
        if not levels or levels[-1].get("model_call_p50_ms") is None:
            reasons.append(f"{side} has no measured request")
        elif levels[-1].get("request_error_rate_avg"):
            reasons.append(f"{side} had request errors")
        # Some families accept only their qualified step counts (Wan2.2-TI2V): the other levels may fail.
        notes += [f"{side} failed at {level.get('steps')} steps" for level in levels[:-1]
                  if level.get("request_error_rate_avg") or level.get("model_call_p50_ms") is None]
    if reasons:
        return {"light": "white", "reasons": reasons, "notes": notes}
    top_candidate, top_reference = candidate[-1], reference[-1]
    result = {"light": light(top_candidate["model_call_p50_ms"], top_reference["model_call_p50_ms"], margin_percent),
              "reasons": [], "notes": notes,
              "speedup": top_reference["model_call_p50_ms"] / top_candidate["model_call_p50_ms"]}
    if top_candidate.get("peak_memory_mb") and top_reference.get("peak_memory_mb"):
        result["memory_ratio"] = top_candidate["peak_memory_mb"] / top_reference["peak_memory_mb"]
    return result


def run_media(environment: Environment, model: Mapping[str, Any], l2: Mapping[str, Any], out: Path,
              python: str, precision: str) -> dict[str, Any]:
    """The media sweep; TRTMC first, then the native model (eager), each alone on the GPU."""
    from .generation import is_video
    from .models import _suite
    from .services import serving
    from .suites import build_suite

    suite = build_suite(_suite(l2.get("suite", "partiprompts-30"), model["catalog_profile"],
                               environment.path("repo")), environment)
    samples = suite.samples[: int(l2.get("prompts", 3))]
    video = is_video(samples[0]["request"])
    endpoint = "video_generation" if video else "image_generation"
    requests = int(l2.get("video_requests" if video else "requests", 2 if video else 3))
    prompts = out / "l2-media-prompts.jsonl"
    prompts.parent.mkdir(parents=True, exist_ok=True)
    prompts.write_text("".join(json.dumps({"text": str(sample["request"]["prompt"])}) + "\n" for sample in samples))
    variants = media_variants(samples[0]["request"])
    sides: dict[str, list[dict[str, Any]]] = {}
    for side, backend, kwargs in (("candidate", "trtmc", {}),
                                  ("reference", "reference", {"mode": "eager", "precision": precision,
                                                              "python": python})):
        with serving(environment, model, backend, out / f"l2-{side}-server", memory_probe=True, **kwargs) as service:
            sides[side] = [_media_level(environment, service, endpoint, prompts, steps, requests,
                                        out / f"l2-{side}-steps{steps or 'catalog'}") for steps in variants]
    return {"kind": "media", "endpoint": endpoint, "prompts": len(samples), "requests": requests,
            "reference_precision": precision, **sides,
            "decomposition": {side: decompose(levels) for side, levels in sides.items()},
            **compare_media(sides["candidate"], sides["reference"], float(l2.get("margin_percent", 5))),
            "note": "model-call time and peak GPU memory from the server records; per-step and fixed times "
                    "assume the call is linear in the denoising steps"}
