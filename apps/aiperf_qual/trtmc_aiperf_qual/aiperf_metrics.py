# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AIPerf's native client statistics, reported per run without acceptance gates."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

METRICS = ("request_latency", "request_throughput", "output_token_throughput", "request_error_rate")
COLUMNS = (("Latency p50", "request_latency", "p50"), ("Latency p99", "request_latency", "p99"),
           ("Request throughput", "request_throughput", "avg"),
           ("Output token throughput", "output_token_throughput", "avg"),
           ("Error rate", "request_error_rate", "avg"))
NOTE = ("AIPerf native client metrics (informational; no gate). Values are from each profiling run's export, "
        "without merging percentiles across runs. Missing metrics are shown as —. "
        "Client measurements include transport overhead; single-lane concurrency measures queueing. "
        "Buffered SSE does not measure token TTFT/ITL.")


def extract(summary: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Keep the original statistics and units; neither synthesize nor reaggregate values."""
    metrics = {}
    for name in METRICS:
        source = summary.get(name)
        if not isinstance(source, Mapping):
            continue
        values = {key: value for key, value in source.items()
                  if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)}
        if values:
            if isinstance(source.get("unit"), str):
                values["unit"] = source["unit"]
            metrics[name] = values
    return metrics


def capture(run: Any) -> dict[str, Any]:
    """A missing or unreadable optional summary cannot change qualification execution."""
    try:
        summary = getattr(run, "summary", {}) or {}
    except (OSError, ValueError):
        summary = {}
    if not isinstance(summary, Mapping):
        summary = {}
    return {"source": str(run.directory / "profile_export_aiperf.json"), "metrics": extract(summary)}


def entries(batches: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Preserve workload, precision, and run boundaries, excluding superseded attempts."""
    results = []
    for batch in batches:
        if batch.get("superseded") or "aiperf_metrics" not in batch:
            continue
        identity, native = batch["identity"], batch["aiperf_metrics"]
        results.append({"workload": batch["workload"], "role": batch["role"], "side": identity.get("side"),
                        "mode": identity.get("mode"), "precision": identity.get("precision"),
                        "concurrency": identity.get("concurrency"), "batch_id": batch["batch_id"],
                        "requests": batch.get("expected_requests"), "aiperf_exit": batch["aiperf_exit"],
                        "gate": False, **native})
    return sorted(results, key=lambda item: (item["workload"], item["role"], item.get("mode") or "",
                                           item.get("precision") or "", item["batch_id"]))


def cells(item: Mapping[str, Any]) -> list[str]:
    """The same labels and original units in Markdown and HTML reports."""
    side = "TRTMC" if item.get("side") == "candidate" else "Native" if item.get("side") == "reference" else "Unknown"
    mode = item.get("mode") if item.get("side") == "reference" else None
    identity = " ".join(str(value) for value in (side, mode, item.get("precision")) if value)
    run = Path(str(item.get("source") or "")).parent.name or "—"
    values = [str(item.get("workload") or "—"), str(item.get("role") or "—"), identity,
              str(item.get("concurrency") or "—"), run, str(item.get("requests") if item.get("requests") is not None else "—")]
    for _, metric, statistic in COLUMNS:
        data = (item.get("metrics") or {}).get(metric) or {}
        value = data.get(statistic)
        values.append("—" if value is None else f"{value:.3f} {data.get('unit') or ''}".strip())
    return values


HEADERS = ("Workload", "Role", "Side / mode / precision", "Concurrency", "Run", "Requests",
           *(label for label, _, _ in COLUMNS))


def markdown(items: Sequence[Mapping[str, Any]]) -> list[str]:
    def escape(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ").replace("\r", " ")

    return ["| " + " | ".join(HEADERS) + " |", "|" + "---|" * len(HEADERS),
            *("| " + " | ".join(escape(cell) for cell in cells(item)) + " |" for item in items)]
