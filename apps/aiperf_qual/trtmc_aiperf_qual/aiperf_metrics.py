# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Original AIPerf client statistics plus compact, informational report summaries."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import median
from typing import Any, Mapping, Sequence

METRICS = ("request_latency", "request_throughput", "output_token_throughput", "request_error_rate")
COLUMNS = (("Latency p50", "request_latency", "p50"), ("Latency p99", "request_latency", "p99"),
           ("Request throughput", "request_throughput", "avg"),
           ("Output token throughput", "output_token_throughput", "avg"),
           ("Error rate", "request_error_rate", "avg"))
NOTE = ("Informational; no gate. Repeated runs use the median of each exported statistic: latency p50/p99 "
        "are medians of run p50/p99 values, not percentiles of pooled requests. Requests are totals. "
        "Different workloads, modes, precisions, and concurrency levels stay separate. "
        "Missing metrics are —; partial metrics show available/total runs. Original runs remain in report.json. "
        "Client latency includes transport overhead. Buffered SSE does not measure token TTFT/ITL.")
BASE_HEADERS = ("Backend", "Precision", "Runs", "Requests")


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


def _statistic(runs: Sequence[Mapping[str, Any]], metric: str, statistic: str) -> dict[str, Any]:
    data = [(run.get("metrics") or {}).get(metric) or {} for run in runs]
    available = [item for item in data if isinstance(item.get(statistic), (int, float))
                 and not isinstance(item[statistic], bool) and math.isfinite(item[statistic])]
    units = {item.get("unit") or "" for item in available}
    values = [item[statistic] for item in available]
    return {"value": median(values) if values and len(units) == 1 else None,
            "unit": next(iter(units)) if len(units) == 1 else "", "unit_conflict": len(units) > 1,
            "available": len(values), "total": len(runs),
            "min": min(values) if values else None, "max": max(values) if values else None}


def summaries(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize repetitions of the same measurement conditions without changing raw evidence."""
    groups = defaultdict(list)
    for item in items:
        if not item.get("superseded"):
            groups[tuple(item.get(key) for key in ("workload", "role", "concurrency", "side", "mode", "precision"))].append(item)
    results = []
    for runs in groups.values():
        requests = [run.get("requests") for run in runs]
        results.append({**{key: runs[0].get(key) for key in ("workload", "role", "concurrency", "side", "mode", "precision")},
                        "runs": len(runs), "requests": sum(requests) if all(isinstance(n, int) for n in requests) else None,
                        "first_batch": min(run.get("batch_id", 0) for run in runs),
                        "failed_runs": sum(run.get("aiperf_exit", 0) != 0 for run in runs),
                        "values": [_statistic(runs, metric, stat) for _, metric, stat in COLUMNS]})
    return results


def workload_label(name: str, profile: str = "") -> str:
    if name.endswith("-catalog-near-capacity") or name == "catalog-near-capacity":
        return "Near capacity"
    if name.endswith("-catalog") or name == "catalog":
        return "Catalog"
    return name.removeprefix(profile + "-") if profile else name


def panels(items: Sequence[Mapping[str, Any]], profile: str = "", native_precision: str | None = None) -> list[dict[str, Any]]:
    """A workload's main Native/TRTMC pair, with alternative reference settings kept aside."""
    groups = defaultdict(list)
    for item in summaries(items):
        groups[(item["workload"] or "—", item["role"], item["concurrency"])].append(item)
    results = []
    for (workload, role, concurrency), variants in groups.items():
        references = sorted((item for item in variants if item["side"] == "reference"),
                            key=lambda item: (item["mode"] != "eager", bool(native_precision and item["precision"] != native_precision),
                                              item["first_batch"]))
        candidates = sorted((item for item in variants if item["side"] == "candidate"), key=lambda item: item["first_batch"])
        main = [*(references[:1] or [{"side": "reference"}]), *(candidates or [{"side": "candidate"}])]
        extras = [*references[1:], *(item for item in variants if item["side"] not in ("reference", "candidate"))]
        selected = [i for i, (_, metric, _) in enumerate(COLUMNS) if metric != "output_token_throughput"
                    or any(item["values"][i]["available"] for item in variants)]
        results.append({"workload": workload, "label": workload_label(workload, profile), "role": role,
                        "concurrency": concurrency, "main": main, "extras": extras, "columns": selected})
    return sorted(results, key=lambda item: ({"performance": 0, "both": 1, "quality": 1, "service": 2}.get(item["role"], 3),
                                             {"Catalog": 0, "Near capacity": 1}.get(item["label"], 2),
                                             item["workload"], item["concurrency"] or 0))


def panel_note(panel: Mapping[str, Any]) -> str:
    role = {"performance": "Fixed request", "both": "Evaluation dataset", "quality": "Quality dataset",
            "service": "Service workload"}.get(panel["role"], "Workload")
    main = panel["main"]
    note = f"{role} · concurrency {panel['concurrency'] or '—'} · Native mode: {main[0].get('mode') or '—'}"
    precisions = {item.get("precision") for item in main if item.get("precision")}
    return note + (" · Precision differs between Native and TRTMC" if len(precisions) > 1 else "")


def headers(panel: Mapping[str, Any]) -> list[str]:
    return [*BASE_HEADERS, *(COLUMNS[i][0] for i in panel["columns"])]


def metric_value(statistic: Mapping[str, Any]) -> str:
    value = statistic.get("value")
    if value is None:
        return "— (units differ)" if statistic.get("unit_conflict") else "—"
    text = f"{value:.3f} {statistic.get('unit') or ''}".strip()
    return text + (f" ({statistic['available']}/{statistic['total']} runs)"
                   if statistic["available"] < statistic["total"] else "")


def cells(item: Mapping[str, Any], columns: Sequence[int]) -> list[str]:
    side = {"candidate": "TRTMC", "reference": "Native"}.get(item.get("side"), "Unknown")
    if item.get("failed_runs"):
        side += f" (failed runs: {item['failed_runs']}/{item['runs']})"
    values = item.get("values") or [{}] * len(COLUMNS)
    return [side, str(item.get("precision") or "—"), str(item.get("runs") or "—"),
            str(item.get("requests") if item.get("requests") is not None else "—"),
            *(metric_value(values[i]) for i in columns)]


def markdown(items: Sequence[Mapping[str, Any]], profile: str = "", native_precision: str | None = None,
             level: int = 3) -> list[str]:
    def escape(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ").replace("\r", " ")

    def table(rows: Sequence[Mapping[str, Any]], panel: Mapping[str, Any]) -> list[str]:
        return ["| " + " | ".join(headers(panel)) + " |", "|" + "---|" * len(headers(panel)),
                *("| " + " | ".join(escape(cell) for cell in cells(item, panel["columns"])) + " |" for item in rows)]

    lines = []
    for panel in panels(items, profile, native_precision):
        lines += ["", "#" * level + " " + escape(panel["label"]), "", panel_note(panel), "", *table(panel["main"], panel)]
        if panel["extras"]:
            lines += ["", "<details><summary>Additional native settings</summary>", ""]
            for item in panel["extras"]:
                lines += [f"Native mode: {escape(str(item.get('mode') or '—'))}", "", *table([item], panel), ""]
            lines += ["</details>"]
    return lines
