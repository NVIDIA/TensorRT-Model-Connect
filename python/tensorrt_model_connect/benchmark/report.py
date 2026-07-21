# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small self-contained HTML report for a benchmark run."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any, Mapping


def write_html_report(result: Mapping[str, Any], path: Path) -> None:
    rows = "\n".join(_cell_row(cell) for cell in result.get("cells", []))
    status = escape(str(result.get("status", "unknown")))
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TRTMC benchmark report</title>
  <style>
    body {{ font: 14px system-ui, sans-serif; margin: 2rem; color: #20242a; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #d8dde5; padding: .6rem; text-align: left; }}
    th {{ background: #f4f6f8; }}
    .completed {{ color: #19713b; }} .failed {{ color: #a32626; }}
    code {{ background: #f4f6f8; padding: .1rem .25rem; }}
  </style>
</head>
<body>
  <h1>TRTMC benchmark</h1>
  <p>Status: <strong class="{status}">{status}</strong>. Timing scope:
     <code>public_pipeline_call_wall</code>. Task quality is not evaluated.</p>
  <table>
    <thead><tr><th>Model</th><th>Case</th><th>Operation</th><th>p50</th><th>p95</th><th>Task rate</th><th>Status</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <p>Machine-readable evidence: <code>result.json</code>; per-case resolved inputs,
     observations, logs, and telemetry are stored in each case directory.</p>
</body>
</html>
"""
    path.write_text(body, encoding="utf-8")


def _cell_row(cell: Mapping[str, Any]) -> str:
    metrics = cell.get("metrics", {})
    latency = metrics.get("latency_ms", {}) if isinstance(metrics, Mapping) else {}
    rate = _task_rate(metrics)
    values = (
        cell.get("model", ""),
        cell.get("name", ""),
        cell.get("operation", ""),
        _number(latency.get("p50"), " ms"),
        _number(latency.get("p95"), " ms"),
        rate,
        cell.get("status", "unknown"),
    )
    columns = "".join(f"<td>{escape(str(value))}</td>" for value in values)
    return f"<tr>{columns}</tr>"


def _task_rate(metrics: Mapping[str, Any]) -> str:
    choices = (
        ("output_tokens_per_s", " token/s"),
        ("images_per_s", " image/s"),
        ("embedding_vectors_per_s", " vector/s"),
        ("windows_per_s", " window/s"),
        ("request_throughput_per_s", " req/s"),
    )
    for field, unit in choices:
        if field in metrics:
            return _number(metrics[field], unit)
    return "—"


def _number(value: Any, suffix: str) -> str:
    return "—" if not isinstance(value, (int, float)) else f"{value:.3f}{suffix}"
