# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-independent JSON and HTML reporting for Performance qualification."""

from __future__ import annotations

from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .types import PerfMatrixError


RESULT_SCHEMA = "trtmc.perf-matrix/v2"
REPORT_SCHEMA = "trtmc.perf-report/v2"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(
    run_directory: Path,
    results: Mapping[str, Any],
    preparation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source_schema = results.get("schema_version", RESULT_SCHEMA)
    row_key = "cases" if source_schema == "trtmc.perf-matrix/v1" else "rows"
    rows = [dict(row) for row in results.get(row_key, []) if isinstance(row, Mapping)]
    selected_ids = results.get("selected_entry_ids")
    if not isinstance(selected_ids, list) or not all(
        isinstance(value, str) for value in selected_ids
    ):
        raise PerfMatrixError("matrix results selected_entry_ids must be strings")
    completed_ids = {str(row.get("id")) for row in rows}
    counts = {
        status: sum(row.get("status") == status for row in rows)
        for status in ("green", "yellow", "red", "contract-mismatch", "white")
    }
    report = {
        "schema_version": REPORT_SCHEMA,
        "source_schema_version": source_schema,
        "generated_at": _now(),
        "status": results.get("status", "unknown"),
        "suite": results.get("suite"),
        "environment": results.get("environment"),
        "summary": {
            "selected": len(selected_ids),
            "pending": sum(entry_id not in completed_ids for entry_id in selected_ids),
            "comparable": counts["green"] + counts["yellow"] + counts["red"],
            **counts,
        },
        "rows": rows,
    }
    if preparation is not None:
        report["preparation"] = dict(preparation)
    _write_json(run_directory / "report.json", report)
    (run_directory / "report.html").write_text(_report_html(report), encoding="utf-8")
    return report


def _measurement_html(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "Not recorded"
    metrics = value.get("metrics", {})
    latency = metrics.get("latency_ms", {}) if isinstance(metrics, Mapping) else {}
    samples = value.get("samples_ms", [])
    measured = (
        sorted(
            float(sample)
            for sample in samples
            if isinstance(sample, (int, float))
            and not isinstance(sample, bool)
            and math.isfinite(sample)
            and sample > 0
        )
        if isinstance(samples, list)
        else []
    )
    parts = []
    for name, percentile in (("p50", 0.5), ("p95", 0.95)):
        number = latency.get(name) if isinstance(latency, Mapping) else None
        if number is None and measured:
            position = (len(measured) - 1) * percentile
            lower, upper = math.floor(position), math.ceil(position)
            number = measured[lower] + (measured[upper] - measured[lower]) * (position - lower)
        if (
            isinstance(number, (int, float))
            and not isinstance(number, bool)
            and math.isfinite(number)
        ):
            parts.append(f"{name}: {number:,.3f} ms")
    policy = value.get("measurement_policy", {})
    scope = value.get("timing_scope") or (
        policy.get("timing_scope") if isinstance(policy, Mapping) else None
    )
    parts.append("Scope: " + html.escape(str(scope or "not recorded")))
    if isinstance(metrics, Mapping):
        parts.extend(
            f"{html.escape(name)}: {number:,.3f}"
            for name, number in metrics.items()
            if name.endswith("_per_s")
            and isinstance(number, (int, float))
            and not isinstance(number, bool)
            and math.isfinite(number)
        )
    return "<br>".join(parts)


def _report_html(report: Mapping[str, Any]) -> str:
    rows = []
    for row in report.get("rows", []):
        comparison = row.get("comparison", {}) if isinstance(row, Mapping) else {}
        measurement_stability = (
            row.get("measurement_stability", {}) if isinstance(row, Mapping) else {}
        )
        candidate = _measurement_html(
            row.get("candidate")
            or {"metrics": {"latency_ms": {"p50": comparison.get("candidate_p50_ms")}}}
        )
        reference = _measurement_html(
            row.get("reference", row.get("baseline"))
            or {"metrics": {"latency_ms": {"p50": comparison.get("reference_p50_ms")}}}
        )
        reason = comparison.get("reason", row.get("error", ""))
        stability = (
            measurement_stability.get("status", "")
            if isinstance(measurement_stability, Mapping)
            else ""
        )
        evidence = {
            key: row[key]
            for key in (
                "commands",
                "comparison",
                "measurement_stability",
                "resolved_settings",
                "baseline_contract",
            )
            if key in row
        }
        detail = html.escape(json.dumps(evidence, indent=2, sort_keys=True))
        link = ""
        artifact = row.get("artifact_dir")
        if (
            isinstance(artifact, str)
            and artifact
            and not Path(artifact).is_absolute()
            and ".." not in Path(artifact).parts
            and not any(value in artifact for value in (":", "\\"))
        ):
            from urllib.parse import quote

            link = f'<a href="{quote(artifact, safe="/.")}/">Case artifacts</a>'
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('id', '')))}</td>"
            f"<td>{html.escape(str(row.get('model', '')))}</td>"
            f"<td>{html.escape(str(row.get('operation', '')))}</td>"
            f"<td>{html.escape(str(row.get('status', '')))}</td>"
            f"<td>{html.escape(str(stability))}</td>"
            f"<td>{candidate}</td>"
            f"<td>{reference}</td>"
            f"<td>{html.escape(str(reason))}</td>"
            f"<td><details><summary>Evidence / commands</summary>{link}<pre>{detail}</pre></details></td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>TRTMC performance</title>
<style>
body {{ font: 14px system-ui, sans-serif; margin: 2rem; color: #222; }}
table {{ border-collapse: collapse; width: 100%; }}
th,td {{ border: 1px solid #ddd; padding: .5rem; text-align: left; }}
th {{ background: #f3f3f3; }}
pre {{ white-space: pre-wrap; overflow-wrap: anywhere; }}
</style></head><body>
<h1>TRTMC performance matrix</h1>
<p>Source data: {html.escape(str(report.get("source_schema_version", RESULT_SCHEMA)))}. Original timing scopes and recorded outcomes are preserved.</p>
<p>Status: {html.escape(str(report.get("status", "unknown")))};
selected: {html.escape(str(report.get("summary", {}).get("selected", 0)))};
pending: {html.escape(str(report.get("summary", {}).get("pending", 0)))}</p>
<label>Filter model, case, or status <input id="filter" type="search"></label>
<table><thead><tr><th>Entry</th><th>Model</th><th>Operation</th><th>Status</th>
<th>Stability</th><th>Candidate measurements</th><th>Reference measurements</th><th>Reason</th><th>Evidence</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>
<p>Green is faster by more than the margin, yellow is within the margin, red is
slower by more than the margin, and white is not comparable.</p>
<p>Output comparison follows the suite's declared contract; matching dimensions or token counts alone does not establish model correctness. Use family correctness tests for acceptance.</p>
<p><a href="results.json">Original results</a> · <a href="report.json">Report data</a></p>
<script>document.querySelector('#filter').addEventListener('input',function(){{
const query=this.value.toLowerCase();document.querySelectorAll('tbody tr').forEach(row=>{{
row.hidden=!row.textContent.toLowerCase().includes(query);}});}});</script>
</body></html>
"""
