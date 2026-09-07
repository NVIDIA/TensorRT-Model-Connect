# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small JSON and HTML reports for benchmark runs."""

from __future__ import annotations

import html
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from .types import BenchmarkError


_RUN_SCHEMA = "trtmc.benchmark-run/v2"
_REPORT_SCHEMA = "trtmc.benchmark-report/v2"

_TASK_RATES = (
    ("output_tokens_per_s", " token/s"),
    ("videos_per_s", " video/s"),
    ("images_per_s", " image/s"),
    ("audio_seconds_per_s", " audio-s/s"),
    ("input_audio_seconds_per_s", " input-audio-s/s"),
    ("output_audio_seconds_per_s", " output-audio-s/s"),
    ("documents_per_s", " document/s"),
    ("embedding_vectors_per_s", " vector/s"),
    ("windows_per_s", " window/s"),
    ("action_steps_per_s", " action-step/s"),
    ("stereo_pairs_per_s", " stereo-pair/s"),
    ("request_throughput_per_s", " request/s"),
)

_DISPLAY_PATH = re.compile(
    r"""(?:
        (?P<quote>["'])
        (?:file://|/|[A-Za-z]:[\\/]|\\\\)
        .*?
        (?P=quote)
        |
        https?://[^\s"'<>]+
        |
        file://[^\s"'<>]+
        |
        (?<![A-Za-z0-9_])[A-Za-z]:[\\/][^\s"'<>]+
        |
        \\\\[^\s"'<>]+
        |
        (?<![-A-Za-z0-9_/.<])/
        [^\s"'<>]+
    )""",
    re.IGNORECASE | re.VERBOSE,
)


def write_html_report(result: Mapping[str, Any], path: Path) -> None:
    cells = result.get("cells", [])
    source_runs = _source_runs(result)
    default_run_id = str(result.get("run_id", ""))
    rows: list[str] = []
    for cell in cells if isinstance(cells, list) else []:
        if not isinstance(cell, Mapping):
            continue
        metrics = cell.get("metrics", {})
        latency = metrics.get("latency_ms", {}) if isinstance(metrics, Mapping) else {}
        p50 = latency.get("p50") if isinstance(latency, Mapping) else None
        p95 = latency.get("p95") if isinstance(latency, Mapping) else None
        run_id = str(cell.get("run_id", default_run_id))
        rows.append(
            "<tr>"
            f"<td>{_escape(cell.get('model', ''))}</td>"
            f"<td>{_escape(cell.get('name', ''))}</td>"
            f"<td>{_escape(cell.get('operation', ''))}</td>"
            f"<td>{_escape(_number(p50, ' ms'))}</td>"
            f"<td>{_escape(_number(p95, ' ms'))}</td>"
            f"<td>{_escape(_task_rate(metrics))}</td>"
            f"<td>{_escape(cell.get('status', 'unknown'))}</td>"
            f"<td>{_cell_evidence(cell, run_id)}</td>"
            "</tr>"
        )
    run_rows = [
        "<tr>"
        f"<td><code>{_escape(run.get('run_id', 'unknown'))}</code></td>"
        f"<td>{_escape(run.get('status', 'unknown'))}</td>"
        f"<td>{_escape(run.get('started_at') or '—')}</td>"
        f"<td>{_escape(run.get('finished_at') or '—')}</td>"
        "<td><code>result.json</code></td>"
        "</tr>"
        for run in source_runs
    ]
    summary = result.get("summary", {})
    summary = summary if isinstance(summary, Mapping) else {}
    run_count = summary.get("runs", len(source_runs))
    warnings = result.get("warnings", [])
    warning_values = warnings if isinstance(warnings, list) else []
    warning_items = "".join(f"<li>{_escape(warning)}</li>" for warning in warning_values)
    warnings_html = f"<h2>Warnings</h2><ul>{warning_items}</ul>" if warning_items else ""
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>TRTMC benchmark report</title>
<style>
body {{ font: 14px system-ui, sans-serif; margin: 2rem; color: #222; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; }}
th, td {{ border: 1px solid #ddd; padding: .5rem; text-align: left; }}
th {{ background: #f3f3f3; }}
code {{ background: #f3f3f3; padding: .1rem .25rem; overflow-wrap: anywhere; }}
.error {{ color: #a32626; }}
</style></head><body>
<h1>TRTMC benchmark</h1>
<p>Status: <strong>{_escape(result.get("status", "unknown"))}</strong>.
Source runs: <strong>{_escape(run_count)}</strong>; cases: <strong>{len(rows)}</strong>.</p>
<h2>Models and cases</h2>
<table><thead><tr><th>Model</th><th>Case</th><th>Operation</th><th>p50</th><th>p95</th><th>Task rate</th><th>Status</th><th>Evidence / error</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>
<h2>Source runs</h2>
<table><thead><tr><th>Run</th><th>Status</th><th>Started</th><th>Finished</th><th>Result evidence</th></tr></thead>
<tbody>{"".join(run_rows)}</tbody></table>
{warnings_html}
<p>Machine-readable evidence remains in <code>result.json</code> for a single run
and <code>report.json</code> for a collection.</p>
</body></html>
"""
    path.write_text(document, encoding="utf-8")


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _number(value: Any, suffix: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    converted = float(value)
    return "—" if not math.isfinite(converted) else f"{converted:.3f}{suffix}"


def _task_rate(metrics: Any) -> str:
    if not isinstance(metrics, Mapping):
        return "—"
    for field, unit in _TASK_RATES:
        if field in metrics:
            return _number(metrics[field], unit)
    return "—"


def _source_runs(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    runs = result.get("runs")
    if isinstance(runs, list):
        return [run for run in runs if isinstance(run, Mapping)]
    run_id = result.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        return []
    return [
        {
            "run_id": run_id,
            "status": result.get("status", "unknown"),
            "started_at": result.get("started_at"),
            "finished_at": result.get("finished_at"),
        }
    ]


def _is_absolute_path(value: str) -> bool:
    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
        return True
    parsed = urlsplit(value)
    return parsed.scheme.lower() == "file" and PurePosixPath(parsed.path).is_absolute()


def _redact_absolute_paths(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        candidate = match.group(0)
        if match.group("quote"):
            candidate = candidate[1:-1]
        return "[absolute path]" if _is_absolute_path(candidate) else match.group(0)

    return _DISPLAY_PATH.sub(replace, value)


def _cell_evidence(cell: Mapping[str, Any], run_id: str) -> str:
    parts: list[str] = []
    if run_id:
        parts.append(f"run <code>{_escape(run_id)}</code>")
    artifact = cell.get("artifact_dir")
    if isinstance(artifact, str) and artifact:
        posix_path = PurePosixPath(artifact)
        windows_path = PureWindowsPath(artifact)
        safe_artifact = not _is_absolute_path(artifact) and all(
            part not in {"", ".", ".."}
            for path in (posix_path, windows_path)
            for part in path.parts
        )
        label = posix_path.as_posix() if safe_artifact else "case artifacts"
        parts.append(f"artifacts <code>{_escape(label)}</code>")
    error = cell.get("error")
    if isinstance(error, str) and error:
        displayed_error = _redact_absolute_paths(error)
        parts.append(f'<span class="error">{_escape(displayed_error)}</span>')
    return "<br>".join(parts) or "—"


def generate_collection_report(
    roots: Sequence[Path], output_dir: Path
) -> tuple[dict[str, Any], tuple[str, ...]]:
    result_paths = _result_paths(roots)
    if not result_paths:
        raise BenchmarkError("no benchmark result.json files were found")
    runs: list[dict[str, Any]] = []
    seen_ids: dict[str, Path] = {}
    warnings: list[str] = []
    cells: list[dict[str, Any]] = []
    for path in result_paths:
        result_label = _result_label(path)
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            warnings.append(f"skipped unreadable result {result_label}: {_error_summary(error)}")
            continue
        if not isinstance(result, Mapping) or result.get("schema_version") != _RUN_SCHEMA:
            warnings.append(f"skipped unsupported result {result_label}")
            continue
        run_id = result.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            warnings.append(f"skipped result without run_id {result_label}")
            continue
        if run_id in seen_ids:
            raise BenchmarkError(f"duplicate run_id {run_id!r}: {seen_ids[run_id]} and {path}")
        seen_ids[run_id] = path
        run_cells = result.get("cells", [])
        if not isinstance(run_cells, list):
            warnings.append(f"skipped malformed cells in {result_label}")
            continue
        source = str(path.parent)
        runs.append(
            {
                "run_id": run_id,
                "result_path": source,
                "status": str(result.get("status", "unknown")),
                "started_at": result.get("started_at"),
                "finished_at": result.get("finished_at"),
            }
        )
        for cell in run_cells:
            if isinstance(cell, Mapping):
                cells.append({"run_id": run_id, **dict(cell)})
    if not runs:
        raise BenchmarkError("no supported benchmark runs were found")
    models = {str(cell.get("model", "")) for cell in cells if cell.get("model")}
    status = "completed" if all(run["status"] == "completed" for run in runs) else "failed"
    report: dict[str, Any] = {
        "schema_version": _REPORT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "summary": {
            "runs": len(runs),
            "models": len(models),
            "cases": len(cells),
            "failed_cases": sum(cell.get("status") != "completed" for cell in cells),
        },
        "runs": runs,
        "cells": cells,
    }
    if warnings:
        report["warnings"] = list(warnings)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_html_report(report, output_dir / "report.html")
    return report, tuple(warnings)


def _result_paths(roots: Sequence[Path]) -> tuple[Path, ...]:
    paths: set[Path] = set()
    for root in roots:
        path = root.expanduser().resolve()
        if path.is_file() and path.name == "result.json":
            paths.add(path)
        elif (path / "result.json").is_file():
            paths.add(path / "result.json")
        elif path.is_dir():
            paths.update(path.rglob("result.json"))
    return tuple(sorted(paths))


def _result_label(path: Path) -> str:
    return (Path(path.parent.name) / path.name).as_posix()


def _error_summary(error: OSError | json.JSONDecodeError) -> str:
    if isinstance(error, json.JSONDecodeError):
        return f"{error.msg} at line {error.lineno} column {error.colno}"
    return error.strerror or type(error).__name__
