# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-contained HTML reports for benchmark runs and result collections."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from html import escape
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote
import uuid

from .types import BenchmarkError


_RUN_SCHEMA = "trtmc.benchmark-run/v1"
_REPORT_SCHEMA = "trtmc.benchmark-report/v1"


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


def generate_collection_report(
    roots: Sequence[Path], output_dir: Path
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Discover benchmark runs below *roots* and atomically rebuild one report."""
    resolved_roots = tuple(_validate_root(root) for root in roots)
    if not resolved_roots:
        raise BenchmarkError("provide at least one benchmark result directory")
    output_dir = output_dir.expanduser().resolve()
    if _is_run_result_directory(output_dir):
        raise BenchmarkError(
            "refusing to replace a single-run report; scan its parent collection directory "
            "or use -o with a separate report directory"
        )
    _prepare_report_output(output_dir)
    runs, warnings = _load_runs(resolved_roots, output_dir)
    if not runs:
        roots_text = ", ".join(str(root) for root in resolved_roots)
        raise BenchmarkError(f"no {_RUN_SCHEMA} result.json files found under {roots_text}")

    cells = [cell for run in runs for cell in run["cells"]]
    statuses = {str(run.get("status", "unknown")) for run in runs}
    status = (
        "failed"
        if "failed" in statuses
        else "incomplete"
        if statuses != {"completed"}
        else "completed"
    )
    models = sorted(
        {
            str(cell["model"])
            for cell in cells
            if isinstance(cell, Mapping) and isinstance(cell.get("model"), str)
        }
    )
    report: dict[str, Any] = {
        "schema_version": _REPORT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "summary": {
            "runs": len(runs),
            "models": len(models),
            "cases": len(cells),
            "completed_runs": sum(run.get("status") == "completed" for run in runs),
            "failed_runs": sum(run.get("status") == "failed" for run in runs),
            "incomplete_runs": sum(
                run.get("status") not in {"completed", "failed"} for run in runs
            ),
        },
        "models": models,
        "runs": runs,
    }
    if warnings:
        report["warnings"] = list(warnings)
    _atomic_write_json(output_dir / "report.json", report)
    _atomic_write_text(output_dir / "report.html", _collection_html(report))
    return report, tuple(warnings)


def _validate_root(root: Path) -> Path:
    expanded = root.expanduser()
    if expanded.is_symlink():
        raise BenchmarkError(f"refusing to scan symlink result directory: {root}")
    resolved = expanded.resolve()
    if not resolved.is_dir():
        raise BenchmarkError(f"benchmark result directory does not exist: {root}")
    return resolved


def _prepare_report_output(output_dir: Path) -> None:
    if output_dir.is_symlink():
        raise BenchmarkError(f"refusing to write report into symlink directory: {output_dir}")
    if output_dir.exists() and not output_dir.is_dir():
        raise BenchmarkError(f"report output exists and is not a directory: {output_dir}")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise BenchmarkError(f"cannot create report output directory {output_dir}: {exc}") from exc


def _is_run_result_directory(path: Path) -> bool:
    try:
        result = json.loads((path / "result.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(result, Mapping) and result.get("schema_version") == _RUN_SCHEMA


def _result_paths(roots: Sequence[Path]) -> tuple[Path, ...]:
    paths: set[Path] = set()
    for root in roots:
        for current, directories, filenames in os.walk(root, followlinks=False):
            current_path = Path(current)
            directories[:] = [
                name
                for name in directories
                if name != ".incoming" and not (current_path / name).is_symlink()
            ]
            if "result.json" in filenames:
                candidate = current_path / "result.json"
                if not candidate.is_symlink():
                    paths.add(candidate)
    return tuple(sorted(paths))


def _load_runs(roots: Sequence[Path], output_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    runs: list[dict[str, Any]] = []
    warnings: list[str] = []
    identities: dict[str, tuple[str, Path]] = {}
    for path in _result_paths(roots):
        try:
            raw = path.read_bytes()
            result = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"skipped unreadable result {path}: {exc}")
            continue
        if not isinstance(result, Mapping) or result.get("schema_version") != _RUN_SCHEMA:
            warnings.append(f"skipped unsupported result {path}")
            continue
        cells = result.get("cells")
        if not isinstance(cells, list) or not all(isinstance(cell, Mapping) for cell in cells):
            warnings.append(f"skipped malformed benchmark result {path}: cells must be a list")
            continue
        digest = hashlib.sha256(
            json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        declared_id = result.get("run_id")
        legacy = not isinstance(declared_id, str) or not declared_id.strip()
        run_id = f"legacy-{digest[:24]}" if legacy else declared_id.strip()
        previous = identities.get(run_id)
        if previous is not None:
            previous_digest, previous_path = previous
            if previous_digest != digest:
                raise BenchmarkError(
                    f"run_id collision for {run_id}: {previous_path} and {path} have different content"
                )
            continue
        identities[run_id] = (digest, path)
        source = Path(os.path.relpath(path, output_dir)).as_posix()
        run: dict[str, Any] = {
            "run_id": run_id,
            "result_sha256": digest,
            "result_path": source,
            "status": str(result.get("status", "unknown")),
            "started_at": result.get("started_at"),
            "finished_at": result.get("finished_at"),
            "measurement_policy": result.get("measurement_policy", {}),
            "environment": result.get("environment", {}),
            "cells": [dict(cell) for cell in cells],
        }
        if legacy:
            run["legacy_run_id"] = True
        runs.append(run)
    runs.sort(key=lambda run: (str(run.get("started_at") or ""), run["run_id"]))
    return runs, warnings


def _collection_html(report: Mapping[str, Any]) -> str:
    summary = report.get("summary", {})
    runs = report.get("runs", [])
    rows: list[str] = []
    run_rows: list[str] = []
    for run in runs if isinstance(runs, list) else []:
        if not isinstance(run, Mapping):
            continue
        run_id = str(run.get("run_id", "unknown"))
        result_path = str(run.get("result_path", ""))
        link = f'<a href="{escape(quote(result_path), quote=True)}">result.json</a>'
        environment = run.get("environment", {})
        hostname = environment.get("hostname", "—") if isinstance(environment, Mapping) else "—"
        gpu = _gpu_name(environment)
        run_rows.append(
            "<tr>"
            f"<td><code>{escape(run_id[:12])}</code></td>"
            f"<td>{escape(str(run.get('started_at') or '—'))}</td>"
            f"<td>{escape(str(hostname))}</td>"
            f"<td>{escape(gpu)}</td>"
            f'<td class="{escape(str(run.get("status", "unknown")))}">'
            f"{escape(str(run.get('status', 'unknown')))}</td><td>{link}</td></tr>"
        )
        for cell in run.get("cells", []):
            if isinstance(cell, Mapping):
                rows.append(_collection_cell_row(run_id, result_path, cell))
    status = escape(str(report.get("status", "unknown")))
    warning_items = "".join(f"<li>{escape(str(item))}</li>" for item in report.get("warnings", []))
    warnings_html = f"<h2>Warnings</h2><ul>{warning_items}</ul>" if warning_items else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TRTMC benchmark collection</title>
  <style>
    body {{ font: 14px system-ui, sans-serif; margin: 2rem; color: #20242a; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; }}
    th, td {{ border-bottom: 1px solid #d8dde5; padding: .6rem; text-align: left; }}
    th {{ background: #f4f6f8; }}
    .completed {{ color: #19713b; }} .failed {{ color: #a32626; }}
    .incomplete, .running {{ color: #8a5a00; }}
    code {{ background: #f4f6f8; padding: .1rem .25rem; }}
  </style>
</head>
<body>
  <h1>TRTMC benchmark collection</h1>
  <p>Status: <strong class="{status}">{status}</strong>.
     Runs: {summary.get("runs", 0)}; models: {summary.get("models", 0)};
     cases: {summary.get("cases", 0)}.</p>
  <h2>Models and cases</h2>
  <table>
    <thead><tr><th>Run</th><th>Model</th><th>Case</th><th>Operation</th><th>p50</th><th>p95</th><th>Task rate</th><th>Status</th><th>Evidence</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
  <h2>Source runs</h2>
  <table>
    <thead><tr><th>Run</th><th>Started</th><th>Host</th><th>GPU</th><th>Status</th><th>Result</th></tr></thead>
    <tbody>{"".join(run_rows)}</tbody>
  </table>
  {warnings_html}
  <p>Machine-readable collection index: <code>report.json</code>. Each source
     <code>result.json</code> remains the authoritative benchmark evidence.</p>
</body>
</html>
"""


def _gpu_name(environment: Any) -> str:
    if not isinstance(environment, Mapping):
        return "—"
    gpus = environment.get("gpus")
    if not isinstance(gpus, list):
        return "—"
    names = [
        str(gpu["name"])
        for gpu in gpus
        if isinstance(gpu, Mapping) and isinstance(gpu.get("name"), str)
    ]
    return ", ".join(names) or "—"


def _collection_cell_row(run_id: str, result_path: str, cell: Mapping[str, Any]) -> str:
    metrics = cell.get("metrics", {})
    metrics = metrics if isinstance(metrics, Mapping) else {}
    latency = metrics.get("latency_ms", {})
    latency = latency if isinstance(latency, Mapping) else {}
    artifact_dir = cell.get("artifact_dir")
    evidence = result_path
    if isinstance(artifact_dir, str) and artifact_dir:
        evidence = (Path(result_path).parent / artifact_dir).as_posix()
    link = f'<a href="{escape(quote(evidence), quote=True)}">artifacts</a>'
    values = (
        f"<code>{escape(run_id[:12])}</code>",
        escape(str(cell.get("model", ""))),
        escape(str(cell.get("name", ""))),
        escape(str(cell.get("operation", ""))),
        escape(_number(latency.get("p50"), " ms")),
        escape(_number(latency.get("p95"), " ms")),
        escape(_task_rate(metrics)),
        escape(str(cell.get("status", "unknown"))),
        link,
    )
    return "<tr>" + "".join(f"<td>{value}</td>" for value in values) + "</tr>"


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.trtmc-bench-{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise BenchmarkError(f"cannot write benchmark report {path}: {exc}") from exc


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
