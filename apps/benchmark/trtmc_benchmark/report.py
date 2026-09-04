# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small JSON and HTML reports for benchmark runs."""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .types import BenchmarkError


_RUN_SCHEMA = "trtmc.benchmark-run/v2"
_REPORT_SCHEMA = "trtmc.benchmark-report/v2"


def write_html_report(result: Mapping[str, Any], path: Path) -> None:
    cells = result.get("cells", [])
    rows = []
    for cell in cells if isinstance(cells, list) else []:
        if not isinstance(cell, Mapping):
            continue
        metrics = cell.get("metrics", {})
        latency = metrics.get("latency_ms", {}) if isinstance(metrics, Mapping) else {}
        p50 = latency.get("p50") if isinstance(latency, Mapping) else None
        detail = (
            f"{float(p50):.3f} ms"
            if isinstance(p50, (int, float)) and not isinstance(p50, bool)
            else html.escape(str(cell.get("error", "")))
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(cell.get('model', '')))}</td>"
            f"<td>{html.escape(str(cell.get('name', '')))}</td>"
            f"<td>{html.escape(str(cell.get('operation', '')))}</td>"
            f"<td>{html.escape(str(cell.get('status', '')))}</td>"
            f"<td>{detail}</td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>TRTMC benchmark</title>
<style>
body {{ font: 14px system-ui, sans-serif; margin: 2rem; color: #222; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: .5rem; text-align: left; }}
th {{ background: #f3f3f3; }}
</style></head><body>
<h1>TRTMC benchmark</h1>
<p>Status: {html.escape(str(result.get("status", "unknown")))}</p>
<table><thead><tr><th>Model</th><th>Case</th><th>Operation</th><th>Status</th><th>p50 / error</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
</body></html>
"""
    path.write_text(document, encoding="utf-8")


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
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            warnings.append(f"skipped unreadable result {path}: {error}")
            continue
        if not isinstance(result, Mapping) or result.get("schema_version") != _RUN_SCHEMA:
            warnings.append(f"skipped unsupported result {path}")
            continue
        run_id = result.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            warnings.append(f"skipped result without run_id {path}")
            continue
        if run_id in seen_ids:
            raise BenchmarkError(
                f"duplicate run_id {run_id!r}: {seen_ids[run_id]} and {path}"
            )
        seen_ids[run_id] = path
        run_cells = result.get("cells", [])
        if not isinstance(run_cells, list):
            warnings.append(f"skipped malformed cells in {path}")
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
