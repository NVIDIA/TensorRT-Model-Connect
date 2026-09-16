# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-contained benchmark reports with the original measurement contracts."""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
from urllib.parse import quote

from .types import BenchmarkError


_RUN_SCHEMAS = {"trtmc.benchmark-run/v1", "trtmc.benchmark-run/v2"}
_REPORT_SCHEMA = "trtmc.benchmark-report/v2"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _escape(value: Any) -> str:
    return html.escape(str(value))


def _number(value: Any, suffix: str = "") -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "—"
    return f"{value:,.3f}{suffix}"


def _environment(value: Any) -> dict[str, Any]:
    """Display hardware/software context without machine identifiers or paths."""
    environment = _mapping(value)
    result = {key: environment[key] for key in ("python", "tensorrt", "cuda") if key in environment}
    result["gpus"] = [
        {key: gpu[key] for key in ("name", "driver_version") if key in gpu}
        for gpu in environment.get("gpus", [])
        if isinstance(gpu, Mapping)
    ]
    return result


def _scope(cell: Mapping[str, Any], run: Mapping[str, Any]) -> str:
    declared = cell.get("timing_scope") or _mapping(run.get("measurement_policy")).get(
        "timing_scope"
    )
    if declared:
        return str(declared)
    if run.get("schema_version") == "trtmc.benchmark-run/v1":
        return "public_pipeline_call_wall (legacy default)"
    return "not recorded"


def _task_rates(metrics: Mapping[str, Any]) -> str:
    rates = [
        f"{_number(value)} {_escape(name)}"
        for name, value in metrics.items()
        if name.endswith("_per_s") or name in {"realtime_factor", "seconds_per_frame"}
    ]
    return "<br>".join(rates) or "—"


def _artifact_directory(cell: Mapping[str, Any], root: Path) -> Path | None:
    raw = cell.get("artifact_dir")
    if not isinstance(raw, str) or not raw or "\\" in raw:
        return None
    relative = PurePosixPath(raw)
    if relative.is_absolute() or ".." in relative.parts or ":" in raw:
        return None
    directory = root / raw
    if not directory.resolve().is_relative_to(root.resolve()):
        return None
    return directory


def _resolved_case(directory: Path | None) -> Mapping[str, Any]:
    if directory is None:
        return {}
    try:
        if not (directory / "resolved-case.json").resolve().is_relative_to(directory.resolve()):
            return {}
        return _mapping(json.loads((directory / "resolved-case.json").read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return {}


def _evidence(directory: Path | None, output: Path) -> str:
    if directory is None:
        return "No case artifacts recorded"
    links = []
    for filename, label in (
        ("resolved-case.json", "Resolved inputs / reproduction"),
        ("observations.jsonl", "Raw measurements"),
        ("telemetry.json", "Telemetry"),
        ("worker.stdout.log", "Worker output"),
        ("worker.stderr.log", "Worker errors"),
    ):
        artifact = directory / filename
        if artifact.is_file() and artifact.resolve().is_relative_to(directory.resolve()):
            href = quote(Path(os.path.relpath(artifact, output)).as_posix(), safe="/.")
            links.append(f'<a href="{href}">{label}</a>')
    return " · ".join(links) or "No case artifacts recorded"


def _artifact_href(value: Any, directory: Path | None, output: Path, suffix: str) -> str | None:
    if directory is None or not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.suffix.lower() != suffix:
        return None
    if not path.is_absolute():
        path = directory / path
    if not path.is_file() or not path.resolve().is_relative_to(directory.resolve()):
        return None
    return quote(Path(os.path.relpath(path, output)).as_posix(), safe="/.")


def _audio_player(href: str, title: str) -> str:
    return (
        f'<p>{_escape(title)}</p><audio controls preload="none" src="{href}">'
        f'<a href="{href}">Download WAV</a></audio>'
    )


def _speech_preview(result: Mapping[str, Any], directory: Path | None, output: Path) -> str:
    events = result.get("events")
    if not isinstance(events, list):
        return ""
    content = ""
    if isinstance(result.get("system_prompt"), str):
        content += f"<p>System prompt (effective)</p><pre>{_escape(result['system_prompt'])}</pre>"
    if "input_audio_artifact" in result:
        audio = _artifact_href(result["input_audio_artifact"], directory, output, ".wav")
        content += _audio_player(audio, "Input audio") if audio else "<p>Input audio: artifact unavailable</p>"
    paths = result.get("event_audio_artifacts")
    paths = paths if isinstance(paths, list) else []
    content += '<p>Speech events (original order; scroll for longer sessions)</p><ol class="speech-events">'
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            continue
        kind = str(event.get("kind", "event")).replace("_", " ")
        label = f"{kind} · epoch {event.get('epoch', '?')} · sequence {event.get('sequence', '?')}"
        content += f"<li><p>{_escape(label)}</p>"
        if isinstance(event.get("text"), str) and event["text"]:
            content += f"<pre>{_escape(event['text'])}</pre>"
        call = _mapping(event.get("tool_call"))
        if call:
            content += (
                f"<p>Tool {_escape(call.get('name', ''))} · {_escape(call.get('call_id', ''))}"
                f" · {_escape(call.get('state', 'unknown'))}</p>"
                f"<pre>{_escape(call.get('arguments_json', ''))}</pre>"
            )
        audio = _artifact_href(paths[index] if index < len(paths) else None, directory, output, ".wav")
        if audio:
            content += _audio_player(audio, "Audio event")
        elif isinstance(event.get("audio_samples"), int) and event["audio_samples"] > 0:
            content += "<p>Audio event: artifact unavailable</p>"
        content += "</li>"
    return content + "</ol>"


def _result_preview(
    cell: Mapping[str, Any], resolved: Mapping[str, Any], directory: Path | None, output: Path
) -> str:
    if cell.get("status") != "completed":
        return ""
    result = _mapping(cell.get("output_summary"))
    content = _speech_preview(result, directory, output)
    if isinstance(result.get("text"), str):
        content += f"<p>Output text</p><pre>{_escape(result['text'])}</pre>"
    audio = _artifact_href(result.get("audio_artifact"), directory, output, ".wav")
    if audio:
        content += _audio_player(audio, "Output audio")
    elif "audio_artifact" in result and result["audio_artifact"] is None and result.get("output_samples") == 0:
        content += "<p>Output audio: empty (0 samples)</p>"
    elif "audio_artifact" in result:
        content += "<p>Output audio: artifact unavailable</p>"
    for key, title in (
        ("input_image_artifacts", "Input images"),
        ("image_artifacts", "Output images"),
        ("frame_artifacts", "Output video frames"),
    ):
        paths = result.get(key)
        if not isinstance(paths, list):
            continue
        figures = []
        times = result.get("timestamps_seconds")
        for index, value in enumerate(paths):
            href = _artifact_href(value, directory, output, ".png")
            if not href:
                continue
            caption = f"{'Frame' if key == 'frame_artifacts' else 'Image'} {index}"
            if key == "frame_artifacts" and isinstance(times, list) and index < len(times):
                timestamp = times[index]
                if not isinstance(timestamp, bool) and isinstance(timestamp, (float, int)) and math.isfinite(timestamp):
                    caption += f" · {_number(timestamp, ' s')}"
            figures.append(
                f'<figure><a href="{href}"><img loading="lazy" src="{href}" alt="{caption}"></a>'
                f"<figcaption>{caption}</figcaption></figure>"
            )
        if figures:
            content += f'<p>{title}</p><div class="media">{"".join(figures)}</div>'
        if len(figures) != len(paths):
            content += f"<p>{title}: {len(paths) - len(figures)} artifact(s) unavailable</p>"
    if not content:
        return ""
    prompt = _mapping(resolved.get("request")).get("prompt")
    if isinstance(prompt, str):
        content = f"<p>Input text</p><pre>{_escape(prompt)}</pre>" + content
    elif isinstance(prompt, list) and all(isinstance(item, str) for item in prompt):
        entries = "".join(f"<li><pre>{_escape(item)}</pre></li>" for item in prompt)
        content = f"<p>Input text</p><ol>{entries}</ol>" + content
    return content


def _history_key(
    cell: Mapping[str, Any], run: Mapping[str, Any], resolved: Mapping[str, Any]
) -> str | None:
    environment = _environment(run.get("environment"))
    model = _mapping(resolved.get("model"))
    measurement = _mapping(resolved.get("measurement"))
    checkpoint = model.get("hf_id")
    revision = model.get("hf_revision")
    preparations = _mapping(run.get("preparation")).get("bundles", [])
    managed = isinstance(preparations, list) and any(
        isinstance(item, Mapping)
        and item.get("model") == cell.get("model")
        and item.get("bundle") == resolved.get("bundle_path")
        and isinstance(item.get("build_identity"), str)
        and re.fullmatch(r"[0-9a-f]{64}", item["build_identity"]) is not None
        for item in preparations
    )
    if (
        not resolved.get("request")
        or not managed
        or not isinstance(checkpoint, str)
        or not checkpoint
        or not isinstance(revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", revision) is None
        or any(
            measurement.get(key) is None
            for key in (
                "warmup",
                "iterations",
                "telemetry",
                "telemetry_interval_ms",
                "timing_scope",
                "asset_loading_included",
            )
        )
        or not environment["gpus"]
        or not all(gpu.get("name") and gpu.get("driver_version") for gpu in environment["gpus"])
        or not environment.get("python")
        or not model.get("precision")
        or not model.get("build")
        or not run.get("started_at")
        or _scope(cell, run) == "not recorded"
    ):
        return None
    return json.dumps(
        [
            run.get("schema_version"),
            cell.get("model"),
            cell.get("name"),
            cell.get("operation"),
            resolved.get("selected_task", model.get("task")),
            _scope(cell, run),
            cell.get("asset_loading_included"),
            resolved.get("request"),
            checkpoint,
            revision,
            model.get("build"),
            model.get("precision"),
            measurement,
            environment,
            run.get("measurement_policy"),
        ],
        sort_keys=True,
    )


def _row(
    cell: Mapping[str, Any], run: Mapping[str, Any], root: Path, output: Path, history: dict
) -> str:
    metrics = _mapping(cell.get("metrics"))
    latency = _mapping(metrics.get("latency_ms"))
    directory = _artifact_directory(cell, root)
    resolved = _resolved_case(directory)
    key = _history_key(cell, run, resolved)
    p50 = latency.get("p50")
    previous = history.get(key) if key else None
    change = "—"
    if (
        isinstance(p50, (int, float))
        and not isinstance(p50, bool)
        and math.isfinite(p50)
        and p50 > 0
        and cell.get("status") == "completed"
    ):
        if previous and previous[0] != run.get("run_id"):
            change = f"{(p50 / previous[1] - 1) * 100:+.1f}% vs {_escape(previous[0])}"
        if key and cell.get("status") == "completed":
            history[key] = (run.get("run_id"), p50)
    details = _evidence(directory, output)
    details += "<p>Input preparation, loading, and warmup follow the recorded timing contract.</p>"
    policy = dict(_mapping(run.get("measurement_policy")))
    policy.update(_mapping(resolved.get("measurement")))
    details += f"<pre>{_escape(json.dumps(policy, indent=2, sort_keys=True))}</pre>"
    if resolved:
        model = _mapping(resolved.get("model"))
        reproduction = {
            "selected_task": resolved.get("selected_task", model.get("task")),
            "request": resolved.get("request"),
            "model": {
                key: model[key]
                for key in ("name", "family", "hf_id", "hf_revision", "precision", "task", "build")
                if key in model
            },
        }
        details += "<p>Recorded workload and build settings:</p>"
        details += f"<pre>{_escape(json.dumps(reproduction, indent=2, sort_keys=True))}</pre>"
    if cell.get("error"):
        details += f"<p class=failed>{_escape(cell['error'])}</p>"
    values = (
        _escape(run.get("run_id", "single run")),
        _escape(cell.get("model", "")),
        _escape(cell.get("name", "")),
        _escape(cell.get("operation", "")),
        _escape(cell.get("status", "unknown")),
        _result_preview(cell, resolved, directory, output)
        + f"<details><summary>Evidence and reproduction</summary>{details}</details>",
        _number(latency.get("p50"), " ms"),
        _number(latency.get("p95"), " ms"),
        _task_rates(metrics),
        _escape(_scope(cell, run)),
        change,
    )
    return "<tr>" + "".join(f"<td>{value}</td>" for value in values) + "</tr>"


def write_html_report(result: Mapping[str, Any], path: Path) -> None:
    runs = result.get("runs") or [result]
    run_index = {run.get("run_id"): run for run in runs if isinstance(run, Mapping)}
    history: dict[str, tuple[str, float]] = {}
    rows = []
    for cell in result.get("cells", []):
        if not isinstance(cell, Mapping):
            continue
        run = run_index.get(cell.get("run_id"), result)
        source = run.get("result_path")
        root = (path.parent / str(source)).parent if source else path.parent
        rows.append(_row(cell, run, root, path.parent, history))
    contexts = []
    for run in run_index.values():
        context = {
            "schema": run.get("schema_version", "not recorded"),
            "started_at": run.get("started_at"),
            "finished_at": run.get("finished_at"),
            "environment": _environment(run.get("environment")),
        }
        contexts.append(
            f"<details><summary>Run {_escape(run.get('run_id', 'single run'))}</summary>"
            f"<pre>{_escape(json.dumps(context, indent=2, sort_keys=True))}</pre></details>"
        )
    document = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>TRTMC benchmark</title><style>
body{font:14px system-ui,sans-serif;margin:2rem;color:#20242a}table{border-collapse:collapse;width:100%}
th,td{border-bottom:1px solid #d8dde5;padding:.6rem;text-align:left;vertical-align:top;overflow-wrap:anywhere}
td:nth-child(6){min-width:320px}td:nth-child(2),td:nth-child(3){max-width:160px}
th,td:nth-child(5),td:nth-child(7),td:nth-child(8),td:nth-child(11){white-space:nowrap}
th{background:#f4f6f8}pre{white-space:pre-wrap;overflow-wrap:anywhere}.failed{color:#a32626}
input,select{font:inherit;padding:.5rem;margin:.5rem}summary{cursor:pointer}.scroll{overflow:auto}
.media{display:flex;flex-wrap:wrap;gap:.5rem}.media figure{margin:0}.media img{width:140px;height:105px;max-width:100%;object-fit:contain}audio{max-width:100%}
.speech-events{max-height:32rem;overflow:auto;padding-left:1.5rem}.speech-events>li{margin-bottom:.4rem}.speech-events p{margin:.2rem 0;font-size:.8rem;line-height:1.25}.speech-events pre{margin:.3rem 0}
</style></head><body><h1>TRTMC benchmark</h1>"""
    document += f"<p>Status: <strong>{_escape(result.get('status', 'unknown'))}</strong></p>"
    document += """<p>These are performance measurements. Task quality is not evaluated; use the family correctness tests for model acceptance.
Loading and warmup are excluded where declared in the recorded timing contract. Legacy timing scopes retain their original meaning.</p>
<label>Filter model, case or run <input id="filter" type="search"></label>
<label>Status <select id="status"><option value="">All</option><option>completed</option><option>failed</option><option>running</option></select></label>
<div class="scroll"><table><thead><tr><th>Run</th><th>Model</th><th>Case</th><th>Operation</th><th>Status</th>
<th>Inputs / outputs and evidence</th><th>p50</th><th>p95</th><th>Task rate</th><th>Timing scope</th><th>p50 change</th></tr></thead><tbody>"""
    document += "".join(rows) + "</tbody></table></div>"
    document += "<p>p50 change compares runs only when the schema, pinned checkpoint revision, workload, build settings, measurement settings, timing policy, and recorded GPU/software fields match. Negative is faster. This is a comparison of recorded measurements, not a correctness gate. Missing comparison evidence is shown as —; different timing scopes are never combined. Runs using local checkpoints, external bundles, or older records without a managed build identity remain readable without a historical delta.</p>"
    document += "".join(contexts)
    document += """<script>
function filterRows(){const query=document.querySelector('#filter').value.toLowerCase();
const status=document.querySelector('#status').value;document.querySelectorAll('tbody tr').forEach(row=>{
row.hidden=!row.textContent.toLowerCase().includes(query)||(status&&row.cells[4].textContent!==status);});}
document.querySelector('#filter').addEventListener('input',filterRows);
document.querySelector('#status').addEventListener('change',filterRows);
</script></body></html>"""
    path.write_text(document, encoding="utf-8")


def generate_collection_report(
    roots: Sequence[Path], output_dir: Path
) -> tuple[dict[str, Any], tuple[str, ...]]:
    result_paths = _result_paths(roots)
    if not result_paths:
        raise BenchmarkError("no benchmark result.json files were found")
    runs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    warnings: list[str] = []
    cells: list[dict[str, Any]] = []
    for path in result_paths:
        try:
            raw = path.read_bytes()
            result = json.loads(raw)
        except (OSError, ValueError) as error:
            warnings.append(f"skipped unreadable result {path}: {error}")
            continue
        if not isinstance(result, Mapping) or result.get("schema_version") not in _RUN_SCHEMAS:
            warnings.append(f"skipped unsupported result {path}")
            continue
        run_id = result.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            if result["schema_version"] != "trtmc.benchmark-run/v1":
                warnings.append(f"skipped result without run_id {path}")
                continue
            run_id = "legacy-" + hashlib.sha256(raw).hexdigest()[:16]
        if run_id in seen_ids:
            raise BenchmarkError(f"duplicate run_id {run_id!r}")
        seen_ids.add(run_id)
        run_cells = result.get("cells", [])
        if not isinstance(run_cells, list) or not all(
            isinstance(cell, Mapping) for cell in run_cells
        ):
            warnings.append(f"skipped malformed cells in {path}")
            continue
        run = {
            key: result.get(key)
            for key in (
                "schema_version",
                "status",
                "started_at",
                "finished_at",
                "measurement_policy",
                "preparation",
                "environment",
            )
        }
        run.update(run_id=run_id, result_path=Path(os.path.relpath(path, output_dir)).as_posix())
        runs.append(run)
        cells.extend({**dict(cell), "run_id": run_id} for cell in run_cells)
    if not runs:
        raise BenchmarkError("no supported benchmark runs were found")
    runs.sort(key=lambda run: (str(run.get("started_at") or ""), run["run_id"]))
    run_order = {run["run_id"]: index for index, run in enumerate(runs)}
    cells.sort(key=lambda cell: run_order[cell["run_id"]])
    statuses = {run.get("status") for run in runs}
    status = (
        "completed"
        if statuses == {"completed"}
        else "failed"
        if "failed" in statuses
        else "incomplete"
    )
    report = {
        "schema_version": _REPORT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "summary": {
            "runs": len(runs),
            "models": len({cell.get("model") for cell in cells}),
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
