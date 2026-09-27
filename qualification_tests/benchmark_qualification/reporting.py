# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-agnostic evidence views for internal Accuracy/Performance runs."""

from __future__ import annotations

import html
import json
import re
import shlex
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
from urllib.parse import quote


def _text(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _link(path: str, label: str) -> str:
    candidate = PurePosixPath(path)
    if not path or candidate.is_absolute() or ".." in candidate.parts:
        return ""
    return f'<a href="{_text(quote(path, safe="/.-_"))}">{_text(label)}</a>'


def _safe_case_base(report_html: Any) -> str | None:
    if not isinstance(report_html, str) or not report_html.endswith("/report.html"):
        return None
    path = PurePosixPath(report_html)
    if path.is_absolute() or ".." in path.parts:
        return None
    return str(path.parent)


def _portable_input(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("/"):
        return f"<staged asset: {Path(value).name}>"
    if isinstance(value, Mapping):
        return {str(key): _portable_input(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_input(item) for item in value]
    return value


def output_preview(value: Any) -> Any:
    """Bound a generic runner output summary without interpreting its semantics."""
    if isinstance(value, str):
        if value.startswith("/"):
            return f"<run asset: {Path(value).name}>"
        return value[:600] + ("…" if len(value) > 600 else "")
    if isinstance(value, Mapping):
        return {str(key): output_preview(item) for key, item in list(value.items())[:12]}
    if isinstance(value, list):
        return [output_preview(item) for item in value[:64]]
    return value


def tail_log(path: Path, limit: int = 1800) -> str:
    with path.open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - limit * 4))
        return stream.read().decode("utf-8", errors="replace")[-limit:]


def _sample_preview(result: Mapping[str, Any], inputs: Sequence[Mapping[str, Any]] = ()) -> str:
    samples = result.get("samples")
    if not isinstance(samples, list):
        samples = []
    failed = [item for item in samples if isinstance(item, Mapping) and item.get("passed") is False]
    if not failed:
        comparison = result.get("comparison_evidence")
        if isinstance(comparison, Mapping) and comparison:
            preview = json.dumps(comparison, ensure_ascii=False, indent=2, default=str)
            return f"<pre>{_text(preview[:2400])}</pre>"
        return ""
    by_id = {item.get("sample_id"): item for item in inputs if isinstance(item, Mapping)}
    parts = [f"<p>{len(failed)} failed / {len(samples)} recorded samples; first 3 shown.</p>"]
    for item in failed[:3]:
        sample_id = item.get("sample_id")
        details: dict[str, Any] = {"result": item}
        if sample_id in by_id:
            details["input"] = _portable_input(by_id[sample_id].get("request", by_id[sample_id]))
        preview = json.dumps(details, ensure_ascii=False, indent=2, default=str)
        if len(preview) > 2400:
            preview = preview[:2400] + "\n… (see result.json for full evidence)"
        parts.append(f"<pre>{_text(preview)}</pre>")
    return "".join(parts)


def _attempted_input_preview(inputs: Sequence[Mapping[str, Any]]) -> str:
    if not inputs:
        return ""
    preview = json.dumps(_portable_input(inputs[0]), ensure_ascii=False, indent=2, default=str)
    if len(preview) > 2400:
        preview = preview[:2400] + "\n… (see request artifact for full input)"
    return f"<p>First attempted input; no output was recorded.</p><pre>{_text(preview)}</pre>"


def _read_case_inputs(output: Path, *, reference_first: bool) -> Sequence[Mapping[str, Any]]:
    names = ("reference-request.json", "candidate-inputs.json") if reference_first else ("candidate-inputs.json", "reference-request.json")
    for name in names:
        path = output / name
        if not path.is_file():
            continue
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(loaded, Mapping):
            loaded = loaded.get("samples")
        if isinstance(loaded, list):
            return loaded
    return ()


def _reason(result: Mapping[str, Any]) -> str:
    error = result.get("error")
    if isinstance(error, str) and error:
        # Historic errors often end with an absolute worker path. The log link
        # below replaces that non-portable location in the human-facing view.
        reason = error.split("; see /", 1)[0]
        return re.sub(r"/(?:raid|runs|mnt|tmp|home)/[^\s)'\"]+", "<local-path>", reason)
    comparison = result.get("comparison_reason")
    if isinstance(comparison, str) and comparison:
        return comparison
    if result.get("comparison_status") == "contract-mismatch":
        return "Candidate/reference output contract mismatch"
    samples = result.get("samples")
    if isinstance(samples, list):
        failed = sum(isinstance(item, Mapping) and item.get("passed") is False for item in samples)
        if failed:
            return f"{failed}/{len(samples)} recorded samples did not pass"
    if result.get("status") == "failed":
        return "Accuracy/Performance gate not met; inspect metrics and raw result"
    return ""


def _metrics(result: Mapping[str, Any]) -> str:
    metrics = result.get("metrics")
    gate = result.get("gate")
    if not isinstance(metrics, Mapping):
        metrics = {}
    if not isinstance(gate, Mapping):
        gate = {}
    observed = result.get("observed_metrics")
    if not isinstance(observed, Mapping):
        observed = {}
    rows = []
    for name, value in metrics.items():
        if value is not None:
            rows.append(f"<tr><th>{_text(name)}</th><td>{_text(value)}</td><td>{_text(gate.get(name, ''))}</td></tr>")
    for name, value in observed.items():
        if value is not None:
            rows.append(f"<tr><th>{_text(name)} (observed; not comparable)</th><td>{_text(value)}</td><td></td></tr>")
    if not rows:
        return '<p class="muted">No valid measurement was produced.</p>'
    return '<table class="metrics"><thead><tr><th>Metric</th><th>Result</th><th>Gate</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table>"


def _headline_metrics(kind: str, result: Mapping[str, Any]) -> str:
    metrics = result.get("metrics")
    if not isinstance(metrics, Mapping):
        return ""
    if kind == "performance":
        candidate = metrics.get("candidate_p50_ms")
        reference = metrics.get("reference_p50_ms")
        if candidate is not None and reference is not None:
            return f"<br><small>p50 TRTMC/HF: {_text(candidate)} / {_text(reference)} ms</small>"
        observed = result.get("observed_metrics")
        if isinstance(observed, Mapping) and observed.get("candidate_p50_ms") is not None:
            return f"<br><small>Observed p50 TRTMC/HF: {_text(observed['candidate_p50_ms'])} / {_text(observed.get('reference_p50_ms', '—'))} ms; not comparable</small>"
        return ""
    gate = result.get("gate")
    selected = []
    if isinstance(gate, Mapping):
        for name in gate:
            metric = name.removeprefix("min_").removeprefix("max_")
            if name in metrics:
                metric = name
            if metric in metrics and metrics[metric] is not None and metric not in selected:
                selected.append(metric)
    if not selected:
        selected = [
            name for name, value in metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            and name not in {"samples", "pairs", "passed_samples", "passed_vectors"}
        ]
    return "".join(f"<br><small>{_text(name)}: {_text(metrics[name])}</small>" for name in selected[:2])


def _repro(result: Mapping[str, Any], dataset: Mapping[str, Any] | None) -> str:
    case_id = result.get("case")
    if not isinstance(case_id, str) or not case_id:
        return ""
    command = [
        "python3", "tools/model_benchmark.py", "run", "--model", case_id,
        "--runtime-root", "/path/to/trtmc-runtime",
        "--worker", "/path/to/trtmc_benchmark_worker",
        "--trtmc-bench", "/path/to/trtmc-bench",
    ]
    if isinstance(dataset, Mapping) and dataset.get("source_mode") in {"manual", "staged", "provided"}:
        dataset_id = dataset.get("id")
        if isinstance(dataset_id, str) and dataset_id:
            command.extend(["--dataset", f"{dataset_id}=/path/to/{dataset_id}"])
    return " ".join(shlex.quote(part) for part in command)


def _quick_evidence(
    result: Mapping[str, Any], report_html: str, available: set[str], dataset: Mapping[str, Any] | None
) -> str:
    base = _safe_case_base(report_html)
    if base is None:
        return ""
    links = []
    logs = sorted(path for path in available if path.startswith(base + "/") and path.endswith(".stderr.log"))
    error = str(result.get("error", "")).lower()
    preferred = "reference" if "reference" in error else "candidate" if "candidate" in error else "prepare" if "bundle" in error or "preparation" in error else ""
    if preferred:
        logs.sort(key=lambda path: (not path.endswith(f"/{preferred}.stderr.log"), path))
    for name, label in (("result.json", "result"), ("candidate-inputs.json", "inputs"), ("reference-request.json", "request"), ("matrix/results.json", "matrix")):
        path = f"{base}/{name}"
        if path in available:
            links.append(_link(path, label))
    if logs:
        links.append(_link(logs[0], "log"))
    samples = result.get("samples")
    sample = next(
        (item for item in samples if isinstance(item, Mapping) and item.get("passed") is False),
        None,
    ) if isinstance(samples, list) else None
    sample_line = f"<br><small>First failed sample: {_text(sample.get('sample_id', 'unknown'))}</small>" if isinstance(sample, Mapping) else ""
    repro = _repro(result, dataset)
    command = f"<br><small>Repro: <code>{_text(repro)}</code></small>" if repro else ""
    return f"<small>{' · '.join(link for link in links if link)}</small>{sample_line}{command}"


def _case_detail(
    result: Mapping[str, Any],
    report_html: str,
    available: set[str],
    *,
    dataset: Mapping[str, Any] | None = None,
    inputs: Sequence[Mapping[str, Any]] = (),
    log_previews: Mapping[str, str] | None = None,
) -> str:
    base = _safe_case_base(report_html)
    if base is None:
        return ""
    dataset = dataset or (result.get("dataset") if isinstance(result.get("dataset"), Mapping) else None)
    status = str(result.get("status", "unknown"))
    reason = _reason(result)
    links = [_link(report_html, "case report")]
    for name in ("result.json", "candidate-inputs.json", "reference-request.json", "reference.json", "matrix/results.json"):
        path = f"{base}/{name}"
        if path in available:
            links.append(_link(path, name))
    logs = sorted(path for path in available if path.startswith(base + "/") and path.endswith(".stderr.log"))
    # Prefer the nearest case-level logs; matrix logs are still available.
    logs.sort(key=lambda path: (path.count("/"), path))
    commands = sorted(path for path in available if path.startswith(base + "/") and path.endswith(".command.json"))
    links.extend(_link(path, path.removeprefix(base + "/")) for path in (logs[:8] + commands[:4]))
    dataset_line = ""
    if dataset:
        dataset_line = f"<p>Dataset: {_text(dataset.get('id', 'unknown'))} · source: {_text(dataset.get('source_mode', 'unknown'))} · SHA-256: {_text(dataset.get('sha256', 'not recorded'))}</p>"
        if isinstance(dataset.get("instructions"), str):
            dataset_line += f"<p>Dataset preparation: {_text(dataset['instructions'])}</p>"
    log_excerpt = ""
    for path in logs:
        preview = (log_previews or {}).get(path)
        if isinstance(preview, str) and preview.strip():
            excerpt = re.sub(r"/(?:raid|runs|mnt|tmp|home)/[^\s)'\"]+", "<local-path>", preview[-1800:])
            log_excerpt = f"<p>Log excerpt ({_text(path.removeprefix(base + '/'))}):</p><pre>{_text(excerpt)}</pre>"
            break
    repro = _repro(result, dataset)
    sample = _sample_preview(result, inputs)
    if not sample and status == "error":
        sample = _attempted_input_preview(inputs)
    if not sample and status != "passed":
        sample = '<p class="muted">No sample-level output was recorded for this case. Check raw artifacts and logs.</p>'
    heading = f"{result.get('kind', 'case')} · {result.get('benchmark', '')} · {status}"
    return (
        f'<details class="case"><summary>{_text(heading)}</summary>'
        + (f"<p><strong>Reason:</strong> {_text(reason)}</p>" if reason else "")
        + dataset_line + _metrics(result)
        + (f"<h4>Failed samples</h4>{sample}" if status != "passed" else "")
        + log_excerpt
        + (f"<p>Reproduce from the repository root (provide the required runtime/worker and dataset assets):</p><pre>{_text(repro)}</pre>" if repro and status != "passed" else "")
        + f"<p>Evidence: {' · '.join(link for link in links if link)}</p>"
        + "</details>"
    )


def render_combined_report(
    rows: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
    *,
    available: set[str],
    title: str = "Internal model benchmark",
    datasets: Mapping[str, Mapping[str, Any]] | None = None,
    inputs: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    log_previews: Mapping[str, str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    by_case = {str(item.get("case")): item for item in cases}
    rank = {"error": 0, "failed": 1, "passed": 2}
    normalized = []
    for original in rows:
        row = dict(original)
        statuses = [
            str(by_case.get(str(entry.get("case")), entry).get("status"))
            for kind in ("accuracy", "performance")
            if isinstance((entry := row.get(kind)), Mapping)
        ]
        row["status"] = "error" if "error" in statuses else "failed" if "failed" in statuses else "passed"
        normalized.append(row)
    ordered = sorted(normalized, key=lambda row: (rank.get(str(row.get("status")), 0), str(row.get("model"))))
    body = []
    for row in ordered:
        model = str(row.get("model", ""))
        cells = []
        details = []
        for kind in ("accuracy", "performance"):
            entry = row.get(kind)
            if not isinstance(entry, Mapping):
                cells.append("—")
                continue
            result = by_case.get(str(entry.get("case")), {})
            status = str(result.get("status", entry.get("status", "unknown")))
            metric_line = _headline_metrics(kind, result)
            report_html = entry.get("report_html")
            case_link = _link(report_html, status) if isinstance(report_html, str) else _text(status)
            cells.append(f'<span class="{_text(status)}">{case_link}</span>{metric_line}')
            if isinstance(report_html, str):
                detail = _case_detail(
                    result or entry, report_html, available,
                    dataset=(datasets or {}).get(str(entry.get("benchmark"))),
                    inputs=(inputs or {}).get(str(entry.get("case")), ()),
                    log_previews=log_previews,
                )
                if detail:
                    details.append(detail)
        reason = "; ".join(
            f"{kind}: {_reason(by_case[str(entry.get('case'))])}"
            for kind in ("accuracy", "performance")
            if isinstance((entry := row.get(kind)), Mapping)
            and str(entry.get("case")) in by_case
            and by_case[str(entry.get("case"))].get("status") != "passed"
        )
        result = _text(row.get("status", "unknown"))
        quick = []
        if row.get("status") != "passed":
            for kind in ("accuracy", "performance"):
                entry = row.get(kind)
                if not isinstance(entry, Mapping) or not isinstance(entry.get("report_html"), str):
                    continue
                result_case = by_case.get(str(entry.get("case")), entry)
                if result_case.get("status") == "passed":
                    continue
                dataset = (datasets or {}).get(str(entry.get("benchmark")))
                if dataset is None and isinstance(result_case.get("dataset"), Mapping):
                    dataset = result_case["dataset"]
                quick.append(f"<p><strong>{_text(kind)}:</strong> {_quick_evidence(result_case, entry['report_html'], available, dataset)}</p>")
        detail_cell = "".join(quick + details) if row.get("status") != "passed" else _link(str(row.get("report_html", "")), "details")
        checkpoint = f"<details><summary>Checkpoint</summary><code>{_text(row.get('checkpoint', ''))}</code><br><small>{_text(row.get('checkpoint_revision', ''))}</small></details>"
        body.append(
            f'<tr class="{result}"><td><strong>{_text(model)}</strong></td><td>{cells[0]}</td><td>{cells[1]}</td>'
            f'<td>{result}<br><small>{_text(reason[:260])}</small></td><td>{detail_cell}{checkpoint}</td></tr>'
        )
    counts = defaultdict(int)
    for row in normalized:
        counts[str(row.get("status", "unknown"))] += 1
    meta = " · ".join(f"{_text(key)}: {_text(value)}" for key, value in (metadata or {}).items())
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>{_text(title)}</title>
<style>body{{font:14px/1.45 system-ui,sans-serif;margin:2rem;max-width:1600px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:.5rem;vertical-align:top;text-align:left}}th{{background:#eee;position:sticky;top:0}}tr.error{{background:#fff0ed}}tr.failed{{background:#fff9eb}}.passed{{color:#167348}}.failed{{color:#895b00}}.error{{color:#ad2828}}.case{{margin:.4rem 0;min-width:20rem}}.case summary{{cursor:pointer}}.metrics{{width:auto;margin:.5rem 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f5f5;padding:.6rem;max-height:22rem;overflow:auto}}small,.muted{{color:#555}}td:nth-child(5){{max-width:36rem}}</style>
<h1>{_text(title)}</h1><p>{meta}</p><p>{len(normalized)} models · {counts['passed']} passed · {counts['failed']} failed · {counts['error']} error. Errors mean execution/evidence was incomplete, not a model gate failure.</p>
<p>Failures and errors are listed first. Accuracy and Performance remain side by side. Expand an evidence cell for metrics, failed samples, logs, and a high-level repro command. Checkpoint metadata is available in its collapsed section.</p>
<table><thead><tr><th>Model</th><th>Accuracy</th><th>Performance</th><th>Result / reason</th><th>Evidence / checkpoint</th></tr></thead><tbody>{''.join(body)}</tbody></table>
<p><a href="report.json">report.json</a> contains the machine-readable qualification results.</p></html>'''


def local_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for case in summary.get("cases", []):
        model = str(case.get("model", ""))
        kind = str(case.get("kind", ""))
        if kind not in {"accuracy", "performance"}:
            continue
        row = rows.setdefault(model, {"model": model, "status": "passed"})
        case_id = str(case.get("case", ""))
        name = case_id.rsplit("/", 1)[-1]
        row[kind] = {
            "status": case.get("status"), "case": case_id,
            "benchmark": case.get("benchmark"),
            "report_html": f"{model}/{kind}/{name}/report.html",
        }
        if case.get("status") == "error":
            row["status"] = "error"
        elif case.get("status") == "failed" and row["status"] != "error":
            row["status"] = "failed"
    return list(rows.values())


def write_local_summary(output: Path, summary: Mapping[str, Any]) -> None:
    (output / "report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    available = set()
    inputs = {}
    previews = {}
    for row in local_rows(summary):
        for kind in ("accuracy", "performance"):
            entry = row.get(kind)
            if not isinstance(entry, Mapping):
                continue
            base = _safe_case_base(entry.get("report_html"))
            if base is None:
                continue
            case_dir = output / base
            if case_dir.is_dir():
                available.update(path.relative_to(output).as_posix() for path in case_dir.rglob("*") if path.is_file())
    for case in summary.get("cases", []):
        if case.get("status") == "passed":
            continue
        case_id = str(case.get("case", ""))
        base = _safe_case_base(f"{case_id}/report.html")
        if base is None:
            continue
        case_dir = output / base
        loaded = _read_case_inputs(case_dir, reference_first="reference" in str(case.get("error", "")).lower())
        if loaded:
            inputs[case_id] = loaded
        for path in case_dir.rglob("*.stderr.log") if case_dir.is_dir() else ():
            if path.is_file() and path.stat().st_size:
                try:
                    previews[path.relative_to(output).as_posix()] = tail_log(path)
                except OSError:
                    pass
    document = render_combined_report(local_rows(summary), summary.get("cases", []), available=available, inputs=inputs, log_previews=previews)
    (output / "report.html").write_text(document, encoding="utf-8")


def write_case_report(output: Path, result: Mapping[str, Any]) -> None:
    available = {f"case/{path.relative_to(output).as_posix()}" for path in output.rglob("*") if path.is_file()}
    record = dict(result)
    record.setdefault("kind", "unknown")
    inputs = _read_case_inputs(output, reference_first="reference" in str(result.get("error", "")).lower())
    previews = {}
    for path in output.rglob("*.stderr.log"):
        if path.is_file() and path.stat().st_size:
            try:
                previews[f"case/{path.relative_to(output).as_posix()}"] = tail_log(path)
            except OSError:
                pass
    detail = _case_detail(record, "case/report.html", available, inputs=inputs, log_previews=previews)
    # Case pages are rooted inside the case directory, not its parent.
    detail = detail.replace('href="case/', 'href="')
    document = f'<!doctype html><meta charset="utf-8"><title>{_text(result.get("case", "qualification"))}</title><h1>{_text(result.get("case", "qualification"))}</h1>{detail}<p><a href="result.json">result.json</a></p>'
    (output / "report.html").write_text(document, encoding="utf-8")
