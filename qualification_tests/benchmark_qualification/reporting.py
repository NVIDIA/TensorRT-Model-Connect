# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-agnostic evidence views for internal Accuracy/Performance runs."""

from __future__ import annotations

import html
import json
import math
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
    names = (
        ("reference-request.json", "candidate-inputs.json")
        if reference_first
        else ("candidate-inputs.json", "reference-request.json")
    )
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
            rows.append(
                f"<tr><th>{_text(name)}</th><td>{_text(value)}</td><td>{_text(gate.get(name, ''))}</td></tr>"
            )
    for name, value in observed.items():
        if value is not None:
            rows.append(
                f"<tr><th>{_text(name)} (observed; not comparable)</th><td>{_text(value)}</td><td></td></tr>"
            )
    if not rows:
        return '<p class="muted">No valid measurement was produced.</p>'
    return (
        '<table class="metrics"><thead><tr><th>Metric</th><th>Result</th><th>Gate</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table>"
    )


def _number(value: Any, *, digits: int = 2) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return _text(value)
    return f"{value:,.{digits}f}".rstrip("0").rstrip(".") or "0"


def _status_badge(status: str) -> str:
    label = {"passed": "Passed", "failed": "Failed", "error": "Error"}.get(status, status)
    return f'<span class="status status-{_text(status)}">{_text(label)}</span>'


def _comparison_signal(status: Any) -> str:
    labels = {
        "green": "Faster than reference",
        "yellow": "Similar to reference",
        "red": "Slower than reference",
        "white": "No valid comparison",
        "contract-mismatch": "No valid comparison",
    }
    if not isinstance(status, str) or status not in labels:
        return ""
    color = status if status in {"green", "yellow", "red"} else "white"
    return (
        f'<span class="signal signal-{color}"><span class="signal-light" aria-hidden="true"></span>'
        f"{_text(labels[status])}</span>"
    )


def _headline_metrics(kind: str, result: Mapping[str, Any]) -> str:
    metrics = result.get("metrics")
    if not isinstance(metrics, Mapping):
        return ""
    if kind == "performance":
        candidate = metrics.get("candidate_p50_ms")
        reference = metrics.get("reference_p50_ms")
        if candidate is not None and reference is not None:
            return (
                '<div class="timing-pair">'
                f"<div><span>TRTMC p50</span><strong>{_number(candidate, digits=3)} ms</strong></div>"
                f"<div><span>Reference p50</span><strong>{_number(reference, digits=3)} ms</strong></div>"
                "</div>"
            )
        observed = result.get("observed_metrics")
        if isinstance(observed, Mapping) and observed.get("candidate_p50_ms") is not None:
            reference = observed.get("reference_p50_ms")
            reference_text = f"{_number(reference, digits=3)} ms" if reference is not None else "—"
            return (
                '<div class="timing-pair observed">'
                f"<div><span>TRTMC p50 (observed)</span><strong>{_number(observed['candidate_p50_ms'], digits=3)} ms</strong></div>"
                f"<div><span>Reference p50 (observed)</span><strong>{reference_text}</strong></div>"
                "<small>Not comparable: output or execution did not meet the comparison contract.</small></div>"
            )
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
            name
            for name, value in metrics.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and name not in {"samples", "pairs", "passed_samples", "passed_vectors"}
        ]
    return "".join(
        f'<div class="accuracy-fact"><span>{_text(name.replace("_", " "))}</span>'
        f"<strong>{_number(metrics[name], digits=4)}</strong></div>"
        for name in selected[:2]
    )


def _repro(result: Mapping[str, Any], dataset: Mapping[str, Any] | None) -> str:
    case_id = result.get("case")
    if not isinstance(case_id, str) or not case_id:
        return ""
    command = [
        "python3",
        "tools/model_benchmark.py",
        "run",
        "--model",
        case_id,
        "--runtime-root",
        "/path/to/trtmc-runtime",
        "--worker",
        "/path/to/trtmc_benchmark_worker",
        "--trtmc-bench",
        "/path/to/trtmc-bench",
    ]
    if isinstance(dataset, Mapping) and dataset.get("source_mode") in {
        "manual",
        "staged",
        "provided",
    }:
        dataset_id = dataset.get("id")
        if isinstance(dataset_id, str) and dataset_id:
            command.extend(["--dataset", f"{dataset_id}=/path/to/{dataset_id}"])
    return " ".join(shlex.quote(part) for part in command)


def _quick_evidence(
    result: Mapping[str, Any],
    report_html: str,
    available: set[str],
    dataset: Mapping[str, Any] | None,
) -> str:
    base = _safe_case_base(report_html)
    if base is None:
        return ""
    links = []
    logs = sorted(
        path for path in available if path.startswith(base + "/") and path.endswith(".stderr.log")
    )
    error = str(result.get("error", "")).lower()
    preferred = (
        "reference"
        if "reference" in error
        else "candidate"
        if "candidate" in error
        else "prepare"
        if "bundle" in error or "preparation" in error
        else ""
    )
    if preferred:
        logs.sort(key=lambda path: (not path.endswith(f"/{preferred}.stderr.log"), path))
    for name, label in (
        ("result.json", "result"),
        ("candidate-inputs.json", "inputs"),
        ("reference-request.json", "request"),
        ("matrix/results.json", "matrix"),
    ):
        path = f"{base}/{name}"
        if path in available:
            links.append(_link(path, label))
    if logs:
        links.append(_link(logs[0], "log"))
    samples = result.get("samples")
    sample = (
        next(
            (item for item in samples if isinstance(item, Mapping) and item.get("passed") is False),
            None,
        )
        if isinstance(samples, list)
        else None
    )
    sample_line = (
        f"<br><small>First failed sample: {_text(sample.get('sample_id', 'unknown'))}</small>"
        if isinstance(sample, Mapping)
        else ""
    )
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
    dataset = dataset or (
        result.get("dataset") if isinstance(result.get("dataset"), Mapping) else None
    )
    status = str(result.get("status", "unknown"))
    reason = _reason(result)
    links = [_link(report_html, "case report")]
    for name in (
        "result.json",
        "candidate-inputs.json",
        "reference-request.json",
        "reference.json",
        "matrix/results.json",
    ):
        path = f"{base}/{name}"
        if path in available:
            links.append(_link(path, name))
    logs = sorted(
        path for path in available if path.startswith(base + "/") and path.endswith(".stderr.log")
    )
    # Prefer the nearest case-level logs; matrix logs are still available.
    logs.sort(key=lambda path: (path.count("/"), path))
    commands = sorted(
        path for path in available if path.startswith(base + "/") and path.endswith(".command.json")
    )
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
            excerpt = re.sub(
                r"/(?:raid|runs|mnt|tmp|home)/[^\s)'\"]+", "<local-path>", preview[-1800:]
            )
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
        + (f"<p><strong>Reason (automated):</strong> {_text(reason)}</p>" if reason else "")
        + dataset_line
        + _metrics(result)
        + (f"<h4>Failed samples</h4>{sample}" if status != "passed" else "")
        + log_excerpt
        + (
            f"<p>Reproduce from the repository root (provide the required runtime/worker and dataset assets):</p><pre>{_text(repro)}</pre>"
            if repro and status != "passed"
            else ""
        )
        + f"<p>Evidence: {' · '.join(link for link in links if link)}</p>"
        + "</details>"
    )


_REPORT_CSS = Path(__file__).with_name("report.css").read_text(encoding="utf-8")


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
        row["status"] = (
            "error"
            if not statuses or "error" in statuses
            else "failed"
            if "failed" in statuses
            else "passed"
        )
        normalized.append(row)
    ordered = sorted(
        normalized, key=lambda row: (rank.get(str(row.get("status")), 0), str(row.get("model")))
    )
    body = []
    for row in ordered:
        model = str(row.get("model", ""))
        cells = []
        details = []
        for kind in ("accuracy", "performance"):
            entry = row.get(kind)
            if not isinstance(entry, Mapping):
                cells.append('<span class="not-run">Not run</span>')
                continue
            result = by_case.get(str(entry.get("case")), {})
            status = str(result.get("status", entry.get("status", "unknown")))
            metric_line = _headline_metrics(kind, result)
            report_html = entry.get("report_html")
            case_link = _link(report_html, "case report") if isinstance(report_html, str) else ""
            comparison = (
                _comparison_signal(result.get("comparison_status")) if kind == "performance" else ""
            )
            cells.append(
                f'<div class="case-status">{_status_badge(status)}{case_link}</div>'
                + (f"<div>{comparison}</div>" if comparison else "")
                + metric_line
            )
            if isinstance(report_html, str):
                detail = _case_detail(
                    result or entry,
                    report_html,
                    available,
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
        result = str(row.get("status", "unknown"))
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
                quick.append(
                    f"<p><strong>{_text(kind)}:</strong> {_quick_evidence(result_case, entry['report_html'], available, dataset)}</p>"
                )
        report_target = row.get("report_html")
        if not isinstance(report_target, str):
            report_target = next(
                (
                    entry["report_html"]
                    for kind in ("accuracy", "performance")
                    if isinstance((entry := row.get(kind)), Mapping)
                    and isinstance(entry.get("report_html"), str)
                ),
                "",
            )
        detail_cell = (
            "".join(quick + details) if result != "passed" else _link(report_target, "details")
        )
        checkpoint = (
            f'<details class="checkpoint"><summary>Checkpoint</summary><code>{_text(row.get("checkpoint", ""))}</code><br><small>{_text(row.get("checkpoint_revision", ""))}</small></details>'
            if row.get("checkpoint") or row.get("checkpoint_revision")
            else ""
        )
        search_text = " ".join(
            str(value.get("benchmark", ""))
            for kind in ("accuracy", "performance")
            if isinstance((value := row.get(kind)), Mapping)
        )
        body.append(
            f'<tr class="{_text(result)}" data-status="{_text(result)}" data-search="{_text(model + " " + search_text)}">'
            f'<td>{_status_badge(result)}<div class="result-reason">{_text(reason[:260])}</div></td>'
            f'<td class="model-name">{_text(model)}</td><td>{cells[0]}</td><td>{cells[1]}</td>'
            f'<td class="evidence-links">{detail_cell}{checkpoint}</td></tr>'
        )
    counts = defaultdict(int)
    for row in normalized:
        counts[str(row.get("status", "unknown"))] += 1
    comparisons = defaultdict(int)
    for row in normalized:
        entry = row.get("performance")
        if not isinstance(entry, Mapping):
            continue
        result = by_case.get(str(entry.get("case")), entry)
        status = result.get("comparison_status")
        comparisons[status if status in {"green", "yellow", "red"} else "white"] += 1
    performance_count = sum(comparisons.values())
    model_count_label = f"{len(normalized)} model{'s' if len(normalized) != 1 else ''}"
    performance_count_label = f"{performance_count} case{'s' if performance_count != 1 else ''}"
    performance_summary = (
        f'<div class="summary-card"><span class="summary-label">Performance vs reference · {performance_count_label}</span>'
        f'<div class="summary-values"><span>{_comparison_signal("green")} <strong>{comparisons["green"]}</strong></span>'
        f"<span>{_comparison_signal('yellow')} <strong>{comparisons['yellow']}</strong></span>"
        f"<span>{_comparison_signal('red')} <strong>{comparisons['red']}</strong></span>"
        f"<span>{_comparison_signal('white')} <strong>{comparisons['white']}</strong></span></div>"
        "<small>Lights indicate relative latency after a valid output comparison; red (slower) is not a qualification failure.</small></div>"
        if performance_count
        else ""
    )
    has_accuracy = any(isinstance(row.get("accuracy"), Mapping) for row in normalized)
    purpose = (
        "Accuracy agreement and Performance against the model reference."
        if has_accuracy and performance_count
        else "Accuracy agreement against the model reference."
        if has_accuracy
        else "Performance against the model reference."
    )
    meta = " · ".join(f"{_text(key)}: {_text(value)}" for key, value in (metadata or {}).items())
    return f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>{_text(title)}</title>
<style>{_REPORT_CSS}</style>
<header class="report-header"><p class="eyebrow">Validation report</p><h1>{_text(title)}</h1><p class="purpose">{purpose}</p><p class="meta">{meta}</p></header>
<section class="outcome-strip{" single" if not performance_count else ""}" aria-label="Run summary">
<div class="summary-card"><span class="summary-label">Qualification · {model_count_label}</span><div class="summary-values"><span>{_status_badge("passed")} <strong>{counts["passed"]}</strong></span><span>{_status_badge("failed")} <strong>{counts["failed"]}</strong></span><span>{_status_badge("error")} <strong>{counts["error"]}</strong></span></div><small>{model_count_label} · {counts["passed"]} passed · {counts["failed"]} failed · {counts["error"]} error. Error means execution or evidence was incomplete, not a model gate failure.</small></div>
{performance_summary}
</section>
<p class="intro">Errors and failures appear first. Reasons come from automated checks or execution logs, never an agent judgment; some rows show a derived sample count or generic fallback. Open evidence for samples, logs, commands and detailed metrics.</p>
<section class="register"><div class="register-head"><h2>Complete qualification register</h2><div class="filters"><input id="model-search" type="search" placeholder="Search model or benchmark" aria-label="Search model or benchmark"><select id="status-filter" aria-label="Filter qualification result"><option value="">All results</option><option value="error">Error</option><option value="failed">Failed</option><option value="passed">Passed</option></select><span class="filter-count" id="filter-count">Showing {len(normalized)} of {len(normalized)}</span></div></div>
<div class="table-wrap"><table class="register-table"><thead><tr><th>Result / reason</th><th>Model</th><th>Accuracy</th><th>Performance</th><th>Evidence / checkpoint</th></tr></thead><tbody id="model-rows">{"".join(body)}</tbody></table></div></section>
<p class="footer"><a href="report.json">report.json</a> contains the machine-readable qualification results and original metric precision.</p>
<script>(()=>{{const search=document.getElementById('model-search'),status=document.getElementById('status-filter'),rows=[...document.querySelectorAll('#model-rows tr')],count=document.getElementById('filter-count');function filter(){{const query=search.value.trim().toLowerCase();let shown=0;for(const row of rows){{row.hidden=!!((query&&!row.dataset.search.toLowerCase().includes(query))||(status.value&&row.dataset.status!==status.value));if(!row.hidden)shown++;}}count.textContent=`Showing ${{shown}} of ${{rows.length}}`;}}search.addEventListener('input',filter);status.addEventListener('change',filter);}})();</script></html>"""


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
            "status": case.get("status"),
            "case": case_id,
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
                available.update(
                    path.relative_to(output).as_posix()
                    for path in case_dir.rglob("*")
                    if path.is_file()
                )
    for case in summary.get("cases", []):
        if case.get("status") == "passed":
            continue
        case_id = str(case.get("case", ""))
        base = _safe_case_base(f"{case_id}/report.html")
        if base is None:
            continue
        case_dir = output / base
        loaded = _read_case_inputs(
            case_dir, reference_first="reference" in str(case.get("error", "")).lower()
        )
        if loaded:
            inputs[case_id] = loaded
        for path in case_dir.rglob("*.stderr.log") if case_dir.is_dir() else ():
            if path.is_file() and path.stat().st_size:
                try:
                    previews[path.relative_to(output).as_posix()] = tail_log(path)
                except OSError:
                    pass
    document = render_combined_report(
        local_rows(summary),
        summary.get("cases", []),
        available=available,
        inputs=inputs,
        log_previews=previews,
    )
    (output / "report.html").write_text(document, encoding="utf-8")


def write_case_report(output: Path, result: Mapping[str, Any]) -> None:
    available = {
        f"case/{path.relative_to(output).as_posix()}"
        for path in output.rglob("*")
        if path.is_file()
    }
    record = dict(result)
    record.setdefault("kind", "unknown")
    inputs = _read_case_inputs(
        output, reference_first="reference" in str(result.get("error", "")).lower()
    )
    previews = {}
    for path in output.rglob("*.stderr.log"):
        if path.is_file() and path.stat().st_size:
            try:
                previews[f"case/{path.relative_to(output).as_posix()}"] = tail_log(path)
            except OSError:
                pass
    detail = _case_detail(
        record, "case/report.html", available, inputs=inputs, log_previews=previews
    )
    # Case pages are rooted inside the case directory, not its parent.
    detail = detail.replace('href="case/', 'href="')
    document = f'<!doctype html><meta charset="utf-8"><title>{_text(result.get("case", "qualification"))}</title><h1>{_text(result.get("case", "qualification"))}</h1>{detail}<p><a href="result.json">result.json</a></p>'
    (output / "report.html").write_text(document, encoding="utf-8")
