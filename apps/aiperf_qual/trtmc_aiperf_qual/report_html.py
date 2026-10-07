# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-contained, failure-first HTML report over result roots (``summary --html``).

One row per model with data only: its result (White, Red, Yellow, Green: ``campaign.signal``) and a short label
for a result that is not Green, both sides' precision, each benchmark's values, both sides' p50 of the catalog
request, a rerun's result when an appendix is linked, and the evidence to expand (the full reason, Acc gates and
failing samples with both outputs, the Perf comparison per reference mode, links to the evidence files, and a
reproduction command). Rows are ordered White, Red, Yellow, Green.
"""

from __future__ import annotations

import html
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .campaign import NO_VERDICT, SIGNAL_NAMES, SIGNALS, ms, reported_perf, request_label, signal, signal_reason
from .report import _fmt, counted

EVIDENCE = ("report.md", "report.json", "build.json", "build/build.log", "error.json", "phase-errors.log",
            "candidate/server.log")
LIGHT_COLORS = {"green": "#1e8e3e", "yellow": "#b06000", "red": "#c5221f", "white": "#5f6368", "n/a": "#5f6368"}
LEGEND = (("green", "Accuracy and performance meet their gates."),
          ("yellow", "Pass: performance within the margin of native, or an accuracy difference not shown either way."),
          ("red", "Accuracy or performance worse than native beyond the margin."),
          ("white", "No valid comparison: a build, run, or environment error, or results that cannot be compared."))
STYLE = """
:root{--bg:#f3f5f2;--panel:#fff;--raised:#f8faf7;--line:rgba(24,39,26,.12);--line-strong:rgba(24,39,26,.22);
--text:#18201a;--muted:#667069;--link:#315500;font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",
sans-serif}*{box-sizing:border-box}body{margin:0;padding:28px;color:var(--text);background:var(--bg);font-size:14px;
line-height:1.45}a,summary{color:var(--link)}summary{cursor:pointer;font-weight:650}
code,pre{font-family:"SFMono-Regular",Consolas,"Liberation Mono",monospace;font-size:12px}
.eyebrow{margin:0 0 6px;color:#5c9600;font-size:11px;font-weight:750;letter-spacing:.13em;text-transform:uppercase}
h1{margin:0 0 5px;font-size:28px;line-height:1.2}.purpose{margin:0;color:var(--muted);font-size:15px}
.meta{margin:6px 0;color:var(--muted);font-size:12px;overflow-wrap:anywhere}
.strip{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin:16px 0 12px}
.card{display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center;padding:11px 14px;border:1px solid var(--line-strong);
border-radius:10px;background:var(--panel)}.card-label{width:100%;color:var(--muted);font-size:11px;font-weight:750;
letter-spacing:.05em;text-transform:uppercase}.card-item{display:inline-flex;align-items:center;gap:6px;white-space:nowrap}
.card-item strong{font-size:15px;font-variant-numeric:tabular-nums}
.legend{margin:0 0 14px;padding:8px 14px;border:1px solid var(--line);border-radius:10px;background:var(--panel)}
.legend div{display:grid;grid-template-columns:90px 1fr;gap:12px;align-items:baseline;padding:3px 0}
.legend dt,.legend dd{margin:0}.legend dd{color:#3f4a42;font-size:13px}
.signal{display:inline-flex;align-items:center;gap:7px;font-weight:650;white-space:nowrap}
.light{width:10px;height:10px;flex-shrink:0;border-radius:50%;background:#80868b;box-shadow:0 0 0 3px #eef0f1}
.signal-green{color:#137333}.signal-green .light{background:#1e8e3e;box-shadow:0 0 0 3px #e6f4ea}
.signal-yellow{color:#8a4f00}.signal-yellow .light{background:#f9ab00;box-shadow:0 0 0 3px #fef7e0}
.signal-red{color:#b3261e}.signal-red .light{background:#d93025;box-shadow:0 0 0 3px #fce8e6}
.signal-white{color:#5f6368}.signal-white .light{background:#fff;border:2px solid #80868b}
.filters{display:flex;flex-wrap:wrap;gap:10px;align-items:end;margin:0 0 12px;padding:10px 12px;
border:1px solid var(--line-strong);border-radius:10px;background:var(--panel)}
.filters label{display:grid;gap:4px;color:var(--muted);font-size:11px;font-weight:750;text-transform:uppercase}
.filters input,.filters select{min-height:32px;padding:5px 9px;border:1px solid var(--line-strong);border-radius:6px;
background:#fff;font:inherit}.filters input{min-width:260px}.count{margin-left:auto;color:var(--muted);font-size:12px}
.wrap{overflow:auto;border:1px solid var(--line-strong);border-radius:10px;background:var(--panel)}
table{width:100%;border-collapse:separate;border-spacing:0}.register{min-width:1240px}
th,td{padding:8px 10px;border-right:1px solid var(--line);border-bottom:1px solid var(--line);text-align:left;
vertical-align:top}th:last-child,td:last-child{border-right:0}
th{position:sticky;top:0;color:#3f4a42;background:#edf1eb;font-size:12px;font-weight:750;white-space:nowrap}
.register>tbody>tr:nth-child(even){background:var(--raised)}.register>tbody>tr:hover{background:#f2f8ea}
.detail{margin-top:3px;color:var(--muted);font-size:11px}
.side{display:flex;justify-content:space-between;gap:10px;white-space:nowrap;font-size:12px}
.side span,.metric span{color:var(--muted)}.side strong{font-family:"SFMono-Regular",Consolas,monospace;font-weight:600}
.metric{display:flex;justify-content:space-between;gap:14px;font-size:12px;font-variant-numeric:tabular-nums}
.metric strong{font-weight:600;white-space:nowrap}.metric.fail strong{color:#b3261e}
.timing{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}.none{color:var(--muted)}
.evidence-body{min-width:min(760px,70vw);padding-top:6px}.evidence-body table{width:auto}
.evidence-body th{position:static}.badge{display:inline-block;padding:0 6px;border-radius:3px;color:#fff;
font-size:11px}.pass{color:#137333}.fail{color:#b3261e}pre{max-height:22rem;overflow:auto;margin:4px 0;
padding:8px;border:1px solid var(--line);border-radius:6px;background:#f4f6f3;white-space:pre-wrap;
overflow-wrap:anywhere}
@media(max-width:760px){body{padding:16px 12px}.strip{grid-template-columns:1fr}
.legend div{grid-template-columns:80px 1fr}.filters input{min-width:0;width:100%}.count{margin-left:0}}
"""
SCRIPT = """
function f(){const q=document.getElementById('q').value.toLowerCase(),s=document.getElementById('s').value;
let n=0;for(const r of document.querySelectorAll('tr.m')){const v=r.dataset.k.includes(q)&&(!s||r.dataset.result===s);
r.style.display=v?'':'none';n+=v}document.getElementById('n').textContent=n}
"""
NO_VERDICT_LABELS = {"error": "run error", "build-failed": "build failed", "config-error": "configuration error",
                     "not-run": "not run", "excluded": "excluded", "smoke-fail": "smoke failed"}
TIMING_SPREAD = re.compile(r"^(TRTMC|native) CI (±[\d.]+%)")
NO_WORK = re.compile(r"^(TRTMC|native): (\d+ responses report no work|no work evidence)")


def _e(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def _signal(result: str) -> str:
    return f"<span class='signal signal-{result}'><span class='light'></span>{_e(SIGNAL_NAMES[result])}</span>"


def _perf_label(reason: str) -> str:
    """A white Perf light's reason as a short label; the full reason stays in the evidence."""
    spread = TIMING_SPREAD.match(reason)
    if spread:
        return f"{spread[1]} CI {spread[2]}"
    if reason.startswith("reference timed at"):
        return "precision differs"
    if reason.startswith(("output check failed", "work differs")):
        return "outputs differ"
    if reason.startswith("not the same workload"):
        return "different workload"
    if NO_WORK.match(reason):
        return "work not reported"
    return reason.split(":")[0].split(" (")[0][:40] or "not comparable"


def _issue(profile: str, row: Mapping[str, Any], result: str) -> str:
    """What makes a result other than Green, in a few words."""
    if result == "green":
        return ""
    accuracy = [item for item in row.get("accuracy", []) if not item.get("informational")]
    if row["category"] in NO_VERDICT:
        status = re.search(r"\b([45]\d\d) (?:Client|Server) Error", str(row.get("notes", "")))
        errors = [str(item.get("error") or "") for item in accuracy if item.get("status") == "error"]
        return (f"HTTP {status[1]}" if status else "no problem fits the bundle" if any(
            text.startswith("no ") and "fits the bundle" in text for text in errors)
            else "Acc incomplete" if errors else NO_VERDICT_LABELS.get(row["category"], row["category"]))
    perf = reported_perf(row)
    lights = {item.get("light") for item in perf}
    parts = []
    if result == "red":  # a gold-scored benchmark is below native; a parity check (no gold) is out of tolerance
        failed = [item for item in accuracy if item.get("status") == "fail"]
        parts += ["Acc below native" if "trtmc_score" in (item.get("metrics") or {}) else "Acc outside tolerance"
                  for item in failed]
        parts += ["Acc failed"] if row["category"] == "acc-issue" and not failed else []
        parts += ["Perf slower than native"] if "red" in lights else []
    elif result == "white":
        parts += ["Acc not applicable" for item in accuracy if item.get("status") in ("not-comparable", "not-covered")]
        parts += [f"Perf {_perf_label(str((item.get('reasons') or [''])[0]))}" for item in perf
                  if item.get("light") == "white"]
        parts += ["Perf not measured"] if row["category"] == "perf-inconclusive" and not perf else []
    else:
        parts += ["Acc inconclusive" for item in accuracy if item.get("status") == "inconclusive"]
        parts += ["Perf within margin"] if "yellow" in lights else []
    return " · ".join(dict.fromkeys(parts)) or row["category"].replace("-", " ")


def _precision(row: Mapping[str, Any]) -> str:
    precision = row.get("precision") or {}
    if not any(precision.values()):
        return "<span class='none'>—</span>"
    return "".join(f"<div class='side'><span>{side}</span><strong>{_e(precision.get(key) or '—')}</strong></div>"
                   for side, key in (("Native", "native"), ("TRTMC", "trtmc")))


def _accuracy_values(items: Sequence[Mapping[str, Any]]) -> str:
    """Each judged benchmark: TRTMC / native values, or the share within tolerance; informational checks stay in
    the evidence."""
    lines = []
    for item in items:
        if item.get("informational"):
            continue
        metrics = item.get("metrics") or {}
        if item.get("status") == "error":
            value = "—"
        elif "trtmc_score" in metrics:
            value = f"{_fmt(metrics['trtmc_score'], 2)} / {_fmt(metrics['native_score'], 2)}"
        elif item.get("required_passes") is not None:
            value = f"{item.get('passed')}/{item.get('samples') or item['required_passes']}"
        else:
            value = counted(item)
        tone = " fail" if item.get("status") == "fail" else ""
        lines.append(f"<div class='metric{tone}'><span>{_e(item.get('suite'))}</span><strong>{_e(value)}</strong></div>")
    return "".join(lines) or "<span class='none'>—</span>"


def _latency(profile: str, items: Sequence[Mapping[str, Any]], side: str) -> str:
    """One side's p50 of each reported request (labelled when there are several)."""
    lines = []
    for item in items:
        timing = item.get(side) or {}
        unit = " / audio s" if timing.get("unit") or (item.get("candidate") or {}).get("unit") else ""
        label = f"<span class='detail'>{_e(request_label(profile, item))}</span> " if len(items) > 1 else ""
        lines.append(f"<div>{label}{_e(ms(timing.get('p50_ms')))}{unit}</div>")
    return "".join(lines) or "<span class='none'>—</span>"


def _links(directory: Path | None, base: Path) -> str:
    if directory is None or not directory.is_dir():
        return ""
    found = [name for name in EVIDENCE if (directory / name).is_file()]
    found += sorted(str(path.relative_to(directory)) for path in directory.glob("family-*/**/result.json"))
    found += sorted(str(path.relative_to(directory)) for path in directory.glob("family-*/**/error.log"))
    return " · ".join(f'<a href="{_e(os.path.relpath(directory / name, base))}">{_e(name)}</a>' for name in found)


def _accuracy(items: Sequence[Mapping[str, Any]]) -> str:
    parts = []
    for item in items:
        status = item.get("status", "")
        head = (f'<b>{_e(item.get("suite"))}</b> <span class="{"pass" if status == "pass" else "fail"}">{_e(status)}'
                f'</span> {_e(counted(item))}'
                f' · gate {_e(json.dumps(item.get("gate", {})))}'
                + (f' · {_e(item.get("benchmark"))} (family case)' if item.get("source") == "family" else "")
                + (f' · {_e(item.get("benchmark"))} (AIPerf, gold answers)' if item.get("source") == "absolute" else "")
                + (f' · isolated {_e(item["isolated_check"].get("status"))}' if item.get("isolated_check") else "")
                + (" · informational (not judged)" if item.get("informational") else "")
                + (" · precision-sensitive" if item.get("precision_sensitive") else "")
                + (" · sampled" if item.get("sampled") else ""))
        rows = "".join(
            f"<tr><td>{_e(f.get('sample_id') or f.get('conversation_id'))}</td><td>{_e(f.get('explanation'))}</td>"
            f"<td><pre>{_e(f.get('actual'))}</pre></td><td><pre>{_e(f.get('expected'))}</pre></td></tr>"
            for f in item.get("failures", []))
        error = "".join(f"<pre>{_e(text)}</pre>" for text in (item.get("error"), "; ".join(item.get("reasons", [])),
                                                              "; ".join(item.get("notes", []))) if text)
        table = (f"<table><tr><th>sample</th><th>reason</th><th>TRTMC</th><th>native</th></tr>{rows}</table>"
                 if rows else "")
        parts.append(f"<div>{head}{error}{table}</div>")
    return "".join(parts)


def _performance(items: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for item in items:
        candidate, reference = item.get("candidate", {}), item.get("reference", {})
        light = item.get("light", "")
        color = LIGHT_COLORS.get(light, "#6e7781")
        reasons = "; ".join([*item.get("reasons", []), *item.get("notes", [])])
        unit = " per audio second" if candidate.get("unit") or reference.get("unit") else ""
        rows.append(f"<tr><td>{_e(item.get('reference_mode'))}</td><td><span class='badge' "
                    f"style='background:{color}'>{_e(light)}</span></td>"
                    f"<td>{_e(_ms(candidate.get('p50_ms')))}{unit}</td><td>{_e(_ms(reference.get('p50_ms')))}{unit}"
                    f" {_e(reference.get('precision') or '')}</td><td>{_e(reasons)}</td></tr>")
    if not rows:
        return ""
    return ("<table><tr><th>native mode</th><th>light</th><th>TRTMC p50 ms</th><th>native p50 ms</th>"
            f"<th>reasons / notes</th></tr>{''.join(rows)}</table>")


def _media_sweep(l2: Mapping[str, Any]) -> str:
    rows = "".join(f"<tr><td>{'TRTMC' if side == 'candidate' else 'native eager'}</td><td>{_e(level.get('steps') or 'catalog')}"
                   f"</td><td>{_e(_ms(level.get('model_call_p50_ms')))}</td><td>{_e(_ms(level.get('request_latency_p50')))}"
                   f"</td><td>{_e(_ms(level.get('peak_memory_mb')))}</td></tr>"
                   for side in ("candidate", "reference") for level in l2.get(side, []))
    parts = "; ".join(f"{'TRTMC' if side == 'candidate' else 'native'} {value['per_step_ms']:.1f} ms/step + "
                      f"{value['fixed_ms']:.1f} ms fixed" for side, value in (l2.get("decomposition") or {}).items()
                      if value)
    return (f"<p>L2 {_e(l2.get('endpoint'))} (informational): light {_e(l2.get('light'))} "
            f"{_e('; '.join(l2.get('reasons', [])))} {_e(parts)}</p><table><tr><th>side</th><th>steps</th>"
            f"<th>model call p50 ms</th><th>request latency p50 ms</th><th>peak GPU memory MiB</th></tr>{rows}</table>")


def _sweep(l2: Mapping[str, Any]) -> str:
    if not l2:
        return ""
    if l2.get("kind") == "media":
        return _media_sweep(l2)
    rows = "".join(f"<tr><td>{'TRTMC' if side == 'candidate' else 'native eager'}</td><td>{_e(level.get('concurrency'))}"
                   f"</td><td>{_e(_ms(level.get('request_throughput_avg')))}</td>"
                   f"<td>{_e(_ms(level.get('request_latency_p50')))}</td><td>{_e(_ms(level.get('request_latency_p99')))}</td>"
                   "</tr>" for side in ("candidate", "reference") for level in l2.get(side, []))
    return (f"<p>L2 serving sweep (informational, ISL {_e(l2.get('isl'))} / OSL {_e(l2.get('osl'))}): light "
            f"{_e(l2.get('light'))} {_e('; '.join(l2.get('reasons', [])))} — {_e(l2.get('note'))}</p><table><tr><th>side</th>"
            f"<th>concurrency</th><th>requests/s</th><th>latency p50 ms</th><th>latency p99 ms</th></tr>{rows}</table>")


def _ms(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "—"


def _evidence(profile: str, row: Mapping[str, Any], base: Path) -> str:
    directory = row.get("directory")
    return ("<details><summary>Evidence</summary><div class='evidence-body'>"
            f"<p>{_e(signal_reason(profile, row))}</p><p class='detail'>harness category {_e(row['category'])} · "
            f"host {_e(row.get('root'))}</p>{_accuracy(row.get('accuracy', []))}{_performance(row.get('perf', []))}"
            f"{_sweep(row.get('l2') or {})}<p>{_links(Path(directory) if directory else None, base)}</p>"
            + (f"<p>reproduce: <code>{_e(row['repro'])}</code></p>" if row.get("repro") else "")
            + "</div></details>")


def render(rows: Mapping[str, Mapping[str, Any]], counts: Mapping[str, int], rank: Mapping[str, int],
           output: Path, title: str = "TRTMC vs native qualification", context: str = "",
           links: Sequence[tuple[str, str]] = (), reruns: Mapping[str, str] | None = None) -> Path:
    """``reruns``: profile -> its result in a linked appendix (its own column)."""
    base, reruns = output.parent.resolve(), reruns or {}
    results = {profile: signal(row) for profile, row in rows.items()}
    order = sorted(rows, key=lambda p: (SIGNALS.index(results[p]), rows[p].get("task") or "", p))
    tally = {result: sum(1 for value in results.values() if value == result) for result in SIGNALS}
    body = []
    for profile in order:
        row, result = rows[profile], results[profile]
        perf = reported_perf(row)
        issue = _issue(profile, row, result)
        key = f"{profile} {row.get('task') or ''} {result} {row.get('root')}".lower()
        rerun = f"<td>{_signal(reruns[profile]) if profile in reruns else ''}</td>" if reruns else ""
        body.append(f"<tr class='m' data-result='{result}' data-k='{_e(key)}'><td>{_signal(result)}"
                    + (f"<div class='detail'>{_e(issue)}</div>" if issue else "")
                    + f"</td><td><code>{_e(profile)}</code><div class='detail'>{_e(row.get('task') or '—')}</div></td>"
                    f"<td>{_precision(row)}</td><td>{_accuracy_values(row.get('accuracy', []))}</td>"
                    f"<td class='timing'>{_latency(profile, perf, 'reference')}</td>"
                    f"<td class='timing'>{_latency(profile, perf, 'candidate')}</td>{rerun}"
                    f"<td>{_evidence(profile, row, base)}</td></tr>")
    cards = ("<section class='strip'><div class='card'><span class='card-label'>Results</span>"
             + "".join(f"<span class='card-item'>{_signal(result)}<strong>{tally[result]}</strong></span>"
                       for result in ("green", "yellow", "red", "white"))
             + "</div><div class='card'><span class='card-label'>Coverage</span>"
             f"<span class='card-item'>Models <strong>{len(rows)}</strong></span>"
             f"<span class='card-item'>Pass (Green + Yellow) <strong>{tally['green'] + tally['yellow']}</strong></span>"
             f"<span class='card-item'>Valid comparisons <strong>{len(rows) - tally['white']} / {len(rows)}"
             "</strong></span></div></section>")
    legend = "<dl class='legend'>" + "".join(f"<div><dt>{_signal(result)}</dt><dd>{_e(text)}</dd></div>"
                                             for result, text in LEGEND) + "</dl>"
    related = " · ".join(f'<a href="{_e(href)}">{_e(label)}</a>' for label, href in links)
    filters = ("<div class='filters'><label>Search<input id='q' type='search' oninput='f()'></label><label>Result"
               "<select id='s' onchange='f()'><option value=''>All</option>"
               + "".join(f"<option value='{result}'>{SIGNAL_NAMES[result]}</option>" for result in SIGNALS)
               + f"</select></label><span class='count'>Showing <span id='n'>{len(rows)}</span> of {len(rows)}"
               "</span></div>")
    columns = ["Result", "Model / Task", "Precision", "Accuracy (TRTMC / native)", "Native p50", "TRTMC p50",
               *(["Rerun"] if reruns else []), "Evidence"]
    document = ("<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' "
                f"content='width=device-width,initial-scale=1'><title>{_e(title)}</title><style>{STYLE}</style>"
                f"<script>{SCRIPT}</script><header><p class='eyebrow'>Qualification report</p><h1>{_e(title)}</h1>"
                "<p class='purpose'>TRTMC against the native model: benchmark accuracy, and the server model-call time "
                "p50 of the catalog request.</p>"
                + (f"<p class='meta'>{_e(context)}</p>" if context else "")
                + (f"<p class='meta'>{related}</p>" if related else "")
                + f"</header>{cards}{legend}{filters}<div class='wrap'><table class='register'><thead><tr>"
                + "".join(f"<th>{column}</th>" for column in columns)
                + f"</tr></thead><tbody>{''.join(body)}</tbody></table></div></html>")
    output.write_text(document)
    return output
