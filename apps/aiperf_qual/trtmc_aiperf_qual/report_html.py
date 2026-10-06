# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-contained, failure-first HTML report over result roots (``summary --html``).

One row per model: its Task, the Acc result of every benchmark and the Perf light of every timed request side
by side, the category with its first reason, and the evidence to expand (Acc gates and failing samples with the
TRTMC and native outputs side by side, the Perf comparison per reference mode, links to the evidence files next
to its result, and a reproduction command). Rows are ordered errors and failed gates first.
"""

from __future__ import annotations

import html
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .report import counted, values

EVIDENCE = ("report.md", "report.json", "build.json", "build/build.log", "error.json", "phase-errors.log",
            "candidate/server.log")
LIGHT_COLORS = {"green": "#1a7f37", "yellow": "#9a6700", "red": "#cf222e", "white": "#6e7781", "n/a": "#6e7781"}
STYLE = """
body{font:14px/1.45 system-ui,sans-serif;margin:2rem;max-width:1700px;color:#1f2328}h1{font-size:22px}
table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:.45rem;text-align:left;
vertical-align:top}th{background:#eee;position:sticky;top:0}tr.error{background:#fff0ed}tr.failed{background:#fff9eb}
tr.warn{background:#fbfbf2}.cat{font-weight:600}.fail{color:#ad2828}.pass{color:#167348}.warn{color:#895b00}
details{margin:4px 0}summary{cursor:pointer}code,pre{font:12px ui-monospace,monospace}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f5f5;padding:.5rem;margin:4px 0;max-height:22rem;
overflow:auto}.light{display:inline-block;padding:0 6px;border-radius:8px;color:#fff;font-size:12px}
small,.muted{color:#555}td.evidence{min-width:14rem}td.evidence table{width:auto}td.reason{max-width:30rem}
.counts{width:auto}#q{width:320px;padding:4px;margin:8px 0}
"""
SCRIPT = """
function f(){const q=document.getElementById('q').value.toLowerCase();
for(const r of document.querySelectorAll('tr.m')){r.style.display=r.dataset.k.includes(q)?'':'none'}}
"""


def _e(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def _css_class(category: str) -> str:
    if category in ("pass",):
        return "pass"
    if category in ("error", "config-error", "build-failed", "not-run", "acc-issue", "acc-session-state", "perf-issue"):
        return "fail"
    return "warn"


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
        rows.append(f"<tr><td>{_e(item.get('reference_mode'))}</td><td><span class='light' "
                    f"style='background:{color}'>{_e(light)}</span></td>"
                    f"<td>{_e(_ms(candidate.get('p50_ms')))}</td><td>{_e(_ms(reference.get('p50_ms')))}"
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


def _row_class(category: str) -> str:
    if category in ("error", "config-error", "build-failed", "not-run"):
        return "error"
    return {"fail": "failed", "warn": "warn"}.get(_css_class(category), "")


def _status_class(status: str) -> str:
    return "pass" if status == "pass" else "fail" if status in ("fail", "error") else "warn"


def _accuracy_cell(items: Sequence[Mapping[str, Any]]) -> str:
    """Each benchmark's status and both sides' values (or the passes of a parity check)."""
    lines = [f'<span class="{_status_class(str(item.get("status", "")))}">{_e(item.get("suite"))}: '
             f'{_e(item.get("status", "—"))}</span>' + (" <small>(informational)</small>" if item.get("informational")
                                                         else "") + f"<br><small>{_e(values(item))}</small>"
             for item in items]
    return "<br>".join(lines) or '<span class="muted">—</span>'


def _time(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    return f"{value:.0f} ms" if value >= 100 else f"{value:.1f} ms" if value >= 1 else f"{value:.3f} ms"


def _performance_cell(items: Sequence[Mapping[str, Any]]) -> str:
    """Each timed request's light and both sides' server model-call time (p50)."""
    lines = []
    for item in items:
        light = item.get("light", "")
        candidate, reference = item.get("candidate") or {}, item.get("reference") or {}
        lines.append(f"<span class='light' style='background:{LIGHT_COLORS.get(light, '#6e7781')}'>{_e(light)}</span> "
                     f"<small>{_e(item.get('request') or item.get('reference_mode') or '')}</small><br>"
                     f"TRTMC {_e(_time(candidate.get('p50_ms')))} · native {_e(_time(reference.get('p50_ms')))}"
                     + (f" <small>({_e(reference['precision'])})</small>" if reference.get("precision") else ""))
    return "<br>".join(lines) or '<span class="muted">—</span>'


def _plain(text: str) -> str:
    """A reason without the server's JSON error envelope: its message only."""
    return re.sub(r'\{"error":\{"message":"((?:[^"\\]|\\.)*)".*?\}\}', r"\1", text)


def _reason(row: Mapping[str, Any]) -> str:
    """The first reason the category is not a pass: a failed build or run, a benchmark, then a timed request
    (its verdict only; the times are in the Performance column)."""
    if row.get("category") == "pass":
        return ""
    found = [str(item.get("error") or "; ".join(item.get("reasons", []))) for item in row.get("accuracy", [])
             if item.get("status") != "pass" and not item.get("informational")]
    found += [f"{item.get('request') or item.get('reference_mode')}: {item['reasons'][0].split(':')[0]}"
              for item in row.get("perf", []) if item.get("light") != "green" and item.get("reasons")]
    found = [text for text in [row.get("notes", ""), *found] if text]
    return _plain(found[0])[:300] if found else ""


def render(rows: Mapping[str, Mapping[str, Any]], counts: Mapping[str, int], rank: Mapping[str, int],
           output: Path, title: str = "TRTMC vs native qualification", context: str = "",
           links: Sequence[tuple[str, str]] = ()) -> Path:
    base = output.parent.resolve()
    order = sorted(rows, key=lambda p: (rank.get(rows[p]["category"], 99), rows[p].get("task") or "", p))
    ranked = sorted(counts, key=lambda c: rank.get(c, 99))
    summary = "".join(f"<tr><td class='{_css_class(c)} cat'>{_e(c)}</td><td>{counts[c]}</td></tr>" for c in ranked)
    tally = " · ".join(f"{counts[c]} {_e(c)}" for c in ranked)
    related = " · ".join(f'<a href="{_e(href)}">{_e(label)}</a>' for label, href in links)
    body = []
    for profile in order:
        row = rows[profile]
        directory = row.get("directory")
        details = (f"<details><summary>evidence</summary>{_accuracy(row.get('accuracy', []))}"
                   f"{_performance(row.get('perf', []))}{_sweep(row.get('l2') or {})}"
                   f"<p>{_links(Path(directory) if directory else None, base)}</p>"
                   + (f"<p>reproduce: <code>{_e(row['repro'])}</code></p>" if row.get("repro") else "")
                   + "</details>")
        key = f"{profile} {row.get('task') or ''} {row['category']} {row.get('root')}".lower()
        reason = _reason(row)
        body.append(f"<tr class='m {_row_class(row['category'])}' data-k='{_e(key)}'><td><b>{_e(profile)}</b>"
                    f"<br><small>{_e(row.get('root'))}</small></td><td>{_e(row.get('task') or '-')}</td>"
                    f"<td>{_accuracy_cell(row.get('accuracy', []))}</td><td>{_performance_cell(row.get('perf', []))}</td>"
                    f"<td class='reason'><span class='{_css_class(row['category'])} cat'>{_e(row['category'])}</span>"
                    + (f"<br><small>{_e(reason)}</small>" if reason else "")
                    + f"</td><td class='evidence'>{details}</td></tr>")
    document = (f"<!doctype html><meta charset='utf-8'><title>{_e(title)}</title><style>{STYLE}</style>"
                f"<script>{SCRIPT}</script><h1>{_e(title)}</h1>"
                + (f"<p>{_e(context)}</p>" if context else "")
                + (f"<p>{related}</p>" if related else "")
                + f"<p>{len(rows)} models · {tally}.</p>"
                "<p>Errors and failed gates first. Acc compares TRTMC with the native model on the same problems (a "
                "paired non-inferiority test against each benchmark's margin, or a parity tolerance); Perf lights compare "
                "their server model-call times (green faster, yellow similar, red slower, white not comparable), and a "
                "light never changes the Acc outcome. <i>error</i> and <i>build-failed</i> mean the run produced no "
                "verdict, not a failed gate. Expand <i>evidence</i> for gates, failing samples with both outputs, Perf "
                "details, files, and the reproduction command.</p>"
                f"<table class='counts'>{summary}</table><input id='q' placeholder='filter (model, Task, category, host)' "
                "oninput='f()'><table><thead><tr><th>Model</th><th>Task</th><th>Accuracy (TRTMC, native)</th><th>Performance "
                "(server model-call time p50)</th><th>Result / reason</th><th>Evidence</th></tr></thead><tbody>"
                f"{''.join(body)}</tbody></table>")
    output.write_text(document)
    return output
