# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-contained, failure-first HTML report over result roots (``summary --html``).

One row per model: its result (White, Red, Yellow, Green: ``campaign.signal``), its Task, both sides' Acc values
and Perf times side by side, one plain reason, and the evidence to expand (Acc gates and failing samples with the
TRTMC and native outputs side by side, the Perf comparison per reference mode, links to the evidence files next
to its result, and a reproduction command). Rows are ordered White, Red, Yellow, Green.
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .campaign import SIGNAL_NAMES, SIGNALS, ms, reported_perf, request_label, signal, signal_reason
from .report import acc_value, counted

EVIDENCE = ("report.md", "report.json", "build.json", "build/build.log", "error.json", "phase-errors.log",
            "candidate/server.log")
LIGHT_COLORS = {"green": "#1a7f37", "yellow": "#9a6700", "red": "#cf222e", "white": "#6e7781", "n/a": "#6e7781"}
STYLE = """
body{font:14px/1.45 system-ui,sans-serif;margin:2rem;max-width:1700px;color:#1f2328}h1{font-size:22px}
table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:.45rem;text-align:left;
vertical-align:top}th{background:#eee;position:sticky;top:0}
.signal{display:inline-flex;align-items:center;gap:.35rem;font-weight:600;
white-space:nowrap}.dot{width:.8rem;height:.8rem;border-radius:50%;display:inline-block;border:1px solid #8c959f}
.dot.green{background:#5c9600}.dot.yellow{background:#e0b000}.dot.red{background:#b93434}.dot.white{background:#fff}
tr[data-result=white]{background:#f6f8fa}tr[data-result=red]{background:#fff0ed}tr[data-result=yellow]{background:#fffbea}
.t-green{color:#3d6b00}.t-yellow{color:#8a6a00}.t-red{color:#b93434}.t-white{color:#57606a}.fail{color:#ad2828}.pass{color:#167348}.warn{color:#895b00}
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
        rows.append(f"<tr><td>{_e(item.get('reference_mode'))}</td><td><span class='light' "
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


def _dot(result: str, text: str = "") -> str:
    return f"<span class='signal'><span class='dot {result}'></span>{_e(text or SIGNAL_NAMES[result])}</span>"


def _accuracy_cell(items: Sequence[Mapping[str, Any]]) -> str:
    """Each benchmark's values only (TRTMC · native), coloured by its own outcome."""
    tone = {"pass": "t-green", "fail": "t-red", "inconclusive": "t-yellow"}
    lines = [f"<span class='{tone.get(str(item.get('status')), 't-white')}'>{_e(acc_value(item))}</span>"
             for item in items if not item.get("informational")]  # informational checks stay in the evidence
    return "<br>".join(lines) or '<span class="muted">—</span>'


def _performance_cell(profile: str, items: Sequence[Mapping[str, Any]]) -> str:
    """Each timed request's server model-call time p50 on both sides, coloured by its light."""
    lines = [f"<span class='t-{_e(item.get('light') or 'white')}'>{_e(request_label(profile, item))}: TRTMC "
             f"{_e(ms((item.get('candidate') or {}).get('p50_ms')))} · native "
             f"{_e(ms((item.get('reference') or {}).get('p50_ms')))}"
             + (" <small>per audio second</small>" if (item.get("candidate") or {}).get("unit") else "") + "</span>"
             for item in items]
    return "<br>".join(lines) or '<span class="muted">—</span>'


def render(rows: Mapping[str, Mapping[str, Any]], counts: Mapping[str, int], rank: Mapping[str, int],
           output: Path, title: str = "TRTMC vs native qualification", context: str = "",
           links: Sequence[tuple[str, str]] = (), reruns: Mapping[str, str] | None = None) -> Path:
    """``reruns``: profile -> its result in a linked appendix (shown under the reason)."""
    base, reruns = output.parent.resolve(), reruns or {}
    results = {profile: signal(row) for profile, row in rows.items()}
    order = sorted(rows, key=lambda p: (SIGNALS.index(results[p]), rows[p].get("task") or "", p))
    tally = {result: sum(1 for value in results.values() if value == result) for result in SIGNALS}
    related = " · ".join(f'<a href="{_e(href)}">{_e(label)}</a>' for label, href in links)
    body = []
    for profile in order:
        row = rows[profile]
        directory = row.get("directory")
        details = (f"<details><summary>evidence</summary><p>harness category: {_e(row['category'])} · host "
                   f"{_e(row.get('root'))}</p>{_accuracy(row.get('accuracy', []))}"
                   f"{_performance(row.get('perf', []))}{_sweep(row.get('l2') or {})}"
                   f"<p>{_links(Path(directory) if directory else None, base)}</p>"
                   + (f"<p>reproduce: <code>{_e(row['repro'])}</code></p>" if row.get("repro") else "")
                   + "</details>")
        key = f"{profile} {row.get('task') or ''} {results[profile]} {row.get('root')}".lower()
        body.append(f"<tr class='m' data-result='{results[profile]}' data-k='{_e(key)}'><td>{_dot(results[profile])}</td>"
                    f"<td><b>{_e(profile)}</b><br><small>{_e(row.get('task') or '-')}</small></td>"
                    f"<td>{_accuracy_cell(row.get('accuracy', []))}</td>"
                    f"<td>{_performance_cell(profile, reported_perf(row))}</td>"
                    f"<td class='reason'>{_e(signal_reason(profile, row))}"
                    + (f"<br>rerun on the fixed harness: {_dot(reruns[profile])}" if profile in reruns else "")
                    + f"</td><td class='evidence'>{details}</td></tr>")
    legend = ("<p>" + _dot("green") + " pass · " + _dot("yellow") + " Perf about equal to native (counts as a pass), "
              "or an Acc difference not shown either way · " + _dot("red") + " Acc or Perf worse than native beyond "
              "its margin · " + _dot("white") + " no valid comparison: an error or a failed build (environment or "
              "runtime), or the comparison does not apply (the native model below a benchmark's floor, timings that "
              "cannot be compared).</p>")
    counted_line = " · ".join(f"{SIGNAL_NAMES[result]} {tally[result]}" for result in SIGNALS)
    document = (f"<!doctype html><meta charset='utf-8'><title>{_e(title)}</title><style>{STYLE}</style>"
                f"<script>{SCRIPT}</script><h1>{_e(title)}</h1>"
                + (f"<p>{_e(context)}</p>" if context else "")
                + (f"<p>{related}</p>" if related else "")
                + f"<p><b>{len(rows)} models · {tally['green'] + tally['yellow']} pass (Green + Yellow)</b> · "
                f"{_e(counted_line)}</p>{legend}"
                "<p class='muted'>Accuracy: each benchmark's value on both sides (TRTMC · native). Performance: server "
                "model-call time p50 of the catalog's own request (other timed requests are in the evidence). Expand <i>evidence</i> for gates, failing samples with both "
                "outputs, timing details, files, and the reproduction command.</p>"
                "<input id='q' placeholder='filter (model, Task, result, host)' oninput='f()'><table><thead><tr>"
                "<th>Result</th><th>Model / Task</th><th>Accuracy (TRTMC · native)</th><th>Performance p50 (TRTMC · "
                f"native)</th><th>Reason</th><th>Evidence</th></tr></thead><tbody>{''.join(body)}</tbody></table>")
    output.write_text(document)
    return output
