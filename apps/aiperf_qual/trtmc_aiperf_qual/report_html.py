# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-contained, failure-first HTML report over result roots (``summary --html``).

Every model row shows its category, the Acc results (status, gate, failing samples with the TRTMC
and native outputs side by side), the Perf comparison per reference mode (light, labelled TRTMC and
native p50, reasons), links to the evidence files next to its result, and a reproduction command.
Rows are ordered errors and failed gates first.
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .report import counted

EVIDENCE = ("report.md", "report.json", "build.json", "build/build.log", "error.json", "phase-errors.log",
            "candidate/server.log")
LIGHT_COLORS = {"green": "#1a7f37", "yellow": "#9a6700", "red": "#cf222e", "white": "#6e7781", "n/a": "#6e7781"}
STYLE = """
body{font:14px/1.45 system-ui,sans-serif;margin:24px;color:#1f2328}h1{font-size:22px}
table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid #d0d7de;padding:4px 8px;text-align:left;
vertical-align:top}th{background:#f6f8fa}.cat{font-weight:600}.fail{color:#cf222e}.pass{color:#1a7f37}
.warn{color:#9a6700}details{margin:4px 0}summary{cursor:pointer}code,pre{font:12px ui-monospace,monospace}
pre{white-space:pre-wrap;background:#f6f8fa;padding:6px;margin:4px 0}.light{display:inline-block;padding:0 6px;
border-radius:8px;color:#fff;font-size:12px}#q{width:320px;padding:4px;margin:8px 0}
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


def render(rows: Mapping[str, Mapping[str, Any]], counts: Mapping[str, int], rank: Mapping[str, int],
           output: Path, title: str = "TRTMC vs native qualification") -> Path:
    base = output.parent.resolve()
    order = sorted(rows, key=lambda p: (rank.get(rows[p]["category"], 99), rows[p].get("task") or "", p))
    summary = "".join(f"<tr><td class='{_css_class(c)} cat'>{_e(c)}</td><td>{counts[c]}</td></tr>"
                      for c in sorted(counts, key=lambda c: rank.get(c, 99)))
    body = []
    for profile in order:
        row = rows[profile]
        directory = row.get("directory")
        details = (f"<details><summary>evidence</summary>{_accuracy(row.get('accuracy', []))}"
                   f"{_performance(row.get('perf', []))}{_sweep(row.get('l2') or {})}"
                   f"<p>{_links(Path(directory) if directory else None, base)}</p>"
                   + (f"<p>reproduce: <code>{_e(row['repro'])}</code></p>" if row.get("repro") else "")
                   + "</details>")
        lights = " ".join(f"<span class='light' style='background:{LIGHT_COLORS.get(p.get('light'), '#6e7781')}'>"
                          f"{_e(p.get('reference_mode'))} {_e(p.get('light'))}</span>" for p in row.get("perf", []))
        key = f"{profile} {row.get('task') or ''} {row['category']} {row.get('root')}".lower()
        body.append(f"<tr class='m' data-k='{_e(key)}'><td><b>{_e(profile)}</b></td><td>{_e(row.get('task') or '-')}"
                    f"</td><td>{_e(row.get('root'))}</td><td class='{_css_class(row['category'])} cat'>"
                    f"{_e(row['category'])}</td><td>{lights}</td><td>{_e(row.get('notes', ''))[:400]}{details}</td></tr>")
    document = (f"<!doctype html><meta charset='utf-8'><title>{_e(title)}</title><style>{STYLE}</style>"
                f"<script>{SCRIPT}</script><h1>{_e(title)}</h1><p>{len(rows)} models; errors and failed gates "
                "first. Lights compare TRTMC with the native model (green faster, yellow similar, red slower, "
                "white not comparable); a light never changes the Acc outcome.</p>"
                f"<table style='width:auto'>{summary}</table><input id='q' placeholder='filter' oninput='f()'>"
                "<table><tr><th>model</th><th>Task</th><th>root</th><th>category</th><th>Perf</th><th>notes</th></tr>"
                f"{''.join(body)}</table>")
    output.write_text(document)
    return output
