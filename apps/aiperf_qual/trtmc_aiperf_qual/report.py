# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""report.json plus a short Markdown summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def _fmt(value: Any, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def counted(item: Mapping[str, Any], limit: int = 3) -> str:
    """``passed/samples``, or the first numeric metrics when the family grades aggregates only."""
    if item.get("passed") is not None:
        return f"{item['passed']}/{item.get('samples')}"
    values = [f"{name} {value:.4g}" for name, value in (item.get("metrics") or {}).items()
              if isinstance(value, float)][:limit]
    samples = f"{item['samples']} samples" if item.get("samples") else ""
    return ": ".join(part for part in (samples, ", ".join(values)) if part) or "—"


def write_report(out: Path, result: Mapping[str, Any]) -> tuple[Path, Path]:
    json_path = out / "report.json"
    json_path.write_text(json.dumps(result, indent=2, default=str))
    verdict = result.get("verdict", {})
    reference = result.get("reference", {})
    lines = [f"# {result['model']} — AIPerf qualification", "",
             f"Task `{result.get('task')}`, operation `{result.get('operation')}`; verdict **{verdict.get('category')}** "
             f"(Acc {verdict.get('acc')}, Perf {verdict.get('perf')}).", "",
             f"Reference: backend `{reference.get('backend')}`, golden precision {reference.get('precision')}, "
             f"noise precision {reference.get('perf_precision')}, perf precision "
             f"{reference.get('timing_precision') or reference.get('perf_precision')}; "
             f"platform `{result.get('platform', {}).get('id')}`; "
             f"aiperf {result['provenance'].get('aiperf')}, plugins {result['provenance'].get('plugins')}.", ""]
    if reference.get("noise_error"):
        lines += [f"Noise floor not available: {reference['noise_error'][:300]}", ""]
    if reference.get("fallback_from"):
        lines += [f"Generic reference failed, fell back to the family script: {reference['fallback_from'][:300]}", ""]
    lines += ["## Accuracy (parity with the native model)", "",
              "| suite | source | status | passed | gate | native at the candidate precision | isolated re-check | "
              "golden / evidence |", "|---|---|---|---|---|---|---|---|"]
    for item in result.get("accuracy", []):
        noise = item.get("noise_floor") or {}
        noise_text = (f"{noise.get('passed')}/{noise.get('total')}"
                      + (" (every failure also native: inconclusive)" if item.get("precision_sensitive") else "")
                      if noise else "—")
        isolated = item.get("isolated_check")
        family = item.get("source") == "family"
        gate = (f"{item['required_passes']} passes" if item.get("required_passes") is not None
                else json.dumps(item.get("gate", {})))
        count = counted(item) if item.get("samples") else json.dumps(item.get("metrics", {}))[:160]
        evidence = item.get("evidence") if family else (item.get("golden") or {}).get("status")
        source = f"{item.get('benchmark')} (family)" if family else item.get("benchmark") or "Task suite"
        lines.append(f"| {item['suite']} | {source} | "
                     f"{item['status']} | {count} | {gate} | {noise_text} | "
                     f"{isolated['status'] + ' ' + str(isolated['passed']) if isolated else '—'} | {evidence} |")
        if item.get("error"):
            lines.append(f"|  | error: {item['error'][:300].replace('|', '/')} |  |  |  |  |  |  |")
    lines += ["", "## Performance L1 (server model-call time, p50)", "",
              "| reference mode | light | TRTMC ms | CI % | reference ms (aggregation) | CI % | speedup | notes |",
              "|---|---|---|---|---|---|---|---|"]
    for item in result.get("performance_l1", []):
        cand, ref = item.get("candidate", {}), item.get("reference", {})
        lines.append(f"| {item['reference_mode']} | {item['light']} | {_fmt(cand.get('p50_ms'))} | "
                     f"{_fmt(cand.get('ci_percent'), 2)} | {_fmt(ref.get('p50_ms'))} ({ref.get('aggregation', 'mean')}) | "
                     f"{_fmt(ref.get('ci_percent'), 2)} | "
                     f"{_fmt(item.get('speedup'), 2)} | {'; '.join(item.get('reasons', []) + item.get('notes', []))} |")
    l2 = result.get("performance_l2") or {}
    if l2:
        lines += ["", f"## Performance L2 (serving sweep, informational; ISL {l2.get('isl')}, OSL {l2.get('osl')})", "",
                  f"Light {l2.get('light')} {'; '.join(l2.get('reasons', []))}"
                  + (f" · TRTMC/native throughput {l2['throughput_ratio']:.2f}x at concurrency {l2.get('concurrency')}"
                     if l2.get("throughput_ratio") else "") + f". {l2.get('note', '')}", "",
                  "| side | concurrency | requests/s | latency p50 ms | latency p99 ms | error % |", "|---|---|---|---|---|---|"]
        for side in ("candidate", "reference"):
            for level in l2.get(side, []):
                lines.append(f"| {'TRTMC' if side == 'candidate' else 'native eager'} | {level.get('concurrency')} | "
                             f"{_fmt(level.get('request_throughput_avg'), 2)} | {_fmt(level.get('request_latency_p50'))} | "
                             f"{_fmt(level.get('request_latency_p99'))} | {_fmt(level.get('request_error_rate_avg'), 1)} |")
    if result.get("errors"):
        lines += ["", "## Phase errors", ""]
        lines += [f"- `{name}`: {message[:500]}" for name, message in result["errors"].items()]
    markdown = out / "report.md"
    markdown.write_text("\n".join(lines) + "\n")
    return json_path, markdown
