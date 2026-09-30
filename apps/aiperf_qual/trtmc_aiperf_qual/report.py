# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""report.json plus a short Markdown summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def _fmt(value: Any, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}" if isinstance(value, float) else str(value)


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
              "| suite | status | passed | required | noise floor (ref @ perf precision) | isolated re-check | golden |",
              "|---|---|---|---|---|---|---|"]
    for item in result.get("accuracy", []):
        noise = item.get("noise_floor") or {}
        noise_text = f"{noise.get('passed')}/{noise.get('total')}" + (" (gate relaxed)" if noise.get("relaxed_gate") else "") \
            if noise else "—"
        isolated = item.get("isolated_check")
        lines.append(f"| {item['suite']} | {item['status']} | {item.get('passed')}/{item.get('samples')} | "
                     f"{item.get('required_passes')} | {noise_text} | "
                     f"{isolated['status'] + ' ' + str(isolated['passed']) if isolated else '—'} | "
                     f"{(item.get('golden') or {}).get('status')} |")
    lines += ["", "## Performance L1 (server model-call time, p50)", "",
              "| reference mode | light | TRTMC ms | CI % | reference ms (aggregation) | CI % | speedup | notes |",
              "|---|---|---|---|---|---|---|---|"]
    for item in result.get("performance_l1", []):
        cand, ref = item.get("candidate", {}), item.get("reference", {})
        lines.append(f"| {item['reference_mode']} | {item['light']} | {_fmt(cand.get('p50_ms'))} | "
                     f"{_fmt(cand.get('ci_percent'), 2)} | {_fmt(ref.get('p50_ms'))} ({ref.get('aggregation', 'mean')}) | "
                     f"{_fmt(ref.get('ci_percent'), 2)} | "
                     f"{_fmt(item.get('speedup'), 2)} | {'; '.join(item.get('reasons', []) + item.get('notes', []))} |")
    if result.get("errors"):
        lines += ["", "## Phase errors", ""]
        lines += [f"- `{name}`: {message[:500]}" for name, message in result["errors"].items()]
    markdown = out / "report.md"
    markdown.write_text("\n".join(lines) + "\n")
    return json_path, markdown
