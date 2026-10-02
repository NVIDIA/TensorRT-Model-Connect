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
    """``passed/samples``, both sides' gold-scored accuracy, or the first numeric metrics when the
    family grades aggregates only."""
    metrics = item.get("metrics") or {}
    if item.get("source") == "absolute" and "trtmc_score" in metrics:
        return (f"TRTMC {metrics['trtmc_score']} vs native {metrics['native_score']} {item.get('benchmark')} "
                f"(Δ {metrics['delta_points']:+.3f}, 95% CI {metrics.get('ci95')}; {metrics.get('units')} units)")
    if item.get("source") == "absolute" and "trtmc_accuracy" in metrics:
        p_value = f", McNemar p {metrics['mcnemar_p_worse']:.3g}" if metrics.get("mcnemar_p_worse") is not None else ""
        return (f"TRTMC {metrics['trtmc_accuracy']}% vs native {metrics['native_accuracy']}% "
                f"(Δ {metrics['delta_points']:+.2f} pt{p_value}; {item.get('samples')} problems)")
    if item.get("passed") is not None:
        return f"{item['passed']}/{item.get('samples')}"
    values = [f"{name} {value:.4g}" for name, value in (item.get("metrics") or {}).items()
              if isinstance(value, float)][:limit]
    samples = f"{item['samples']} samples" if item.get("samples") else ""
    return ": ".join(part for part in (samples, ", ".join(values)) if part) or "—"


def _media_l2(l2: Mapping[str, Any]) -> list[str]:
    speedup = f" · TRTMC {l2['speedup']:.2f}x faster per call" if l2.get("speedup") else ""
    memory = f", peak memory {l2['memory_ratio']:.2f}x native" if l2.get("memory_ratio") else ""
    lines = ["", f"## Performance L2 (AIPerf {l2.get('endpoint')}, informational; {l2.get('prompts')} prompts, "
             f"{l2.get('requests')} requests per level)", "",
             f"Light {l2.get('light')} {'; '.join(l2.get('reasons', []) + l2.get('notes', []))}{speedup}{memory}. "
             f"{l2.get('note', '')}", "",
             "| side | steps | model call p50 ms | request latency p50 ms | peak GPU memory MiB | measured |",
             "|---|---|---|---|---|---|"]
    for side in ("candidate", "reference"):
        name = "TRTMC" if side == "candidate" else f"native eager {l2.get('reference_precision', '')}".strip()
        for level in l2.get(side, []):
            lines.append(f"| {name} | {level.get('steps') or 'catalog'} | {_fmt(level.get('model_call_p50_ms'))} | "
                         f"{_fmt(level.get('request_latency_p50'))} | {_fmt(level.get('peak_memory_mb'), 0)} | "
                         f"{level.get('measured')} |")
    for side, parts in (l2.get("decomposition") or {}).items():
        if parts:
            lines.append(f"\n{'TRTMC' if side == 'candidate' else 'native'}: {parts['per_step_ms']:.1f} ms per "
                         f"denoising step + {parts['fixed_ms']:.1f} ms fixed (text encoders, VAE decode).")
    return lines


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
    if result.get("coverage"):
        lines += [f"Coverage: {result['coverage']}", ""]
    if result.get("candidate_note"):
        lines += [f"TRTMC bundle: {result['candidate_note']}", ""]
    if reference.get("noise_error"):
        lines += [f"Noise floor not available: {reference['noise_error'][:300]}", ""]
    if reference.get("fallback_from"):
        lines += [f"Generic reference failed, fell back to the family script: {reference['fallback_from'][:300]}", ""]
    lines += ["## Accuracy (absolute: both sides scored against gold answers; otherwise parity with the native model)", "",
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
        source = (f"{item.get('benchmark')} (family)" if family else
                  f"{item.get('benchmark')} (AIPerf, gold answers, {item.get('endpoint')})" if item.get("source") == "absolute"
                  else item.get("benchmark") or "Task suite")
        lines.append(f"| {item['suite']} | {source} | "
                     f"{item['status']} | {count} | {gate} | {noise_text} | "
                     f"{' '.join(str(part) for part in (isolated['status'], isolated.get('passed')) if part is not None) if isolated else '—'} | {evidence} |")
        if item.get("error"):
            lines.append(f"|  | error: {item['error'][:300].replace('|', '/')} |  |  |  |  |  |  |")
        if item.get("reasons") or item.get("notes"):
            text = "; ".join([*item.get("reasons", []), *item.get("notes", [])])
            lines.append(f"|  | {text[:300].replace('|', '/')} |  |  |  |  |  |  |")
    workloads = [item for item in result.get("accuracy", []) if (item.get("workload_perf") or {}).get("pairs")]
    if workloads:
        lines += ["", "## Performance on the benchmark requests (informational; server model-call time, p50)", "",
                  "| benchmark | light | TRTMC ms | native ms | speedup | problems (same output length) | prompt tokens p50 |",
                  "|---|---|---|---|---|---|---|"]
        for item in workloads:
            perf = item["workload_perf"]
            lines.append(f"| {item['suite']} | {perf['light']} | {perf['trtmc_p50_ms']} | {perf['native_p50_ms']} | "
                         f"{perf['speedup']}x | {perf['pairs']} | {perf.get('prompt_tokens_p50')} |")
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
    if l2.get("kind") == "media":
        lines += _media_l2(l2)
    elif l2:
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
