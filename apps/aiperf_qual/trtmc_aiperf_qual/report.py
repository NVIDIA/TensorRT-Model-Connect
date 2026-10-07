# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""report.json plus a short Markdown summary."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from .compat import report as normalized_report


def _fmt(value: Any, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def counted(item: Mapping[str, Any], limit: int = 3) -> str:
    """``passed/samples``, both sides' gold-scored score and the regression test, or the first numeric
    metrics of a check that reports aggregates only."""
    metrics = item.get("metrics") or {}
    if "trtmc_score" in metrics:
        test = metrics.get("test") or {}
        bound = (f"z {test['z']:+.2f}" if "z" in test else
                 f"excess 90% interval {[round(v, 3) for v in test['excess_interval90']]}" if "excess_interval90" in test
                 else f"bounds {_fmt(test.get('lower'))}..{_fmt(test.get('upper'))}" if "upper" in test else "")
        unit = f"{metrics['units']} units" if metrics.get("units") is not None else f"{item.get('samples')} problems"
        return (f"TRTMC {_fmt(metrics['trtmc_score'], 2)} vs native {_fmt(metrics['native_score'], 2)} "
                f"(regression {_fmt(test.get('regression_points', metrics.get('regression_points')), 3)} pt, {bound}; {unit})")
    if item.get("passed") is not None:
        return f"{item['passed']}/{item.get('samples')}"
    values = [f"{name} {value:.4g}" for name, value in (item.get("metrics") or {}).items()
              if isinstance(value, float)][:limit]
    samples = f"{item['samples']} samples" if item.get("samples") else ""
    return ": ".join(part for part in (samples, ", ".join(values)) if part) or "—"


# What a gold-scored entry's score is, by its metric (right/wrong benchmarks score the percent answered right).
SCORE_NAMES = {"wer": "WER %", "chrf": "chrF++", "coco_map": "COCO mAP", "miou": "mIoU %", "mask_iou": "mask IoU %",
               "sts_spearman": "Spearman x100", "rerank_ndcg": "nDCG", "retrieval_ndcg": "nDCG",
               "retrieval_ndcg10": "nDCG@10", "forecast_mse": "MSE", "precomputed_mean": "mean score"}


METRIC_NAMES = {"candidate_vs_native_psnr_db": "PSNR vs native (dB)", "candidate_vs_native_ssim": "SSIM vs native",
                "candidate_vs_reference_psnr_db": "PSNR vs fp32 (dB)"}


def score_name(item: Mapping[str, Any]) -> str:
    benchmark = str(item.get("benchmark") or "")
    if benchmark in SCORE_NAMES:
        return SCORE_NAMES[benchmark]
    if "(" in benchmark and benchmark.rstrip(")").rsplit("(", 1)[-1] in SCORE_NAMES:  # an AIPerf corpus metric
        return SCORE_NAMES[benchmark.rstrip(")").rsplit("(", 1)[-1]]
    if (item.get("metrics") or {}).get("units") is not None:
        return benchmark.split(" (")[0] or "score"
    return "accuracy %"


def values(item: Mapping[str, Any]) -> str:
    """Both sides' values of an Acc entry and what they cover, without test statistics: ``accuracy %: TRTMC 70.62,
    native 70.67 (1947 of 2241 problems)``, ``400/400 within tolerance``."""
    metrics = item.get("metrics") or {}
    expected, samples = item.get("expected_samples"), item.get("samples")
    if "trtmc_score" in metrics:
        count = metrics["units"] if metrics.get("units") is not None else samples
        of = f" of {expected}" if expected and samples is not None and samples < expected else ""
        unit = "items" if metrics.get("units") is not None else "problems"
        return (f"{score_name(item)}: TRTMC {_fmt(metrics['trtmc_score'], 2)}, native {_fmt(metrics['native_score'], 2)}"
                f" ({count}{of} {unit})")
    if item.get("required_passes") is not None:
        return f"{item.get('passed')}/{samples or item['required_passes']} within tolerance"
    found = [f"{METRIC_NAMES.get(name, name)} {value:.4g}" for name, value in metrics.items()
             if isinstance(value, float) and (name in METRIC_NAMES or not name.startswith("native_vs"))][:3]
    if found:
        return ", ".join(found)
    if item.get("passed") is not None and samples:
        return f"{item['passed']}/{samples}"
    return f"{samples} of {expected} answered" if expected else "—"


def plain(text: str) -> str:
    """A reason without a server's JSON error envelope: its message only."""
    return re.sub(r'\{"error":\{"message":"((?:[^"\\]|\\.)*)".*?\}\}', r"\1", str(text))


def acc_value(item: Mapping[str, Any]) -> str:
    """One benchmark's values only: ``mmlu-0shot (accuracy %): TRTMC 82.84 · native 83.21``; nothing for an error."""
    metrics = item.get("metrics") or {}
    if item.get("status") == "error":
        return f"{item.get('suite')}: —"
    if "trtmc_score" in metrics:
        return (f"{item.get('suite')} ({score_name(item)}): TRTMC {_fmt(metrics['trtmc_score'], 2)} · "
                f"native {_fmt(metrics['native_score'], 2)}")
    return f"{item.get('suite')}: {values(item)}"


def _media_service_metrics(service_metrics: Mapping[str, Any]) -> list[str]:
    speedup = f" · TRTMC {service_metrics['speedup']:.2f}x faster per call" if service_metrics.get("speedup") else ""
    memory = f", peak memory {service_metrics['memory_ratio']:.2f}x native" if service_metrics.get("memory_ratio") else ""
    lines = ["", f"## Optional service metrics (AIPerf {service_metrics.get('endpoint')}, informational; {service_metrics.get('prompts')} prompts, "
             f"{service_metrics.get('requests')} requests per level)", "",
             f"Light {service_metrics.get('light')} {'; '.join(service_metrics.get('reasons', []) + service_metrics.get('notes', []))}{speedup}{memory}. "
             f"{service_metrics.get('note', '')}", "",
             "| side | steps | model call p50 ms | request latency p50 ms | peak GPU memory MiB | measured |",
             "|---|---|---|---|---|---|"]
    for side in ("candidate", "reference"):
        name = "TRTMC" if side == "candidate" else f"native eager {service_metrics.get('reference_precision', '')}".strip()
        for level in service_metrics.get(side, []):
            lines.append(f"| {name} | {level.get('steps') or 'catalog'} | {_fmt(level.get('model_call_p50_ms'))} | "
                         f"{_fmt(level.get('request_latency_p50'))} | {_fmt(level.get('peak_memory_mb'), 0)} | "
                         f"{level.get('measured')} |")
    for side, parts in (service_metrics.get("decomposition") or {}).items():
        if parts:
            lines.append(f"\n{'TRTMC' if side == 'candidate' else 'native'}: {parts['per_step_ms']:.1f} ms per "
                         f"denoising step + {parts['fixed_ms']:.1f} ms fixed (text encoders, VAE decode).")
    return lines


def write_report(out: Path, result: Mapping[str, Any]) -> tuple[Path, Path]:
    result = normalized_report(result)
    json_path = out / "report.json"
    json_path.write_text(json.dumps(result, indent=2, default=str))
    verdict = result.get("verdict", {})
    reference = result.get("reference", {})
    lines = [f"# {result['model']} — AIPerf qualification", "",
             f"Task `{result.get('task')}`, operation `{result.get('operation')}`; verdict **{verdict.get('category')}** "
             f"(Acc {verdict.get('acc')}, Perf {verdict.get('perf')}).", "",
             f"Reference: backend `{reference.get('backend')}`, perf precision "
             f"{reference.get('timing_precision') or reference.get('perf_precision')}; "
             f"platform `{result.get('platform', {}).get('id')}`; "
             f"aiperf {result['provenance'].get('aiperf')}, plugins {result['provenance'].get('plugins')}.", ""]
    if result.get("coverage"):
        lines += [f"Coverage: {result['coverage']}", ""]
    if result.get("accuracy_note"):
        lines += [f"Accuracy not applicable: {result['accuracy_note']}", ""]
    lines += ["## Quality (task scores or explicitly labelled conversion parity)", "",
              "| suite | source | status | passed | gate |",
              "|---|---|---|---|---|"]
    for item in result.get("accuracy", []):
        gate = (f"{item['required_passes']} passes" if item.get("required_passes") is not None
                else json.dumps(item.get("gate", {})))
        count = counted(item) if item.get("samples") else json.dumps(item.get("metrics", {}))[:160]
        status = item["status"] + (" (informational)" if item.get("informational") else "")
        source = (f"{item.get('benchmark')} (AIPerf, gold answers, {item.get('endpoint')})" if item.get("source") == "absolute"
                  else item.get("benchmark") or "Task check")
        lines.append(f"| {item['suite']} | {source} | {status} | {count} | {gate} |")
        if item.get("error"):
            lines.append(f"|  | error: {item['error'][:300].replace('|', '/')} |  |  |  |")
        if item.get("reasons") or item.get("notes"):
            text = "; ".join([*item.get("reasons", []), *item.get("notes", [])])
            lines.append(f"|  | {text[:300].replace('|', '/')} |  |  |  |")
    lines += ["", "## Performance (server task-call time)", "",
              "| reference mode | light | TRTMC ms | CI % | reference ms (aggregation) | CI % | speedup | notes |",
              "|---|---|---|---|---|---|---|---|"]
    for item in result.get("performance", []):
        cand, ref = item.get("candidate", {}), item.get("reference", {})
        if item.get("kind") == "natural_dataset":
            interval = item.get("speedup_interval90")
            ratio = _fmt(item.get("speedup"), 2) + "x" if item.get("comparable") else "not comparable"
            scope = "dataset units 90% CI " + str(interval) if interval else "no dataset-unit interval"
            notes = "; ".join([*item.get("reasons", []), *item.get("notes", []), scope])
            lines.append(f"| {item['request']} (shared quality outputs) | {item['light']} | "
                         f"{_fmt(cand.get('p50_ms'))} | — | {_fmt(ref.get('p50_ms'))} | — | {ratio} | {notes} |")
            lines += ["", f"Natural task total-time ratio: {_fmt(item.get('natural_task_speedup'), 2)}x; "
                      f"{item.get('matched_pairs')}/{item.get('pairs')} paired responses have matching work. "
                      "This describes the collected tasks and is not an equal-work claim when work differs.", ""]
            continue
        unit = " per audio second" if cand.get("unit") or ref.get("unit") else ""
        lines.append(f"| {item['reference_mode']}{' ' + item['request'] if item.get('request') else ''} | {item['light']} | {_fmt(cand.get('p50_ms'))}{unit} | "
                     f"{_fmt(cand.get('ci_percent'), 2)} | {_fmt(ref.get('p50_ms'))}{unit} ({ref.get('aggregation', 'mean')}) | "
                     f"{_fmt(ref.get('ci_percent'), 2)} | "
                     f"{_fmt(item.get('speedup'), 2)} | {'; '.join(item.get('reasons', []) + item.get('notes', []))} |")
    service_metrics = result.get("service_metrics") or {}
    if service_metrics.get("kind") == "media":
        lines += _media_service_metrics(service_metrics)
    elif service_metrics:
        lines += ["", f"## Optional service metrics (serving sweep, informational; ISL {service_metrics.get('isl')}, OSL {service_metrics.get('osl')})", "",
                  f"Light {service_metrics.get('light')} {'; '.join(service_metrics.get('reasons', []))}"
                  + (f" · TRTMC/native throughput {service_metrics['throughput_ratio']:.2f}x at concurrency {service_metrics.get('concurrency')}"
                     if service_metrics.get("throughput_ratio") else "") + f". {service_metrics.get('note', '')}", "",
                  "| side | concurrency | requests/s | latency p50 ms | latency p99 ms | error % |", "|---|---|---|---|---|---|"]
        for side in ("candidate", "reference"):
            for level in service_metrics.get(side, []):
                lines.append(f"| {'TRTMC' if side == 'candidate' else 'native eager'} | {level.get('concurrency')} | "
                             f"{_fmt(level.get('request_throughput_avg'), 2)} | {_fmt(level.get('request_latency_p50'))} | "
                             f"{_fmt(level.get('request_latency_p99'))} | {_fmt(level.get('request_error_rate_avg'), 1)} |")
    if result.get("errors"):
        lines += ["", "## Phase errors", ""]
        lines += [f"- `{name}`: {message[:500]}" for name, message in result["errors"].items()]
    markdown = out / "report.md"
    markdown.write_text("\n".join(lines) + "\n")
    return json_path, markdown
