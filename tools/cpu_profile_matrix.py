#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-strategy CPU phase bottleneck harness.

Runs the CPU-phase profiler across representative family-owned specs and prints
a side-by-side comparison table showing where host-side time is spent per
runtime strategy.

Usage:
    # Use pre-built bundles from engine-dir (recommended)
    python tools/cpu_profile_matrix.py \\
      --engine-dir /workspace/users/yifeif/tensorrt-model-connect/engines \\
      --hf-cache /mnt/storage/tensorrt-model-connect/model-weights

    # Profile specific strategies only
    python tools/cpu_profile_matrix.py \\
      --engine-dir /path/to/engines \\
      --strategies qwen_decoder_kv_cache gpt_oss_decoder_moe

    # Override the representative model for a strategy
    python tools/cpu_profile_matrix.py \\
      --engine-dir /path/to/engines \\
      --model-override strategy_name=org/model:model.trtfb

    # Save JSON and HTML report
    python tools/cpu_profile_matrix.py \\
      --engine-dir /path/to/engines \\
      --json /tmp/cpu_matrix.json \\
      --html /tmp/cpu_matrix.html

    # Tune profiling parameters
    python tools/cpu_profile_matrix.py \\
      --engine-dir /path/to/engines \\
      --warmup 3 --iterations 20 --max-new-tokens 10
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------

class StrategySpec(NamedTuple):
    strategy: str          # runtime_strategy string
    label: str             # display name in table
    hf_id: str             # HuggingFace model ID
    bundle: str            # bundle filename (relative to engine-dir)
    runner: str            # "decoder" or "family"
    trust_remote_code: bool = False


def _family_root() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "python"
        / "tensorrt_model_connect"
        / "families"
    )


@lru_cache(maxsize=1)
def _family_matrix_modules() -> tuple[ModuleType, ...]:
    """Load optional family-owned matrix specs."""
    modules: list[ModuleType] = []
    for hook_path in sorted(_family_root().glob("*/cpu_profile_matrix.py")):
        module_name = f"_trtmc_cpu_profile_matrix_{hook_path.parent.name}"
        spec = importlib.util.spec_from_file_location(module_name, hook_path)
        if spec is None or spec.loader is None:
            print(f"[matrix] WARN: cannot load family matrix hook "
                  f"{hook_path}", file=sys.stderr)
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            print(f"[matrix] WARN: failed to import family matrix hook "
                  f"{hook_path}: {exc}", file=sys.stderr)
            continue
        if callable(getattr(module, "cpu_profile_matrix_specs", None)):
            modules.append(module)
    return tuple(modules)


def _strategy_spec_from_mapping(raw: dict, source: str) -> StrategySpec:
    required = ("strategy", "label", "hf_id", "bundle", "runner")
    missing = [key for key in required if not raw.get(key)]
    if missing:
        raise ValueError(
            f"{source} returned an incomplete CPU profile matrix spec; "
            f"missing {missing}"
        )
    runner = str(raw["runner"])
    if runner not in {"decoder", "family"}:
        raise ValueError(
            f"{source} returned unsupported runner {runner!r}; "
            "expected 'decoder' or 'family'"
        )
    return StrategySpec(
        strategy=str(raw["strategy"]),
        label=str(raw["label"]),
        hf_id=str(raw["hf_id"]),
        bundle=str(raw["bundle"]),
        runner=runner,
        trust_remote_code=bool(raw.get("trust_remote_code", False)),
    )


def _load_default_specs() -> list[StrategySpec]:
    """Collect default representative specs from model-family hooks."""
    ordered: list[tuple[int, str, int, StrategySpec]] = []
    for module in _family_matrix_modules():
        source = getattr(module, "__file__", module.__name__)
        for index, raw in enumerate(module.cpu_profile_matrix_specs()):
            if not isinstance(raw, dict):
                raise TypeError(
                    f"{source} returned non-dict CPU profile matrix spec "
                    f"{raw!r}"
                )
            order = int(raw.get("order", 1000))
            ordered.append((
                order,
                str(source),
                index,
                _strategy_spec_from_mapping(raw, str(source)),
            ))
    return [item[3] for item in sorted(ordered, key=lambda item: item[:3])]


_DEFAULT_SPECS: list[StrategySpec] = _load_default_specs()

# All phases across both runner types (union)
_ALL_PHASES = ("mask_build", "h2d", "tensor_bind", "execute",
               "d2d_cache", "d2d_state", "d2h", "argmax")


# ---------------------------------------------------------------------------
# Core: profile one strategy
# ---------------------------------------------------------------------------

def _profile_strategy(
    spec: StrategySpec,
    engine_dir: str | None,
    hf_cache: str | None,
    warmup: int,
    iterations: int,
    max_new_tokens: int,
    prompt: str,
    verbose: bool,
) -> dict:
    """Run cpu_profile for one strategy. Returns the JSON-serializable result."""
    import gc

    tools_dir = Path(__file__).parent
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))

    from cpu_profile import (
        _TimedDecoderRunner,
        _run_profile, _aggregate,
    )
    from perf_compare import _handler_attr, build_trt_engine, load_trt_from_bundle
    from tool_helpers import runtime_strategy_from_config

    # Resolve bundle path
    bundle_path: str | None = None
    if engine_dir and spec.bundle:
        candidate = Path(engine_dir) / spec.bundle
        if candidate.exists():
            bundle_path = str(candidate)

    # Resolve model directory
    model_dir = spec.hf_id
    if hf_cache:
        hf_name = spec.hf_id.replace("/", "--")
        for prefix in ("models--",):
            candidate = Path(hf_cache) / f"{prefix}{hf_name}"
            if candidate.exists():
                model_dir = str(candidate)
                break

    # Tokenize
    from tensorrt_model_connect.engine_builder import _resolve_model
    from transformers import AutoTokenizer

    print(f"[matrix] [{spec.strategy}] loading tokenizer ...", file=sys.stderr)
    resolved = _resolve_model(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        resolved, trust_remote_code=spec.trust_remote_code)
    input_ids = tokenizer.encode(prompt)
    eos_token_id = tokenizer.eos_token_id

    # Build or load engine
    if bundle_path:
        print(f"[matrix] [{spec.strategy}] loading bundle: {bundle_path}",
              file=sys.stderr)
        engine_plan, num_layers, max_cache_length, bundle_config, perf_handler = \
            load_trt_from_bundle(bundle_path)
        runtime_strategy = str(bundle_config.get("runtime_strategy") or spec.strategy)
        runner_config = bundle_config
        runner_bundle_path = bundle_path
        runner_type = _handler_attr(
            perf_handler, "cpu_profile_runner_type", "decoder")
    else:
        print(f"[matrix] [{spec.strategy}] building engine for {spec.hf_id} ...",
              file=sys.stderr)
        engine_plan, config, _, perf_handler = build_trt_engine(
            model_dir, 256, verbose)
        num_layers = config.num_hidden_layers
        max_cache_length = 256
        runtime_strategy = runtime_strategy_from_config(config)
        runner_config = config
        runner_bundle_path = ""
        runner_type = _handler_attr(
            perf_handler, "cpu_profile_runner_type", spec.runner)

    # Build timed runner
    make_family_runner = getattr(perf_handler, "make_cpu_profile_runner", None)
    if callable(make_family_runner):
        runner = make_family_runner(
            engine_plan=engine_plan,
            num_layers=num_layers,
            max_cache_length=max_cache_length,
        )
    else:
        runner = _TimedDecoderRunner(
            engine_plan,
            max_cache_length,
            num_layers,
            runtime_strategy,
            config=runner_config,
            bundle_path=runner_bundle_path,
        )
    del engine_plan
    gc.collect()

    # Run profiling
    print(f"[matrix] [{spec.strategy}] profiling "
          f"({warmup} warmup + {iterations} iters × {max_new_tokens} steps) ...",
          file=sys.stderr)
    _run_profile(runner, input_ids, max_new_tokens,
                 warmup, iterations, eos_token_id, verbose)

    rows = _aggregate(runner.phase_times)
    total_ms = round(sum(r["mean_ms"] for r in rows), 4)
    bottleneck = max(rows, key=lambda r: r["mean_ms"])["phase"] if rows else "N/A"

    del runner
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return {
        "strategy": spec.strategy,
        "model": spec.hf_id,
        "runner_type": runner_type,
        "num_layers": num_layers,
        "phases": rows,
        "total_ms": total_ms,
        "bottleneck": bottleneck,
    }


# ---------------------------------------------------------------------------
# Reporting: console table
# ---------------------------------------------------------------------------

def _print_matrix(results: list[dict], gpu: str, trt_ver: str,
                  prompt: str, max_new_tokens: int) -> None:
    if not results:
        print("No results.", file=sys.stderr)
        return

    # Collect all phases that appear in at least one result
    phases_seen: list[str] = []
    for ph in _ALL_PHASES:
        if any(any(r["phase"] == ph for r in res["phases"])
               for res in results):
            phases_seen.append(ph)

    def _phase_val(res: dict, phase: str) -> tuple[float, float]:
        for r in res["phases"]:
            if r["phase"] == phase:
                return r["mean_ms"], r["pct"]
        return 0.0, 0.0

    col_w = 22
    n = len(results)
    sep_wide = "=" * (22 + n * (col_w + 2))

    print(f"\n{sep_wide}")
    print("  CPU Phase Bottleneck Matrix")
    print(f"  GPU: {gpu}  |  TRT: {trt_ver}")
    print(f"  Prompt: \"{prompt[:50]}\"  |  Decode steps: {max_new_tokens}")
    print("  (ms = mean per decode step;  % = fraction of total step time)")
    print(sep_wide)

    # Header row: strategy labels
    header = f"  {'Phase':<18s}"
    for res in results:
        label = res["strategy"].replace("_", "_\n")  # won't wrap in terminal
        label = res["strategy"]
        header += f"  {label:>{col_w}s}"
    print(header)

    # Sub-header: model names
    subheader = f"  {'':18s}"
    for res in results:
        model_short = res["model"].split("/")[-1]
        if len(model_short) > col_w:
            model_short = model_short[:col_w - 1] + "…"
        subheader += f"  {model_short:>{col_w}s}"
    print(subheader)
    print(f"  {'─'*18}" + f"  {'─'*col_w}" * n)

    # Phase rows
    for ph in phases_seen:
        row = f"  {ph:<18s}"
        for res in results:
            ms, pct = _phase_val(res, ph)
            if ms == 0.0:
                row += f"  {'—':>{col_w}s}"
            else:
                cell = f"{ms:.3f}ms ({pct:.0f}%)"
                row += f"  {cell:>{col_w}s}"
        print(row)

    print(f"  {'─'*18}" + f"  {'─'*col_w}" * n)

    # Total row
    row = f"  {'TOTAL':<18s}"
    for res in results:
        row += f"  {res['total_ms']:>{col_w}.3f}"
    print(row)

    # Bottleneck row (highlighted)
    row = f"  {'BOTTLENECK':<18s}"
    for res in results:
        bn = res["bottleneck"]
        ms, pct = _phase_val(res, bn)
        cell = f"{bn} ({pct:.0f}%)"
        if len(cell) > col_w:
            cell = cell[:col_w - 1] + "…"
        row += f"  {cell:>{col_w}s}"
    print(row)

    print(sep_wide)
    print()

    # Per-strategy analysis
    print("  Analysis:")
    for res in results:
        bn = res["bottleneck"]
        ms, pct = _phase_val(res, bn)
        layers = res["num_layers"]
        print(f"    {res['strategy']:<28s}  bottleneck={bn:<14s} "
              f"({pct:.0f}% of {res['total_ms']:.3f}ms/step, "
              f"{layers} layers)")
    print()


# ---------------------------------------------------------------------------
# Reporting: HTML
# ---------------------------------------------------------------------------

def _heat(pct: float) -> str:
    """Colour scale: white (0%) → orange-red (100%)."""
    if pct <= 0:
        return "#f8f8f8"
    t = min(pct / 100.0, 1.0)
    if t < 0.5:
        r, g, b = 255, int(255 - t * 2 * 80), int(255 - t * 2 * 80)
    else:
        r, g, b = 255, int(175 - (t - 0.5) * 2 * 175), 0
    return f"rgb({r},{g},{b})"


def _phase_row(res: dict, phase: str) -> dict:
    for r in res["phases"]:
        if r["phase"] == phase:
            return r
    return {"phase": phase, "mean_ms": 0.0, "pct": 0.0}


def _build_html(results: list[dict], gpu: str, trt_ver: str,
                prompt: str, max_new_tokens: int,
                warmup: int, iterations: int) -> str:
    if not results:
        return "<p>No results.</p>"

    phases_seen: list[str] = []
    for ph in _ALL_PHASES:
        if any(any(r["phase"] == ph for r in res["phases"])
               for res in results):
            phases_seen.append(ph)

    # Build Chart.js dataset
    chart_labels = [res["strategy"] for res in results]
    datasets = []
    _phase_colors = {
        "mask_build":  "#a8d8ea", "h2d": "#aa96da",
        "tensor_bind": "#fcbad3", "execute": "#e63946",
        "d2d_cache":   "#f4a261", "d2d_state": "#e9c46a",
        "d2h":         "#2a9d8f", "argmax": "#264653",
    }
    for ph in phases_seen:
        data = [_phase_row(res, ph)["mean_ms"] for res in results]
        datasets.append({
            "label": ph,
            "data": data,
            "backgroundColor": _phase_colors.get(ph, "#999"),
        })

    chart_json = json.dumps({"labels": chart_labels, "datasets": datasets})

    # Build HTML table
    thead = "<tr><th>Phase</th>"
    for res in results:
        model_short = res["model"].split("/")[-1]
        thead += (f"<th>{res['strategy']}<br>"
                  f"<small>{model_short}</small></th>")
    thead += "</tr>"

    tbody = ""
    for ph in phases_seen:
        tbody += f"<tr><td><code>{ph}</code></td>"
        for res in results:
            row = _phase_row(res, ph)
            ms, pct = row["mean_ms"], row["pct"]
            bg = _heat(pct)
            if ms == 0.0:
                tbody += f'<td style="background:{bg}">—</td>'
            else:
                is_bn = res["bottleneck"] == ph
                bold = " font-weight:bold;" if is_bn else ""
                tbody += (f'<td style="background:{bg};{bold}">'
                          f'{ms:.3f}ms<br><small>{pct:.0f}%</small></td>')
        tbody += "</tr>"

    # Total + bottleneck rows
    tbody += "<tr style='border-top:2px solid #333'><td><b>TOTAL</b></td>"
    for res in results:
        tbody += f"<td><b>{res['total_ms']:.3f}ms</b></td>"
    tbody += "</tr>"

    tbody += "<tr><td><b>BOTTLENECK</b></td>"
    for res in results:
        bn = res["bottleneck"]
        row = _phase_row(res, bn)
        bg = _heat(row["pct"])
        pct = row["pct"]
        tbody += (f"<td style='background:{bg};font-weight:bold'>"
                  f"{bn}<br><small>{pct:.0f}%</small></td>")
    tbody += "</tr>"

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CPU Phase Bottleneck Matrix</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 32px; color: #222; }}
  h1 {{ font-size: 1.4em; }}
  .meta {{ color: #666; font-size: 0.9em; margin-bottom: 24px; }}
  table {{ border-collapse: collapse; margin-top: 24px; font-size: 0.9em; }}
  th, td {{ border: 1px solid #ccc; padding: 8px 14px; text-align: right; white-space: nowrap; }}
  th {{ background: #f0f0f0; text-align: center; }}
  td:first-child {{ text-align: left; }}
  th:first-child {{ text-align: left; }}
  small {{ color: #555; }}
  canvas {{ max-width: 900px; margin-top: 32px; }}
  .analysis {{ margin-top: 24px; padding: 16px; background: #f9f9f9;
               border-left: 4px solid #e63946; font-size: 0.9em; }}
  .analysis li {{ margin: 6px 0; }}
</style>
</head>
<body>
<h1>CPU Phase Bottleneck Matrix</h1>
<div class="meta">
  GPU: <b>{gpu}</b> &nbsp;|&nbsp; TRT: <b>{trt_ver}</b><br>
  Prompt: <i>{prompt[:80]}</i><br>
  Decode steps: {max_new_tokens} &nbsp;|&nbsp;
  Warmup: {warmup} &nbsp;|&nbsp; Iterations: {iterations}<br>
  Generated: {ts}
</div>

<table>
  <thead>{thead}</thead>
  <tbody>{tbody}</tbody>
</table>

<p style="margin-top:8px;font-size:0.8em;color:#888">
  Heat: <span style="background:#fff;padding:2px 6px;border:1px solid #ccc">0%</span>
  → <span style="background:rgb(255,175,175);padding:2px 6px">50%</span>
  → <span style="background:rgb(255,0,0);color:#fff;padding:2px 6px">100%</span>
  of per-step time. Bold = bottleneck phase for that strategy.
</p>

<canvas id="chart"></canvas>
<script>
const DATA = {chart_json};
new Chart(document.getElementById("chart"), {{
  type: "bar",
  data: DATA,
  options: {{
    plugins: {{
      title: {{ display: true,
                text: "CPU Phase Time per Strategy (ms/step, stacked)" }},
      legend: {{ position: "bottom" }}
    }},
    responsive: true,
    scales: {{
      x: {{ stacked: true }},
      y: {{ stacked: true, title: {{ display: true, text: "ms / decode step" }} }}
    }}
  }}
}});
</script>

<div class="analysis">
  <b>Per-strategy bottleneck summary:</b>
  <ul>
{"".join(f"    <li><code>{res['strategy']}</code> ({res['model'].split('/')[-1]}) — "
         f"bottleneck: <b>{res['bottleneck']}</b> "
         f"({_phase_row(res, res['bottleneck'])['pct']:.0f}% of "
         f"{res['total_ms']:.3f} ms/step, {res['num_layers']} layers)</li>"
         for res in results)}
  </ul>
  <p style="margin:0;font-size:0.85em;color:#666">
    <b>execute</b> &gt; 75%: GPU compute bound — look at per-layer profile.<br>
    <b>tensor_bind</b> &gt; 10%: CPU overhead scales with num_layers.<br>
    <b>d2d_cache</b> large: KV-cache update D2D traffic (scales with num_layers × head_dim).<br>
    <b>h2d</b> large: input transfer bottleneck.
  </p>
</div>

</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-strategy CPU phase bottleneck comparison harness")
    parser.add_argument("--engine-dir",
                        help="Directory containing pre-built .trtfb bundles")
    parser.add_argument("--hf-cache",
                        help="HuggingFace model cache dir "
                             "(default: ~/.cache/huggingface/hub)")
    parser.add_argument("--strategies", nargs="+",
                        choices=[s.strategy for s in _DEFAULT_SPECS],
                        default=None,
                        help="Strategies to profile (default: all supported)")
    parser.add_argument(
        "--model-override", nargs="+", metavar="STRATEGY=HF_ID:BUNDLE",
        help="Override the model for a strategy, e.g. "
             "strategy_name=org/model:model.trtfb")
    parser.add_argument("--prompt",
                        default="The capital of France is",
                        help="Input prompt for all strategies")
    parser.add_argument("--max-new-tokens", type=int, default=10,
                        help="Decode steps per profiling run (default: 10)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup iterations (default: 3)")
    parser.add_argument("--iterations", type=int, default=20,
                        help="Timed iterations (default: 20)")
    parser.add_argument("--json", dest="json_path", metavar="PATH",
                        help="Save results JSON to this path")
    parser.add_argument("--html", dest="html_path", metavar="PATH",
                        help="Save HTML report to this path")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Build spec list
    specs = list(_DEFAULT_SPECS)

    # Apply --model-override
    overrides: dict[str, tuple[str, str]] = {}
    for ov in (args.model_override or []):
        try:
            strategy, rest = ov.split("=", 1)
            hf_id, bundle = rest.rsplit(":", 1)
            overrides[strategy] = (hf_id, bundle)
        except ValueError:
            print(f"WARNING: ignoring malformed --model-override {ov!r}",
                  file=sys.stderr)
    if overrides:
        new_specs = []
        for s in specs:
            if s.strategy in overrides:
                hf_id, bundle = overrides[s.strategy]
                new_specs.append(s._replace(hf_id=hf_id, bundle=bundle))
            else:
                new_specs.append(s)
        specs = new_specs

    # Filter by --strategies
    if args.strategies:
        specs = [s for s in specs if s.strategy in args.strategies]

    if not specs:
        print("ERROR: no strategies selected.", file=sys.stderr)
        sys.exit(1)

    print(f"[matrix] Profiling {len(specs)} strategies: "
          f"{[s.strategy for s in specs]}", file=sys.stderr)

    # Run profiling for each strategy
    results: list[dict] = []
    failed: list[str] = []
    for spec in specs:
        try:
            res = _profile_strategy(
                spec=spec,
                engine_dir=args.engine_dir,
                hf_cache=args.hf_cache,
                warmup=args.warmup,
                iterations=args.iterations,
                max_new_tokens=args.max_new_tokens,
                prompt=args.prompt,
                verbose=args.verbose,
            )
            results.append(res)
            print(f"[matrix] [{spec.strategy}] done — "
                  f"bottleneck={res['bottleneck']} "
                  f"({res['total_ms']:.3f} ms/step)", file=sys.stderr)
        except Exception as exc:
            print(f"[matrix] [{spec.strategy}] FAILED: {exc}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()
            failed.append(spec.strategy)

    if not results:
        print("ERROR: all strategies failed.", file=sys.stderr)
        sys.exit(1)

    # Gather metadata
    tools_dir = Path(__file__).parent
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    from cpu_profile import _get_gpu_name, _get_trt_version
    gpu = _get_gpu_name()
    trt_ver = _get_trt_version()

    # Console report
    _print_matrix(results, gpu, trt_ver, args.prompt, args.max_new_tokens)

    if failed:
        print(f"  Skipped strategies (failed): {failed}", file=sys.stderr)

    # JSON output
    if args.json_path:
        output = {
            "metadata": {
                "gpu": gpu,
                "trt_version": trt_ver,
                "prompt": args.prompt,
                "max_new_tokens": args.max_new_tokens,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "results": results,
            "failed": failed,
        }
        with open(args.json_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"[matrix] Results saved to {args.json_path}", file=sys.stderr)

    # HTML output
    if args.html_path:
        html = _build_html(results, gpu, trt_ver, args.prompt,
                           args.max_new_tokens, args.warmup, args.iterations)
        with open(args.html_path, "w") as f:
            f.write(html)
        print(f"[matrix] HTML report saved to {args.html_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
