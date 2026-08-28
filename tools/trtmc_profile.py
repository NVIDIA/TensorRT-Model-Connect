#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified profiling entry point for a single model.

Runs all profiling passes in one command and produces a combined console
report plus optional JSON artifacts:

    python tools/profile.py --model example-org/example-decoder

Passes executed (all in-process, serial to avoid GPU memory contention):
  1. TRT + IProfiler    — e2e latency AND per-layer kernel timing in one run
  2. HF eager           — baseline latency
  3. HF torch.compile   — compiled latency (skip with --no-compile)

Artifacts saved to --output-dir when --json is passed:
  perf_compare.json    — e2e 3-way comparison (same schema as perf_compare.py)
  layer_profile.json   — per-layer TRT timing (IProfiler)

Usage:
    # Minimal: builds engine on the fly, all passes
    python tools/profile.py --model example-org/example-decoder

    # Pre-built bundle, custom prompt, save JSONs
    python tools/profile.py \\
      --model example-org/example-decoder \\
      --bundle /path/to/model.bundle \\
      --prompt "The capital of France is" \\
      --max-new-tokens 20 \\
      --warmup 3 --iterations 10 \\
      --output-dir /tmp/model_profile \\
      --json

    # Skip torch.compile (e.g. on environments without inductor)
    python tools/profile.py --model example-org/example-decoder --no-compile

    # Skip per-layer profiling (faster — just e2e perf compare)
    python tools/profile.py --model example-org/example-decoder --no-layer-profile
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Helpers shared with perf_compare (avoid re-implementing)
# ---------------------------------------------------------------------------

def _import_perf_compare():
    tools_dir = Path(__file__).parent
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    import perf_compare as pc
    return pc


# ---------------------------------------------------------------------------
# TRT + IProfiler combined pass
# ---------------------------------------------------------------------------

def bench_trt_with_layer_profile(
    engine_plan: bytes,
    num_layers: int,
    max_cache_length: int,
    input_ids: list[int],
    max_new_tokens: int,
    warmup: int,
    iterations: int,
    eos_token_id: int | None,
    verbose: bool,
    *,
    runtime_strategy: str,
    config=None,
    bundle_path: str = "",
) -> tuple[dict, dict]:
    """Run TRT inference with LayerProfiler attached.

    Returns (trt_bench_result, layer_profile_dict) collected in a single run.
    trt_bench_result has the same schema as perf_compare.bench_trt().
    """
    from layer_profiler import LayerProfiler
    from tool_helpers import make_family_debug_runner

    profiler = LayerProfiler()
    runner = make_family_debug_runner(
        engine_plan=engine_plan,
        runtime_strategy=runtime_strategy,
        max_cache_length=max_cache_length,
        num_layers=num_layers,
        config=config,
        bundle_path=bundle_path,
        profiler=profiler,
    )

    prefill_times: list[float] = []
    decode_times: list[float] = []
    decode_token_counts: list[int] = []
    gen_ids: list[int] = []

    total_runs = warmup + iterations
    for run_idx in range(total_runs):
        is_warmup = run_idx < warmup
        if run_idx == warmup:
            profiler.reset()  # discard warmup layer timings

        runner.reset()

        # Prefill
        t0 = time.perf_counter()
        result = None
        for tid in input_ids:
            result = runner.step(tid)
        logits = result["logits"].flatten()
        prefill_ms = (time.perf_counter() - t0) * 1000

        # Decode
        tokens_generated = 0
        run_gen_ids: list[int] = []
        t0 = time.perf_counter()
        for _ in range(max_new_tokens):
            next_token = int(np.argmax(logits))
            run_gen_ids.append(next_token)
            if eos_token_id is not None and next_token == eos_token_id:
                break
            result = runner.step(next_token)
            logits = result["logits"].flatten()
            tokens_generated += 1
        decode_ms = (time.perf_counter() - t0) * 1000

        if not is_warmup:
            prefill_times.append(prefill_ms)
            decode_times.append(decode_ms)
            decode_token_counts.append(tokens_generated)
            gen_ids = run_gen_ids

        if verbose:
            tag = "warmup" if is_warmup else f"iter {run_idx - warmup + 1}"
            print(f"  [trt {tag}] prefill={prefill_ms:.2f}ms "
                  f"decode={decode_ms:.2f}ms ({tokens_generated} tokens)",
                  file=sys.stderr)

    trt_res = {
        "prefill_times": prefill_times,
        "decode_times": decode_times,
        "decode_token_counts": decode_token_counts,
        "gen_ids": gen_ids,
    }
    layer_data = profiler.to_dict()
    return trt_res, layer_data


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"mean": m, "std": s}


def _fmt(mean: float, std: float) -> str:
    return f"{mean:.1f} +/- {std:.1f}"


def _speedup(baseline: float, target: float) -> str:
    return f"{baseline / target:.2f}x" if target > 0 else "N/A"


def print_combined_report(
    model_name: str,
    prompt: str,
    num_input_tokens: int,
    max_new_tokens: int,
    warmup: int,
    iterations: int,
    hf_dtype: str,
    trt_res: dict,
    hf_res: dict,
    compile_res: dict | None,
    compile_mode: str,
    layer_data: dict | None,
    gpu: str,
    trt_ver: str,
    cpp_res: dict | None = None,
) -> None:
    sep_wide = "=" * 80
    print(f"\n{sep_wide}")
    print(f"Profile: {model_name}")
    print(f"GPU: {gpu}   TRT: {trt_ver}   HF dtype: {hf_dtype}")
    print(f'Prompt: "{prompt[:70]}{"..." if len(prompt) > 70 else ""}" '
          f"({num_input_tokens} tokens)")
    print(f"{iterations} iters × {max_new_tokens} decode steps  |  "
          f"{warmup} warmup")

    # --- E2E comparison table ---
    trt_pf = _stats(trt_res["prefill_times"])
    trt_dc = _stats(trt_res["decode_times"])
    hf_pf = _stats(hf_res["prefill_times"])
    hf_dc = _stats(hf_res["decode_times"])

    trt_avg_tok = (statistics.mean(trt_res["decode_token_counts"])
                   if trt_res["decode_token_counts"] else 0)
    hf_avg_tok = (statistics.mean(hf_res["decode_token_counts"])
                  if hf_res["decode_token_counts"] else 0)

    def _tps(dc: dict, avg_tok: float) -> str:
        if avg_tok > 0 and dc["mean"] > 0:
            return f"{1000.0 * avg_tok / dc['mean']:.0f}"
        return "N/A"

    backends: list[tuple[str, dict, dict, float]] = [
        ("TRT (Python)", trt_pf, trt_dc, trt_avg_tok),
        ("HF (eager)", hf_pf, hf_dc, hf_avg_tok),
    ]
    if cpp_res:
        cpp_pf = _stats(cpp_res["prefill_times"])
        cpp_dc = _stats(cpp_res["decode_times"])
        cpp_avg_tok = (statistics.mean(cpp_res["decode_token_counts"])
                       if cpp_res["decode_token_counts"] else 0)
        backends.insert(0, ("TRT (C++)", cpp_pf, cpp_dc, cpp_avg_tok))
    if compile_res:
        cp_pf = _stats(compile_res["prefill_times"])
        cp_dc = _stats(compile_res["decode_times"])
        cp_avg_tok = (statistics.mean(compile_res["decode_token_counts"])
                      if compile_res["decode_token_counts"] else 0)
        backends.append((f"HF ({compile_mode})", cp_pf, cp_dc, cp_avg_tok))

    col_w = 22
    print(f"\n{'─' * 80}")
    print("  E2E Latency Comparison")
    print(f"{'─' * 80}")
    header = f"  {'':18s}"
    for label, _, _, _ in backends:
        header += f"  {label:>{col_w}s}"
    print(header)
    print(f"  {'─'*18}" + (f"  {'─'*col_w}" * len(backends)))

    for row_label, getpf, getdc in [
        ("Prefill (ms)", lambda b: b[1], None),
        ("Decode (ms)", lambda b: b[2], None),
        ("Throughput (t/s)", None, None),
    ]:
        row = f"  {row_label:18s}"
        for label, pf, dc, avg_tok in backends:
            if row_label == "Prefill (ms)":
                row += f"  {_fmt(pf['mean'], pf['std']):>{col_w}s}"
            elif row_label == "Decode (ms)":
                row += f"  {_fmt(dc['mean'], dc['std']):>{col_w}s}"
            else:
                row += f"  {_tps(dc, avg_tok):>{col_w}s}"
        print(row)

    # Speedup row — baseline is the fastest TRT variant (C++ if present)
    if len(backends) > 1:
        baseline_dc_mean = backends[0][2]["mean"]
        baseline_label = backends[0][0]
        row_label = f"Speedup vs {baseline_label}"[:18]
        row = f"  {row_label:18s}"
        for i, (label, pf, dc, avg_tok) in enumerate(backends):
            if i == 0:
                row += f"  {'—':>{col_w}s}"
            else:
                row += f"  {_speedup(dc['mean'], baseline_dc_mean):>{col_w}s}"
        print(row)

    token_match = trt_res["gen_ids"] == hf_res["gen_ids"]
    print(f"\n  Token match (TRT Python vs HF eager): {token_match}")
    print("  * Prefill: HF batches all tokens; TRT runs token-by-token")

    # --- Per-layer summary ---
    if layer_data and layer_data.get("layers"):
        layers = layer_data["layers"]
        total_layer_ms = layer_data["total_ms"]
        top_n = layers[:15]  # top 15 slowest

        print(f"\n{'─' * 80}")
        print(f"  Per-Layer TRT Kernel Timing  "
              f"(total: {total_layer_ms:.3f} ms/step)  — top 15 slowest")
        print(f"{'─' * 80}")
        print(f"  {'Layer':<44s}  {'Mean (ms)':>9s}  {'Std':>7s}  {'%':>6s}")
        print(f"  {'─'*44}  {'─'*9}  {'─'*7}  {'─'*6}")
        cumulative = 0.0
        for layer in top_n:
            name = layer["name"]
            if len(name) > 44:
                name = name[:41] + "..."
            print(f"  {name:<44s}  {layer['mean_ms']:>9.4f}"
                  f"  {layer['std_ms']:>7.4f}  {layer['pct']:>5.1f}%")
            cumulative += layer["pct"]
        if len(layers) > 15:
            remaining_ms = total_layer_ms - sum(
                l["mean_ms"] for l in top_n)
            remaining_pct = 100.0 - cumulative
            print(f"  {'... (' + str(len(layers) - 15) + ' more layers)':<44s}"
                  f"  {remaining_ms:>9.4f}  {'':>7s}  {remaining_pct:>5.1f}%")
        print(f"  Bottleneck: {layers[0]['name']!r}  "
              f"({layers[0]['mean_ms']:.4f} ms, {layers[0]['pct']:.1f}%)")

    print(f"\n{sep_wide}\n")


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

def _generate_html_report(out_dir: Path, nsight_cpp_data: dict | None) -> None:
    """Call profile_report.py to produce report.html from saved JSON artifacts."""
    import subprocess as _sp
    report_script = Path(__file__).parent / "profile_report.py"
    if not report_script.exists():
        return

    cmd = [sys.executable, str(report_script), "--output-dir", str(out_dir),
           "-o", str(out_dir / "report.html")]
    if nsight_cpp_data:
        cmd += ["--nsight-cpp", str(out_dir / "nsight_cpp.json")]

    try:
        _sp.run(cmd, check=True)
    except Exception as exc:
        print(f"[profile] HTML report generation failed: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unified profiling: per-layer + e2e perf for a single model")
    parser.add_argument("--model", required=True,
                        help="HF repo ID or local model directory")
    parser.add_argument("--bundle",
                        help="Pre-built .bundle artifact (skips engine build)")
    parser.add_argument("--prompt", default="The capital of France is",
                        help="Input prompt")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--max-cache-length", type=int, default=256,
                        help="TRT KV cache length (ignored with --bundle)")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Warmup iterations (not timed)")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Timed iterations")
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "float32", "bfloat16"],
                        help="HF model dtype (default: float16)")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-compile", action="store_true",
                        help="Skip torch.compile benchmark")
    parser.add_argument("--compile-mode", default="reduce-overhead",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode (default: reduce-overhead)")
    parser.add_argument("--trtmc-binary",
                        help="Path to compiled trtmc binary for C++ benchmark pass "
                             "(e.g. ./build/trtmc).  Skipped when not provided.")
    parser.add_argument("--no-layer-profile", action="store_true",
                        help="Skip per-layer IProfiler pass (faster)")
    parser.add_argument("--cpu-profile", action="store_true",
                        help="Run CPU phase breakdown pass (adds ~1 min)")
    parser.add_argument("--nsight", action="store_true",
                        help="Run nsys capture of the C++ binary (requires "
                             "--trtmc-binary and --bundle)")
    parser.add_argument("--output-dir", default=".",
                        help="Directory for JSON artifacts (default: .)")
    parser.add_argument("--json", action="store_true",
                        help="Save JSON artifacts to --output-dir")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    pc = _import_perf_compare()

    # -- Tokenize --
    from tensorrt_model_connect.engine_builder import _resolve_model
    from transformers import AutoTokenizer

    print("[profile] Loading tokenizer ...", file=sys.stderr)
    model_dir = _resolve_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=args.trust_remote_code)
    input_ids = tokenizer.encode(args.prompt)
    eos_token_id = tokenizer.eos_token_id
    print(f"[profile] Prompt: {len(input_ids)} tokens", file=sys.stderr)

    # -- Build / load TRT engine --
    if args.bundle:
        print(f"[profile] Loading bundle: {args.bundle}", file=sys.stderr)
        engine_plan, num_layers, max_cache_length, bundle_config, perf_handler = \
            pc.load_trt_from_bundle(args.bundle)
        runtime_strategy = str(bundle_config.get("runtime_strategy") or "")
        runner_config = bundle_config
        runner_bundle_path = args.bundle
    else:
        from tool_helpers import runtime_strategy_from_config
        engine_plan, config, _, perf_handler = pc.build_trt_engine(
            args.model, args.max_cache_length, args.verbose)
        num_layers = config.num_hidden_layers
        max_cache_length = args.max_cache_length
        runtime_strategy = runtime_strategy_from_config(config)
        runner_config = config
        runner_bundle_path = ""

    if not pc._handler_supports(perf_handler, "supports_layer_profile", True):
        print(pc._handler_attr(
            perf_handler,
            "layer_profile_skip_message",
            "[profile] Family runtime does not support per-layer IProfiler; skipping.",
        ), file=sys.stderr)
        args.no_layer_profile = True

    # -- Pass 0: C++ binary (optional) --
    cpp_res: dict | None = None
    if args.trtmc_binary and args.bundle:
        print(f"[profile] C++ binary pass ({args.warmup} warmup + "
              f"{args.iterations} iters) ...", file=sys.stderr)
        cpp_res = pc.bench_trtmc_cpp(
            binary=args.trtmc_binary,
            bundle_path=args.bundle,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            warmup=args.warmup,
            iterations=args.iterations,
            verbose=args.verbose,
        )
        if cpp_res is None:
            print("[profile] C++ pass failed; continuing without it.",
                  file=sys.stderr)
    elif args.trtmc_binary and not args.bundle:
        print("[profile] --trtmc-binary requires --bundle (no on-the-fly engine "
              "build for C++ pass); skipping C++ pass.", file=sys.stderr)

    # -- Pass 1: TRT e2e timing (no IProfiler — accurate latency) --
    print(f"[profile] TRT pass ({args.warmup} warmup + "
          f"{args.iterations} iters) ...", file=sys.stderr)

    if perf_handler is not None:
        trt_res = pc.bench_trt_family(
            perf_handler, engine_plan, num_layers, max_cache_length,
            input_ids, args.max_new_tokens,
            args.warmup, args.iterations, eos_token_id, args.verbose)
    else:
        trt_res = pc.bench_trt(
            engine_plan, num_layers, max_cache_length,
            input_ids, args.max_new_tokens,
            args.warmup, args.iterations, eos_token_id, args.verbose,
            runtime_strategy=runtime_strategy,
            config=runner_config,
            bundle_path=runner_bundle_path)

    # -- Pass 1b: TRT per-layer profiling (IProfiler attached — timing discarded) --
    layer_data = None
    if not args.no_layer_profile and pc._handler_supports(
        perf_handler, "supports_layer_profile", True
    ):
        print("[profile] Per-layer profiling pass (IProfiler) ...",
              file=sys.stderr)
        _discard_res, layer_data = bench_trt_with_layer_profile(
            engine_plan, num_layers, max_cache_length,
            input_ids, args.max_new_tokens,
            warmup=1, iterations=1,  # minimal — only need layer %
            eos_token_id=eos_token_id, verbose=False,
            runtime_strategy=runtime_strategy,
            config=runner_config,
            bundle_path=runner_bundle_path)

    del engine_plan
    gc.collect()
    import torch
    torch.cuda.empty_cache()

    # -- Pass 2: HF eager --
    print("[profile] HF eager pass ...", file=sys.stderr)
    hf_model = pc.load_hf_model(model_dir, args.dtype, args.trust_remote_code)
    hf_res = pc.bench_hf(
        hf_model, input_ids, args.max_new_tokens,
        args.warmup, args.iterations, eos_token_id, args.verbose)
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()

    # -- Pass 3: HF torch.compile --
    compile_res = None
    if not args.no_compile and pc._handler_supports(
        perf_handler, "supports_hf_compile", True
    ):
        print(f"[profile] HF torch.compile({args.compile_mode!r}) pass ...",
              file=sys.stderr)
        hf_model2 = pc.load_hf_model(
            model_dir, args.dtype, args.trust_remote_code)
        try:
            compile_res = pc.bench_hf_compiled(
                hf_model2, input_ids, args.max_new_tokens,
                args.warmup, args.iterations, eos_token_id,
                args.compile_mode, args.verbose)
        except Exception as exc:
            # reduce-overhead uses CUDA graphs which can conflict with
            # some HF model architectures (e.g. RoPE tensor reuse).
            # Fall back to 'default' compile mode before giving up.
            if args.compile_mode != "default":
                fallback = "default"
                print(f"[profile] torch.compile({args.compile_mode!r}) failed, "
                      f"retrying with mode={fallback!r} ...", file=sys.stderr)
                del hf_model2
                gc.collect()
                torch.cuda.empty_cache()
                hf_model2 = pc.load_hf_model(
                    model_dir, args.dtype, args.trust_remote_code)
                try:
                    compile_res = pc.bench_hf_compiled(
                        hf_model2, input_ids, args.max_new_tokens,
                        args.warmup, args.iterations, eos_token_id,
                        fallback, args.verbose)
                    args.compile_mode = fallback
                except Exception as exc2:
                    print(f"[profile] torch.compile({fallback!r}) also failed "
                          f"({exc2}); skipping.", file=sys.stderr)
            else:
                print(f"[profile] torch.compile failed ({exc}); skipping.",
                      file=sys.stderr)
        del hf_model2
        gc.collect()
        torch.cuda.empty_cache()

    # -- Pass 4: CPU phase breakdown (optional) --
    cpu_profile_data: dict | None = None
    if getattr(args, "cpu_profile", False) and pc._handler_supports(
        perf_handler, "supports_cpu_phase_profile", True
    ):
        print("[profile] CPU phase breakdown pass ...", file=sys.stderr)
        import subprocess as _sp
        import tempfile as _tf
        # Ensure output dir exists before subprocess tries to write there
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        cpu_json = str(Path(args.output_dir) / "cpu_profile.json") if args.json else \
                   str(Path(_tf.gettempdir()) / "cpu_profile_tmp.json")
        cpu_cmd = [
            sys.executable, str(Path(__file__).parent / "cpu_profile.py"),
            "--model", args.model,
            "--bundle", args.bundle,
            "--prompt", args.prompt,
            "--max-new-tokens", str(args.max_new_tokens),
            "--warmup", str(args.warmup),
            "--iterations", str(args.iterations),
            "--json", cpu_json,
        ]
        if args.bundle:
            pass  # bundle already in cmd
        try:
            env = {**os.environ}
            result = _sp.run(cpu_cmd, capture_output=False, env=env)
            if result.returncode == 0:
                import json as _json
                cpu_profile_data = _json.loads(Path(cpu_json).read_text())
                print(f"[profile] CPU profile saved to {cpu_json}", file=sys.stderr)
        except Exception as exc:
            print(f"[profile] CPU phase pass failed: {exc}", file=sys.stderr)

    # -- Pass 5: nsight C++ capture (optional) --
    nsight_cpp_data: dict | None = None
    if getattr(args, "nsight", False) and args.trtmc_binary and args.bundle:
        print("[profile] Nsight Systems C++ pass ...", file=sys.stderr)
        import subprocess as _sp
        import tempfile as _tf
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        nsight_json = str(Path(args.output_dir) / "nsight_cpp.json") if args.json else \
                      str(Path(_tf.gettempdir()) / "nsight_cpp_tmp.json")
        nsight_cmd = [
            sys.executable, str(Path(__file__).parent / "nsight_collect.py"),
            "--mode", "nsys",
            "--backend", "cpp",
            "--model", args.model,
            "--bundle", args.bundle,
            "--trtmc-binary", args.trtmc_binary,
            "--prompt", args.prompt,
            "--max-new-tokens", str(min(args.max_new_tokens, 20)),
            "--output-dir", str(Path(args.output_dir) / "nsight_files"),
            "--top-n", "15",
            "--json", nsight_json,
        ]
        try:
            env = {**os.environ}
            result = _sp.run(nsight_cmd, capture_output=False, env=env)
            if result.returncode == 0:
                nsight_cpp_data = json.loads(Path(nsight_json).read_text())
                print(f"[profile] Nsight C++ results saved to {nsight_json}",
                      file=sys.stderr)
            else:
                print(f"[profile] Nsight pass exited with code {result.returncode}",
                      file=sys.stderr)
        except Exception as exc:
            print(f"[profile] Nsight pass failed: {exc}", file=sys.stderr)
    elif getattr(args, "nsight", False):
        print("[profile] --nsight requires --trtmc-binary and --bundle; skipping.",
              file=sys.stderr)

    # -- Report --
    gpu = pc._get_gpu_name()
    trt_ver = pc._get_trt_version()

    if layer_data:
        layer_data["metadata"] = {
            "model": args.model,
            "gpu": gpu,
            "trt_version": trt_ver,
            "prompt_tokens": len(input_ids),
            "max_new_tokens": args.max_new_tokens,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    print_combined_report(
        model_name=args.model,
        prompt=args.prompt,
        num_input_tokens=len(input_ids),
        max_new_tokens=args.max_new_tokens,
        warmup=args.warmup,
        iterations=args.iterations,
        hf_dtype=args.dtype,
        trt_res=trt_res,
        hf_res=hf_res,
        compile_res=compile_res,
        compile_mode=args.compile_mode,
        layer_data=layer_data,
        gpu=gpu,
        trt_ver=trt_ver,
        cpp_res=cpp_res,
    )

    # -- Save JSON artifacts --
    if args.json:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        perf_data = pc.build_json_output(
            args.model, args.prompt, len(input_ids),
            args.max_new_tokens, args.iterations, args.warmup,
            args.dtype, trt_res, hf_res,
            compile_res=compile_res, compile_mode=args.compile_mode,
            cpp_res=cpp_res)
        perf_path = out_dir / "perf_compare.json"
        perf_path.write_text(json.dumps(perf_data, indent=2))
        print(f"[profile] Saved {perf_path}", file=sys.stderr)

        if layer_data:
            layer_path = out_dir / "layer_profile.json"
            layer_path.write_text(json.dumps(layer_data, indent=2))
            print(f"[profile] Saved {layer_path}", file=sys.stderr)

        if cpu_profile_data:
            cpu_path = out_dir / "cpu_profile.json"
            if not cpu_path.exists():  # may have been written by subprocess already
                cpu_path.write_text(json.dumps(cpu_profile_data, indent=2))
            print(f"[profile] CPU profile at {cpu_path}", file=sys.stderr)

        if nsight_cpp_data:
            # nsight_cpp.json was already written by the subprocess; confirm it
            print(f"[profile] Nsight C++ data at {out_dir / 'nsight_cpp.json'}",
                  file=sys.stderr)

        # Generate HTML report when JSON artifacts have been saved
        _generate_html_report(out_dir, nsight_cpp_data)

    if trt_res["gen_ids"] != hf_res["gen_ids"]:
        print("WARNING: TRT and HF generated different tokens.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
