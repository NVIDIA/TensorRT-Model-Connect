#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-side timing breakdown for TRT inference.

Instruments each phase of a single TRT inference step to show where
host-side time is spent across different runtime strategies:

  mask_build    -- attention mask construction + host buffer prep (pure CPU)
  h2d           -- H2D memcpy dispatch: token_id, position_id, mask
  tensor_bind   -- context.set_tensor_address() calls (scales with num_layers)
  execute       -- execute_async_v3 dispatch + GPU kernel execution
  d2d_cache     -- D2D cache/state update memcpy dispatch + completion
  d2h           -- D2H logits readback + stream sync
  argmax        -- np.argmax on host logits

Phases are separated by explicit cudaStreamSynchronize so each is measured
independently. This makes the run ~2-3x slower than production but gives
accurate per-phase attribution.

Supported runtime strategies:
  family-owned decoder runtimes   -- use family-owned debug runners
  family-owned custom runtimes    -- use model-owned CPU profiling hooks

Usage:
    # Standard decoder
    python tools/cpu_profile.py \\
      --model example-org/example-decoder \\
      --prompt "The capital of France is" \\
      --max-new-tokens 10 \\
      --warmup 3 --iterations 20 \\
      --json cpu_profile.json

    # Family-owned runtime model
    python tools/cpu_profile.py \\
      --model example-org/example-family-runtime \\
      --runner family \\
      --max-new-tokens 10 \\
      --json cpu_profile_family.json

    # Use a pre-built bundle (skips engine build)
    python tools/cpu_profile.py \\
      --model example-org/example-decoder \\
      --bundle /path/to/model.trtfb \\
      --max-new-tokens 10
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

# cuda-python >= 13 uses cuda.bindings.runtime; older versions use cuda.cudart.
try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart  # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Phase constants
# ---------------------------------------------------------------------------

DECODER_PHASES = ("mask_build", "h2d", "tensor_bind", "execute",
                  "d2d_cache", "d2h", "argmax")


# ---------------------------------------------------------------------------
# Timed runner — decoder (TrtRunner subclass)
# ---------------------------------------------------------------------------

class _TimedDecoderRunner:
    """Wraps a family-owned debug runner and provides timed_step().

    Each phase is separated by cudaStreamSynchronize so times are accurate.
    VL/DeepStack inputs are intentionally omitted — this tool profiles the
    standard text-generation path only.
    """

    PHASES = DECODER_PHASES

    def __init__(self, engine_plan: bytes, max_cache_length: int,
                 num_layers: int, runtime_strategy: str, config=None,
                 bundle_path: str = ""):
        from tool_helpers import make_family_debug_runner
        self._runner = make_family_debug_runner(
            engine_plan=engine_plan,
            runtime_strategy=runtime_strategy,
            max_cache_length=max_cache_length,
            num_layers=num_layers,
            config=config,
            bundle_path=bundle_path,
        )
        self._phase_times: dict[str, list[float]] = {p: [] for p in self.PHASES}

    def reset(self) -> None:
        self._runner.reset()

    def reset_timing(self) -> None:
        for p in self.PHASES:
            self._phase_times[p].clear()

    def step(self, token_id: int) -> np.ndarray:
        """Delegate to the underlying runner (used for warmup)."""
        return self._runner.step(token_id)["logits"].flatten()

    def timed_step(self, token_id: int) -> np.ndarray:
        """Like step() but measures each phase with explicit syncs."""
        H2D = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        D2H = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        D2D = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice

        r = self._runner
        stream = r.stream
        attention_window = r.max_cache_length + 1

        # ------------------------------------------------------------------
        # Phase 1: mask_build (pure CPU — no GPU ops)
        # ------------------------------------------------------------------
        t = time.perf_counter()
        position_id = min(r.cache_length, r.max_cache_length)
        r._h_mask[:] = -1e9
        valid = min(r.cache_length, r.max_cache_length)
        r._h_mask[0, :valid] = 0.0
        r._h_mask[0, -1] = 0.0
        r._h_token_id[0] = token_id
        r._h_position_id[0] = position_id
        self._phase_times["mask_build"].append((time.perf_counter() - t) * 1000)

        # ------------------------------------------------------------------
        # Phase 2: h2d — async dispatches + sync
        # ------------------------------------------------------------------
        t = time.perf_counter()
        cudart.cudaMemcpyAsync(
            r._d_token_id, r._h_token_id.ctypes.data, 4, H2D, stream)
        cudart.cudaMemcpyAsync(
            r._d_position_id, r._h_position_id.ctypes.data, 4, H2D, stream)
        cudart.cudaMemcpyAsync(
            r._d_mask, r._h_mask.ctypes.data,
            attention_window * 4, H2D, stream)
        cudart.cudaStreamSynchronize(stream)
        self._phase_times["h2d"].append((time.perf_counter() - t) * 1000)

        # ------------------------------------------------------------------
        # Phase 3: tensor_bind — pure CPU, scales with num_layers
        # ------------------------------------------------------------------
        t = time.perf_counter()
        r.context.set_tensor_address("token_id", r._d_token_id)
        r.context.set_tensor_address("position_id", r._d_position_id)
        r.context.set_tensor_address("attention_mask", r._d_mask)
        r.context.set_tensor_address("logits", r._d_logits)
        for i in range(r.num_layers):
            r.context.set_tensor_address(f"cache_k_{i}", r._d_cache_k[i])
            r.context.set_tensor_address(f"cache_v_{i}", r._d_cache_v[i])
            r.context.set_tensor_address(f"present_k_{i}", r._d_present_k[i])
            r.context.set_tensor_address(f"present_v_{i}", r._d_present_v[i])
        for name in r._debug_output_names:
            r.context.set_tensor_address(name, r._d_debug[name])
        self._phase_times["tensor_bind"].append(
            (time.perf_counter() - t) * 1000)

        # ------------------------------------------------------------------
        # Phase 4: execute — dispatch + wait for GPU kernels
        # ------------------------------------------------------------------
        t = time.perf_counter()
        r.context.execute_async_v3(stream)
        cudart.cudaStreamSynchronize(stream)
        self._phase_times["execute"].append((time.perf_counter() - t) * 1000)

        # ------------------------------------------------------------------
        # Phase 5: d2d_cache — KV-cache update (scales with num_layers)
        # ------------------------------------------------------------------
        row_bytes = r.attention_size * 4
        t = time.perf_counter()
        for i in range(r.num_layers):
            for cache_buf, present_buf in [
                (r._d_cache_k[i], r._d_present_k[i]),
                (r._d_cache_v[i], r._d_present_v[i]),
            ]:
                if r.cache_length < r.max_cache_length:
                    offset = r.cache_length * row_bytes
                    cudart.cudaMemcpyAsync(
                        cache_buf + offset, present_buf, row_bytes, D2D, stream)
                else:
                    cudart.cudaMemcpyAsync(
                        cache_buf, cache_buf + row_bytes,
                        (r.max_cache_length - 1) * row_bytes, D2D, stream)
                    offset = (r.max_cache_length - 1) * row_bytes
                    cudart.cudaMemcpyAsync(
                        cache_buf + offset, present_buf, row_bytes, D2D, stream)
        cudart.cudaStreamSynchronize(stream)
        self._phase_times["d2d_cache"].append(
            (time.perf_counter() - t) * 1000)

        # ------------------------------------------------------------------
        # Phase 6: d2h — logits readback + sync
        # ------------------------------------------------------------------
        t = time.perf_counter()
        cudart.cudaMemcpyAsync(
            r._h_logits.ctypes.data, r._d_logits,
            r._logits_numel * 4, D2H, stream)
        cudart.cudaStreamSynchronize(stream)
        self._phase_times["d2h"].append((time.perf_counter() - t) * 1000)

        r.cache_length = min(r.cache_length + 1, r.max_cache_length)
        logits = r._h_logits.flatten()

        # ------------------------------------------------------------------
        # Phase 7: argmax (pure CPU)
        # ------------------------------------------------------------------
        t = time.perf_counter()
        int(np.argmax(logits))
        self._phase_times["argmax"].append((time.perf_counter() - t) * 1000)

        return logits

    @property
    def phase_times(self) -> dict[str, list[float]]:
        return self._phase_times


# ---------------------------------------------------------------------------
# Profiling loop
# ---------------------------------------------------------------------------

def _run_profile(runner, input_ids: list[int], max_new_tokens: int,
                 warmup: int, iterations: int,
                 eos_token_id: int | None, verbose: bool) -> None:
    """Run warmup + timed iterations, accumulating phase times in runner."""
    total_runs = warmup + iterations
    for run_idx in range(total_runs):
        is_warmup = run_idx < warmup
        runner.reset()
        if is_warmup:
            runner.reset_timing()

        logits = None
        for tid in input_ids:
            logits = runner.step(tid)  # warmup uses plain step()

        next_token = int(np.argmax(logits))
        for step_i in range(max_new_tokens):
            if is_warmup:
                logits = runner.step(next_token)
            else:
                logits = runner.timed_step(next_token)
            next_token = int(np.argmax(logits))
            if eos_token_id is not None and next_token == eos_token_id:
                break

        if verbose:
            tag = "warmup" if is_warmup else f"iter {run_idx - warmup + 1}"
            print(f"  [{tag}] done", file=sys.stderr)

    # Discard the final warmup timing residue accumulated during prefill steps
    runner.reset_timing()

    # Re-run timed iterations only (exclude prefill from per-phase stats)
    for run_idx in range(iterations):
        runner.reset()
        logits = None
        for tid in input_ids:
            logits = runner.step(tid)  # prefill not timed

        next_token = int(np.argmax(logits))
        for _ in range(max_new_tokens):
            logits = runner.timed_step(next_token)
            next_token = int(np.argmax(logits))
            if eos_token_id is not None and next_token == eos_token_id:
                break


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _aggregate(phase_times: dict[str, list[float]]) -> list[dict]:
    """Compute mean/std/pct for each phase."""
    total = sum(
        statistics.mean(v) for v in phase_times.values() if v
    )
    rows = []
    for phase, times in phase_times.items():
        if not times:
            continue
        mean_ms = statistics.mean(times)
        std_ms = statistics.stdev(times) if len(times) > 1 else 0.0
        pct = 100.0 * mean_ms / total if total > 0 else 0.0
        rows.append({
            "phase": phase,
            "mean_ms": round(mean_ms, 4),
            "std_ms": round(std_ms, 4),
            "pct": round(pct, 1),
            "samples": len(times),
        })
    return rows


def _print_table(rows: list[dict], runner_type: str,
                 model_name: str, num_layers: int,
                 prompt_tokens: int, max_new_tokens: int) -> None:
    total_ms = sum(r["mean_ms"] for r in rows)
    sep = "-" * 62
    print(f"\n{'=' * 62}")
    print(f"CPU Phase Breakdown: {model_name}  [{runner_type}]")
    print(f"  Layers: {num_layers}  |  Prompt: {prompt_tokens} tokens  "
          f"|  Decode steps: {max_new_tokens}")
    print("  (times are per decode step, averaged over timed iterations)")
    print(f"{'=' * 62}")
    print(f"  {'Phase':<16s}  {'Mean (ms)':>10s}  {'Std':>8s}  {'%':>6s}")
    print(sep)
    for r in rows:
        print(f"  {r['phase']:<16s}  {r['mean_ms']:>10.4f}  "
              f"{r['std_ms']:>8.4f}  {r['pct']:>5.1f}%")
    print(sep)
    print(f"  {'TOTAL':<16s}  {total_ms:>10.4f}")
    print()
    bottleneck = max(rows, key=lambda r: r["mean_ms"])
    print(f"  Bottleneck: {bottleneck['phase']}  "
          f"({bottleneck['mean_ms']:.4f} ms, {bottleneck['pct']:.1f}%)")


def _get_gpu_name() -> str:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            return r.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return "unknown"


def _get_trt_version() -> str:
    try:
        import tensorrt as trt
        return trt.__version__
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CPU-side per-phase timing breakdown for TRT inference")
    parser.add_argument("--model", required=True,
                        help="HF repo ID or local model directory")
    parser.add_argument("--bundle",
                        help="Pre-built .trtfb bundle (skips engine build)")
    parser.add_argument("--prompt", default="The capital of France is",
                        help="Input prompt")
    parser.add_argument("--max-new-tokens", type=int, default=10,
                        help="Decode steps to profile per iteration")
    parser.add_argument("--max-cache-length", type=int, default=256,
                        help="TRT KV cache length (ignored with --bundle)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup iterations (not timed)")
    parser.add_argument("--iterations", type=int, default=20,
                        help="Timed iterations")
    parser.add_argument("--runner", choices=["decoder", "family"],
                        default="decoder",
                        help="Runner type matching the model's runtime strategy")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--json", dest="json_path", metavar="PATH",
                        help="Save results to JSON file")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # -- Resolve model and tokenize --
    from tensorrt_model_connect.engine_builder import _resolve_model
    from transformers import AutoTokenizer

    model_dir = _resolve_model(args.model)
    print("[cpu_profile] Loading tokenizer ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=args.trust_remote_code)
    input_ids = tokenizer.encode(args.prompt)
    eos_token_id = tokenizer.eos_token_id
    print(f"[cpu_profile] Prompt: {len(input_ids)} tokens", file=sys.stderr)

    # -- Build or load TRT engine --
    from perf_compare import (
        _handler_attr,
        build_trt_engine,
        load_trt_from_bundle,
    )

    if args.bundle:
        print(f"[cpu_profile] Loading bundle: {args.bundle}", file=sys.stderr)
        engine_plan, num_layers, max_cache_length, bundle_config, perf_handler = \
            load_trt_from_bundle(args.bundle)
        runtime_strategy = str(bundle_config.get("runtime_strategy") or "")
        runner_config = bundle_config
        runner_bundle_path = args.bundle
        runner_type = _handler_attr(
            perf_handler, "cpu_profile_runner_type", "decoder")
        if args.runner != runner_type:
            print(f"[cpu_profile] Note: bundle is {runner_type!r}, "
                  f"ignoring --runner={args.runner!r}", file=sys.stderr)
    else:
        from tool_helpers import runtime_strategy_from_config
        engine_plan, config, _, perf_handler = build_trt_engine(
            args.model, args.max_cache_length, args.verbose)
        num_layers = config.num_hidden_layers
        max_cache_length = args.max_cache_length
        runtime_strategy = runtime_strategy_from_config(config)
        runner_config = config
        runner_bundle_path = ""
        runner_type = _handler_attr(
            perf_handler, "cpu_profile_runner_type", args.runner)

    # -- Build timed runner --
    print(f"[cpu_profile] Building {runner_type} runner ...", file=sys.stderr)
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

    # -- Run profiling --
    print(f"[cpu_profile] Warmup ({args.warmup}) + "
          f"profiling ({args.iterations} iters × {args.max_new_tokens} steps) ...",
          file=sys.stderr)
    _run_profile(runner, input_ids, args.max_new_tokens,
                 args.warmup, args.iterations, eos_token_id, args.verbose)

    # -- Report --
    rows = _aggregate(runner.phase_times)
    _print_table(rows, runner_type, args.model, num_layers,
                 len(input_ids), args.max_new_tokens)

    # -- JSON output --
    if args.json_path:
        data = {
            "metadata": {
                "model": args.model,
                "runner_type": runner_type,
                "gpu": _get_gpu_name(),
                "trt_version": _get_trt_version(),
                "num_layers": num_layers,
                "prompt": args.prompt,
                "prompt_tokens": len(input_ids),
                "max_new_tokens": args.max_new_tokens,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "phases": rows,
            "total_ms": round(sum(r["mean_ms"] for r in rows), 4),
            "bottleneck": max(rows, key=lambda r: r["mean_ms"])["phase"],
        }
        with open(args.json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[cpu_profile] Results saved to {args.json_path}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
