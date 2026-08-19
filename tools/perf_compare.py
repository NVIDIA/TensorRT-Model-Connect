#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TRT vs HuggingFace inference performance comparison.

Runs both backends in-process Python for a controlled, apples-to-apples
comparison. TRT uses the family-owned debug runner for the runtime strategy; HF uses
AutoModelForCausalLM on CUDA with KV cache enabled.

TRT and HF run serially (not simultaneously), so large models that
exceed GPU memory when loaded together are supported.  Use --dtype
float16 (default) to reduce HF memory usage.

Usage:
    # Build TRT engine on the fly from HF model
    python3 tools/perf_compare.py \
      --model example-org/example-decoder \
      --prompt "The capital of France is" \
      --max-new-tokens 20 \
      --max-cache-length 256 \
      --warmup 2 --iterations 5

    # Use a pre-built bundle (skips engine build)
    python3 tools/perf_compare.py \
      --model example-org/example-decoder \
      --bundle /path/to/model.bundle \
      --prompt "The capital of France is" \
      --max-new-tokens 20

    # Save results as JSON
    python3 tools/perf_compare.py \
      --model example-org/example-decoder \
      --prompt "Hello" --max-new-tokens 20 \
      --json results.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import numpy as np

from tool_helpers import (
    load_hf_model as _load_hf_model_base,
    make_family_debug_runner,
    runtime_strategy_from_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _get_peak_memory_mb() -> float | None:
    """Return peak GPU memory usage in MB, or None if unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    return None


@lru_cache(maxsize=1)
def _family_perf_modules() -> tuple[ModuleType, ...]:
    """Load optional model-owned performance hooks from family folders."""
    repo_root = Path(__file__).resolve().parents[1]
    roots = (repo_root / "python/tensorrt_model_connect/models",)
    handlers: dict[str, Path] = {}
    for root in reversed(roots):
        handlers.update(
            {path.parent.name: path for path in root.glob("*/perf_hooks.py")}
        )
    modules: list[ModuleType] = []
    for family in sorted(handlers):
        handler_path = handlers[family]
        module_name = f"_trtmc_perf_hooks_{handler_path.parent.name}"
        spec = importlib.util.spec_from_file_location(module_name, handler_path)
        if spec is None or spec.loader is None:
            print(f"[perf] WARN: cannot load family perf hook "
                  f"{handler_path}", file=sys.stderr)
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            print(f"[perf] WARN: failed to import family perf hook "
                  f"{handler_path}: {exc}", file=sys.stderr)
            continue
        if callable(getattr(module, "handles_runtime_strategy", None)):
            modules.append(module)
    return tuple(modules)


def find_family_perf_handler(runtime_strategy: str) -> ModuleType | None:
    """Return the model-owned performance hook for a runtime strategy."""
    for module in _family_perf_modules():
        handles = getattr(module, "handles_runtime_strategy")
        if handles(runtime_strategy):
            return module
    return None


def _handler_attr(handler: ModuleType | None, name: str, default):
    if handler is None:
        return default
    value = getattr(handler, name, default)
    return value() if callable(value) else value


def _handler_supports(handler: ModuleType | None, name: str, default: bool) -> bool:
    return bool(_handler_attr(handler, name, default))


def build_trt_engine(model_id_or_path: str, max_cache_length: int,
                     verbose: bool):
    """Build TRT engine and return (engine_plan_bytes, config, model_dir)."""
    from tensorrt_model_connect.engine_builder import _resolve_model
    from tensorrt_model_connect.config import ModelConfig
    from tensorrt_model_connect.models import find_model

    model_dir = _resolve_model(model_id_or_path)
    config = ModelConfig.from_dir(model_dir)
    model = find_model(config)
    if model is None:
        raise ValueError(f"No family model for model_type={config.model_type!r}")

    # Reject unsupported model types
    rt = getattr(model, "runtime_strategy", None)
    if rt == "vision_language":
        raise SystemExit(
            "ERROR: Vision-language models are not supported by perf_compare. "
            "Use tools/diff_vl.py instead.")

    print(f"[perf] Loading weights ({config.model_type}) ...", file=sys.stderr)
    weights = model.load_weights(model_dir, config)
    print(f"[perf] Building TRT engine (cache={max_cache_length}) ...",
          file=sys.stderr)
    engine_plan = model.build_engine(
        config, weights, max_cache_length, verbose=verbose)
    print(f"[perf] Engine built ({len(engine_plan) / 1e6:.1f} MB)",
          file=sys.stderr)

    perf_handler = find_family_perf_handler(rt or "")
    return engine_plan, config, model_dir, perf_handler


def load_trt_from_bundle(bundle_path: str):
    """Load TRT engine from a pre-built bundle.

    Returns (engine_plan_bytes, num_layers, max_cache_length, bundle_config,
             family_perf_handler).
    """
    import struct

    with open(bundle_path, "rb") as f:
        magic = f.read(8)
        if magic != b"BUNDLE\x01\x00":
            raise ValueError(f"Not a valid .bundle artifact: {bundle_path}")
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        sections = header.get("sections", {})
        engine_meta = sections.get("engine_plan")
        if engine_meta is None:
            raise KeyError(
                f"Bundle {bundle_path!r} does not contain section 'engine_plan'")
        data_start = 16 + header_len
        f.seek(data_start + engine_meta["offset"])
        engine_plan = f.read(engine_meta["size"])

        config_meta = sections.get("config.json")
        if config_meta is None:
            bundle_config = {}
        else:
            f.seek(data_start + config_meta["offset"])
            bundle_config = json.loads(f.read(config_meta["size"]).decode("utf-8"))

    # Reject unsupported runtime strategies
    rt = str(bundle_config.get("runtime_strategy") or "")
    if not rt:
        raise SystemExit(
            "ERROR: bundle config.json is missing runtime_strategy; "
            "runtime dispatch requires an explicit model-owned strategy."
        )
    if rt == "vision_language":
        raise SystemExit(
            "ERROR: VL bundles are not supported. Use tools/diff_vl.py.")

    perf_handler = find_family_perf_handler(rt)
    return (engine_plan, header["num_layers"],
            header.get("max_cache_length", 0), bundle_config, perf_handler)


def load_hf_model(model_dir: str, dtype: str, trust_remote_code: bool):
    """Load HF model on CUDA with the specified dtype."""
    import torch

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype)
    if torch_dtype is None:
        raise ValueError(f"Unsupported dtype: {dtype!r}. "
                         f"Choose from: {list(dtype_map)}")

    model = _load_hf_model_base(
        model_dir, trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype, tag="perf")
    model = model.to("cuda")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Timing stats helper
# ---------------------------------------------------------------------------

def _stats(values: list[float]) -> dict:
    """Return mean/std/values dict for a list of measurements."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "values": []}
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"mean": m, "std": s, "values": values}


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def bench_trt(engine_plan: bytes, num_layers: int, max_cache_length: int,
              input_ids: list[int], max_new_tokens: int,
              warmup: int, iterations: int, eos_token_id: int | None,
              verbose: bool, *, runtime_strategy: str, config=None,
              bundle_path: str = "") -> dict:
    """Benchmark TRT inference via the family-owned debug runner.

    Returns dict with timing lists and generated token IDs.
    """
    runner = make_family_debug_runner(
        engine_plan=engine_plan,
        runtime_strategy=runtime_strategy,
        max_cache_length=max_cache_length,
        num_layers=num_layers,
        config=config,
        bundle_path=bundle_path,
    )

    prefill_times: list[float] = []
    decode_times: list[float] = []
    decode_token_counts: list[int] = []
    gen_ids: list[int] = []

    total_runs = warmup + iterations
    for run_idx in range(total_runs):
        is_warmup = run_idx < warmup

        # Reset device-side cache
        runner.reset()

        # -- Prefill --
        t0 = time.perf_counter()
        for tid in input_ids:
            result = runner.step(tid)
        logits = result["logits"].flatten()
        prefill_ms = (time.perf_counter() - t0) * 1000

        # -- Decode --
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

    return {
        "prefill_times": prefill_times,
        "decode_times": decode_times,
        "decode_token_counts": decode_token_counts,
        "gen_ids": gen_ids,
    }


def bench_trt_family(
    handler: ModuleType,
    engine_plan: bytes,
    num_layers: int,
    max_cache_length: int,
    input_ids: list[int],
    max_new_tokens: int,
    warmup: int,
    iterations: int,
    eos_token_id: int | None,
    verbose: bool,
) -> dict:
    """Benchmark TRT inference using a model-owned family performance hook."""
    bench = getattr(handler, "bench_trt", None)
    if not callable(bench):
        raise TypeError(
            f"Family perf hook {handler.__file__} does not define bench_trt()"
        )
    return bench(
        engine_plan=engine_plan,
        num_layers=num_layers,
        max_cache_length=max_cache_length,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        warmup=warmup,
        iterations=iterations,
        eos_token_id=eos_token_id,
        verbose=verbose,
    )


def bench_trtmc_cpp(
    binary: str,
    bundle_path: str,
    prompt: str,
    max_new_tokens: int,
    warmup: int,
    iterations: int,
    hf_python: str | None,
    verbose: bool,
) -> dict | None:
    """Benchmark the C++ trtmc binary using --benchmark / --warmup flags.

    Parses timing from lines printed to stderr by the binary:
      [trtmc.benchmark] setup_ms=X prefill_ms=Y decode_ms=Z
                        generated_tokens_mean=N tokens_per_sec=T

    Returns a dict with the same schema as bench_trt(), or None on error.
    """
    import re

    cmd = [
        binary, "run", bundle_path,
        "--prompt", prompt,
        "--max-new-tokens", str(max_new_tokens),
        "--benchmark", str(iterations),
        "--warmup", str(warmup),
    ]
    if hf_python:
        cmd += ["--hf-python", hf_python]

    if verbose:
        print(f"  [cpp] running: {' '.join(cmd)}", file=sys.stderr)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except Exception as exc:
        print(f"[perf] C++ binary failed: {exc}", file=sys.stderr)
        return None

    if result.returncode != 0:
        print(f"[perf] C++ binary exited {result.returncode}: {result.stderr}",
              file=sys.stderr)
        return None

    # Parse the mean values reported by the C++ benchmark.
    m = re.search(
        r"\[trtmc\.benchmark\]\s+setup_ms=[\d.]+\s+prefill_ms=([\d.]+)"
        r"\s+decode_ms=([\d.]+)\s+generated_tokens_mean=([\d.]+)"
        r"\s+tokens_per_sec=([\d.]+)",
        result.stderr)
    if not m:
        print("[perf] C++ binary: could not parse benchmark output from stderr.",
              file=sys.stderr)
        if verbose:
            print(result.stderr, file=sys.stderr)
        return None

    prefill_ms = float(m.group(1))
    decode_ms  = float(m.group(2))
    generated_tokens_mean = float(m.group(3))
    tps        = float(m.group(4))

    if verbose:
        print(f"  [cpp] prefill={prefill_ms:.2f}ms decode={decode_ms:.2f}ms "
              f"tps={tps:.1f}", file=sys.stderr)

    # The C++ binary reports the mean over all timed iterations; synthesise
    # single-element lists so the same stats helpers work.
    return {
        "prefill_times": [prefill_ms],
        "decode_times": [decode_ms],
        "decode_token_counts": [generated_tokens_mean],
        "gen_ids": [],  # C++ binary doesn't return token IDs
    }


def bench_hf(model, input_ids: list[int], max_new_tokens: int,
             warmup: int, iterations: int, eos_token_id: int | None,
             verbose: bool, _cudagraph_mark: bool = False) -> dict:
    """Benchmark HF inference with KV cache.

    Returns dict with timing lists and generated token IDs.
    """
    import torch

    ids_tensor = torch.tensor([input_ids], dtype=torch.long, device="cuda")

    prefill_times: list[float] = []
    decode_times: list[float] = []
    decode_token_counts: list[int] = []
    gen_ids: list[int] = []

    total_runs = warmup + iterations
    for run_idx in range(total_runs):
        is_warmup = run_idx < warmup

        with torch.no_grad():
            # -- Prefill --
            if _cudagraph_mark:
                torch.compiler.cudagraph_mark_step_begin()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = model(ids_tensor, use_cache=True)
            torch.cuda.synchronize()
            prefill_ms = (time.perf_counter() - t0) * 1000

            # HF decoders use past_key_values; some recurrent models use cache_params.
            uses_recurrent_cache = hasattr(outputs, "cache_params")
            if uses_recurrent_cache:
                past = outputs.cache_params
                seq_len = ids_tensor.shape[1]
            else:
                past = outputs.past_key_values
            logits = outputs.logits  # (1, seq_len, vocab)

            # -- Decode --
            tokens_generated = 0
            run_gen_ids: list[int] = []
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for step in range(max_new_tokens):
                next_token = int(logits[0, -1].argmax())
                run_gen_ids.append(next_token)
                if eos_token_id is not None and next_token == eos_token_id:
                    break
                next_input = torch.tensor(
                    [[next_token]], dtype=torch.long, device="cuda")
                if _cudagraph_mark:
                    torch.compiler.cudagraph_mark_step_begin()
                if uses_recurrent_cache:
                    cache_pos = torch.tensor(
                        [seq_len + step], dtype=torch.long, device="cuda")
                    outputs = model(
                        next_input, cache_params=past,
                        cache_position=cache_pos, use_cache=True)
                    past = outputs.cache_params
                else:
                    outputs = model(
                        next_input, past_key_values=past,
                        use_cache=True)
                    past = outputs.past_key_values
                logits = outputs.logits
                tokens_generated += 1
            torch.cuda.synchronize()
            decode_ms = (time.perf_counter() - t0) * 1000

        if not is_warmup:
            prefill_times.append(prefill_ms)
            decode_times.append(decode_ms)
            decode_token_counts.append(tokens_generated)
            gen_ids = run_gen_ids

        if verbose:
            tag = "warmup" if is_warmup else f"iter {run_idx - warmup + 1}"
            print(f"  [hf  {tag}] prefill={prefill_ms:.2f}ms "
                  f"decode={decode_ms:.2f}ms ({tokens_generated} tokens)",
                  file=sys.stderr)

    return {
        "prefill_times": prefill_times,
        "decode_times": decode_times,
        "decode_token_counts": decode_token_counts,
        "gen_ids": gen_ids,
    }


def bench_hf_compiled(model, input_ids: list[int], max_new_tokens: int,
                      warmup: int, iterations: int, eos_token_id: int | None,
                      compile_mode: str, verbose: bool) -> dict:
    """Benchmark torch.compile(model) inference.

    Applies torch.compile before the warmup loop; graph tracing happens on the
    first forward passes (included in warmup, excluded from timing). Returns the
    same dict format as bench_hf().
    """
    import torch
    print(f"[perf] torch.compile(mode={compile_mode!r}) ...", file=sys.stderr)
    compiled = torch.compile(model, mode=compile_mode)
    return bench_hf(compiled, input_ids, max_new_tokens,
                    warmup, iterations, eos_token_id, verbose,
                    _cudagraph_mark=True)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(mean: float, std: float) -> str:
    """Format mean +/- std."""
    return f"{mean:.1f} +/- {std:.1f}"


def _speedup(hf_mean: float, trt_mean: float) -> str:
    """Compute and format speedup (HF/TRT)."""
    if trt_mean <= 0:
        return "N/A"
    return f"{hf_mean / trt_mean:.2f}x"


def print_report(model_name: str, prompt: str, num_input_tokens: int,
                 max_new_tokens: int, iterations: int, warmup: int,
                 hf_dtype: str, trt_res: dict, hf_res: dict,
                 runtime_note: str | None = None,
                 compile_res: dict | None = None,
                 compile_mode: str = "reduce-overhead"):
    """Print formatted comparison table to stdout."""
    gpu = _get_gpu_name()
    trt_ver = _get_trt_version()

    trt_prefill = _stats(trt_res["prefill_times"])
    trt_decode = _stats(trt_res["decode_times"])
    hf_prefill = _stats(hf_res["prefill_times"])
    hf_decode = _stats(hf_res["decode_times"])

    has_compile = compile_res is not None
    if has_compile:
        cp_prefill = _stats(compile_res["prefill_times"])
        cp_decode = _stats(compile_res["decode_times"])

    # Per-token and throughput (from decode phase)
    trt_avg_tokens = (statistics.mean(trt_res["decode_token_counts"])
                      if trt_res["decode_token_counts"] else 0)
    hf_avg_tokens = (statistics.mean(hf_res["decode_token_counts"])
                     if hf_res["decode_token_counts"] else 0)

    if trt_avg_tokens > 0 and trt_decode["mean"] > 0:
        trt_per_tok = trt_decode["mean"] / trt_avg_tokens
        trt_per_tok_std = trt_decode["std"] / trt_avg_tokens
        trt_tps = 1000.0 * trt_avg_tokens / trt_decode["mean"]
        trt_tps_std = (1000.0 * trt_avg_tokens * trt_decode["std"]
                       / trt_decode["mean"] ** 2)
    else:
        trt_per_tok = trt_per_tok_std = 0.0
        trt_tps = trt_tps_std = 0.0

    if hf_avg_tokens > 0 and hf_decode["mean"] > 0:
        hf_per_tok = hf_decode["mean"] / hf_avg_tokens
        hf_per_tok_std = hf_decode["std"] / hf_avg_tokens
        hf_tps = 1000.0 * hf_avg_tokens / hf_decode["mean"]
        hf_tps_std = (1000.0 * hf_avg_tokens * hf_decode["std"]
                      / hf_decode["mean"] ** 2)
    else:
        hf_per_tok = hf_per_tok_std = 0.0
        hf_tps = hf_tps_std = 0.0

    if has_compile:
        cp_avg_tokens = (statistics.mean(compile_res["decode_token_counts"])
                         if compile_res["decode_token_counts"] else 0)
        if cp_avg_tokens > 0 and cp_decode["mean"] > 0:
            cp_per_tok = cp_decode["mean"] / cp_avg_tokens
            cp_per_tok_std = cp_decode["std"] / cp_avg_tokens
            cp_tps = 1000.0 * cp_avg_tokens / cp_decode["mean"]
            cp_tps_std = (1000.0 * cp_avg_tokens * cp_decode["std"]
                          / cp_decode["mean"] ** 2)
        else:
            cp_per_tok = cp_per_tok_std = 0.0
            cp_tps = cp_tps_std = 0.0

    trt_total = _stats([p + d for p, d in zip(trt_res["prefill_times"],
                                              trt_res["decode_times"])])
    hf_total = _stats([p + d for p, d in zip(hf_res["prefill_times"],
                                             hf_res["decode_times"])])
    if has_compile:
        cp_total = _stats([p + d for p, d in zip(compile_res["prefill_times"],
                                                  compile_res["decode_times"])])

    # Check token agreement
    trt_gen = trt_res["gen_ids"]
    hf_gen = hf_res["gen_ids"]
    text_match = trt_gen == hf_gen

    prompt_display = prompt[:60] + ("..." if len(prompt) > 60 else "")
    sep = "=" * (60 if not has_compile else 80)

    print(f"\n{sep}")
    print(f"Perf Comparison: {model_name}")
    print(f"GPU: {gpu}, TRT: {trt_ver}")
    print(f'Prompt: "{prompt_display}" ({num_input_tokens} tokens)')
    print(f"Max new tokens: {max_new_tokens}, "
          f"{iterations} iterations, {warmup} warmup")
    print(f"Token match: {text_match}"
          + ("" if text_match
             else f" (TRT={len(trt_gen)} tokens, HF={len(hf_gen)} tokens)"))
    if has_compile:
        compile_label = f"HF (compile/{compile_mode})"
        print(sep)
        hdr = (f"{'':>20s}  {'TRT':>16s}  {'HF (eager)':>16s}"
               f"  {compile_label:>22s}  {'TRT/compile':>11s}")
        print(hdr)

        def _row3(label, trt_v, hf_v, cp_v, sp_col):
            print(f"  {label:>18s}:  {trt_v:>16s}  {hf_v:>16s}"
                  f"  {cp_v:>22s}  {sp_col:>11s}")

        _row3("Prefill (ms)",
              _fmt(trt_prefill["mean"], trt_prefill["std"]),
              _fmt(hf_prefill["mean"], hf_prefill["std"]),
              _fmt(cp_prefill["mean"], cp_prefill["std"]),
              _speedup(cp_prefill["mean"], trt_prefill["mean"]) + "  *")
        _row3("Decode (ms)",
              _fmt(trt_decode["mean"], trt_decode["std"]),
              _fmt(hf_decode["mean"], hf_decode["std"]),
              _fmt(cp_decode["mean"], cp_decode["std"]),
              _speedup(cp_decode["mean"], trt_decode["mean"]))
        if trt_per_tok > 0 and cp_per_tok > 0:
            _row3("Per-token (ms)",
                  f"{trt_per_tok:.2f} +/- {trt_per_tok_std:.2f}",
                  f"{hf_per_tok:.2f} +/- {hf_per_tok_std:.2f}",
                  f"{cp_per_tok:.2f} +/- {cp_per_tok_std:.2f}",
                  _speedup(cp_per_tok, trt_per_tok))
            _row3("Throughput (t/s)",
                  f"{trt_tps:.1f} +/- {trt_tps_std:.1f}",
                  f"{hf_tps:.1f} +/- {hf_tps_std:.1f}",
                  f"{cp_tps:.1f} +/- {cp_tps_std:.1f}",
                  _speedup(trt_tps, cp_tps))
        _row3("Total (ms)",
              _fmt(trt_total["mean"], trt_total["std"]),
              _fmt(hf_total["mean"], hf_total["std"]),
              _fmt(cp_total["mean"], cp_total["std"]),
              _speedup(cp_total["mean"], trt_total["mean"]))
    else:
        print(sep)
        hdr = f"{'':>20s}  {'TRT':>16s}  {'HF':>16s}  {'Speedup':>8s}"
        print(hdr)

        rows = [
            ("Prefill (ms)",
             _fmt(trt_prefill["mean"], trt_prefill["std"]),
             _fmt(hf_prefill["mean"], hf_prefill["std"]),
             _speedup(hf_prefill["mean"], trt_prefill["mean"]) + "  *"),
            ("Decode (ms)",
             _fmt(trt_decode["mean"], trt_decode["std"]),
             _fmt(hf_decode["mean"], hf_decode["std"]),
             _speedup(hf_decode["mean"], trt_decode["mean"])),
        ]

        if trt_per_tok > 0 and hf_per_tok > 0:
            rows.append(("Per-token (ms)",
                          f"{trt_per_tok:.2f} +/- {trt_per_tok_std:.2f}",
                          f"{hf_per_tok:.2f} +/- {hf_per_tok_std:.2f}",
                          _speedup(hf_per_tok, trt_per_tok)))
            rows.append(("Throughput (t/s)",
                          f"{trt_tps:.1f} +/- {trt_tps_std:.1f}",
                          f"{hf_tps:.1f} +/- {hf_tps_std:.1f}",
                          _speedup(trt_tps, hf_tps)))

        rows.append(("Total (ms)",
                      _fmt(trt_total["mean"], trt_total["std"]),
                      _fmt(hf_total["mean"], hf_total["std"]),
                      _speedup(hf_total["mean"], trt_total["mean"])))

        for label, trt_val, hf_val, sp in rows:
            print(f"  {label:>18s}:  {trt_val:>16s}  {hf_val:>16s}  {sp:>8s}")

    print()
    if runtime_note:
        print(runtime_note)
    else:
        print("* Prefill: HF batches all tokens; TRT processes token-by-token")
        print("  Decode: both token-by-token with KV cache (apples-to-apples)")
    if has_compile:
        print(f"  TRT/compile speedup = TRT vs HF (compile/{compile_mode})")
    print("  Excludes: model loading, tokenization, engine build")
    print(f"  HF dtype: {hf_dtype}")


def build_json_output(model_name: str, prompt: str, num_input_tokens: int,
                      max_new_tokens: int, iterations: int, warmup: int,
                      hf_dtype: str, trt_res: dict, hf_res: dict,
                      compile_res: dict | None = None,
                      compile_mode: str = "reduce-overhead",
                      cpp_res: dict | None = None) -> dict:
    """Build structured JSON output."""
    trt_prefill = _stats(trt_res["prefill_times"])
    trt_decode = _stats(trt_res["decode_times"])
    hf_prefill = _stats(hf_res["prefill_times"])
    hf_decode = _stats(hf_res["decode_times"])

    trt_avg_tokens = (statistics.mean(trt_res["decode_token_counts"])
                      if trt_res["decode_token_counts"] else 0)
    hf_avg_tokens = (statistics.mean(hf_res["decode_token_counts"])
                     if hf_res["decode_token_counts"] else 0)

    def _per_token_stats(decode_stat: dict, avg_tokens: float) -> dict:
        if avg_tokens > 0 and decode_stat["mean"] > 0:
            pt = decode_stat["mean"] / avg_tokens
            pt_std = decode_stat["std"] / avg_tokens
            tps = 1000.0 * avg_tokens / decode_stat["mean"]
            tps_std = (1000.0 * avg_tokens * decode_stat["std"]
                       / decode_stat["mean"] ** 2)
        else:
            pt = pt_std = tps = tps_std = 0.0
        return {
            "per_token_ms": {"mean": pt, "std": pt_std},
            "throughput_tps": {"mean": tps, "std": tps_std},
        }

    trt_tok = _per_token_stats(trt_decode, trt_avg_tokens)
    hf_tok = _per_token_stats(hf_decode, hf_avg_tokens)

    trt_total = _stats([p + d for p, d in zip(trt_res["prefill_times"],
                                              trt_res["decode_times"])])
    hf_total = _stats([p + d for p, d in zip(hf_res["prefill_times"],
                                             hf_res["decode_times"])])

    def _safe_div(a: float, b: float) -> float | None:
        return round(a / b, 3) if b > 0 else None

    peak_memory_mb = _get_peak_memory_mb()

    out: dict = {
        "metadata": {
            "model": model_name,
            "gpu": _get_gpu_name(),
            "trt_version": _get_trt_version(),
            "prompt": prompt,
            "num_input_tokens": num_input_tokens,
            "max_new_tokens": max_new_tokens,
            "warmup": warmup,
            "iterations": iterations,
            "hf_dtype": hf_dtype,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "trt": {
            "prefill_ms": trt_prefill,
            "decode_ms": trt_decode,
            "decode_ms_per_token": trt_tok["per_token_ms"]["mean"],
            "per_token_ms": trt_tok["per_token_ms"],
            "throughput_tps": trt_tok["throughput_tps"],
            "tokens_per_second": trt_tok["throughput_tps"]["mean"],
            "total_ms": trt_total,
            "total_latency_ms": trt_total["mean"],
            "num_decode_tokens": int(trt_avg_tokens),
        },
        "hf": {
            "prefill_ms": hf_prefill,
            "decode_ms": hf_decode,
            "decode_ms_per_token": hf_tok["per_token_ms"]["mean"],
            "per_token_ms": hf_tok["per_token_ms"],
            "throughput_tps": hf_tok["throughput_tps"],
            "tokens_per_second": hf_tok["throughput_tps"]["mean"],
            "total_ms": hf_total,
            "total_latency_ms": hf_total["mean"],
            "num_decode_tokens": int(hf_avg_tokens),
        },
        "speedup": {
            "prefill": _safe_div(hf_prefill["mean"], trt_prefill["mean"]),
            "decode": _safe_div(hf_decode["mean"], trt_decode["mean"]),
            "per_token": _safe_div(hf_tok["per_token_ms"]["mean"],
                                   trt_tok["per_token_ms"]["mean"]),
            "throughput": _safe_div(trt_tok["throughput_tps"]["mean"],
                                    hf_tok["throughput_tps"]["mean"]),
            "total": _safe_div(hf_total["mean"], trt_total["mean"]),
        },
        "token_match": trt_res["gen_ids"] == hf_res["gen_ids"],
        "peak_memory_mb": peak_memory_mb,
    }

    if cpp_res is not None:
        cpp_prefill = _stats(cpp_res["prefill_times"])
        cpp_decode = _stats(cpp_res["decode_times"])
        cpp_avg_tokens = (statistics.mean(cpp_res["decode_token_counts"])
                          if cpp_res["decode_token_counts"] else 0)
        cpp_tok = _per_token_stats(cpp_decode, cpp_avg_tokens)
        cpp_total = _stats([p + d for p, d in zip(cpp_res["prefill_times"],
                                                   cpp_res["decode_times"])])
        out["trt_cpp"] = {
            "prefill_ms": cpp_prefill,
            "decode_ms": cpp_decode,
            "per_token_ms": cpp_tok["per_token_ms"],
            "throughput_tps": cpp_tok["throughput_tps"],
            "total_ms": cpp_total,
            "num_decode_tokens": int(cpp_avg_tokens),
        }
        out["speedup"]["cpp_vs_hf_decode"] = _safe_div(
            hf_decode["mean"], cpp_decode["mean"])
        out["speedup"]["cpp_vs_trt_python_decode"] = _safe_div(
            trt_decode["mean"], cpp_decode["mean"])

    if compile_res is not None:
        cp_prefill = _stats(compile_res["prefill_times"])
        cp_decode = _stats(compile_res["decode_times"])
        cp_avg_tokens = (statistics.mean(compile_res["decode_token_counts"])
                         if compile_res["decode_token_counts"] else 0)
        cp_tok = _per_token_stats(cp_decode, cp_avg_tokens)
        cp_total = _stats([p + d for p, d in zip(compile_res["prefill_times"],
                                                  compile_res["decode_times"])])
        out["hf_compiled"] = {
            "compile_mode": compile_mode,
            "prefill_ms": cp_prefill,
            "decode_ms": cp_decode,
            "per_token_ms": cp_tok["per_token_ms"],
            "throughput_tps": cp_tok["throughput_tps"],
            "total_ms": cp_total,
            "num_decode_tokens": int(cp_avg_tokens),
        }
        out["speedup"]["trt_vs_compile_prefill"] = _safe_div(
            cp_prefill["mean"], trt_prefill["mean"])
        out["speedup"]["trt_vs_compile_decode"] = _safe_div(
            cp_decode["mean"], trt_decode["mean"])
        out["speedup"]["trt_vs_compile_total"] = _safe_div(
            cp_total["mean"], trt_total["mean"])

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TRT vs HuggingFace inference performance comparison")
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
                        help="Warmup iterations (not counted)")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Timed iterations")
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "float32", "bfloat16"],
                        help="HF model dtype (default: float16)")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Allow custom code from the HF model repo")
    parser.add_argument("--trt-only", action="store_true",
                        help="Benchmark TRT only (skip HF reference)")
    parser.add_argument("--no-compile", action="store_true",
                        help="Skip torch.compile benchmark")
    parser.add_argument("--compile-mode", default="reduce-overhead",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode (default: reduce-overhead)")
    parser.add_argument("--json", dest="json_path", metavar="PATH",
                        help="Save results to JSON file")
    parser.add_argument("--perf-db", dest="perf_db_path", metavar="PATH",
                        help="SQLite perf database path (enables perf tracking)")
    parser.add_argument("--trtmc-binary", dest="trtmc_binary", metavar="PATH",
                        help="Path to trtmc C++ binary for C++ runtime benchmark "
                             "(requires --bundle)")
    parser.add_argument("--hf-python", dest="hf_python", metavar="PATH",
                        help="Path to Python interpreter for HF tokenizer in C++ binary "
                             "(passed to --trtmc-binary runs)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # -- Resolve model directory --
    from tensorrt_model_connect.engine_builder import _resolve_model
    model_dir = _resolve_model(args.model)

    # -- Tokenize --
    from transformers import AutoTokenizer
    print("[perf] Loading tokenizer ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=args.trust_remote_code)
    input_ids = tokenizer.encode(args.prompt)
    print(f"[perf] Prompt: {len(input_ids)} tokens", file=sys.stderr)

    # Determine EOS token ID
    eos_token_id = None
    if tokenizer.eos_token_id is not None:
        eos_token_id = tokenizer.eos_token_id

    # -- Load / build TRT engine --
    if args.bundle:
        print(f"[perf] Loading bundle: {args.bundle}", file=sys.stderr)
        engine_plan, num_layers, max_cache_length, bundle_config, perf_handler = \
            load_trt_from_bundle(args.bundle)
        runtime_strategy = str(bundle_config.get("runtime_strategy") or "")
        runner_config = bundle_config
        runner_bundle_path = args.bundle
    else:
        engine_plan, config, _, perf_handler = build_trt_engine(
            args.model, args.max_cache_length, args.verbose)
        num_layers = config.num_hidden_layers
        max_cache_length = args.max_cache_length
        runtime_strategy = runtime_strategy_from_config(config)
        runner_config = config
        runner_bundle_path = ""

    # -- Bench TRT (GPU-exclusive) --
    backend_label = _handler_attr(perf_handler, "backend_label", "TRT")
    print(f"[perf] Benchmarking {backend_label} ({args.warmup} warmup + "
          f"{args.iterations} iterations) ...", file=sys.stderr)
    if perf_handler is not None:
        trt_res = bench_trt_family(
            perf_handler, engine_plan, num_layers, max_cache_length,
            input_ids, args.max_new_tokens,
            args.warmup, args.iterations, eos_token_id, args.verbose)
    else:
        trt_res = bench_trt(
            engine_plan, num_layers, max_cache_length,
            input_ids, args.max_new_tokens,
            args.warmup, args.iterations, eos_token_id, args.verbose,
            runtime_strategy=runtime_strategy,
            config=runner_config,
            bundle_path=runner_bundle_path)
    del engine_plan

    # Free TRT GPU memory before loading HF
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()

    # -- Record TRT-only results to PerfDB early (before HF may crash) --
    trt_prefill_early = _stats(trt_res["prefill_times"])
    trt_decode_early = _stats(trt_res["decode_times"])
    trt_avg_tokens_early = (statistics.mean(trt_res["decode_token_counts"])
                            if trt_res["decode_token_counts"] else 0)
    trt_tps_early = 0.0
    trt_pt_early = 0.0
    if trt_avg_tokens_early > 0 and trt_decode_early["mean"] > 0:
        trt_tps_early = 1000.0 * trt_avg_tokens_early / trt_decode_early["mean"]
        trt_pt_early = trt_decode_early["mean"] / trt_avg_tokens_early
    trt_total_early = _stats([p + d for p, d in zip(trt_res["prefill_times"],
                                                     trt_res["decode_times"])])
    trt_only_json = {
        "metadata": {
            "model": args.model,
            "gpu": _get_gpu_name(),
            "trt_version": _get_trt_version(),
            "prompt": args.prompt,
            "num_input_tokens": len(input_ids),
            "max_new_tokens": args.max_new_tokens,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "hf_dtype": args.dtype,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "trt": {
            "prefill_ms": trt_prefill_early,
            "decode_ms": trt_decode_early,
            "decode_ms_per_token": trt_pt_early,
            "per_token_ms": {"mean": trt_pt_early, "std": 0.0},
            "throughput_tps": {"mean": trt_tps_early, "std": 0.0},
            "tokens_per_second": trt_tps_early,
            "total_ms": trt_total_early,
            "total_latency_ms": trt_total_early["mean"],
            "num_decode_tokens": int(trt_avg_tokens_early),
        },
        "hf": {},
        "speedup": {},
        "token_match": None,
        "peak_memory_mb": _get_peak_memory_mb(),
    }

    if args.trt_only:
        # -- TRT-only mode: report, save, and exit --
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"Perf (TRT only): {args.model}")
        print(f"GPU: {trt_only_json['metadata']['gpu']}, "
              f"TRT: {trt_only_json['metadata']['trt_version']}")
        print(f'Prompt: "{args.prompt[:60]}" ({len(input_ids)} tokens)')
        print(f"Throughput: {trt_tps_early:.1f} t/s")
        print(f"Decode: {trt_decode_early['mean']:.1f} ms")
        print(f"Prefill: {trt_prefill_early['mean']:.1f} ms")
        print(sep)

        if args.json_path:
            with open(args.json_path, "w") as f:
                json.dump(trt_only_json, f, indent=2)
            print(f"\n[perf] Results saved to {args.json_path}",
                  file=sys.stderr)

        if args.perf_db_path:
            try:
                from perfdb import PerfDB
                pdb = PerfDB(args.perf_db_path)
                run_id = pdb.record_perf_compare(trt_only_json)
                pdb.close()
                print(f"[perf] Recorded to perf DB (run_id={run_id})",
                      file=sys.stderr)
            except Exception as e:
                print(f"[perf] WARNING: PerfDB recording failed: {e}",
                      file=sys.stderr)
        return

    # -- Record TRT results to PerfDB early (before HF may crash) --
    trt_early_run_id = None
    if args.perf_db_path:
        try:
            from perfdb import PerfDB
            pdb = PerfDB(args.perf_db_path)
            trt_early_run_id = pdb.record_perf_compare(trt_only_json)
            pdb.close()
            print(f"[perf] TRT results recorded to perf DB "
                  f"(run_id={trt_early_run_id})", file=sys.stderr)
        except Exception as e:
            print(f"[perf] WARNING: PerfDB early recording failed: {e}",
                  file=sys.stderr)

    # -- Bench HF eager (GPU-exclusive) --
    print(f"[perf] Loading HF model (dtype={args.dtype}) ...", file=sys.stderr)
    hf_model = load_hf_model(model_dir, args.dtype, args.trust_remote_code)
    print(f"[perf] Benchmarking HF eager ({args.warmup} warmup + "
          f"{args.iterations} iterations) ...", file=sys.stderr)
    hf_res = bench_hf(
        hf_model, input_ids, args.max_new_tokens,
        args.warmup, args.iterations, eos_token_id, args.verbose)
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()

    # -- Bench HF compiled (optional) --
    compile_res = None
    if not args.no_compile and _handler_supports(
        perf_handler, "supports_hf_compile", True
    ):
        print("[perf] Loading HF model for torch.compile ...", file=sys.stderr)
        hf_model2 = load_hf_model(model_dir, args.dtype, args.trust_remote_code)
        print(f"[perf] Benchmarking HF compiled ({args.warmup} warmup + "
              f"{args.iterations} iterations) ...", file=sys.stderr)
        try:
            compile_res = bench_hf_compiled(
                hf_model2, input_ids, args.max_new_tokens,
                args.warmup, args.iterations, eos_token_id,
                args.compile_mode, args.verbose)
        except Exception as exc:
            print(f"[perf] torch.compile failed ({exc}); skipping.",
                  file=sys.stderr)
        del hf_model2
        gc.collect()
        torch.cuda.empty_cache()

    # -- Bench C++ binary (optional) --
    cpp_res = None
    trtmc_binary = getattr(args, "trtmc_binary", None)
    if trtmc_binary and args.bundle:
        hf_python = getattr(args, "hf_python", None)
        print(f"[perf] Benchmarking C++ binary ({args.warmup} warmup + "
              f"{args.iterations} iterations) ...", file=sys.stderr)
        cpp_res = bench_trtmc_cpp(
            trtmc_binary, args.bundle, args.prompt, args.max_new_tokens,
            args.warmup, args.iterations, hf_python, args.verbose)
        if cpp_res is None:
            print("[perf] WARNING: C++ benchmark failed; omitting from report.",
                  file=sys.stderr)
    elif trtmc_binary and not args.bundle:
        print("[perf] WARNING: --trtmc-binary requires --bundle; skipping C++ bench.",
              file=sys.stderr)

    # -- Report --
    print_report(
        args.model, args.prompt, len(input_ids),
        args.max_new_tokens, args.iterations, args.warmup,
        args.dtype, trt_res, hf_res,
        runtime_note=_handler_attr(perf_handler, "perf_report_note", None),
        compile_res=compile_res, compile_mode=args.compile_mode)

    # -- JSON output (full TRT+HF+CPP) --
    json_output_data = build_json_output(
        args.model, args.prompt, len(input_ids),
        args.max_new_tokens, args.iterations, args.warmup,
        args.dtype, trt_res, hf_res,
        compile_res=compile_res, compile_mode=args.compile_mode,
        cpp_res=cpp_res)
    if args.json_path:
        with open(args.json_path, "w") as f:
            json.dump(json_output_data, f, indent=2)
        print(f"\n[perf] Results saved to {args.json_path}", file=sys.stderr)

    # -- Perf DB: update with full TRT+HF results --
    if args.perf_db_path:
        try:
            from perfdb import PerfDB
            pdb = PerfDB(args.perf_db_path)
            run_id = pdb.record_perf_compare(json_output_data)
            # Remove the TRT-only early record (now superseded by full result)
            if trt_early_run_id is not None:
                pdb._conn.execute(
                    "DELETE FROM perf_runs WHERE run_id = ?",
                    (trt_early_run_id,))
                pdb._conn.commit()
            pdb.close()
            print(f"[perf] Full results recorded to perf DB "
                  f"(run_id={run_id})", file=sys.stderr)
        except Exception as e:
            print(f"[perf] WARNING: PerfDB recording failed: {e}",
                  file=sys.stderr)

    # Warn if tokens differ
    if trt_res["gen_ids"] != hf_res["gen_ids"]:
        print("\nWARNING: TRT and HF generated different tokens. "
              "Per-token metrics may not be directly comparable.",
              file=sys.stderr)


def run_as_diff_test(ctx, include_compile: bool = False):
    """Framework entry point. Returns DiffResult."""
    from diff_framework.protocol import DiffResult
    import time as _time

    t0 = _time.monotonic()
    try:
        from tensorrt_model_connect.engine_builder import _resolve_model
        model_dir = _resolve_model(ctx.model)

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=ctx.trust_remote_code)
        input_ids = tokenizer.encode("The capital of France is")

        eos_token_id = tokenizer.eos_token_id

        # Build or load TRT engine
        if ctx.bundle_path:
            engine_plan, num_layers, max_cache_length, bundle_config, perf_handler = \
                load_trt_from_bundle(ctx.bundle_path)
            runtime_strategy = str(bundle_config.get("runtime_strategy") or "")
            runner_config = bundle_config
            runner_bundle_path = ctx.bundle_path
        else:
            engine_plan, config, _, perf_handler = build_trt_engine(
                ctx.model, ctx.max_cache_length, ctx.verbose)
            num_layers = config.num_hidden_layers
            max_cache_length = ctx.max_cache_length
            runtime_strategy = runtime_strategy_from_config(config)
            runner_config = config
            runner_bundle_path = ""

        warmup, iterations = 1, 3
        if perf_handler is not None:
            trt_res = bench_trt_family(
                perf_handler, engine_plan, num_layers, max_cache_length,
                input_ids, ctx.max_new_tokens,
                warmup, iterations, eos_token_id, ctx.verbose)
        else:
            trt_res = bench_trt(
                engine_plan, num_layers, max_cache_length,
                input_ids, ctx.max_new_tokens,
                warmup, iterations, eos_token_id, ctx.verbose,
                runtime_strategy=runtime_strategy,
                config=runner_config,
                bundle_path=runner_bundle_path)
        del engine_plan

        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()

        hf_model = load_hf_model(model_dir, "float16", ctx.trust_remote_code)
        hf_res = bench_hf(
            hf_model, input_ids, ctx.max_new_tokens,
            warmup, iterations, eos_token_id, ctx.verbose)
        del hf_model
        gc.collect()
        torch.cuda.empty_cache()

        trt_decode = _stats(trt_res["decode_times"])
        hf_decode = _stats(hf_res["decode_times"])
        speedup = (hf_decode["mean"] / trt_decode["mean"]
                   if trt_decode["mean"] > 0 else 0.0)
        token_match = trt_res["gen_ids"] == hf_res["gen_ids"]

        return DiffResult(
            test_name="perf_benchmark", model=ctx.model,
            runtime_strategy=ctx.runtime_strategy,
            passed=True,  # perf tests always "pass" — they report metrics
            status="PASS",
            message=(f"decode_speedup={speedup:.2f}x, "
                     f"token_match={token_match}"),
            metrics={
                "trt_decode_ms": trt_decode["mean"],
                "hf_decode_ms": hf_decode["mean"],
                "decode_speedup": round(speedup, 2),
                "token_match": token_match,
            },
            duration_s=_time.monotonic() - t0)
    except Exception as e:
        return DiffResult.error(
            "perf_benchmark", ctx.model, ctx.runtime_strategy, str(e))


if __name__ == "__main__":
    main()
