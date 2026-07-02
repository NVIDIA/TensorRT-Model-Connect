#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Automated performance tuning: profile → classify → optimize → verify.

Runs the full /perf-tune loop non-interactively for any model and pipeline type.
No agent needed — pure Python orchestration.

Usage:
    # Single model
    python3 tools/auto_perf_tune.py --model org/model

    # With specific pipeline type and precision
    python3 tools/auto_perf_tune.py --model org/model --precision fp16

    # Non-default runtime strategy
    python3 tools/auto_perf_tune.py --model org/model --pipeline-type runtime_strategy

    # Batch validation across models
    python3 tools/auto_perf_tune.py --batch models.json --output results/

    # Dry run (show what would be done)
    python3 tools/auto_perf_tune.py --model org/model --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.e2e_harness.runtime_strategy_metadata import runtime_strategy_performance_mode  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NSYS_DEB_URL = (
    "https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/"
    "NsightSystems-linux-cli-public-2026.2.1.210-3763964.deb"
)
DEFAULT_PERF_VALIDATION_ROOT = PROJECT_ROOT / "tests" / "e2e" / "models"

# Default prompts by mode
DEFAULT_PROMPTS = {
    "decode": "The capital of France is",
    "diffusion": "a beautiful sunset over mountains",
    "enc_dec": "The capital of France is",
    "single_pass": "The quick brown fox jumps over the lazy dog",
    "multi_stage": "Hello, how are you today?",
}


@dataclass
class TuneResult:
    """Result of one auto-tune run."""
    model: str
    pipeline_type: str
    mode: str
    # Baseline
    baseline_tps: float = 0
    baseline_precision: str = "fp32"
    # Classification
    bottleneck: str = ""
    confidence: str = ""
    evidence: list[str] = field(default_factory=list)
    recommended_techniques: list[str] = field(default_factory=list)
    # After optimization
    optimized_tps: float = 0
    optimized_precision: str = ""
    optimized_runtime_flags: str = ""
    speedup: float = 0
    # SOL (both rooflines)
    sol_tps: float = 0
    bw_sol_tps: float = 0
    compute_sol_tps: float = 0
    sol_bottleneck: str = ""       # "bandwidth" or "compute"
    utilization_before: float = 0
    utilization_after: float = 0
    bw_sol_tps_fp16: float = 0
    compute_sol_tps_fp16: float = 0
    sol_bottleneck_fp16: str = ""
    # Nsys
    nsys_kernel_count: int = 0
    nsys_top_kernel: str = ""
    nsys_top_kernel_pct: float = 0
    # Status
    status: str = "pending"  # pending, success, failed, skipped
    error: str = ""


def run_cmd(cmd: str, timeout: int = 600, dry_run: bool = False) -> tuple[int, str]:
    """Run a shell command, return (exit_code, stdout+stderr)."""
    if dry_run:
        print(f"  [dry-run] {cmd[:120]}")
        return 0, ""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT"


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_build(model: str, output: str, precision: str = "fp32",
               max_cache: int = 256, dry_run: bool = False) -> bool:
    """Build a .trtfb bundle."""
    cmd = (f"./build/trtmc build {model} -o {output} "
           f"--max-cache-length {max_cache} --precision {precision}")
    print(f"\n[build] {precision.upper()} bundle: {model}")
    rc, out = run_cmd(cmd, timeout=600, dry_run=dry_run)
    if rc != 0 and not dry_run:
        print(f"[build] FAILED: {out[-300:]}")
        return False
    print(f"[build] OK → {output}")
    return True


DEFAULT_BENCHMARK = {
    "label": "run",
    "gpu_argmax_label": "run",
    "metric": "tok/s",
    "command": [
        "{binary}",
        "run",
        "{bundle}",
        "--prompt",
        "{prompt}",
        "--max-new-tokens",
        "{max_tokens}",
        "{hf_python_args}",
        "{config_args}",
    ],
}


def _benchmark_context(
    bundle: str,
    prompt: str,
    max_tokens: int,
    gpu_argmax: bool,
) -> dict[str, object]:
    pid = os.getpid()
    return {
        "binary": "/tmp/build/trtmc",
        "repo_root": str(PROJECT_ROOT),
        "bundle": bundle,
        "prompt": prompt,
        "max_tokens": str(max_tokens),
        "hf_python_args": ["--hf-python", "/opt/venv/bin/python"],
        "config_args": (
            ["--set", "platform.trt_log_stderr=true"]
            + (["--set", "runtime.prefer_gpu_greedy=true"] if gpu_argmax else [])
        ),
        "generated_output_wav": f"/tmp/bench_audio_out_{pid}.wav",
        "generated_output_image": f"/tmp/bench_image_out_{pid}.png",
        "generated_output_dir": f"/tmp/bench_media_out_{pid}",
    }


def _expand_command_template(command: list[str], context: dict[str, object]) -> list[str]:
    expanded: list[str] = []
    for token in command:
        if token in {"{hf_python_args}", "{config_args}"}:
            expanded.extend(str(value) for value in context[token[1:-1]])
            continue
        try:
            expanded.append(token.format(**context))
        except KeyError as exc:
            raise ValueError(f"Unknown benchmark command placeholder {exc.args[0]!r}") from exc
    return [token for token in expanded if token]


def _build_bench_cmd(
    bundle: str,
    prompt: str,
    max_tokens: int,
    gpu_argmax: bool,
    benchmark: dict | None = None,
) -> tuple[str, str, str]:
    """Build a benchmark command from a model-owned template.

    Returns (shell_command, metric_name, label).
    """
    spec = benchmark or DEFAULT_BENCHMARK
    command = spec.get("command", DEFAULT_BENCHMARK["command"])
    if not isinstance(command, list) or not all(isinstance(token, str) for token in command):
        raise ValueError("benchmark.command must be a list of strings")

    context = _benchmark_context(bundle, prompt, max_tokens, gpu_argmax)
    argv = _expand_command_template(command, context)
    cmd = " ".join(shlex.quote(arg) for arg in argv) + " 2>&1"
    metric = str(spec.get("metric", DEFAULT_BENCHMARK["metric"]))
    label_key = "gpu_argmax_label" if gpu_argmax else "label"
    label = str(spec.get(label_key, spec.get("label", DEFAULT_BENCHMARK["label"])))
    return cmd, metric, label


def _parse_metric(out: str, metric_name: str) -> float:
    """Parse performance metric from trtmc output.

    Supported metrics:
        tok/s       — from "Decode: N tokens, X ms, Y tok/s"
        pipeline_ms — from "Total pipeline: X ms" or timed externally
        rtf         — from "RTF (real-time factor): X"
    """
    for line in out.split("\n"):
        # tok/s: "[trtmc] Decode: 20 tokens, 67.4 ms, 296.6 tok/s"
        if metric_name == "tok/s" and "Decode:" in line and "tok/s" in line:
            parts = line.split(",")
            for part in parts:
                if "tok/s" in part:
                    try:
                        return float(part.strip().split()[0])
                    except (ValueError, IndexError):
                        pass

        # pipeline_ms: "[model]   Total pipeline: 867.361 ms"
        # or "[trtmc] Inference: X ms"
        if metric_name == "pipeline_ms" and "Total pipeline:" in line:
            try:
                ms = float(line.split("Total pipeline:")[1].strip().split()[0])
                return ms
            except (ValueError, IndexError):
                pass

        # rtf: "RTF (real-time factor): 0.889"
        if metric_name == "rtf" and "RTF" in line and "real-time" in line:
            try:
                return float(line.split(":")[1].strip().split()[0])
            except (ValueError, IndexError):
                pass

    return 0


def step_benchmark(bundle: str, prompt: str, max_tokens: int = 100,
                   gpu_argmax: bool = False, dry_run: bool = False,
                   mode: str = "decode",
                   pipeline_type: str = "",
                   benchmark: dict | None = None) -> float:
    """Benchmark with C++ binary, return performance metric.

    Returns tok/s for decode/enc_dec, or pipeline_ms for others.
    The metric type depends on the mode.
    """
    cmd, metric_name, label = _build_bench_cmd(
        bundle, prompt, max_tokens, gpu_argmax, benchmark)

    print(f"[bench] {label}")

    # For pipeline_ms: measure wall-clock time if output doesn't contain timing
    t0 = time.time()
    rc, out = run_cmd(cmd, timeout=300, dry_run=dry_run)
    elapsed_ms = (time.time() - t0) * 1000
    if dry_run:
        return 0

    value = _parse_metric(out, metric_name)

    # Fallback: use wall-clock time if no metric parsed and command succeeded
    if value == 0 and rc == 0 and elapsed_ms > 0:
        value = elapsed_ms
        if metric_name == "tok/s":
            # Convert wall-clock to approximate tok/s
            # (inaccurate — includes loading, but better than 0)
            value = max_tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
            metric_name = "tok/s"  # keep unit consistent
    if value == 0 and metric_name == "pipeline_ms":
        # Fallback: try to extract any "X ms" timing from output
        for line in out.split("\n"):
            if "ms" in line and ("pipeline" in line.lower() or "inference" in line.lower()):
                try:
                    for word in line.split():
                        if word.replace(".", "").isdigit():
                            value = float(word)
                            break
                except (ValueError, IndexError):
                    pass
                if value > 0:
                    break

    if value == 0:
        print(f"[bench] Could not parse {metric_name} from output")

    return value


def step_nsys_profile(bundle: str, prompt: str, output_prefix: str,
                      max_tokens: int = 50, dry_run: bool = False,
                      mode: str = "decode",
                      pipeline_type: str = "",
                      benchmark: dict | None = None) -> str | None:
    """Run nsys profile, return path to .sqlite or None."""
    nsys = "/tmp/nsys_install/opt/nvidia/nsight-systems-cli/2026.2.1/target-linux-x64/nsys"

    # Check nsys exists
    if not dry_run and not os.path.exists(nsys):
        print("[nsys] Not installed, skipping profile")
        return None

    # Build the trtmc command for profiling (same logic as benchmark)
    trtmc_cmd, _, _ = _build_bench_cmd(
        bundle, prompt, max_tokens, gpu_argmax=False, benchmark=benchmark)
    # Strip the "2>&1" from the end and env prefix — nsys wraps the command
    # Extract just the trtmc binary + args part
    trtmc_part = trtmc_cmd
    # Remove env vars prefix (everything before /tmp/build/trtmc)
    if "/tmp/build/trtmc" in trtmc_part:
        idx = trtmc_part.index("/tmp/build/trtmc")
        env_part = trtmc_part[:idx].strip()
        trtmc_part = trtmc_part[idx:]
    else:
        env_part = ""
    # Remove trailing 2>&1
    trtmc_part = trtmc_part.replace("2>&1", "").strip()

    rep = f"{output_prefix}.nsys-rep"
    cmd = (f"{env_part} {nsys} profile -t cuda,nvtx --cuda-graph-trace=node "
           f"-o {output_prefix} --force-overwrite true "
           f"{trtmc_part} 2>&1")
    print("[nsys] Profiling...")
    rc, out = run_cmd(cmd, timeout=300, dry_run=dry_run)
    if dry_run:
        return f"{output_prefix}.sqlite"

    if rc != 0 or not os.path.exists(rep):
        print("[nsys] Profile failed")
        return None

    # Export to SQLite
    cmd2 = f"{nsys} stats {rep} --force-export true 2>&1"
    run_cmd(cmd2, timeout=60)
    sqlite = f"{output_prefix}.sqlite"
    if os.path.exists(sqlite):
        print(f"[nsys] OK → {sqlite}")
        return sqlite
    return None


def step_classify(sqlite_path: str | None, pipeline_type: str,
                  results_jsonl: str | None = None,
                  engine_section: str = "all",
                  dry_run: bool = False) -> dict:
    """Run classify_bottleneck.py via direct import (avoids subprocess issues)."""
    if not sqlite_path and not dry_run:
        return {"classification": "unknown", "confidence": "low",
                "techniques": [], "evidence": ["no nsys data"]}

    print(f"[classify] pipeline_type={pipeline_type}, engine_section={engine_section}")
    if dry_run:
        return {"classification": "pending", "techniques": []}

    try:
        from classify_bottleneck import classify_from_nsys, to_json
        result = classify_from_nsys(sqlite_path, pipeline_type,
                                    engine_section=engine_section)
        return to_json(result)
    except Exception as e:
        print(f"[classify] Error: {e}")
        return {"classification": "error", "confidence": "low",
                "techniques": [], "evidence": [str(e)]}


def step_sol(model: str, dtype: str, cache_length: int,
             benchmark_json: str | None = None,
             actual_tps: float = 0,
             dry_run: bool = False) -> dict:
    """Run sol_estimate via direct import."""
    print(f"[sol] {dtype.upper()}")
    if dry_run:
        return {}

    try:
        from sol_estimate import (
            load_model_arch_from_hf, GPU_SPECS, estimate_sol,
            detect_gpu, to_json, parse_benchmark_json,
        )
        arch = load_model_arch_from_hf(model)
        gpu_key = detect_gpu() or "B200"
        gpu = GPU_SPECS[gpu_key]

        tps = actual_tps
        if benchmark_json and os.path.exists(benchmark_json):
            try:
                tps = parse_benchmark_json(benchmark_json)
            except Exception:
                pass

        est = estimate_sol(arch, gpu, dtype, cache_length, tps)
        return to_json(est)
    except Exception as e:
        print(f"[sol] Error: {e}")
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def auto_tune_model(
    model: str,
    pipeline_type: str = "",
    max_cache: int = 256,
    max_tokens: int = 100,
    output_dir: str = "/tmp/auto_perf",
    engine_section: str = "all",
    dry_run: bool = False,
    benchmark: dict | None = None,
) -> TuneResult:
    """Run full auto-tune loop for one model."""
    if not pipeline_type:
        raise ValueError("pipeline_type is required")
    mode = runtime_strategy_performance_mode(pipeline_type, default="decode")
    prompt = DEFAULT_PROMPTS.get(mode, "Hello")
    safe_name = model.split("/")[-1].lower().replace("-", "_")

    os.makedirs(output_dir, exist_ok=True)

    result = TuneResult(model=model, pipeline_type=pipeline_type, mode=mode)

    print(f"\n{'='*60}")
    print(f"  Auto Perf-Tune: {model}")
    print(f"  Pipeline: {pipeline_type} → mode: {mode}")
    print(f"{'='*60}")

    # --- Step 1: Build FP32 baseline ---
    fp32_bundle = f"{output_dir}/{safe_name}_fp32.trtfb"
    if not step_build(model, fp32_bundle, "fp32", max_cache, dry_run):
        result.status = "failed"
        result.error = "FP32 build failed"
        return result

    # Determine metric name for this mode
    _, metric_name, _ = _build_bench_cmd(
        fp32_bundle, prompt, max_tokens, False, benchmark)
    metric_unit = metric_name  # "tok/s" or "pipeline_ms"

    # --- Step 2: Baseline benchmark ---
    result.baseline_tps = step_benchmark(
        fp32_bundle, prompt, max_tokens, gpu_argmax=False, dry_run=dry_run,
        mode=mode, pipeline_type=pipeline_type, benchmark=benchmark)
    result.baseline_precision = "fp32"
    print(f"[baseline] FP32: {result.baseline_tps:.1f} {metric_unit}")

    # --- Step 3: Nsys profile ---
    nsys_prefix = f"{output_dir}/{safe_name}_nsys"
    sqlite = step_nsys_profile(
        fp32_bundle, prompt, nsys_prefix, max_tokens=50, dry_run=dry_run,
        mode=mode, pipeline_type=pipeline_type, benchmark=benchmark)

    # --- Step 4: Classify bottleneck ---
    classification = step_classify(sqlite, pipeline_type,
                                   engine_section=engine_section, dry_run=dry_run)
    result.bottleneck = classification.get("classification", "unknown")
    result.confidence = classification.get("confidence", "low")
    result.evidence = classification.get("evidence", [])
    result.recommended_techniques = [
        t.get("name", "") for t in classification.get("techniques", [])]

    print(f"[classify] {result.bottleneck}-bound "
          f"(confidence: {result.confidence})")
    print(f"[classify] Recommended: {result.recommended_techniques}")

    # --- Step 5: SOL estimation (both BW and compute rooflines) ---
    sol_data = step_sol(
        model, "fp32", max_cache, actual_tps=result.baseline_tps, dry_run=dry_run)
    result.sol_tps = sol_data.get("sol_tps", 0)
    result.utilization_before = sol_data.get("utilization_pct", 0)
    result.bw_sol_tps = sol_data.get("bw_sol_tps", 0)
    result.compute_sol_tps = sol_data.get("compute_sol_tps", 0)
    result.sol_bottleneck = sol_data.get("bottleneck", "")

    if result.sol_tps > 0:
        print(f"[sol] FP32 BW SOL: {result.bw_sol_tps:.0f} tok/s, "
              f"Compute SOL: {result.compute_sol_tps:.0f} tok/s")
        print(f"[sol] Bottleneck: {result.sol_bottleneck}, "
              f"SOL: {result.sol_tps:.0f} tok/s, "
              f"util: {result.utilization_before:.1f}%")

    # --- Step 6: Apply optimizations (based on mode) ---
    # Try GPU argmax (decode/enc_dec modes only — single_pass and diffusion don't use argmax)
    if mode in ("decode", "enc_dec", "multi_stage"):
        argmax_tps = step_benchmark(
            fp32_bundle, prompt, max_tokens, gpu_argmax=True, dry_run=dry_run,
            mode=mode, pipeline_type=pipeline_type, benchmark=benchmark)
        if argmax_tps > result.baseline_tps:
            pct = (argmax_tps / result.baseline_tps - 1) * 100 if result.baseline_tps > 0 else 0
            print(f"[optimize] GPU argmax: {argmax_tps:.1f} {metric_unit} (+{pct:.0f}%)")

    # Try all precision variants
    best_tps = result.baseline_tps
    best_precision = "fp32"
    best_runtime_flags = ""
    all_results = []

    for precision in ("fp16", "bf16"):
        bundle = f"{output_dir}/{safe_name}_{precision}.trtfb"
        if not step_build(model, bundle, precision, max_cache, dry_run):
            print(f"[optimize] {precision.upper()} build failed, skipping")
            continue

        tps = step_benchmark(
            bundle, prompt, max_tokens, gpu_argmax=False, dry_run=dry_run,
            mode=mode, pipeline_type=pipeline_type, benchmark=benchmark)
        all_results.append((precision, False, tps))
        print(f"[optimize] {precision.upper()}: {tps:.1f} {metric_unit}")

        if mode in ("decode", "enc_dec", "multi_stage"):
            argmax_tps = step_benchmark(
                bundle, prompt, max_tokens, gpu_argmax=True, dry_run=dry_run,
                mode=mode, pipeline_type=pipeline_type, benchmark=benchmark)
            all_results.append((precision, True, argmax_tps))
            print(f"[optimize] {precision.upper()} + GPU argmax: {argmax_tps:.1f} {metric_unit}")

    # Find best config.
    # For tok/s: higher is better. For pipeline_ms: lower is better.
    lower_is_better = metric_name == "pipeline_ms"

    for prec, argmax, tps in all_results:
        if tps <= 0:
            continue
        if lower_is_better:
            if best_tps <= 0 or tps < best_tps:
                best_tps = tps
                best_precision = prec
                best_runtime_flags = "--set runtime.prefer_gpu_greedy=true" if argmax else ""
        else:
            if tps > best_tps:
                best_tps = tps
                best_precision = prec
                best_runtime_flags = "--set runtime.prefer_gpu_greedy=true" if argmax else ""

    result.optimized_tps = best_tps
    result.optimized_precision = best_precision
    result.optimized_runtime_flags = best_runtime_flags
    if result.baseline_tps > 0 and best_tps > 0:
        if lower_is_better:
            result.speedup = result.baseline_tps / best_tps  # ms ratio (baseline/optimized)
        else:
            result.speedup = best_tps / result.baseline_tps  # tok/s ratio
    else:
        result.speedup = 0

    argmax_label = " + GPU argmax" if result.optimized_runtime_flags else ""
    print(f"[optimize] Best: {best_tps:.1f} {metric_unit} "
          f"({result.optimized_precision.upper()}{argmax_label}, "
          f"{result.speedup:.2f}x vs baseline)")

    # --- Step 7: SOL re-check (best precision, both rooflines) ---
    if result.optimized_tps > 0:
        sol16 = step_sol(
            model, result.optimized_precision, max_cache,
            actual_tps=result.optimized_tps, dry_run=dry_run)
        result.utilization_after = sol16.get("utilization_pct", 0)
        result.bw_sol_tps_fp16 = sol16.get("bw_sol_tps", 0)
        result.compute_sol_tps_fp16 = sol16.get("compute_sol_tps", 0)
        result.sol_bottleneck_fp16 = sol16.get("bottleneck", "")
        if result.utilization_after > 0:
            print(f"[sol] FP16 BW SOL: {result.bw_sol_tps_fp16:.0f}, "
                  f"Compute SOL: {result.compute_sol_tps_fp16:.0f}")
            print(f"[sol] Bottleneck: {result.sol_bottleneck_fp16}, "
                  f"util: {result.utilization_after:.1f}% "
                  f"(was {result.utilization_before:.1f}%)")

    # --- Step 8: Nsys kernel summary ---
    if sqlite and not dry_run:
        try:
            import sqlite3
            db = sqlite3.connect(sqlite)
            rows = db.execute("""
                SELECT s.value, SUM(k.end - k.start) as total_ns, COUNT(*)
                FROM CUPTI_ACTIVITY_KIND_KERNEL k
                JOIN StringIds s ON k.shortName = s.id
                GROUP BY s.value ORDER BY total_ns DESC
            """).fetchall()
            db.close()
            if rows:
                total = sum(r[1] for r in rows)
                result.nsys_kernel_count = sum(r[2] for r in rows)
                result.nsys_top_kernel = rows[0][0][:60]
                result.nsys_top_kernel_pct = rows[0][1] / total * 100 if total > 0 else 0
        except Exception:
            pass

    result.status = "success"

    # --- Report ---
    print(f"\n{'='*70}")
    print(f"  Result: {model}")
    print(f"{'='*70}")
    print(f"  Mode:           {mode}")
    print(f"  Bottleneck (nsys):  {result.bottleneck}-bound ({result.confidence})")
    print(f"  Bottleneck (SOL):   {result.sol_bottleneck} (FP32) → "
          f"{result.sol_bottleneck_fp16} (FP16)")
    print(f"  Baseline:       {result.baseline_tps:.1f} {metric_unit} (FP32)")
    print(f"  Optimized:      {result.optimized_tps:.1f} {metric_unit} "
          f"({result.optimized_precision.upper()}"
          f"{' + GPU argmax' if result.optimized_runtime_flags else ''})")
    print(f"  Speedup:        {result.speedup:.2f}x")
    print(f"  SOL (FP32):     BW={result.bw_sol_tps:.0f}  "
          f"Compute={result.compute_sol_tps:.0f}  "
          f"→ {result.sol_bottleneck} {result.sol_tps:.0f} tok/s  "
          f"util={result.utilization_before:.1f}%")
    print(f"  SOL (FP16):     BW={result.bw_sol_tps_fp16:.0f}  "
          f"Compute={result.compute_sol_tps_fp16:.0f}  "
          f"→ {result.sol_bottleneck_fp16}  "
          f"util={result.utilization_after:.1f}%")
    print(f"  Kernels:        {result.nsys_kernel_count} total, "
          f"top: {result.nsys_top_kernel} ({result.nsys_top_kernel_pct:.0f}%)")
    print(f"{'='*70}")

    return result


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

def load_default_validation_models(
    root: Path = DEFAULT_PERF_VALIDATION_ROOT,
) -> list[dict]:
    """Load model-owned default performance validation entries."""
    models: list[dict] = []
    for path in sorted(root.glob("*/perf_validation.json")):
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        entries = raw.get("models", raw) if isinstance(raw, dict) else raw
        if not isinstance(entries, list):
            raise ValueError(f"{path}: expected a list or object with 'models'")
        for index, entry in enumerate(entries, 1):
            if not isinstance(entry, dict) or not entry.get("model"):
                raise ValueError(f"{path}: entry {index} must be an object with 'model'")
            item = dict(entry)
            pipeline_type = str(item.get("pipeline_type") or "")
            if not pipeline_type:
                raise ValueError(
                    f"{path}: entry {index} must declare pipeline_type"
                )
            item["pipeline_type"] = pipeline_type
            item.setdefault("label", f"{path.parent.name}-{index}")
            models.append(item)
    return models


def run_batch(models: list[dict], output_dir: str, dry_run: bool) -> list[TuneResult]:
    """Run auto-tune on multiple models."""
    results = []
    for entry in models:
        try:
            r = auto_tune_model(
                model=entry["model"],
                pipeline_type=str(entry.get("pipeline_type") or ""),
                output_dir=f"{output_dir}/{entry.get('label', 'model')}",
                benchmark=entry.get("benchmark"),
                dry_run=dry_run,
            )
            results.append(r)
        except Exception as e:
            print(f"\n[ERROR] {entry['model']}: {e}")
            r = TuneResult(
                model=entry["model"],
                pipeline_type=entry.get("pipeline_type", ""),
                mode=runtime_strategy_performance_mode(
                    entry.get("pipeline_type", ""),
                    default="",
                ),
                status="failed", error=str(e))
            results.append(r)

    # Summary table
    print(f"\n{'='*110}")
    print(f"  Batch Summary ({len(results)} models)")
    print(f"{'='*110}")
    print(f"  {'Model':<25} {'Mode':<10} {'Nsys':<10} {'SOL':<8} "
          f"{'Baseline':>8} {'Best':>8} {'Speedup':>7} "
          f"{'BW SOL':>8} {'CMP SOL':>8} {'Util':>5}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*8} "
          f"{'-'*8} {'-'*8} {'-'*7} "
          f"{'-'*8} {'-'*8} {'-'*5}")
    for r in results:
        print(f"  {r.model.split('/')[-1]:<25} {r.mode:<10} "
              f"{r.bottleneck:<10} {r.sol_bottleneck_fp16 or r.sol_bottleneck:<8} "
              f"{r.baseline_tps:>7.1f} {r.optimized_tps:>7.1f} "
              f"{r.speedup:>6.2f}x "
              f"{r.bw_sol_tps_fp16 or r.bw_sol_tps:>7.0f} "
              f"{r.compute_sol_tps_fp16 or r.compute_sol_tps:>7.0f} "
              f"{r.utilization_after or r.utilization_before:>4.1f}%")
    print(f"{'='*110}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Automated performance tuning for TRT models.")
    parser.add_argument("--model",
                        help="HuggingFace model ID")
    parser.add_argument("--pipeline-type", default="",
                        help="Runtime strategy")
    parser.add_argument("--max-cache-length", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--output-dir", default="/tmp/auto_perf")
    parser.add_argument("--batch", action="store_true",
                        help="Run model-owned default performance validation entries")
    parser.add_argument("--batch-json",
                        help="Custom batch config JSON file")
    parser.add_argument("--engine-section", default="all",
                        help="Engine section for multi-engine analysis: "
                             "'all' (default), 'primary', 'secondary', or graph ID")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without running")

    args = parser.parse_args()

    if args.batch or args.batch_json:
        if args.batch_json:
            with open(args.batch_json, encoding="utf-8") as f:
                models = json.load(f)
        else:
            models = load_default_validation_models()
            if not models:
                parser.error(
                    "No model-owned perf_validation.json files found; "
                    "pass --batch-json to provide an explicit batch."
                )
        results = run_batch(models, args.output_dir, args.dry_run)
        # Save results
        out_path = f"{args.output_dir}/batch_results.json"
        os.makedirs(args.output_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([vars(r) for r in results], f, indent=2)
        print(f"\nResults saved: {out_path}")
    elif args.model:
        if not args.pipeline_type:
            parser.error("--pipeline-type is required with --model")
        result = auto_tune_model(
            model=args.model,
            pipeline_type=args.pipeline_type,
            max_cache=args.max_cache_length,
            max_tokens=args.max_tokens,
            output_dir=args.output_dir,
            engine_section=args.engine_section,
            dry_run=args.dry_run,
        )
        # Save result
        out_path = f"{args.output_dir}/result.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(vars(result), f, indent=2)
        print(f"\nResult saved: {out_path}")
    else:
        parser.error("Specify --model or --batch")


if __name__ == "__main__":
    main()
