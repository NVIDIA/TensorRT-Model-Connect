#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-owned end-to-end benchmark: TRT decomposed attention vs FlashInfer.

Runs Qwen3-0.6B (or similar) through:
  1. Standard TRT engine (decomposed attention via graph_ops)
  2. FlashInfer fused attention kernel (replacing the TRT attention core)

Reports per-token latency and total generation time.

Usage:
  python -m tensorrt_model_connect.families.qwen.bench_flashinfer_e2e \
    --model Qwen/Qwen3-0.6B \
    --prompt "The capital of France is" \
    --max-new-tokens 20
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


def _run_trtmc_binary(binary, bundle, prompt, max_new_tokens, hf_python):
    """Run the C++ trtmc binary and measure wall-clock time."""
    cmd = [
        binary, "run", bundle,
        "--prompt", prompt,
        "--max-new-tokens", str(max_new_tokens),
    ]
    if hf_python:
        cmd += ["--hf-python", hf_python]

    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        print(f"[trtmc] stderr: {result.stderr[:500]}", file=sys.stderr)
        raise RuntimeError(f"trtmc binary failed with code {result.returncode}")

    output_text = result.stdout.strip()
    return elapsed, output_text


def _run_flashinfer_inference(model_id, prompt, max_new_tokens):
    """Run HF model with FlashInfer attention backend for comparison."""
    import torch

    try:
        import flashinfer  # noqa: F401
    except ImportError:
        print("FlashInfer not available, skipping", file=sys.stderr)
        return None, None

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[flashinfer] Loading model {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2",
    )
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda:0")

    # Warmup
    with torch.no_grad():
        _ = model.generate(input_ids, max_new_tokens=2, do_sample=False)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, max_new_tokens=max_new_tokens, do_sample=False,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return elapsed, output_text


def _run_hf_baseline(model_id, prompt, max_new_tokens):
    """Run HF model with default (eager) attention for baseline."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[hf-eager] Loading model {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        attn_implementation="eager",
    )
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda:0")

    # Warmup
    with torch.no_grad():
        _ = model.generate(input_ids, max_new_tokens=2, do_sample=False)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, max_new_tokens=max_new_tokens, do_sample=False,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return elapsed, output_text


def main():
    parser = argparse.ArgumentParser(description="TRT vs FlashInfer E2E benchmark")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--bundle", default=None, help="Pre-built .trtfb bundle path")
    parser.add_argument("--binary", default="./build/trtmc")
    parser.add_argument("--hf-python", default="/opt/venv/bin/python")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument(
        "--engine-dir",
        default=os.environ.get("TRTMC_ENGINE_DIR", "./engines"),
    )
    parser.add_argument("--skip-trt", action="store_true")
    parser.add_argument("--skip-flashinfer", action="store_true")
    parser.add_argument("--skip-hf-eager", action="store_true")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    # Resolve bundle path
    bundle = args.bundle
    if bundle is None:
        model_slug = args.model.split("/")[-1].lower()
        bundle = os.path.join(args.engine_dir, f"{model_slug}.trtfb")
        if not os.path.exists(bundle):
            print(f"Bundle not found at {bundle}, building...")
            subprocess.run([
                sys.executable, "-m", "tensorrt_model_connect", "build",
                args.model, "-o", bundle,
                "--max-cache-length", "256",
            ], check=True)

    print("=" * 80)
    print(f"E2E Benchmark: {args.model}")
    print(f"Prompt: {args.prompt!r}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Bundle: {bundle}")
    print(f"Runs: {args.runs}")
    print("=" * 80)

    results = {}

    # 1. TRT C++ binary (decomposed attention)
    if not args.skip_trt and os.path.exists(args.binary) and os.path.exists(bundle):
        print("\n[1/3] TRT C++ runtime (decomposed attention)...")
        times = []
        output = None
        for i in range(args.runs):
            t, text = _run_trtmc_binary(
                args.binary, bundle, args.prompt,
                args.max_new_tokens, args.hf_python,
            )
            times.append(t)
            output = text
            print(f"  Run {i+1}: {t:.3f}s")
        avg = sum(times) / len(times)
        results["trt_decomposed"] = avg
        print(f"  Average: {avg:.3f}s")
        print(f"  Output: {output[:100]}...")
    else:
        print("\n[1/3] TRT C++ runtime: SKIPPED")

    # 2. HF with FlashAttention2 backend
    if not args.skip_flashinfer:
        print("\n[2/3] HF + FlashAttention2...")
        try:
            times = []
            output = None
            for i in range(args.runs):
                t, text = _run_flashinfer_inference(
                    args.model, args.prompt, args.max_new_tokens,
                )
                if t is None:
                    break
                times.append(t)
                output = text
                print(f"  Run {i+1}: {t:.3f}s")
            if times:
                avg = sum(times) / len(times)
                results["hf_flash_attn"] = avg
                print(f"  Average: {avg:.3f}s")
                print(f"  Output: {output[:100]}...")
        except Exception as e:
            print(f"  FAILED: {e}")
    else:
        print("\n[2/3] HF + FlashAttention2: SKIPPED")

    # 3. HF with eager attention (baseline)
    if not args.skip_hf_eager:
        print("\n[3/3] HF + Eager attention (baseline)...")
        try:
            times = []
            output = None
            for i in range(args.runs):
                t, text = _run_hf_baseline(
                    args.model, args.prompt, args.max_new_tokens,
                )
                times.append(t)
                output = text
                print(f"  Run {i+1}: {t:.3f}s")
            if times:
                avg = sum(times) / len(times)
                results["hf_eager"] = avg
                print(f"  Average: {avg:.3f}s")
                print(f"  Output: {output[:100]}...")
        except Exception as e:
            print(f"  FAILED: {e}")
    else:
        print("\n[3/3] HF + Eager: SKIPPED")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for name, t in sorted(results.items(), key=lambda x: x[1]):
        tok_per_sec = args.max_new_tokens / t
        print(f"  {name:<25} {t:.3f}s  ({tok_per_sec:.1f} tok/s)")

    if "trt_decomposed" in results and "hf_eager" in results:
        speedup = results["hf_eager"] / results["trt_decomposed"]
        print(f"\n  TRT vs HF-eager speedup: {speedup:.2f}x")

    if "hf_flash_attn" in results and "hf_eager" in results:
        speedup = results["hf_eager"] / results["hf_flash_attn"]
        print(f"  FlashAttn vs HF-eager speedup: {speedup:.2f}x")

    if "trt_decomposed" in results and "hf_flash_attn" in results:
        speedup = results["hf_flash_attn"] / results["trt_decomposed"]
        print(f"  TRT vs FlashAttn speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
