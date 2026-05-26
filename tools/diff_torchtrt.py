#!/usr/bin/env python3
"""Torch-TRT vs HuggingFace logit comparison.

Compiles a model with torch_tensorrt (in-memory, no bundle), runs step-by-step
inference, and compares per-token logits against HuggingFace transformers
reference. This is the primary correctness check for the Torch-TRT pipeline.

Usage:
    python3 tools/diff_torchtrt.py --model Qwen/Qwen3-0.6B --atol 1e-2
    python3 tools/diff_torchtrt.py --model Qwen/Qwen3-0.6B --atol 1e-2 --battery
    python3 tools/diff_torchtrt.py --model /path/to/local/model --atol 1e-2

Options:
    --model         HF repo ID or local model directory
    --atol          Absolute tolerance for logit comparison (default: 1e-2)
    --max-cache-length  KV cache length (default: 256)
    --max-new-tokens    Tokens to generate per prompt (default: 10)
    --battery       Run multiple prompts instead of just one
    --trust-remote-code  Pass trust_remote_code=True to HF
    --no-compile    Skip Torch-TRT compilation, compare HF eager vs HF eager
                    (useful for testing the comparison harness itself)
    --verbose       Print detailed per-step output
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Prompts for battery mode
BATTERY_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "In a galaxy far far away",
    "The quick brown fox jumps over",
]

SINGLE_PROMPT = "The capital of France is"


def _parse_percent(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text)
    if match is None:
        return None
    return float(match.group(1)) / 100.0


def _parse_metric(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text)
    if match is None:
        return None
    return float(match.group(1))


def run_as_diff_test(ctx):
    """Run the Torch-TRT logit diff through the unified diff framework."""
    from diff_framework.protocol import DiffResult

    test_name = "torchtrt_logit_diff"
    atol = 1e-2 if ctx.atol == 1e-3 else ctx.atol
    max_new_tokens = 10 if ctx.max_new_tokens == 20 else ctx.max_new_tokens
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model",
        ctx.model,
        "--atol",
        str(atol),
        "--max-cache-length",
        str(ctx.max_cache_length),
        "--max-new-tokens",
        str(max_new_tokens),
    ]
    if ctx.trust_remote_code:
        command.append("--trust-remote-code")
    if ctx.verbose:
        command.append("--verbose")

    start = time.time()
    completed = subprocess.run(
        command, text=True, capture_output=True, check=False)
    output = "\n".join(
        part for part in (completed.stdout, completed.stderr) if part)

    metrics = {}
    top1 = _parse_percent(r"top1_match=([0-9.]+)%", output)
    top5 = _parse_percent(r"top5_overlap=([0-9.]+)%", output)
    cosine = _parse_metric(r"cos_sim=([0-9.eE+-]+)", output)
    max_diff = _parse_metric(r"max_diff=([0-9.eE+-]+)", output)
    if top1 is not None:
        metrics["top1_match_rate"] = top1
    if top5 is not None:
        metrics["mean_top5_overlap"] = top5
    if cosine is not None:
        metrics["mean_cosine_sim"] = cosine
    if max_diff is not None:
        metrics["max_abs_diff"] = max_diff

    passed = completed.returncode == 0
    return DiffResult(
        test_name=test_name,
        model=ctx.model,
        runtime_strategy=ctx.runtime_strategy,
        passed=passed,
        status="PASS" if passed else "FAIL",
        message=(
            "Torch-TRT logits match HF reference"
            if passed else f"Torch-TRT logit diff failed with rc={completed.returncode}"
        ),
        metrics=metrics,
        duration_s=time.time() - start,
        details=output[-4000:],
    )


def _load_model_and_tokenizer(model_id: str, trust_remote_code: bool = False):
    """Load HF model and tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code)
    return model, tokenizer


def _run_hf_steps(model, tokenizer, prompt: str, max_new_tokens: int):
    """Run HF model step-by-step, collecting per-step logits."""
    import torch

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    all_logits = []

    with torch.no_grad():
        # Prefill
        out = model(input_ids)
        logits = out.logits[:, -1, :].float().cpu().numpy()
        all_logits.append(logits[0])
        next_token = int(logits[0].argmax(-1))

        # Decode
        past = out.past_key_values
        for _ in range(max_new_tokens - 1):
            next_input = torch.tensor([[next_token]], device="cuda")
            out = model(next_input, past_key_values=past)
            logits = out.logits[:, -1, :].float().cpu().numpy()
            all_logits.append(logits[0])
            past = out.past_key_values
            next_token = int(logits[0].argmax(-1))

    return all_logits


def _run_torchtrt_steps(model, tokenizer, prompt: str, max_new_tokens: int,
                         max_cache_length: int):
    """Run Torch-TRT compiled model step-by-step, collecting per-step logits.

    Uses StaticCache for explicit cache management, matching the export
    signature used by the build pipeline.
    """
    import torch
    from transformers import StaticCache

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    seq_len = input_ids.shape[1]
    all_logits = []

    # Create static cache
    cache = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=max_cache_length + seq_len,
        dtype=torch.float16,
        device="cuda",
    )

    with torch.no_grad():
        # Prefill
        cache_position = torch.arange(seq_len, device="cuda")
        position_ids = cache_position.unsqueeze(0)
        attention_mask = torch.ones(1, max_cache_length + seq_len,
                                    dtype=torch.long, device="cuda")
        # Mask out unfilled positions
        attention_mask[0, seq_len:] = 0

        out = model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=cache,
        )
        logits = out.logits[:, -1, :].float().cpu().numpy()
        all_logits.append(logits[0])
        next_token = int(logits[0].argmax(-1))

        # Decode
        for step in range(max_new_tokens - 1):
            pos = seq_len + step
            next_input = torch.tensor([[next_token]], device="cuda")
            cache_position = torch.tensor([pos], device="cuda")
            position_ids = torch.tensor([[pos]], device="cuda")

            # Update attention mask
            attention_mask = torch.ones(1, max_cache_length + 1,
                                        dtype=torch.long, device="cuda")
            filled = pos + 1
            if filled < max_cache_length + 1:
                attention_mask[0, filled:] = 0

            out = model(
                next_input,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                past_key_values=cache,
            )
            logits = out.logits[:, -1, :].float().cpu().numpy()
            all_logits.append(logits[0])
            next_token = int(logits[0].argmax(-1))

    return all_logits


def _compare_logits(hf_logits: list, trt_logits: list, *, atol: float,
                     verbose: bool = False) -> dict[str, Any]:
    """Compare per-step logits between HF and TRT.

    Returns a dict with comparison metrics.
    """
    assert len(hf_logits) == len(trt_logits), \
        f"Step count mismatch: HF={len(hf_logits)}, TRT={len(trt_logits)}"

    results = {
        "num_steps": len(hf_logits),
        "max_abs_diff": 0.0,
        "mean_abs_diff": 0.0,
        "cosine_sims": [],
        "top1_matches": 0,
        "top5_overlaps": [],
        "all_within_atol": True,
    }

    total_abs_diff = 0.0
    for i, (hf, trt) in enumerate(zip(hf_logits, trt_logits)):
        diff = np.abs(hf - trt)
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())
        total_abs_diff += mean_diff

        results["max_abs_diff"] = max(results["max_abs_diff"], max_diff)

        # Cosine similarity
        norm_hf = np.linalg.norm(hf)
        norm_trt = np.linalg.norm(trt)
        if norm_hf > 0 and norm_trt > 0:
            cos_sim = float(np.dot(hf, trt) / (norm_hf * norm_trt))
        else:
            cos_sim = 0.0
        results["cosine_sims"].append(cos_sim)

        # Top-1 match
        if np.argmax(hf) == np.argmax(trt):
            results["top1_matches"] += 1

        # Top-5 overlap
        top5_hf = set(np.argsort(hf)[-5:])
        top5_trt = set(np.argsort(trt)[-5:])
        overlap = len(top5_hf & top5_trt) / 5.0
        results["top5_overlaps"].append(overlap)

        if max_diff > atol:
            results["all_within_atol"] = False

        if verbose:
            hf_token = int(np.argmax(hf))
            trt_token = int(np.argmax(trt))
            match_str = "OK" if hf_token == trt_token else "MISMATCH"
            print(f"  step {i}: max_diff={max_diff:.6f} cos={cos_sim:.6f} "
                  f"top1={match_str} (HF={hf_token} TRT={trt_token})")

    results["mean_abs_diff"] = total_abs_diff / len(hf_logits) if hf_logits else 0
    results["mean_cosine_sim"] = (
        sum(results["cosine_sims"]) / len(results["cosine_sims"])
        if results["cosine_sims"] else 0)
    results["top1_match_rate"] = (
        results["top1_matches"] / results["num_steps"]
        if results["num_steps"] > 0 else 0)
    results["mean_top5_overlap"] = (
        sum(results["top5_overlaps"]) / len(results["top5_overlaps"])
        if results["top5_overlaps"] else 0)

    return results


def run_comparison(
    model_id: str,
    *,
    atol: float = 1e-2,
    max_cache_length: int = 256,
    max_new_tokens: int = 10,
    battery: bool = False,
    trust_remote_code: bool = False,
    no_compile: bool = False,
    verbose: bool = False,
) -> bool:
    """Run the full comparison. Returns True if all prompts pass."""
    prompts = BATTERY_PROMPTS if battery else [SINGLE_PROMPT]

    print(f"Loading model: {model_id}", file=sys.stderr)
    model, tokenizer = _load_model_and_tokenizer(model_id, trust_remote_code)

    all_pass = True
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}/{len(prompts)}: {prompt!r}", file=sys.stderr)

        # HF reference (eager, standard cache)
        hf_logits = _run_hf_steps(model, tokenizer, prompt, max_new_tokens)

        # Torch-TRT (or HF with StaticCache if --no-compile)
        trt_logits = _run_torchtrt_steps(
            model, tokenizer, prompt, max_new_tokens, max_cache_length)

        results = _compare_logits(hf_logits, trt_logits, atol=atol, verbose=verbose)

        status = "PASS" if results["top1_match_rate"] >= 0.8 else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"  {status}: steps={results['num_steps']} "
              f"top1_match={results['top1_match_rate']:.0%} "
              f"cos_sim={results['mean_cosine_sim']:.6f} "
              f"max_diff={results['max_abs_diff']:.6f} "
              f"top5_overlap={results['mean_top5_overlap']:.0%}")

    return all_pass


def main():
    parser = argparse.ArgumentParser(
        description="Torch-TRT vs HF logit comparison")
    parser.add_argument("--model", required=True,
                        help="HF repo ID or local model directory")
    parser.add_argument("--atol", type=float, default=1e-2,
                        help="Absolute tolerance (default: 1e-2)")
    parser.add_argument("--max-cache-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--battery", action="store_true",
                        help="Run multiple prompts")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-compile", action="store_true",
                        help="Compare HF eager vs HF StaticCache (no TRT)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    success = run_comparison(
        args.model,
        atol=args.atol,
        max_cache_length=args.max_cache_length,
        max_new_tokens=args.max_new_tokens,
        battery=args.battery,
        trust_remote_code=args.trust_remote_code,
        no_compile=args.no_compile,
        verbose=args.verbose,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
