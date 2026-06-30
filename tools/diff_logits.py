#!/usr/bin/env python3
"""Pure-Python E2E logit comparison between TRT engine and HF transformers.

No C++ binary needed. Builds a TRT engine via tensorrt_model_connect, runs inference
in Python, and compares per-step logits against HF transformers.

Usage:
    python3 tools/diff_logits.py \
      --model example-org/example-causal-lm \
      --prompt "The capital of France is" \
      --max-new-tokens 10 --atol 1e-3

    python3 tools/diff_logits.py \
      --model models/hf/example-causal-lm --battery

    # Machine-readable JSON output for automated accuracy gating:
    python3 tools/diff_logits.py \
      --model example-org/example-causal-lm --battery --json results.json
"""
import argparse
import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import numpy as np

from tool_helpers import make_family_debug_runner, runtime_strategy_from_config

STANDARD_PROMPTS = [
    ("factual", "The capital of France is"),
    ("reasoning", "Explain why water boils at 100 degrees Celsius."),
    ("code", "Write a Python function that checks if a number is prime:"),
    ("multi-turn", "User: What is 2+2?\nAssistant:"),
]


@lru_cache(maxsize=1)
def _family_diff_logits_modules() -> tuple[ModuleType, ...]:
    """Load optional model-owned logit diff hooks from family folders."""
    family_root = Path(__file__).resolve().parent / "families"
    modules: list[ModuleType] = []
    for handler_path in sorted(family_root.glob("*/diff_logits.py")):
        module_name = f"_trtmc_diff_logits_{handler_path.parent.name}"
        spec = importlib.util.spec_from_file_location(module_name, handler_path)
        if spec is None or spec.loader is None:
            print(f"[diff] WARN: cannot load family logit diff handler "
                  f"{handler_path}", file=sys.stderr)
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            print(f"[diff] WARN: failed to import family logit diff handler "
                  f"{handler_path}: {exc}", file=sys.stderr)
            continue
        if callable(getattr(module, "handles_model_type", None)):
            modules.append(module)
    return tuple(modules)


def _find_family_diff_logits_handler(model_type: str) -> ModuleType | None:
    """Return the model-owned logit diff hook module for a model type."""
    for module in _family_diff_logits_modules():
        handles = getattr(module, "handles_model_type")
        if handles(model_type):
            return module
    return None


def build_trt_engine(model_id_or_path, max_cache_length, verbose):
    """Build TRT engine and return (engine_plan_bytes, config, model_dir)."""
    from tensorrt_model_connect.engine_builder import _resolve_model
    from tensorrt_model_connect.config import ModelConfig
    from tensorrt_model_connect.families import find_plugin

    model_dir = _resolve_model(model_id_or_path)
    config = ModelConfig.from_dir(model_dir)
    plugin = find_plugin(config.model_type)
    if plugin is None:
        raise ValueError(f"No plugin for model_type={config.model_type!r}")

    print(f"[diff] Loading weights ({config.model_type}) ...", file=sys.stderr)
    weights = plugin.load_weights(model_dir, config)
    print(f"[diff] Building TRT engine (cache={max_cache_length}) ...",
          file=sys.stderr)
    engine_plan = plugin.build_engine(
        config, weights, max_cache_length, verbose=verbose)
    print(f"[diff] Engine built ({len(engine_plan) / 1e6:.1f} MB)",
          file=sys.stderr)

    handler = _find_family_diff_logits_handler(config.model_type)
    attach_plans = getattr(handler, "attach_additional_plans", None)
    if callable(attach_plans):
        attach_plans(plugin, model_dir, config, weights, verbose=verbose)

    return engine_plan, config, model_dir


def run_trt(engine_plan, config, input_ids, max_new_tokens, max_cache_length):
    """Run TRT inference, return list of logit arrays (one per step)."""
    handler = _find_family_diff_logits_handler(config.model_type)
    make_runner = getattr(handler, "make_trt_runner", None)
    if callable(make_runner):
        runner = make_runner(engine_plan, config, max_cache_length)
    else:
        runner = make_family_debug_runner(
            engine_plan=engine_plan,
            runtime_strategy=runtime_strategy_from_config(config),
            max_cache_length=max_cache_length,
            num_layers=config.num_hidden_layers,
            config=config,
        )

    results = runner.generate(input_ids, max_new_tokens)
    return [r["logits"].flatten() for r in results]


def _load_hf_model(model_dir, trust_remote_code=False):
    """Load HF model. Uses native transformers support by default.

    If the model requires custom code (e.g. older repos without native
    transformers support), pass --trust-remote-code to enable it.
    """
    import json
    import torch
    from transformers import AutoModelForCausalLM

    config_path = Path(model_dir) / "config.json"
    model_type = ""
    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        model_type = cfg.get("model_type", "").lower()

    handler = _find_family_diff_logits_handler(model_type)
    load_hf_model = getattr(handler, "load_hf_model", None)
    if callable(load_hf_model):
        return load_hf_model(model_dir)

    try:
        return AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=False, torch_dtype=torch.float32)
    except (ValueError, KeyError, ImportError) as e:
        if trust_remote_code:
            print(f"[diff] Native loading failed ({e}), "
                  f"retrying with trust_remote_code=True ...",
                  file=sys.stderr)
            return AutoModelForCausalLM.from_pretrained(
                model_dir, trust_remote_code=True, torch_dtype=torch.float32)
        raise ValueError(
            f"Failed to load model from {model_dir} without custom code. "
            f"If this model requires custom code, re-run with "
            f"--trust-remote-code. Original error: {e}"
        ) from e


def run_hf(model_dir, config, input_ids, max_new_tokens, trust_remote_code=False):
    """Run HF transformers, return list of logit arrays (one per step)."""
    import torch

    model = _load_hf_model(model_dir, trust_remote_code=trust_remote_code)
    model.eval()

    handler = _find_family_diff_logits_handler(config.model_type)
    run_hf_model = getattr(handler, "run_hf", None)
    if callable(run_hf_model):
        return run_hf_model(model, config, input_ids, max_new_tokens)

    # Standard causal LM
    ids_tensor = torch.tensor([input_ids], dtype=torch.long)
    all_logits = []

    with torch.no_grad():
        # Prefill: get logits at each input position
        outputs = model(ids_tensor)
        prefill_logits = outputs.logits[0].numpy()  # (seq_len, vocab)
        for i in range(len(input_ids)):
            all_logits.append(prefill_logits[i])

        # Generate: autoregressive
        gen_ids = list(input_ids)
        for _ in range(max_new_tokens):
            next_token = int(np.argmax(all_logits[-1]))
            gen_ids.append(next_token)
            ids_tensor = torch.tensor([gen_ids], dtype=torch.long)
            outputs = model(ids_tensor)
            all_logits.append(outputs.logits[0, -1].numpy())

    return all_logits


def _prompt_cases_for_handler(prompts, handler):
    """Return (label, display_text, input_ids_override) prompt cases."""
    get_prompt_cases = getattr(handler, "prompt_cases", None)
    if callable(get_prompt_cases):
        return get_prompt_cases(prompts)
    return [(label, prompt, None) for label, prompt in prompts]


def _cosine_similarity(a, b):
    """Cosine similarity between two 1-D arrays. Returns 0.0 if either is zero."""
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def compare_logits(trt_logits, hf_logits, atol, top_k=10):
    """Compare logit arrays step by step.

    Returns (max_diff, report_lines, step_metrics) where step_metrics is a
    list of dicts with per-step structured data (cosine_sim, argmax_match,
    mean_abs_diff).  Steps with shape mismatches are omitted from
    step_metrics.
    """
    n = min(len(trt_logits), len(hf_logits))
    max_diff = 0.0
    lines = []
    step_metrics = []

    for step in range(n):
        trt_l = trt_logits[step]
        hf_l = hf_logits[step]

        if trt_l.shape != hf_l.shape:
            lines.append(f"  step {step}: shape mismatch "
                         f"trt={trt_l.shape} hf={hf_l.shape}")
            continue

        # Full logit comparison
        diff = np.abs(trt_l - hf_l)
        step_max = float(diff.max())
        step_mean = float(diff.mean())
        max_diff = max(max_diff, step_max)

        # Cosine similarity
        cosine = _cosine_similarity(trt_l, hf_l)

        # Top-K token agreement
        trt_top = set(np.argsort(trt_l)[-top_k:])
        hf_top = set(np.argsort(hf_l)[-top_k:])
        overlap = len(trt_top & hf_top)

        trt_argmax = int(np.argmax(trt_l))
        hf_argmax = int(np.argmax(hf_l))
        argmax_match = "Y" if trt_argmax == hf_argmax else "N"

        lines.append(
            f"  step {step:3d}: max_diff={step_max:10.6f}  "
            f"argmax_match={argmax_match}  "
            f"top{top_k}_overlap={overlap}/{top_k}")

        step_metrics.append({
            "step": step,
            "cosine_sim": cosine,
            "argmax_match": trt_argmax == hf_argmax,
            "mean_abs_diff": step_mean,
            "max_abs_diff": step_max,
        })

    return max_diff, lines, step_metrics


def _build_json_report(prompt_results, atol):
    """Build a machine-readable JSON report from accumulated prompt results.

    Args:
        prompt_results: list of dicts, each with keys:
            - label: prompt label string
            - passed: bool, whether this prompt passed the atol gate
            - max_diff: float, max absolute logit diff for this prompt
            - step_metrics: list of per-step metric dicts from compare_logits
            - trt_text: decoded TRT output text
            - hf_text: decoded HF output text
        atol: float, absolute tolerance used for the comparison

    Returns:
        dict with top-level summary fields and per-prompt details.
    """
    # Collect all step metrics across all prompts
    all_cosines = []
    all_argmax_matches = []
    all_mean_abs_diffs = []
    for pr in prompt_results:
        for sm in pr["step_metrics"]:
            all_cosines.append(sm["cosine_sim"])
            all_argmax_matches.append(sm["argmax_match"])
            all_mean_abs_diffs.append(sm["mean_abs_diff"])

    overall_pass = all(pr["passed"] for pr in prompt_results)

    # Compute aggregate metrics (safe defaults when no steps exist)
    if all_cosines:
        cosine_p5 = float(np.percentile(all_cosines, 5))
    else:
        cosine_p5 = 0.0

    if all_argmax_matches:
        top1_match_rate = float(
            sum(all_argmax_matches) / len(all_argmax_matches))
    else:
        top1_match_rate = 0.0

    if all_mean_abs_diffs:
        mean_abs_diff = float(np.mean(all_mean_abs_diffs))
    else:
        mean_abs_diff = 0.0

    # Token agreement: fraction of prompts where TRT and HF produce the
    # same decoded text (stripped).
    if prompt_results:
        text_matches = sum(
            1 for pr in prompt_results
            if pr["trt_text"].strip() == pr["hf_text"].strip()
        )
        token_agreement = float(text_matches / len(prompt_results))
    else:
        token_agreement = 0.0

    report = {
        "pass": overall_pass,
        "cosine_p5": cosine_p5,
        "top1_match_rate": top1_match_rate,
        "token_agreement": token_agreement,
        "mean_abs_diff": mean_abs_diff,
        "atol": atol,
        "num_prompts": len(prompt_results),
        "prompts": [],
    }

    for pr in prompt_results:
        prompt_entry = {
            "label": pr["label"],
            "passed": pr["passed"],
            "max_diff": pr["max_diff"],
            "trt_text": pr["trt_text"],
            "hf_text": pr["hf_text"],
            "num_steps": len(pr["step_metrics"]),
        }
        report["prompts"].append(prompt_entry)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Pure-Python E2E logit comparison: TRT vs HF transformers")
    parser.add_argument("--model", required=True,
                        help="HF repo ID or local model directory")
    parser.add_argument("--prompt", default="",
                        help="Single prompt (overrides --battery)")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--max-cache-length", type=int, default=64)
    parser.add_argument("--atol", type=float, default=1e-3,
                        help="Absolute tolerance for logit comparison")
    parser.add_argument("--battery", action="store_true",
                        help="Run standard prompt battery")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Allow executing custom Python code from the "
                             "model repository (required for models without "
                             "native transformers support)")
    parser.add_argument("--json", metavar="PATH", default=None,
                        help="Write machine-readable JSON report to PATH")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    prompts = []
    if args.prompt:
        prompts = [("custom", args.prompt)]
    elif args.battery:
        prompts = STANDARD_PROMPTS
    else:
        prompts = [("default", "The capital of France is")]

    # Build engine once
    engine_plan, config, model_dir = build_trt_engine(
        args.model, args.max_cache_length, args.verbose)

    # Load HF tokenizer for encoding prompts
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=args.trust_remote_code)

    handler = _find_family_diff_logits_handler(config.model_type)
    prompt_cases = _prompt_cases_for_handler(prompts, handler)

    all_passed = True
    prompt_results = []
    for label, prompt, input_ids_override in prompt_cases:
        print(f"\n{'=' * 60}")
        print(f"Prompt [{label}]: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        print(f"{'=' * 60}")

        if input_ids_override is None:
            input_ids = tokenizer.encode(prompt)
        else:
            input_ids = input_ids_override
        print(f"  Input tokens: {len(input_ids)}")

        # Run TRT
        print("  Running TRT ...", file=sys.stderr)
        trt_logits = run_trt(
            engine_plan, config, input_ids,
            args.max_new_tokens, args.max_cache_length)

        # Run HF
        print("  Running HF ...", file=sys.stderr)
        hf_logits = run_hf(model_dir, config, input_ids, args.max_new_tokens,
                           trust_remote_code=args.trust_remote_code)

        # Compare
        max_diff, report, step_metrics = compare_logits(
            trt_logits, hf_logits, args.atol)

        # Decode generated text
        trt_gen_ids = [int(np.argmax(l)) for l in trt_logits[len(input_ids) - 1:]]
        hf_gen_ids = [int(np.argmax(l)) for l in hf_logits[len(input_ids) - 1:]]
        trt_text = tokenizer.decode(trt_gen_ids, skip_special_tokens=True)
        hf_text = tokenizer.decode(hf_gen_ids, skip_special_tokens=True)

        print(f"  TRT text: {trt_text[:120]}")
        print(f"  HF  text: {hf_text[:120]}")
        print(f"  Text match: {trt_text.strip() == hf_text.strip()}")
        print()

        for line in report:
            print(line)

        passed = max_diff <= args.atol
        print(f"\n  max_abs_logit_diff: {max_diff:.6f}")
        print(f"  atol: {args.atol}")
        print(f"  {'PASS' if passed else 'FAIL'}")

        if not passed:
            all_passed = False

        prompt_results.append({
            "label": label,
            "passed": passed,
            "max_diff": max_diff,
            "step_metrics": step_metrics,
            "trt_text": trt_text,
            "hf_text": hf_text,
        })

    # Write JSON report if requested
    if args.json:
        report_dict = _build_json_report(prompt_results, args.atol)
        json_path = Path(args.json)
        json_path.write_text(json.dumps(report_dict, indent=2))
        print(f"\n[diff] JSON report written to {json_path}", file=sys.stderr)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
