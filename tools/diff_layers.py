#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-layer TRT-vs-HF hidden state comparison.

Builds a TRT debug engine with per-layer outputs marked, runs a single
forward pass, and compares per-layer hidden states against HF transformers.

Usage:
    python3 tools/diff_layers.py \
      --model org/model-name \
      --prompt "Hello" --atol 0.05

    python3 tools/diff_layers.py \
      --model models/hf/example-model --atol 0.05
"""
import argparse
import sys

import numpy as np

from tool_helpers import (
    load_hf_model,
    make_family_debug_runner,
    runtime_strategy_from_config,
)


def build_debug_engine(model_id_or_path, max_cache_length, verbose):
    """Build TRT engine with debug layer outputs marked."""
    from tensorrt_model_connect.engine_builder import _resolve_model
    from tensorrt_model_connect.config import ModelConfig
    from tensorrt_model_connect.families import family_has_capability, find_model

    model_dir = _resolve_model(model_id_or_path)
    config = ModelConfig.from_dir(model_dir)
    model = find_model(config)
    if model is None:
        raise ValueError(f"No family model for model_type={config.model_type!r}")
    if not family_has_capability(config, "debug_layer_outputs"):
        raise ValueError(
            f"Family for model_type={config.model_type!r} does not declare "
            "debug_layer_outputs support")

    print(f"[diff-layers] Model: {config.model_type} "
          f"(layers={config.num_hidden_layers}, hidden={config.hidden_size})",
          file=sys.stderr)

    print("[diff-layers] Loading weights ...", file=sys.stderr)
    weights = model.load_weights(model_dir, config)

    # Build with debug outputs through the family-owned model module.
    print(f"[diff-layers] Building debug TRT engine (cache={max_cache_length}) ...",
          file=sys.stderr)
    engine_plan = model.build_engine(
        config, weights, max_cache_length,
        verbose=verbose, debug_layer_outputs=True)
    print(f"[diff-layers] Debug engine built ({len(engine_plan) / 1e6:.1f} MB)",
          file=sys.stderr)

    return engine_plan, config, model_dir


def run_trt_single_step(engine_plan, config, token_id, max_cache_length):
    """Run one TRT step at position 0. Returns dict with all outputs."""
    runner = make_family_debug_runner(
        engine_plan=engine_plan,
        runtime_strategy=runtime_strategy_from_config(config),
        max_cache_length=max_cache_length,
        num_layers=config.num_hidden_layers,
        config=config,
    )
    return runner.step(token_id)


def run_hf_single_step(model_dir, token_id, trust_remote_code=False):
    """Run HF model on a single token, return per-layer hidden states."""
    import torch

    model = load_hf_model(model_dir, trust_remote_code=trust_remote_code,
                          tag="diff-layers")
    model.eval()

    ids_tensor = torch.tensor([[token_id]], dtype=torch.long)
    with torch.no_grad():
        outputs = model(ids_tensor, output_hidden_states=True)

    # outputs.hidden_states: tuple of (batch, seq, hidden) tensors
    # [0] = embedding, [1] = after layer 0, [2] = after layer 1, ...
    hidden_states = []
    for hs in outputs.hidden_states:
        hidden_states.append(hs[0, 0].numpy())  # (hidden,)

    logits = outputs.logits[0, 0].numpy()  # (vocab,)
    return hidden_states, logits


def main():
    parser = argparse.ArgumentParser(
        description="Per-layer TRT-vs-HF hidden state comparison")
    parser.add_argument("--model", required=True,
                        help="HF repo ID or local model directory")
    parser.add_argument("--prompt", default="Hello",
                        help="Input prompt (first token used)")
    parser.add_argument("--max-cache-length", type=int, default=64)
    parser.add_argument("--atol", type=float, default=0.05,
                        help="Absolute tolerance for hidden state comparison")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Allow executing custom Python code from the "
                             "model repository (required for models without "
                             "native transformers support)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Build debug engine
    engine_plan, config, model_dir = build_debug_engine(
        args.model, args.max_cache_length, args.verbose)

    # Tokenize — use first token only for single-step comparison
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=args.trust_remote_code)
    input_ids = tokenizer.encode(args.prompt)
    token_id = input_ids[0]
    print(f"\n[diff-layers] Token: id={token_id} "
          f"text={tokenizer.decode([token_id])!r}")

    # Run TRT (single step at position 0)
    print("[diff-layers] Running TRT ...", file=sys.stderr)
    trt_results = run_trt_single_step(
        engine_plan, config, token_id, args.max_cache_length)

    # Run HF (single token forward pass)
    print("[diff-layers] Running HF ...", file=sys.stderr)
    hf_hidden, hf_logits = run_hf_single_step(
        model_dir, token_id, trust_remote_code=args.trust_remote_code)

    # Compare
    print(f"\n{'=' * 72}")
    print(f"{'Layer':<20} {'Shape':>14} {'MaxDiff':>10} {'MeanDiff':>10} "
          f"{'TRT_std':>10} {'HF_std':>10} {'Status':>8}")
    print(f"{'-' * 72}")

    max_overall = 0.0
    all_passed = True

    # Embedding
    if "debug_embed" in trt_results:
        trt_embed = trt_results["debug_embed"].flatten()
        hf_embed = hf_hidden[0]  # outputs.hidden_states[0] = embedding
        diff = np.abs(trt_embed - hf_embed)
        md = float(diff.max())
        max_overall = max(max_overall, md)
        status = "OK" if md <= args.atol else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"{'embed':<20} {str(trt_embed.shape):>14} "
              f"{md:>10.6f} {float(diff.mean()):>10.6f} "
              f"{float(trt_embed.std()):>10.4f} {float(hf_embed.std()):>10.4f} "
              f"{status:>8}")

    # Per-layer hidden states
    #
    # In transformers 5.x, the `check_model_inputs` wrapper with
    # tie_last_hidden_states=True (default for LMs) overwrites
    # hidden_states[-1] with last_hidden_state (which is post-final-norm).
    # So hidden_states has num_layers+2 entries:
    #   [0]=embed, [1]..[N]=after layers 0..N-1, [N+1]=post-final-norm
    # The last entry does NOT correspond to the raw output of the last
    # decoder layer. We skip the comparison for the last layer and rely
    # on the logits comparison to validate correctness instead.
    num_layers = config.num_hidden_layers
    for i in range(num_layers):
        hidden_key = f"debug_hidden_{i}"
        if hidden_key not in trt_results:
            print(f"  Layer {i}: debug output not found")
            continue

        trt_h = trt_results[hidden_key].flatten()
        # HF hidden_states[i+1] = output of layer i (for all but the last).
        # For the last layer, hf_hidden[num_layers] is the post-final-norm
        # output (due to tie_last_hidden_states), so skip the comparison.
        is_last = (i == num_layers - 1)
        if is_last and len(hf_hidden) == num_layers + 1:
            # The last HF entry is post-norm; skip direct comparison.
            print(f"{'layer.' + str(i) + '.hidden':<20} {str(trt_h.shape):>14} "
                  f"{'---':>10} {'---':>10} "
                  f"{float(trt_h.std()):>10.4f} {'---':>10} "
                  f"{'(skip: post-norm in HF)':>8}")
            continue

        hf_h = hf_hidden[i + 1] if (i + 1) < len(hf_hidden) else None

        if hf_h is None:
            print(f"  Layer {i}: HF hidden state not available")
            continue

        diff = np.abs(trt_h - hf_h)
        md = float(diff.max())
        max_overall = max(max_overall, md)
        status = "OK" if md <= args.atol else "FAIL"
        if status == "FAIL":
            all_passed = False

        print(f"{'layer.' + str(i) + '.hidden':<20} {str(trt_h.shape):>14} "
              f"{md:>10.6f} {float(diff.mean()):>10.6f} "
              f"{float(trt_h.std()):>10.4f} {float(hf_h.std()):>10.4f} "
              f"{status:>8}")

    # Per-layer post-attention
    for i in range(num_layers):
        attn_key = f"debug_post_attn_{i}"
        if attn_key not in trt_results:
            continue

        trt_a = trt_results[attn_key].flatten()
        # No direct HF equivalent for post-attention residual without hooks,
        # so just report the TRT values for reference
        print(f"{'layer.' + str(i) + '.post_attn':<20} {str(trt_a.shape):>14} "
              f"{'---':>10} {'---':>10} "
              f"{float(trt_a.std()):>10.4f} {'---':>10} "
              f"{'(ref)':>8}")

    # Logits
    trt_logits = trt_results["logits"].flatten()
    diff = np.abs(trt_logits - hf_logits)
    md = float(diff.max())
    max_overall = max(max_overall, md)
    status = "OK" if md <= args.atol else "FAIL"
    if status == "FAIL":
        all_passed = False

    trt_argmax = int(np.argmax(trt_logits))
    hf_argmax = int(np.argmax(hf_logits))
    print(f"{'logits':<20} {str(trt_logits.shape):>14} "
          f"{md:>10.6f} {float(diff.mean()):>10.6f} "
          f"{float(trt_logits.std()):>10.4f} {float(hf_logits.std()):>10.4f} "
          f"{status:>8}")
    print(f"\n  TRT argmax: {trt_argmax} ({tokenizer.decode([trt_argmax])!r})")
    print(f"  HF  argmax: {hf_argmax} ({tokenizer.decode([hf_argmax])!r})")
    print(f"  Argmax match: {trt_argmax == hf_argmax}")

    print(f"\n  Overall max diff: {max_overall:.6f}")
    print(f"  Tolerance: {args.atol}")
    print(f"  {'PASS' if all_passed else 'FAIL'}")
    sys.exit(0 if all_passed else 1)


def run_as_diff_test(ctx):
    """Framework entry point. Returns DiffResult."""
    from diff_framework.protocol import DiffResult
    import time as _time

    t0 = _time.monotonic()
    try:
        engine_plan, config, model_dir = build_debug_engine(
            ctx.model, ctx.max_cache_length, ctx.verbose)

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=ctx.trust_remote_code)
        input_ids = tokenizer.encode("Hello")
        token_id = input_ids[0]

        trt_results = run_trt_single_step(
            engine_plan, config, token_id, ctx.max_cache_length)
        hf_hidden, hf_logits = run_hf_single_step(
            model_dir, token_id, trust_remote_code=ctx.trust_remote_code)

        max_overall = 0.0
        all_passed = True
        layer_atol = ctx.layer_atol

        # Embedding
        if "debug_embed" in trt_results:
            trt_embed = trt_results["debug_embed"].flatten()
            hf_embed = hf_hidden[0]
            diff = np.abs(trt_embed - hf_embed)
            md = float(diff.max())
            max_overall = max(max_overall, md)
            if md > layer_atol:
                all_passed = False

        # Per-layer hidden states
        num_layers = config.num_hidden_layers
        for i in range(num_layers):
            hidden_key = f"debug_hidden_{i}"
            if hidden_key not in trt_results:
                continue
            trt_h = trt_results[hidden_key].flatten()
            is_last = (i == num_layers - 1)
            if is_last and len(hf_hidden) == num_layers + 1:
                continue
            hf_h = hf_hidden[i + 1] if (i + 1) < len(hf_hidden) else None
            if hf_h is None:
                continue
            diff = np.abs(trt_h - hf_h)
            md = float(diff.max())
            max_overall = max(max_overall, md)
            if md > layer_atol:
                all_passed = False

        # Logits
        trt_logits_arr = trt_results["logits"].flatten()
        diff = np.abs(trt_logits_arr - hf_logits)
        md = float(diff.max())
        max_overall = max(max_overall, md)
        if md > layer_atol:
            all_passed = False

        return DiffResult(
            test_name="layer_diff", model=ctx.model,
            runtime_strategy=ctx.runtime_strategy,
            passed=all_passed,
            status="PASS" if all_passed else "FAIL",
            message=f"max_overall_diff={max_overall:.6f} (atol={layer_atol})",
            metrics={"max_overall_diff": max_overall, "atol": layer_atol},
            duration_s=_time.monotonic() - t0)
    except Exception as e:
        return DiffResult.error(
            "layer_diff", ctx.model, ctx.runtime_strategy, str(e))


if __name__ == "__main__":
    main()
