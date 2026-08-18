# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for standard decoder/comparison tools.

Used by: diff_logits.py, diff_layers.py, perf_compare.py,
         debug_diffusion_pipeline.py
"""
from __future__ import annotations

import sys

import numpy as np


def build_trt_engine(model_id_or_path, max_cache_length, verbose, *, tag="diff"):
    """Build TRT engine and return (engine_plan_bytes, config, model_dir)."""
    from tensorrt_model_connect.engine_builder import _resolve_model
    from tensorrt_model_connect.config import ModelConfig
    from tensorrt_model_connect.families import find_model

    model_dir = _resolve_model(model_id_or_path)
    config = ModelConfig.from_dir(model_dir)
    model = find_model(config)
    if model is None:
        raise ValueError(f"No family model for model_type={config.model_type!r}")

    print(f"[{tag}] Loading weights ({config.model_type}) ...", file=sys.stderr)
    weights = model.load_weights(model_dir, config)
    print(f"[{tag}] Building TRT engine (cache={max_cache_length}) ...",
          file=sys.stderr)
    engine_plan = model.build_engine(
        config, weights, max_cache_length, verbose=verbose)
    print(f"[{tag}] Engine built ({len(engine_plan) / 1e6:.1f} MB)",
          file=sys.stderr)

    return engine_plan, config, model_dir


def runtime_strategy_from_config(config) -> str:
    """Return the family-owned runtime strategy for a ModelConfig-like object."""
    from tensorrt_model_connect.families import find_model

    model = find_model(config)
    strategy = str(getattr(model, "runtime_strategy", "") or "")
    if not strategy:
        raise ValueError(
            f"No runtime_strategy declared for model_type="
            f"{getattr(config, 'model_type', config)!r}"
        )
    return strategy


def _debug_runner_config_dict(config, runtime_strategy: str, num_layers: int) -> dict:
    if isinstance(config, dict):
        result = dict(config)
    else:
        result = {
            "model_type": str(getattr(config, "model_type", "")),
            "num_hidden_layers": int(
                getattr(config, "num_hidden_layers", num_layers)
            ),
        }
        raw = getattr(config, "raw", None)
        if isinstance(raw, dict):
            result.update(raw)
    result.setdefault("runtime_strategy", runtime_strategy)
    result.setdefault("num_hidden_layers", num_layers)
    return result


def make_family_debug_runner(
    *,
    engine_plan: bytes,
    runtime_strategy: str,
    max_cache_length: int,
    num_layers: int,
    config=None,
    bundle_path: str = "",
    profiler=None,
):
    """Instantiate the debug runner owned by runtime_strategy's family."""
    from tensorrt_model_connect.families import resolve_debug_runner

    strategy = str(runtime_strategy or "")
    if not strategy:
        raise ValueError("runtime_strategy is required for family debug runner dispatch")
    runner = resolve_debug_runner(
        strategy,
        config=_debug_runner_config_dict(config, strategy, num_layers),
        header={
            "max_cache_length": max_cache_length,
            "num_layers": num_layers,
        },
        engine_plan=engine_plan,
        bundle_path=bundle_path,
    )
    if runner is None:
        raise RuntimeError(
            f"No family-owned debug_runner adapter handles {strategy!r}"
        )
    if profiler is not None:
        runner.context.profiler = profiler
    return runner


def load_hf_model(model_dir, *, trust_remote_code=False, torch_dtype=None,
                   tag="diff"):
    """Load a standard causal-LM HF model with trust_remote_code fallback.

    Returns the model on CPU. Callers should move to device and call eval()
    as needed.
    """
    import torch
    from transformers import AutoModelForCausalLM

    if torch_dtype is None:
        torch_dtype = torch.float32

    try:
        return AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=False, torch_dtype=torch_dtype)
    except (ValueError, KeyError, ImportError) as e:
        if trust_remote_code:
            print(f"[{tag}] Native loading failed ({e}), "
                  f"retrying with trust_remote_code=True ...",
                  file=sys.stderr)
            return AutoModelForCausalLM.from_pretrained(
                model_dir, trust_remote_code=True, torch_dtype=torch_dtype)
        raise ValueError(
            f"Failed to load model from {model_dir} without custom code. "
            f"If this model requires custom code, re-run with "
            f"--trust-remote-code. Original error: {e}"
        ) from e


def cosine_sim(a, b):
    """Compute cosine similarity between two arrays."""
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def compare_arrays(name, ours, ref, atol):
    """Compare two arrays and print PASS/FAIL. Returns True if within tolerance."""
    diff = np.abs(ours.flatten() - ref.flatten())
    mx, mn = float(diff.max()), float(diff.mean())
    cs = cosine_sim(ours, ref)
    ok = mx <= atol
    tag = "PASS" if ok else "FAIL"
    print(f"   {tag}: max_diff={mx:.6f}, mean_diff={mn:.6f}, cosine_sim={cs:.6f}")
    return ok
