#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-0.6B E2E: build bundle with FlashInfer attention, run inference, compare.

This script:
1. JIT-compiles FlashInfer single_decode kernel (native CUDA)
2. Registers it as a TVM-FFI global function
3. Builds a Qwen3-0.6B TRT engine with FlashInfer attention in all 28 layers
4. Runs autoregressive generation via the debug_runner
5. Compares latency against the baseline (standard decomposed attention)
"""
if __name__ != "__main__":
    import pytest

    pytest.skip(
        "Qwen FlashInfer smoke script requires explicit direct execution.",
        allow_module_level=True,
    )

import ctypes as ct
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import tvm_ffi

# ---------------------------------------------------------------------------
# 1. JIT compile and register FlashInfer kernel
# ---------------------------------------------------------------------------

import flashinfer.decode as fi_dec

HEAD_DIM = 64  # Qwen3-0.6B

print("JIT compiling FlashInfer kernel for Qwen3-0.6B (fp16, head_dim=64)...")
fi_module = fi_dec.gen_single_decode_module(
    torch.float16, torch.float16, torch.float16,
    HEAD_DIM, HEAD_DIM,
    pos_encoding_mode=0, use_sliding_window=False, use_logits_soft_cap=False,
).build_and_load()

tvm_ffi.register_global_func("flashinfer.decode_f16_d64", fi_module.run, override=True)
print("  Registered flashinfer.decode_f16_d64 (native CUDA, zero Python callback)")

# ---------------------------------------------------------------------------
# 2. Load plugin shared library
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[4]
shared_lib = str(REPO_ROOT / "build_shared" / "libtrtmc_core.so")
if not os.path.exists(shared_lib):
    shared_lib = str(REPO_ROOT / "build" / "libtrtmc_tvm_ffi_plugin.so")
if not os.path.exists(shared_lib):
    print("SKIP: No plugin .so found")
    sys.exit(0)

lib = ct.CDLL(shared_lib, mode=ct.RTLD_GLOBAL)
lib.tvm_ffi_plugin_force_link()
print(f"  Loaded plugin: {shared_lib}")

# ---------------------------------------------------------------------------
# 3. Build Qwen3 bundle with FlashInfer attention
# ---------------------------------------------------------------------------

# Set env var so the builder uses FlashInfer attention
os.environ["TRTMC_FFI_ATTENTION_KERNEL"] = "flashinfer.decode_f16_d64"

import tensorrt_model_connect  # noqa: E402

MODEL_ID = "Qwen/Qwen3-0.6B"
MAX_CACHE = 256
TMP_SIZE = 32 * 1024 * 1024  # 32MB FlashInfer workspace

bundle_path = "/tmp/qwen3_flashinfer.bundle"
baseline_bundle = os.environ.get(
    "TRTMC_QWEN3_BASELINE_BUNDLE",
    str(REPO_ROOT / "engines" / "qwen3-0.6b.bundle"),
)

if os.path.exists(bundle_path) and "--rebuild" not in sys.argv:
    print(f"\nUsing existing bundle: {bundle_path}")
else:
    print("\nBuilding Qwen3-0.6B bundle with FlashInfer attention...")
    print(f"  Model: {MODEL_ID}")
    print(f"  Max cache: {MAX_CACHE}")

    # Build with FlashInfer attention (uses the env var)
    tensorrt_model_connect.build(MODEL_ID, bundle_path, max_cache_length=MAX_CACHE, verbose=False)
    print(f"  Bundle: {bundle_path}")

# ---------------------------------------------------------------------------
# 4. Run inference with debug_runner
# ---------------------------------------------------------------------------

from tensorrt_model_connect.families.qwen.debug_runner import TrtRunner  # noqa: E402

def run_generation(bundle_path, prompt_tokens, max_new_tokens, label):
    """Run autoregressive generation and return (tokens, latency_ms)."""
    from tensorrt_model_connect.families.qwen.debug_runner import load_section_from_bundle
    import json

    engine_plan = load_section_from_bundle(bundle_path, "engine_plan")
    config_json = load_section_from_bundle(bundle_path, "config.json").decode()

    cfg = json.loads(config_json)
    num_layers = cfg.get("num_hidden_layers", 28)
    attention_size = cfg.get("num_attention_heads", 16) * cfg.get("head_dim", 64)

    runner = TrtRunner(engine_plan, MAX_CACHE, num_layers, attention_size)

    # Generate
    tokens = list(prompt_tokens)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(max_new_tokens):
        logits = runner.step(tokens[-1])
        next_token = int(np.argmax(logits))
        tokens.append(next_token)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    del runner
    return tokens, elapsed_ms

# Tokenize prompt
from transformers import AutoTokenizer  # noqa: E402
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
prompt = "The capital of France is"
prompt_tokens = tokenizer.encode(prompt)
max_new_tokens = 20

print(f"\nRunning generation: '{prompt}' + {max_new_tokens} tokens")

# Run FlashInfer bundle
print("\n[FlashInfer attention]")
fi_tokens, fi_ms = run_generation(bundle_path, prompt_tokens, max_new_tokens, "flashinfer")
fi_text = tokenizer.decode(fi_tokens)
print(f"  Latency: {fi_ms:.1f} ms ({max_new_tokens / (fi_ms/1000):.1f} tok/s)")
print(f"  Output: {fi_text[:100]}")

# Run baseline bundle
if os.path.exists(baseline_bundle):
    print("\n[Baseline decomposed attention]")
    base_tokens, base_ms = run_generation(baseline_bundle, prompt_tokens, max_new_tokens, "baseline")
    base_text = tokenizer.decode(base_tokens)
    print(f"  Latency: {base_ms:.1f} ms ({max_new_tokens / (base_ms/1000):.1f} tok/s)")
    print(f"  Output: {base_text[:100]}")

    # Compare
    print(f"\n{'='*60}")
    print(f"Speedup: {base_ms / fi_ms:.2f}x")
    print(f"Token match: {fi_tokens == base_tokens}")
    print(f"{'='*60}")
else:
    print(f"\n(Baseline bundle not found at {baseline_bundle})")
