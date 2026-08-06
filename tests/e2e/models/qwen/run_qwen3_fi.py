#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-4B-Instruct: FlashInfer attention via TVM-FFI plugin vs baseline.

Builds the FlashInfer engine, runs autoregressive generation with proper
prompt formatting, and compares latency against the standard TRT engine.
"""
import ctypes as ct
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import tvm_ffi
import flashinfer.decode as fd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[4]
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
BASELINE_BUNDLE = os.environ.get(
    "TRTMC_QWEN3_BASELINE_BUNDLE",
    str(REPO_ROOT / "engines" / "qwen3-4b-instruct-2507.bundle"),
)
FI_BUNDLE = "/tmp/qwen3_4b_fi.bundle"
HEAD_DIM = 128
MAX_CACHE = 256
MAX_NEW = 30
PROMPT = "What is the capital of France? Answer in one sentence."

# ---------------------------------------------------------------------------
# 1. Register FlashInfer kernel (native CUDA)
# ---------------------------------------------------------------------------

print(f"JIT compiling FlashInfer kernel (head_dim={HEAD_DIM})...")
fi_mod = fd.gen_single_decode_module(
    torch.float16, torch.float16, torch.float16,
    HEAD_DIM, HEAD_DIM, 0, False, False,
).build_and_load()
tvm_ffi.register_global_func("flashinfer.decode_f16_d128", fi_mod.run, override=True)
print("  Registered (native CUDA, zero Python callback)")

# Load plugin
lib = ct.CDLL(str(REPO_ROOT / "build_shared" / "libtrtmc_core.so"), mode=ct.RTLD_GLOBAL)
lib.tvm_ffi_plugin_force_link()

# ---------------------------------------------------------------------------
# 2. Build FlashInfer engine if needed
# ---------------------------------------------------------------------------

if not os.path.exists(FI_BUNDLE):
    os.environ["TRTMC_FFI_ATTENTION_KERNEL"] = "flashinfer.decode_f16_d128"
    from tensorrt_model_connect.engine_builder import build
    print(f"\nBuilding {MODEL_ID} with FlashInfer attention...")
    build(MODEL_ID, FI_BUNDLE, max_cache_length=MAX_CACHE)
    print(f"  Saved: {FI_BUNDLE}")
    del os.environ["TRTMC_FFI_ATTENTION_KERNEL"]
else:
    print(f"  FlashInfer bundle exists: {FI_BUNDLE}")

# ---------------------------------------------------------------------------
# 3. Runner helper
# ---------------------------------------------------------------------------

from tensorrt_model_connect.families.qwen.debug_runner import TrtRunner  # noqa: E402
from tensorrt_model_connect.families.qwen.debug_runner import load_section_from_bundle  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Format prompt with chat template
messages = [{"role": "user", "content": PROMPT}]
formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
prompt_tokens = tokenizer.encode(formatted)
print(f"\nPrompt ({len(prompt_tokens)} tokens): {formatted[:100]}...")


def run_generation(bundle_path, label, extra_inputs=None):
    """Feed all prompt tokens, then generate max_new tokens. Return (text, ms)."""
    engine_plan = load_section_from_bundle(bundle_path, "engine_plan")
    cfg = json.loads(load_section_from_bundle(bundle_path, "config.json").decode())
    num_layers = cfg["num_hidden_layers"]
    attention_size = cfg["num_attention_heads"] * cfg["head_dim"]

    runner = TrtRunner(engine_plan, MAX_CACHE, num_layers, attention_size)

    if extra_inputs:
        for name, ptr in extra_inputs.items():
            runner.context.set_tensor_address(name, ptr)

    # Prefill: feed all prompt tokens one by one to fill the cache
    for tok in prompt_tokens[:-1]:
        runner.step(tok)

    # Feed last prompt token to get first logits
    logits = runner.step(prompt_tokens[-1])
    first_tok = int(np.argmax(logits))

    # Generate decode tokens (timed)
    generated = [first_tok]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(MAX_NEW - 1):
        logits = runner.step(generated[-1])
        next_tok = int(np.argmax(logits))
        generated.append(next_tok)
        if next_tok == tokenizer.eos_token_id:
            break
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    num_generated = len(generated)
    text = tokenizer.decode(generated[len(prompt_tokens):], skip_special_tokens=True)
    tok_s = num_generated / elapsed if elapsed > 0 else 0
    print(f"[{label}] {elapsed*1000:.1f} ms, {tok_s:.1f} tok/s, {num_generated} tokens")
    print(f"  Output: {text[:120]}")
    return elapsed, num_generated


# ---------------------------------------------------------------------------
# 4. Run comparisons
# ---------------------------------------------------------------------------

# Allocate FlashInfer tmp buffer
tmp_elems = 16777216  # 32MB as fp16
err, tmp_ptr = cudart.cudaMalloc(tmp_elems * 2)

print(f"\n{'='*70}")

# FlashInfer
fi_time, fi_ntok = run_generation(
    FI_BUNDLE, "FlashInfer", extra_inputs={"ffi_attn_tmp": tmp_ptr})

# Baseline
base_time, base_ntok = run_generation(BASELINE_BUNDLE, "Baseline")

# Cleanup
cudart.cudaFree(tmp_ptr)

# Summary
print(f"\n{'='*70}")
print(f"Qwen3-4B-Instruct — {MAX_NEW} decode steps")
print(f"  Baseline:   {base_time*1000:.1f} ms  ({base_ntok/base_time:.1f} tok/s)")
print(f"  FlashInfer: {fi_time*1000:.1f} ms  ({fi_ntok/fi_time:.1f} tok/s)")
print(f"  Speedup:    {base_time/fi_time:.2f}x")
print(f"{'='*70}")
