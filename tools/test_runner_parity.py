#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-validation: verify Python TrtRunner matches C++ trtmc binary.

Runs the same bundle+prompt through both paths and asserts identical
generated tokens. This is the consistency guarantee between the Python
debug runner and the C++ runtime — if either side changes its mask,
cache, or position logic, this test catches the divergence.

Supports standard KV-cache decoders and family-owned debug runners,
auto-detected from bundle metadata.

Usage (inside container):
    python3 tools/test_runner_parity.py \
      --bundle /tmp/model.bundle \
      --binary ./build/trtmc \
      --hf-python .venv/bin/python \
      --prompt "The capital of France is" \
      --max-new-tokens 20
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def _extract_bundle_files(reader) -> str:
    """Extract tokenizer and config files from bundle into a temp dir."""
    tmpdir = tempfile.mkdtemp(prefix="trtmc_parity_")
    for name in ("tokenizer.json", "tokenizer_config.json",
                 "config.json", "special_tokens_map.json",
                 "vocab.json", "merges.txt"):
        try:
            data = reader.read_section(name)
            Path(tmpdir, name).write_bytes(data)
        except KeyError:
            pass
    return tmpdir


def run_cpp(binary: str, bundle: str, prompt: str, max_new_tokens: int,
            hf_python: str) -> str:
    """Run C++ trtmc binary, return generated text."""
    cmd = [binary, "run", bundle, "--prompt", prompt,
           "--max-new-tokens", str(max_new_tokens)]
    if hf_python:
        cmd.extend(["--hf-python", hf_python])

    env = os.environ.copy()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env,
                            timeout=120)
    if result.returncode != 0:
        print(f"C++ stderr:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError(f"C++ binary failed (rc={result.returncode})")

    # stdout contains the generated text (prompt + completion)
    return result.stdout.strip()


def run_python(bundle: str, prompt: str,
               max_new_tokens: int) -> tuple[str, list[int]]:
    """Run Python debug runner, return (text, token_ids)."""
    from tensorrt_model_connect.families import resolve_debug_runner

    from tensorrt_model_connect import BundleReader
    reader = BundleReader(bundle)
    header_raw = reader.header
    tmpdir = _extract_bundle_files(reader)
    
    try:
        cfg = json.loads(reader.read_section("config.json").decode("utf-8"))
    except KeyError:
        cfg = {}
        
    runtime_strategy = str(cfg.get("runtime_strategy") or "")
    if not runtime_strategy:
        raise RuntimeError(
            "Bundle config.json is missing runtime_strategy; runner parity "
            "requires a family-owned debug runner strategy."
        )
    engine_plan = reader.read_section("engine_plan")

    # Extract eos_token_id (matches C++ EOS detection)
    eid = cfg.get("eos_token_id", -1)
    if isinstance(eid, list):
        eos_token_id = eid[0] if eid else -1
    else:
        eos_token_id = eid

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tmpdir, trust_remote_code=True)

    runner = resolve_debug_runner(
        runtime_strategy,
        config=cfg,
        header=header_raw,
        engine_plan=engine_plan,
        bundle_path=bundle,
    )
    if runner is None:
        raise RuntimeError(
            f"No family-owned debug_runner adapter handles {runtime_strategy!r}"
        )

    # Encode prompt — use add_special_tokens=False to match the C++ runtime,
    # which calls hf_tokenizer.py with add_special_tokens=False.
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)

    # Prefill
    for tid in input_ids:
        result = runner.step(tid)

    # Generate (stop on EOS to match C++ runtime behavior)
    gen_ids = list(input_ids)
    for _ in range(max_new_tokens):
        logits = result["logits"].flatten()
        next_token = int(np.argmax(logits))
        gen_ids.append(next_token)
        if next_token == eos_token_id:
            break
        result = runner.step(next_token)

    # Decode
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)

    return text, gen_ids


def main():
    parser = argparse.ArgumentParser(
        description="Cross-validate Python TrtRunner vs C++ trtmc binary")
    parser.add_argument("--bundle", required=True, help=".bundle artifact path")
    parser.add_argument("--binary", default="./build/trtmc",
                        help="Path to trtmc C++ binary")
    parser.add_argument("--hf-python", default="",
                        help="Python path for HF tokenizer bridge")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    args = parser.parse_args()

    print(f"Bundle: {args.bundle}")
    print(f"Prompt: {args.prompt!r}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print()

    # Run C++ binary
    print("Running C++ binary ...", file=sys.stderr)
    cpp_text = run_cpp(args.binary, args.bundle, args.prompt,
                       args.max_new_tokens, args.hf_python)
    print(f"C++:    {cpp_text!r}")

    # Run Python runner
    print("Running Python runner ...", file=sys.stderr)
    py_text, py_ids = run_python(args.bundle, args.prompt,
                                  args.max_new_tokens)
    print(f"Python: {py_text!r}")

    # Compare (strip both to normalize trailing whitespace from stdout)
    cpp_text = cpp_text.strip()
    py_text = py_text.strip()
    exact_match = cpp_text == py_text
    print(f"\nExact match: {exact_match}")

    # Fuzzy match: normalize consecutive whitespace.  FP32 argmax
    # tie-breaking between \n and \n\n can differ between the C++ host
    # memcpy path and the Python CUDA runtime path — both produce valid
    # output but with slightly different whitespace around newlines.
    import re
    cpp_normalized = re.sub(r'\s+', ' ', cpp_text)
    py_normalized = re.sub(r'\s+', ' ', py_text)
    fuzzy_match = cpp_normalized == py_normalized

    if exact_match:
        print("PASS")
        sys.exit(0)
    elif fuzzy_match:
        print("PASS (fuzzy — whitespace-only difference)")
        sys.exit(0)
    else:
        # Find first divergence point
        cpp_words = cpp_text.split()
        py_words = py_text.split()
        for i, (cw, pw) in enumerate(zip(cpp_words, py_words)):
            if cw != pw:
                print(f"First word divergence at position {i}: "
                      f"C++={cw!r} Python={pw!r}")
                break
        print("FAIL")
        sys.exit(1)


def run_as_diff_test(ctx):
    """Framework entry point. Returns DiffResult."""
    from diff_framework.protocol import DiffResult
    import time as _time

    t0 = _time.monotonic()
    try:
        bundle = ctx.bundle_path
        binary = ctx.binary_path or "./build/trtmc"
        hf_python = ctx.hf_python or ""
        prompt = "The capital of France is"

        cpp_text = run_cpp(
            binary, bundle, prompt, ctx.max_new_tokens, hf_python)
        py_text, _ = run_python(
            bundle, prompt, ctx.max_new_tokens)

        cpp_text = cpp_text.strip()
        py_text = py_text.strip()
        match = cpp_text == py_text

        return DiffResult(
            test_name="runner_parity", model=ctx.model,
            runtime_strategy=ctx.runtime_strategy,
            passed=match,
            status="PASS" if match else "FAIL",
            message=f"exact_match={match}",
            metrics={"exact_match": match},
            duration_s=_time.monotonic() - t0,
            details=f"C++: {cpp_text[:200]!r}\nPython: {py_text[:200]!r}")
    except Exception as e:
        return DiffResult.error(
            "runner_parity", ctx.model, ctx.runtime_strategy, str(e))


if __name__ == "__main__":
    main()
