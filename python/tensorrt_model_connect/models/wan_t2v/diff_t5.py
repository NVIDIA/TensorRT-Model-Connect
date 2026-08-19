#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T5 encoder TRT vs HuggingFace comparison.

Validates that the TRT T5 encoder engine produces the same text embeddings
as the HuggingFace UMT5EncoderModel / T5EncoderModel.

Usage:
    python tools/diff_t5.py --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers --atol 1e-3

Requires: torch, transformers, tensorrt, cuda-python
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PYTHON_DIR = _REPO_ROOT / "python"


def handles_diff_t5_args(argv: list[str]) -> bool:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", default="")
    ns, _ = parser.parse_known_args(argv)
    return "wan" in ns.model.lower()


def main():
    parser = argparse.ArgumentParser(description="T5 encoder TRT vs HF comparison")
    parser.add_argument("--model", required=True, help="HF model ID or local path (diffusers format)")
    parser.add_argument("--atol", type=float, default=1e-3, help="Absolute tolerance")
    parser.add_argument("--max-seq-len", type=int, default=64, help="Max sequence length for test")
    parser.add_argument("--prompt", default="A cat on a beach", help="Test prompt")
    args = parser.parse_args()

    print(f"[diff-t5] Model: {args.model}", file=sys.stderr)
    print(f"[diff-t5] Prompt: {args.prompt!r}", file=sys.stderr)

    # 1. Load HF model
    print("[diff-t5] Loading HF T5 encoder ...", file=sys.stderr)
    import torch
    from transformers import AutoTokenizer, T5EncoderModel

    model_path = Path(args.model)
    if model_path.is_dir() and (model_path / "text_encoder").is_dir():
        te_path = str(model_path / "text_encoder")
    else:
        te_path = args.model

    tokenizer = AutoTokenizer.from_pretrained(te_path)
    hf_model = T5EncoderModel.from_pretrained(te_path, torch_dtype=torch.float32)
    hf_model.eval()

    # 2. Tokenize
    encoding = tokenizer(
        args.prompt,
        max_length=args.max_seq_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"]  # [1, seq_len]

    # 3. HF forward pass
    with torch.no_grad():
        hf_output = hf_model(input_ids=input_ids).last_hidden_state
    hf_embeddings = hf_output.numpy()  # [1, seq_len, d_model]

    print(f"[diff-t5] HF output shape: {hf_embeddings.shape}", file=sys.stderr)
    print(f"[diff-t5] HF output range: [{hf_embeddings.min():.4f}, {hf_embeddings.max():.4f}]",
          file=sys.stderr)

    # 4. Build TRT engine
    print("[diff-t5] Building TRT T5 engine ...", file=sys.stderr)
    if str(_PYTHON_DIR) not in sys.path:
        sys.path.insert(0, str(_PYTHON_DIR))
    from tensorrt_model_connect.models.wan_t2v.t5_encoder_builder import build_t5_encoder_engine, load_t5_weights

    config = hf_model.config
    t5_weights = load_t5_weights(
        te_path,
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_kv=config.d_kv,
        d_ff=config.d_ff,
        num_layers=config.num_layers,
        vocab_size=config.vocab_size,
    )
    engine_plan = build_t5_encoder_engine(
        t5_weights,
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_kv=config.d_kv,
        d_ff=config.d_ff,
        num_layers=config.num_layers,
        vocab_size=config.vocab_size,
        max_seq_len=args.max_seq_len,
    )

    # 5. Run TRT engine
    print("[diff-t5] Running TRT engine ...", file=sys.stderr)
    from tensorrt_model_connect import trt_compat
    from cuda import cudart

    trt = trt_compat.get_trt()
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_plan)
    context = engine.create_execution_context()

    # Allocate buffers and run
    input_np = input_ids.numpy().astype(np.int32)
    d_model = config.d_model

    # Simple engine execution
    stream = cudart.cudaStreamCreate()[1]

    # Bind input
    d_input = cudart.cudaMalloc(input_np.nbytes)[1]
    cudart.cudaMemcpyAsync(d_input, input_np.ctypes.data, input_np.nbytes,
                           cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    context.set_tensor_address("input_ids", d_input)

    # Bind output
    output_shape = (1, args.max_seq_len, d_model)
    output_np = np.empty(output_shape, dtype=np.float32)
    d_output = cudart.cudaMalloc(output_np.nbytes)[1]
    context.set_tensor_address("text_embeddings", d_output)

    # Execute
    context.execute_async_v3(stream)
    cudart.cudaStreamSynchronize(stream)

    # Copy output
    cudart.cudaMemcpy(output_np.ctypes.data, d_output, output_np.nbytes,
                      cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    trt_embeddings = output_np  # [1, seq_len, d_model]

    # Cleanup
    cudart.cudaFree(d_input)
    cudart.cudaFree(d_output)
    cudart.cudaStreamDestroy(stream)

    # 6. Compare
    print(f"[diff-t5] TRT output shape: {trt_embeddings.shape}", file=sys.stderr)
    print(f"[diff-t5] TRT output range: [{trt_embeddings.min():.4f}, {trt_embeddings.max():.4f}]",
          file=sys.stderr)

    max_diff = np.max(np.abs(hf_embeddings - trt_embeddings))
    mean_diff = np.mean(np.abs(hf_embeddings - trt_embeddings))

    print(f"[diff-t5] Max abs diff: {max_diff:.6f}", file=sys.stderr)
    print(f"[diff-t5] Mean abs diff: {mean_diff:.6f}", file=sys.stderr)

    if max_diff <= args.atol:
        print(f"PASS: T5 encoder match (max_diff={max_diff:.6f} <= atol={args.atol})")
    else:
        print(f"FAIL: T5 encoder mismatch (max_diff={max_diff:.6f} > atol={args.atol})")
        sys.exit(1)


if __name__ == "__main__":
    main()
