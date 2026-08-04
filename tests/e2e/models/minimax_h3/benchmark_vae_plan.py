# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the fixed-shape native MiniMax-H3 VAE tile plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import numpy as np
import tensorrt as trt
import torch


INPUT_SHAPE = (7, 24, 7, 16, 16)
OUTPUT_SHAPE = (7, 3, 28, 256, 256)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("warmup must be non-negative and iterations must be positive")

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(Path(args.plan).read_bytes())
    if engine is None:
        raise RuntimeError("TensorRT failed to deserialize the MiniMax-H3 VAE plan")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("TensorRT failed to create the MiniMax-H3 VAE context")

    latent = torch.randn(INPUT_SHAPE, device="cuda", dtype=torch.float32)
    decoded = torch.empty(OUTPUT_SHAPE, device="cuda", dtype=torch.float32)
    context.set_tensor_address("latent_tiles", latent.data_ptr())
    context.set_tensor_address("decoded_tiles", decoded.data_ptr())
    stream = torch.cuda.Stream()

    def execute() -> None:
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT failed to execute the MiniMax-H3 VAE plan")

    for _ in range(args.warmup):
        execute()
    stream.synchronize()
    timings = []
    for _ in range(args.iterations):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record(stream)
        execute()
        end.record(stream)
        end.synchronize()
        timings.append(begin.elapsed_time(end))

    receipt = {
        "component": "minimax_h3_vae_tile_decoder",
        "checkpoint_revision": "48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc",
        "input_shape": list(INPUT_SHAPE),
        "output_shape": list(OUTPUT_SHAPE),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "latency_ms": timings,
        "median_latency_ms": float(np.median(timings)),
        "mean_latency_ms": float(np.mean(timings)),
        "p95_latency_ms": float(np.percentile(timings, 95)),
        "tensorrt": trt.__version__,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "host": platform.node(),
    }
    Path(args.output).write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
