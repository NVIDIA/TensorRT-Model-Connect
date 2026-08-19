# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the fixed-shape native MiniMax-H3 VAE tile plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch
from tensorrt_model_connect.models.minimax_h3.config import SOL_ENGINE_1344X768_124F
from tensorrt_model_connect.models.minimax_h3.provenance import (
    CHECKPOINT_REVISION,
    atomic_write_json,
    file_identity,
    stable_file_record,
    validate_component_build_receipt,
    validate_file_identity,
    validate_source_revision,
)

INPUT_SHAPE = (28, 24, 7, 16, 16)
OUTPUT_SHAPE = (28, 3, 28, 256, 256)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--build-receipt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    source_revision = validate_source_revision(args.source_revision)
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("warmup must be non-negative and iterations must be positive")

    plan_path = Path(args.plan).resolve(strict=True)
    receipt_path = Path(args.build_receipt).resolve(strict=True)
    receipt_identity = file_identity(receipt_path)
    build_receipt = json.loads(receipt_path.read_text())
    build_receipt_record, receipt_hashed_identity = stable_file_record(
        receipt_path, "native build receipt"
    )
    if receipt_hashed_identity != receipt_identity:
        raise ValueError("MiniMax-H3 build receipt changed while it was being read")

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    plan_identity = file_identity(plan_path)
    plan = plan_path.read_bytes()
    plan_bytes = len(plan)
    plan_sha256 = hashlib.sha256(plan).hexdigest()
    validate_file_identity(plan_path, plan_identity, "VAE plan")
    build_source_sha, plan_record, snapshot_record = validate_component_build_receipt(
        build_receipt,
        component="vae_tile_decoder.plan",
        artifact=plan_path,
        build_helper=Path(__file__).with_name("build_native_components.py"),
        source_revision=source_revision,
        profile=SOL_ENGINE_1344X768_124F,
        hash_file=False,
    )
    if plan_record["bytes"] != plan_bytes or plan_record["sha256"] != plan_sha256:
        raise ValueError("MiniMax-H3 VAE plan does not match its native build receipt")
    engine = runtime.deserialize_cuda_engine(plan)
    del plan
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
        "checkpoint_revision": CHECKPOINT_REVISION,
        "source_revision": source_revision,
        "builder_source_sha256": build_source_sha,
        "checkpoint_inventory_sha256": snapshot_record["inventory_sha256"],
        "workspace_limit_bytes": build_receipt["workspace_limit_bytes"],
        "build_receipt": build_receipt_record,
        "plan_bytes": plan_bytes,
        "plan_sha256": plan_sha256,
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
    validate_file_identity(plan_path, plan_identity, "VAE plan")
    validate_file_identity(receipt_path, receipt_hashed_identity, "native build receipt")
    atomic_write_json(Path(args.output), receipt)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
