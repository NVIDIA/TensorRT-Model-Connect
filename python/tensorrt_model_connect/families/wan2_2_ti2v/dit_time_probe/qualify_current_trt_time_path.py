#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Locate the first mismatch in the current TensorRT Wan2.2 time path."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

FAMILY_DIR = Path(__file__).resolve().parents[1]
PYTHON_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PYTHON_ROOT))
from tensorrt_model_connect.families.wan2_2_ti2v import trt_ops as op  # noqa: E402


ORDER = (
    "expanded_time_features",
    "time_linear1",
    "time_silu",
    "time_embed",
    "projection_silu",
    "time_projection_flat",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_array(path: Path) -> np.ndarray:
    return np.ascontiguousarray(np.load(path, allow_pickle=False))


def mark_output(network, tensor, name: str) -> None:
    tensor.name = name
    network.mark_output(tensor)


def build_plan(capture_dir: Path) -> bytes:
    manifest = json.loads((capture_dir / "manifest.json").read_text())
    seq_len = int(manifest["seq_len"])
    freq_dim = int(manifest["freq_dim"])
    linear1_weight = load_array(capture_dir / "time_linear1_weight.npy")
    linear1_bias = load_array(capture_dir / "time_linear1_bias.npy")
    linear2_weight = load_array(capture_dir / "time_linear2_weight.npy")
    linear2_bias = load_array(capture_dir / "time_linear2_bias.npy")
    projection_weight = load_array(capture_dir / "projection_linear_weight.npy")
    projection_bias = load_array(capture_dir / "projection_linear_bias.npy")

    op.set_bf16_gemm_emulation(False)
    op.set_source_attention_plugin(False)
    op.set_cuda_bf16_barriers(False)
    op.set_dit_cuda_numerics(False)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 96 << 30)
    singleton_features = network.add_input("time_features", trt.float32, (1, freq_dim))
    expanded = network.add_elementwise(
        singleton_features,
        op.constant(network, np.zeros((seq_len, freq_dim), dtype=np.float32)),
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    mark_output(network, expanded, "expanded_time_features")
    time_linear1 = op.linear(network, expanded, linear1_weight, linear1_bias, bf16=False)
    mark_output(network, time_linear1, "time_linear1")
    time_silu = op.silu(network, time_linear1)
    mark_output(network, time_silu, "time_silu")
    time_embed = op.linear(network, time_silu, linear2_weight, linear2_bias, bf16=False)
    mark_output(network, time_embed, "time_embed")
    projection_silu = op.silu(network, time_embed)
    mark_output(network, projection_silu, "projection_silu")
    projection = op.linear(network, projection_silu, projection_weight, projection_bias, bf16=False)
    mark_output(network, projection, "time_projection_flat")
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT could not build the current time-path probe")
    return bytes(plan)


def metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    actual = actual.reshape(reference.shape)
    exact = actual.contiguous().view(torch.int32) == reference.contiguous().view(torch.int32)
    exact_elements = int(torch.count_nonzero(exact).item())
    delta = actual.double() - reference.double()
    return {
        "bitwise_exact": exact_elements == reference.numel(),
        "exact_elements": exact_elements,
        "total_elements": reference.numel(),
        "max_abs_error": float(delta.abs().max().item()),
        "mean_abs_error": float(delta.abs().mean().item()),
        "rmse": float(delta.square().mean().sqrt().item()),
        "cosine_similarity": float(
            torch.nn.functional.cosine_similarity(
                actual.flatten().double(), reference.flatten().double(), dim=0
            ).item()
        ),
    }


def main() -> int:
    args = parse_args()
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if not args.plan.is_file():
        args.plan.write_bytes(build_plan(args.capture_dir))

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    logger = trt.Logger(trt.Logger.WARNING)
    engine = trt.Runtime(logger).deserialize_cuda_engine(args.plan.read_bytes())
    if engine is None:
        raise RuntimeError("Could not deserialize the current time-path probe")
    context = engine.create_execution_context()
    full_features = load_array(args.capture_dir / "time_features.npy")
    if not np.array_equal(
        full_features, np.broadcast_to(full_features[:, :1], full_features.shape)
    ):
        raise RuntimeError("Call-0 time features are not identical across the expanded sequence")
    singleton = torch.from_numpy(full_features[:, 0]).to(device)
    outputs = {
        name: torch.empty(tuple(engine.get_tensor_shape(name)), device=device, dtype=torch.float32)
        for name in ORDER
    }
    context.set_tensor_address("time_features", singleton.data_ptr())
    for name, tensor in outputs.items():
        context.set_tensor_address(name, tensor.data_ptr())
    stream = torch.cuda.current_stream(device)

    def execute() -> None:
        if not context.execute_async_v3(stream_handle=stream.cuda_stream):
            raise RuntimeError("Current TensorRT time-path execution failed")

    for _ in range(args.warmup):
        execute()
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(args.iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        execute()
        end.record(stream)
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))

    reference_files = {
        "expanded_time_features": "time_features.npy",
        "time_linear1": "time_linear1.npy",
        "time_silu": "time_silu.npy",
        "time_embed": "time_embed.npy",
        "projection_silu": "projection_silu.npy",
        "time_projection_flat": "time_projection_flat.npy",
    }
    comparisons = {}
    earliest = None
    for name in ORDER:
        reference = torch.from_numpy(load_array(args.capture_dir / reference_files[name])).to(
            device
        )
        comparisons[name] = metrics(outputs[name], reference)
        if earliest is None and not comparisons[name]["bitwise_exact"]:
            earliest = name
        del reference
        torch.cuda.empty_cache()

    first_output_path = args.report.parent / "current_trt_time_linear1.npy"
    np.save(first_output_path, outputs["time_linear1"].detach().cpu().numpy(), allow_pickle=False)
    inspector = engine.create_engine_inspector()
    inspector_path = args.report.parent / "current_trt_time_path_inspector.json"
    inspector_path.write_text(
        inspector.get_engine_information(trt.LayerInformationFormat.JSON) + "\n",
        encoding="utf-8",
    )
    report = {
        "kind": "wan2_2_ti2v_current_trt_call0_time_path",
        "device": torch.cuda.get_device_name(device),
        "capture_manifest_sha256": sha256_file(args.capture_dir / "manifest.json"),
        "plan": {
            "path": str(args.plan.resolve()),
            "bytes": args.plan.stat().st_size,
            "sha256": sha256_file(args.plan),
            "device_memory_bytes": int(engine.device_memory_size_v2),
        },
        "latency": {
            "samples_ms": samples,
            "min_ms": min(samples),
            "median_ms": statistics.median(samples),
            "mean_ms": statistics.mean(samples),
        },
        "comparisons": comparisons,
        "earliest_non_bitwise": earliest,
        "time_linear1_output": {
            "path": str(first_output_path.resolve()),
            "sha256": sha256_file(first_output_path),
        },
        "inspector": {
            "path": str(inspector_path.resolve()),
            "sha256": sha256_file(inspector_path),
        },
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
