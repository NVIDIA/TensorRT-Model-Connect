#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bracket PyTorch and TensorRT FP32 boundaries in Wan2.2 Q RMSNorm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch
from safetensors import safe_open


def metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | int | bool]:
    reference = reference.float()
    candidate = candidate.float()
    delta = candidate - reference
    exact = int(
        torch.count_nonzero(
            candidate.contiguous().view(torch.int32) == reference.contiguous().view(torch.int32)
        )
    )
    total = reference.numel()
    return {
        "bitwise_exact": exact == total,
        "exact_elements": exact,
        "mismatched_elements": total - exact,
        "total_elements": total,
        "max_abs_error": float(delta.abs().max()),
        "mean_abs_error": float(delta.abs().mean()),
        "rmse": float(delta.square().mean().sqrt()),
    }


def mark_output(network, tensor, name: str) -> None:
    tensor.name = name
    network.mark_output(tensor)


def build_plan(*, expose_boundaries: bool) -> bytes:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    hidden = network.add_input("hidden", trt.bfloat16, (27_280, 3_072))
    gamma = network.add_input("gamma", trt.float32, (1, 3_072))
    hidden_fp32 = network.add_cast(hidden, trt.float32).get_output(0)
    squared = network.add_elementwise(
        hidden_fp32, hidden_fp32, trt.ElementWiseOperation.PROD
    ).get_output(0)
    mean = network.add_reduce(squared, trt.ReduceOperation.AVG, 1 << 1, True).get_output(0)
    epsilon = network.add_constant((1, 1), np.array([[1.0e-6]], dtype=np.float32)).get_output(0)
    variance = network.add_elementwise(mean, epsilon, trt.ElementWiseOperation.SUM).get_output(0)
    root = network.add_unary(variance, trt.UnaryOperation.SQRT).get_output(0)
    inverse = network.add_unary(root, trt.UnaryOperation.RECIP).get_output(0)
    normalized_fp32 = network.add_elementwise(
        hidden_fp32, inverse, trt.ElementWiseOperation.PROD
    ).get_output(0)
    normalized_bf16 = network.add_cast(normalized_fp32, trt.bfloat16).get_output(0)
    normalized_bf16_fp32 = network.add_cast(normalized_bf16, trt.float32).get_output(0)
    output = network.add_elementwise(
        normalized_bf16_fp32, gamma, trt.ElementWiseOperation.PROD
    ).get_output(0)
    if expose_boundaries:
        for name, tensor in {
            "squared": squared,
            "mean": mean,
            "variance": variance,
            "inverse": inverse,
            "normalized_fp32": normalized_fp32,
            "normalized_bf16_fp32": normalized_bf16_fp32,
        }.items():
            mark_output(network, tensor, name)
    mark_output(network, output, "output")
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("Could not build TensorRT RMSNorm boundary plan")
    return bytes(plan)


def execute_plan(plan: bytes, hidden: torch.Tensor, gamma: torch.Tensor) -> dict[str, torch.Tensor]:
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(plan)
    if engine is None:
        raise RuntimeError("Could not deserialize TensorRT RMSNorm boundary plan")
    context = engine.create_execution_context()
    outputs = {}
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if name in {"hidden", "gamma"}:
            continue
        outputs[name] = torch.empty(
            tuple(engine.get_tensor_shape(name)), device=hidden.device, dtype=torch.float32
        )
    for name, value in {"hidden": hidden, "gamma": gamma, **outputs}.items():
        if not context.set_tensor_address(name, value.data_ptr()):
            raise RuntimeError(f"Could not bind RMSNorm tensor {name}")
    stream = torch.cuda.current_stream(hidden.device)
    if not context.execute_async_v3(stream_handle=stream.cuda_stream):
        raise RuntimeError("TensorRT RMSNorm boundary execution failed")
    torch.cuda.synchronize(hidden.device)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-linear", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--trt-output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    hidden_fp32 = torch.load(args.q_linear, map_location="cpu", weights_only=True).reshape(
        27_280, 3_072
    )
    hidden = hidden_fp32.to(device=device, dtype=torch.bfloat16)
    source_output = (
        torch.load(args.source_output, map_location="cpu", weights_only=True)
        .reshape(27_280, 3_072)
        .to(device)
    )
    trt_output = (
        torch.load(args.trt_output, map_location="cpu", weights_only=True)
        .reshape(27_280, 3_072)
        .to(device)
    )
    shard = args.checkpoint / "diffusion_pytorch_model-00001-of-00003.safetensors"
    with safe_open(shard, framework="pt", device="cpu") as checkpoint:
        gamma = checkpoint.get_tensor("blocks.0.self_attn.norm_q.weight")
    gamma = gamma.reshape(1, 3_072).to(device=device, dtype=torch.float32)

    source_hidden = hidden.float()
    source_squared = source_hidden.pow(2)
    source_mean = source_squared.mean(dim=-1, keepdim=True)
    source_variance = source_mean + 1.0e-6
    source_inverse = torch.rsqrt(source_variance)
    source_normalized_fp32 = source_hidden * source_inverse
    source_normalized_bf16_fp32 = source_normalized_fp32.to(torch.bfloat16).float()
    recomputed_source_output = source_normalized_bf16_fp32 * gamma
    source_stages = {
        "squared": source_squared,
        "mean": source_mean,
        "variance": source_variance,
        "inverse": source_inverse,
        "normalized_fp32": source_normalized_fp32,
        "normalized_bf16_fp32": source_normalized_bf16_fp32,
        "output": recomputed_source_output,
    }
    torch.cuda.synchronize(device)

    args.plan_dir.mkdir(parents=True, exist_ok=True)
    plans = {
        "production_graph": build_plan(expose_boundaries=False),
        "exposed_boundaries": build_plan(expose_boundaries=True),
    }
    executions = {}
    for name, plan in plans.items():
        (args.plan_dir / f"{name}.plan").write_bytes(plan)
        executions[name] = execute_plan(plan, hidden, gamma)

    report = {
        "kind": "wan2_2_ti2v_rms_norm_boundary_diagnosis",
        "device": torch.cuda.get_device_name(device),
        "shape": [27_280, 3_072],
        "source_recompute_vs_captured": metrics(source_output, recomputed_source_output),
        "production_micro_vs_captured_trt": metrics(
            trt_output, executions["production_graph"]["output"]
        ),
        "variants": {},
    }
    for variant, outputs in executions.items():
        report["variants"][variant] = {
            stage: metrics(source_stages[stage], value) for stage, value in outputs.items()
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
