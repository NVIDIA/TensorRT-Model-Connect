#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualify the source-exact TRT cross-attention affine LayerNorm graph."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path

import numpy as np
from tensorrt_model_connect.trt_compat import trt
import torch
from safetensors import safe_open


ROWS = 27_280
COLUMNS = 3_072
EPSILON = 1.0e-6


def metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | int | bool]:
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


def load_affine(checkpoint: Path) -> tuple[np.ndarray, np.ndarray]:
    index_path = checkpoint / "diffusion_pytorch_model.safetensors.index.json"
    shard = checkpoint / "diffusion_pytorch_model-00001-of-00003.safetensors"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        shard = checkpoint / index["weight_map"]["blocks.0.norm3.weight"]
    with safe_open(shard, framework="pt", device="cpu") as tensors:
        weight = tensors.get_tensor("blocks.0.norm3.weight").float().numpy()
        bias = tensors.get_tensor("blocks.0.norm3.bias").float().numpy()
    return weight.reshape(1, COLUMNS), bias.reshape(1, COLUMNS)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    input_fp32 = (
        torch.load(args.input, map_location="cpu", weights_only=True)
        .reshape(ROWS, COLUMNS)
        .to(device)
    )
    source_output = (
        torch.load(args.source_output, map_location="cpu", weights_only=True)
        .reshape(ROWS, COLUMNS)
        .to(device)
    )
    weight_np, bias_np = load_affine(args.checkpoint)
    source_normalized = torch.nn.functional.layer_norm(input_fp32, (COLUMNS,), None, None, EPSILON)

    ctypes.CDLL(str(args.plugin.resolve()), mode=ctypes.RTLD_GLOBAL)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    x = network.add_input("input", trt.float32, (ROWS, COLUMNS))
    creator = trt.get_plugin_registry().get_creator("Wan22DitLayerNormFp32", "1", "")
    if creator is None:
        raise RuntimeError("Wan22DitLayerNormFp32 plugin creator is not registered")
    plugin = creator.create_plugin("qualified_cross_norm", trt.PluginFieldCollection([]))
    layer = network.add_plugin_v2([x], plugin)
    if layer is None:
        raise RuntimeError("Could not add Wan22DitLayerNormFp32 plugin")
    normalized = layer.get_output(0)
    normalized.name = "normalized"
    network.mark_output(normalized)

    gamma = network.add_constant(weight_np.shape, np.ascontiguousarray(weight_np)).get_output(0)
    beta = network.add_constant(bias_np.shape, np.ascontiguousarray(bias_np)).get_output(0)
    scaled = network.add_elementwise(normalized, gamma, trt.ElementWiseOperation.PROD).get_output(0)
    affine = network.add_elementwise(scaled, beta, trt.ElementWiseOperation.SUM).get_output(0)
    affine.name = "affine"
    network.mark_output(affine)

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("Could not build affine LayerNorm qualification engine")
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    if engine is None:
        raise RuntimeError("Could not deserialize affine LayerNorm qualification engine")
    context = engine.create_execution_context()
    normalized_output = torch.empty_like(input_fp32)
    affine_output = torch.empty_like(input_fp32)
    context.set_tensor_address("input", input_fp32.data_ptr())
    context.set_tensor_address("normalized", normalized_output.data_ptr())
    context.set_tensor_address("affine", affine_output.data_ptr())
    stream = torch.cuda.current_stream(device)
    if not context.execute_async_v3(stream_handle=stream.cuda_stream):
        raise RuntimeError("Affine LayerNorm qualification execution failed")
    torch.cuda.synchronize(device)

    normalized_metrics = metrics(source_normalized, normalized_output)
    output_metrics = metrics(source_output, affine_output)
    primary_pass = bool(normalized_metrics["bitwise_exact"] and output_metrics["bitwise_exact"])
    report = {
        "kind": "wan2_2_ti2v_source_exact_cross_affine_layer_norm_qualification",
        "device": torch.cuda.get_device_name(device),
        "shape": [ROWS, COLUMNS],
        "epsilon": EPSILON,
        "graph": "Wan22DitLayerNormFp32 then explicit TensorRT PROD gamma and SUM beta",
        "predefined_gate": {
            "primary": "bitwise exact normalized and affine output over all 83,804,160 elements",
        },
        "normalized_metrics": normalized_metrics,
        "output_metrics": output_metrics,
        "gate": {"primary_pass": primary_pass, "pass": primary_pass},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if primary_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
