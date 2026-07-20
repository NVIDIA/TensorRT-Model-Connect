#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bracket Wan2.2 cross-attention affine LayerNorm on a saved source call."""

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


def load_rows(path: Path, device: torch.device) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=True).reshape(ROWS, COLUMNS).to(device)


def load_affine(checkpoint: Path) -> tuple[torch.Tensor, torch.Tensor]:
    index_path = checkpoint / "diffusion_pytorch_model.safetensors.index.json"
    shard = checkpoint / "diffusion_pytorch_model-00001-of-00003.safetensors"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        shard = checkpoint / index["weight_map"]["blocks.0.norm3.weight"]
    with safe_open(shard, framework="pt", device="cpu") as tensors:
        weight = tensors.get_tensor("blocks.0.norm3.weight").float()
        bias = tensors.get_tensor("blocks.0.norm3.bias").float()
    return weight.reshape(COLUMNS), bias.reshape(COLUMNS)


def run_trt_bracket(
    input_fp32: torch.Tensor,
    source_normalized: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> dict[str, torch.Tensor]:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    x = network.add_input("input", trt.float32, (ROWS, COLUMNS))
    normalized = network.add_input("normalized", trt.float32, (ROWS, COLUMNS))

    def constant(value: np.ndarray):
        array = np.ascontiguousarray(value, dtype=np.float32)
        return network.add_constant(array.shape, array).get_output(0)

    gamma = constant(weight.detach().cpu().numpy())
    beta = constant(bias.detach().cpu().numpy())
    ones = constant(np.ones((1, COLUMNS), dtype=np.float32))
    zeros = constant(np.zeros((1, COLUMNS), dtype=np.float32))

    affine = network.add_normalization_v2(x, gamma, beta, 1 << 1)
    affine.epsilon = EPSILON
    affine_output = affine.get_output(0)
    affine_output.name = "trt_affine_normalization"
    network.mark_output(affine_output)

    unit = network.add_normalization_v2(x, ones, zeros, 1 << 1)
    unit.epsilon = EPSILON
    unit_output = unit.get_output(0)
    unit_output.name = "trt_unit_normalization"
    network.mark_output(unit_output)

    product = network.add_elementwise(normalized, gamma, trt.ElementWiseOperation.PROD).get_output(
        0
    )
    explicit_output = network.add_elementwise(
        product, beta, trt.ElementWiseOperation.SUM
    ).get_output(0)
    explicit_output.name = "trt_explicit_affine"
    network.mark_output(explicit_output)

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("Could not build affine LayerNorm diagnostic engine")
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    if engine is None:
        raise RuntimeError("Could not deserialize affine LayerNorm diagnostic engine")
    context = engine.create_execution_context()
    outputs = {
        name: torch.empty_like(input_fp32)
        for name in (
            "trt_affine_normalization",
            "trt_unit_normalization",
            "trt_explicit_affine",
        )
    }
    context.set_tensor_address("input", input_fp32.data_ptr())
    context.set_tensor_address("normalized", source_normalized.data_ptr())
    for name, output in outputs.items():
        context.set_tensor_address(name, output.data_ptr())
    stream = torch.cuda.current_stream(input_fp32.device)
    if not context.execute_async_v3(stream_handle=stream.cuda_stream):
        raise RuntimeError("Affine LayerNorm diagnostic engine execution failed")
    torch.cuda.synchronize(input_fp32.device)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-input", type=Path, required=True)
    parser.add_argument("--trt-input", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--trt-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    source_input = load_rows(args.source_input, device)
    trt_input = load_rows(args.trt_input, device)
    source_output = load_rows(args.source_output, device)
    production_trt_output = load_rows(args.trt_output, device)
    weight_1d, bias_1d = load_affine(args.checkpoint)
    weight_1d = weight_1d.to(device)
    bias_1d = bias_1d.to(device)

    source_direct, source_mean, source_rstd = torch.native_layer_norm(
        source_input, (COLUMNS,), weight_1d, bias_1d, EPSILON
    )
    weight = weight_1d.reshape(1, COLUMNS)
    bias = bias_1d.reshape(1, COLUMNS)
    source_normalized = torch.nn.functional.layer_norm(
        source_input, (COLUMNS,), None, None, EPSILON
    )
    plugin_normalized = torch.empty_like(source_input)
    library = ctypes.CDLL(str(args.plugin.resolve()), mode=ctypes.RTLD_GLOBAL)
    launch = library.trtmc_wan22_dit_layer_norm_fp32_launch
    launch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_float,
        ctypes.c_void_p,
    ]
    launch.restype = ctypes.c_int
    stream = torch.cuda.current_stream(device)
    status = launch(
        ctypes.c_void_p(source_input.data_ptr()),
        ctypes.c_void_p(plugin_normalized.data_ptr()),
        ROWS,
        COLUMNS,
        ctypes.c_float(EPSILON),
        ctypes.c_void_p(stream.cuda_stream),
    )
    if status != 0:
        raise RuntimeError(f"Wan22 LayerNorm launch failed with status {status}")
    torch.cuda.synchronize(device)

    torch_product = source_normalized * weight
    torch_separate_affine = torch_product + bias
    torch_fused_affine = torch.addcmul(bias, source_normalized, weight)
    centered = source_input - source_mean
    combined_scale = source_rstd * weight
    torch_centered_separate = centered * combined_scale + bias
    torch_centered_fused = torch.addcmul(bias, centered, combined_scale)
    trt_outputs = run_trt_bracket(source_input, source_normalized, weight, bias)

    report = {
        "kind": "wan2_2_ti2v_affine_layer_norm_boundary_diagnosis",
        "device": torch.cuda.get_device_name(device),
        "shape": [ROWS, COLUMNS],
        "epsilon": EPSILON,
        "input_contract": {
            "source": {
                "dtype": str(source_input.dtype),
                "shape": list(source_input.shape),
                "stride": list(source_input.stride()),
                "contiguous": source_input.is_contiguous(),
            },
            "tensorrt": {
                "dtype": str(trt_input.dtype),
                "shape": list(trt_input.shape),
                "stride": list(trt_input.stride()),
                "contiguous": trt_input.is_contiguous(),
            },
            "comparison": metrics(source_input, trt_input),
        },
        "affine_parameters": {
            "dtype": str(weight.dtype),
            "weight_min": float(weight.min()),
            "weight_max": float(weight.max()),
            "bias_min": float(bias.min()),
            "bias_max": float(bias.max()),
        },
        "source_reproduction": {
            "native_layer_norm_vs_captured_source": metrics(source_output, source_direct),
        },
        "production_trt_reproduction": {
            "standalone_trt_affine_vs_captured_trt": metrics(
                production_trt_output, trt_outputs["trt_affine_normalization"]
            ),
        },
        "reduction_and_normalization": {
            "source_exact_plugin_vs_source_non_affine": metrics(
                source_normalized, plugin_normalized
            ),
            "production_trt_unit_vs_source_non_affine": metrics(
                source_normalized, trt_outputs["trt_unit_normalization"]
            ),
            "production_trt_unit_vs_source_exact_plugin": metrics(
                plugin_normalized, trt_outputs["trt_unit_normalization"]
            ),
        },
        "affine_boundary_candidates_vs_source": {
            "torch_separate_multiply_then_add": metrics(source_output, torch_separate_affine),
            "torch_addcmul_fused": metrics(source_output, torch_fused_affine),
            "torch_centered_combined_scale_separate": metrics(
                source_output, torch_centered_separate
            ),
            "torch_centered_combined_scale_addcmul": metrics(source_output, torch_centered_fused),
            "trt_explicit_multiply_then_add": metrics(
                source_output, trt_outputs["trt_explicit_affine"]
            ),
        },
        "production_trt_vs_source": metrics(source_output, production_trt_output),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
