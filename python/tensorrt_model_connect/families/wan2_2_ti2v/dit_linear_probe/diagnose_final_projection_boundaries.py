#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bracket Wan2.2 final FP32 projection numerics on a saved source call."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch
from safetensors import safe_open


ROWS = 27_280
INPUT_COLUMNS = 3_072
OUTPUT_COLUMNS = 192


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


def load_rows(path: Path, columns: int, device: torch.device) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=True).reshape(ROWS, columns).to(device)


def load_head(checkpoint: Path) -> tuple[np.ndarray, np.ndarray]:
    index_path = checkpoint / "diffusion_pytorch_model.safetensors.index.json"
    shard = checkpoint / "diffusion_pytorch_model-00003-of-00003.safetensors"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        shard = checkpoint / index["weight_map"]["head.head.weight"]
    with safe_open(shard, framework="pt", device="cpu") as tensors:
        weight = tensors.get_tensor("head.head.weight").float().numpy()
        bias = tensors.get_tensor("head.head.bias").float().numpy()
    return weight.reshape(OUTPUT_COLUMNS, INPUT_COLUMNS), bias.reshape(OUTPUT_COLUMNS)


def run_trt_matmul(
    input_fp32: torch.Tensor,
    weight: np.ndarray,
    bias: np.ndarray | None,
    *,
    tf32: bool,
) -> tuple[torch.Tensor, bool]:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    if not tf32:
        config.clear_flag(trt.BuilderFlag.TF32)
    actual_tf32 = bool(config.get_flag(trt.BuilderFlag.TF32))
    x = network.add_input("input", trt.float32, (ROWS, INPUT_COLUMNS))
    rhs_values = np.ascontiguousarray(weight.T, dtype=np.float32)
    rhs = network.add_constant(rhs_values.shape, rhs_values).get_output(0)
    output = network.add_matrix_multiply(
        x, trt.MatrixOperation.NONE, rhs, trt.MatrixOperation.NONE
    ).get_output(0)
    if bias is not None:
        bias_values = np.ascontiguousarray(bias.reshape(1, OUTPUT_COLUMNS), dtype=np.float32)
        bias_tensor = network.add_constant(bias_values.shape, bias_values).get_output(0)
        output = network.add_elementwise(
            output, bias_tensor, trt.ElementWiseOperation.SUM
        ).get_output(0)
    output.name = "output"
    network.mark_output(output)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("Could not build final projection diagnostic engine")
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    if engine is None:
        raise RuntimeError("Could not deserialize final projection diagnostic engine")
    context = engine.create_execution_context()
    result = torch.empty((ROWS, OUTPUT_COLUMNS), device=input_fp32.device, dtype=torch.float32)
    context.set_tensor_address("input", input_fp32.data_ptr())
    context.set_tensor_address("output", result.data_ptr())
    stream = torch.cuda.current_stream(input_fp32.device)
    if not context.execute_async_v3(stream_handle=stream.cuda_stream):
        raise RuntimeError("Final projection diagnostic execution failed")
    torch.cuda.synchronize(input_fp32.device)
    return result, actual_tf32


def run_trt_bias(input_fp32: torch.Tensor, bias: np.ndarray) -> torch.Tensor:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    x = network.add_input("input", trt.float32, (ROWS, OUTPUT_COLUMNS))
    values = np.ascontiguousarray(bias.reshape(1, OUTPUT_COLUMNS), dtype=np.float32)
    bias_tensor = network.add_constant(values.shape, values).get_output(0)
    output = network.add_elementwise(x, bias_tensor, trt.ElementWiseOperation.SUM).get_output(0)
    output.name = "output"
    network.mark_output(output)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("Could not build final projection bias diagnostic engine")
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    if engine is None:
        raise RuntimeError("Could not deserialize final projection bias diagnostic engine")
    context = engine.create_execution_context()
    result = torch.empty_like(input_fp32)
    context.set_tensor_address("input", input_fp32.data_ptr())
    context.set_tensor_address("output", result.data_ptr())
    stream = torch.cuda.current_stream(input_fp32.device)
    if not context.execute_async_v3(stream_handle=stream.cuda_stream):
        raise RuntimeError("Final projection bias diagnostic execution failed")
    torch.cuda.synchronize(input_fp32.device)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-input", type=Path, required=True)
    parser.add_argument("--trt-input", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--trt-output", type=Path, required=True)
    parser.add_argument("--pipeline-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    source_input = load_rows(args.source_input, INPUT_COLUMNS, device)
    trt_input = load_rows(args.trt_input, INPUT_COLUMNS, device)
    source_output = load_rows(args.source_output, OUTPUT_COLUMNS, device)
    production_trt_output = load_rows(args.trt_output, OUTPUT_COLUMNS, device)
    weight_np, bias_np = load_head(args.checkpoint)
    weight = torch.from_numpy(weight_np).to(device)
    bias = torch.from_numpy(bias_np).to(device)

    source_linear = torch.nn.functional.linear(source_input, weight, bias)
    source_addmm = torch.addmm(bias, source_input, weight.t())
    source_mm = torch.mm(source_input, weight.t())
    source_separate_bias = source_mm + bias
    trt_default_mm, default_mm_tf32 = run_trt_matmul(source_input, weight_np, None, tf32=True)
    trt_fp32_mm, fp32_mm_tf32 = run_trt_matmul(source_input, weight_np, None, tf32=False)
    trt_default_linear, default_linear_tf32 = run_trt_matmul(
        source_input, weight_np, bias_np, tf32=True
    )
    trt_fp32_linear, fp32_linear_tf32 = run_trt_matmul(source_input, weight_np, bias_np, tf32=False)
    trt_bias_on_source_mm = run_trt_bias(source_mm, bias_np)

    pipeline = json.loads(args.pipeline_report.read_text())
    unpatchify = pipeline["unpatchify_diagnostics"]
    report = {
        "kind": "wan2_2_ti2v_final_projection_boundary_diagnosis",
        "device": torch.cuda.get_device_name(device),
        "shape": [ROWS, INPUT_COLUMNS, OUTPUT_COLUMNS],
        "source_runtime": {
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        },
        "dtype_and_materialization": {
            "source_input": {
                "dtype": str(source_input.dtype),
                "shape": list(source_input.shape),
                "stride": list(source_input.stride()),
                "contiguous": source_input.is_contiguous(),
            },
            "tensorrt_input": {
                "dtype": str(trt_input.dtype),
                "shape": list(trt_input.shape),
                "stride": list(trt_input.stride()),
                "contiguous": trt_input.is_contiguous(),
            },
            "weight_dtype": str(weight.dtype),
            "weight_shape": list(weight.shape),
            "bias_dtype": str(bias.dtype),
            "bias_shape": list(bias.shape),
            "input_source_vs_tensorrt": metrics(source_input, trt_input),
            "torch_linear_vs_captured_source": metrics(source_output, source_linear),
        },
        "gemm_accumulation_and_order": {
            "tensorrt_flags": {
                "default_mm_tf32": default_mm_tf32,
                "fp32_mm_tf32": fp32_mm_tf32,
                "default_linear_tf32": default_linear_tf32,
                "fp32_linear_tf32": fp32_linear_tf32,
            },
            "trt_default_mm_vs_torch_mm": metrics(source_mm, trt_default_mm),
            "trt_tf32_disabled_mm_vs_torch_mm": metrics(source_mm, trt_fp32_mm),
            "trt_default_linear_vs_source": metrics(source_output, trt_default_linear),
            "trt_tf32_disabled_linear_vs_source": metrics(source_output, trt_fp32_linear),
        },
        "bias_and_fusion": {
            "torch_addmm_vs_captured_source": metrics(source_output, source_addmm),
            "torch_separate_mm_then_bias_vs_source": metrics(source_output, source_separate_bias),
            "trt_bias_only_vs_torch_separate_bias": metrics(
                source_separate_bias, trt_bias_on_source_mm
            ),
            "standalone_trt_default_linear_vs_production_trt": metrics(
                production_trt_output, trt_default_linear
            ),
        },
        "production_trt_vs_source": metrics(source_output, production_trt_output),
        "output_reshape_and_materialization": {
            "final_rows_shape_source": list(source_output.shape),
            "final_rows_shape_tensorrt": list(production_trt_output.shape),
            "trt_rows_bf16_then_source_layout_vs_engine_output": unpatchify["bf16_source_layout"],
            "trt_fp32_rows_source_layout_vs_engine_output": unpatchify["source_layout"],
            "source_fp32_rows_source_layout_vs_source_output": unpatchify[
                "native_layout_vs_reference"
            ],
        },
        "qualified_bf16_linear_plugin_applicability": {
            "applicable": False,
            "reason": (
                "final projection is FP32 [27280,3072] x [3072,192]; the qualified "
                "BF16 plugin requires BF16 inputs and does not include N=192"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
