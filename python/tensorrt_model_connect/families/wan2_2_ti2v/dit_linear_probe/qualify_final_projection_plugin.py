#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualify the production TensorRT FP32 Wan2.2 output-head plugin."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path

import numpy as np
from tensorrt_model_connect.trt_compat import trt
import torch
from safetensors import safe_open


M = 27_280
N = 192
K = 3_072


class PlanInfo(ctypes.Structure):
    _fields_ = [
        ("heuristic_index", ctypes.c_int32),
        ("algorithm_id", ctypes.c_int32),
        ("tile_id", ctypes.c_int32),
        ("stages_id", ctypes.c_int32),
        ("split_k", ctypes.c_int32),
        ("reduction_scheme", ctypes.c_int32),
        ("cta_swizzle", ctypes.c_int32),
        ("custom_option", ctypes.c_int32),
        ("inner_shape_id", ctypes.c_int32),
        ("cluster_shape_id", ctypes.c_int32),
        ("algorithm_workspace_bytes", ctypes.c_uint64),
        ("workspace_limit_bytes", ctypes.c_uint64),
        ("waves_count", ctypes.c_float),
    ]

    def as_dict(self) -> dict[str, int | float]:
        return {name: getattr(self, name) for name, _ctype in self._fields_}


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


def load_head(checkpoint: Path) -> tuple[np.ndarray, np.ndarray]:
    index_path = checkpoint / "diffusion_pytorch_model.safetensors.index.json"
    shard = checkpoint / "diffusion_pytorch_model-00003-of-00003.safetensors"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        shard = checkpoint / index["weight_map"]["head.head.weight"]
    with safe_open(shard, framework="pt", device="cpu") as tensors:
        weight = tensors.get_tensor("head.head.weight").float().numpy()
        bias = tensors.get_tensor("head.head.bias").float().numpy()
    return weight.reshape(N, K), bias.reshape(N)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    x = (
        torch.load(args.input, map_location="cpu", weights_only=True)
        .reshape(M, K)
        .to(device)
        .contiguous()
    )
    reference = (
        torch.load(args.reference, map_location="cpu", weights_only=True)
        .reshape(M, N)
        .to(device)
        .contiguous()
    )
    weight, bias = load_head(args.checkpoint)

    library = ctypes.CDLL(str(args.plugin.resolve()), mode=ctypes.RTLD_GLOBAL)
    query = library.trtmc_wan22_dit_final_projection_fp32_plan_info
    query.argtypes = [ctypes.POINTER(PlanInfo)]
    query.restype = ctypes.c_int32
    plan_info = PlanInfo()
    if query(ctypes.byref(plan_info)) != 0:
        raise RuntimeError("Could not query production final projection tactic")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    input_tensor = network.add_input("input", trt.float32, (M, K))
    weight_values = np.ascontiguousarray(weight, dtype=np.float32)
    bias_values = np.ascontiguousarray(bias, dtype=np.float32)
    weight_tensor = network.add_constant(weight_values.shape, weight_values).get_output(0)
    bias_tensor = network.add_constant(bias_values.shape, bias_values).get_output(0)
    creator = trt.get_plugin_registry().get_creator("Wan22DitFinalProjectionFp32", "1", "")
    if creator is None:
        raise RuntimeError("Wan22DitFinalProjectionFp32 creator is not registered")
    plugin = creator.create_plugin("qualified_final_projection", trt.PluginFieldCollection([]))
    layer = network.add_plugin_v2([input_tensor, weight_tensor, bias_tensor], plugin)
    if layer is None:
        raise RuntimeError("Could not add Wan22DitFinalProjectionFp32 plugin")
    output = layer.get_output(0)
    output.name = "output"
    network.mark_output(output)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 << 20)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Could not build final projection qualification plan")
    plan = bytes(serialized)
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    args.plan.write_bytes(plan)

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    if engine is None:
        raise RuntimeError("Could not deserialize final projection qualification plan")
    context = engine.create_execution_context()
    candidate = torch.empty((M, N), device=device, dtype=torch.float32)
    context.set_tensor_address("input", x.data_ptr())
    context.set_tensor_address("output", candidate.data_ptr())
    stream = torch.cuda.current_stream(device)
    if not context.execute_async_v3(stream_handle=stream.cuda_stream):
        raise RuntimeError("Final projection qualification execution failed")
    torch.cuda.synchronize(device)
    first = candidate.clone()
    if not context.execute_async_v3(stream_handle=stream.cuda_stream):
        raise RuntimeError("Final projection repeat execution failed")
    torch.cuda.synchronize(device)

    result = metrics(reference, candidate)
    repeat = metrics(first, candidate)
    passed = bool(result["bitwise_exact"] and repeat["bitwise_exact"])
    report = {
        "kind": "wan2_2_ti2v_production_final_projection_fp32_qualification",
        "status": "PASS" if passed else "FAIL",
        "device": torch.cuda.get_device_name(device),
        "shape": {"m": M, "n": N, "k": K},
        "plugin": str(args.plugin.resolve()),
        "plan": str(args.plan.resolve()),
        "plan_bytes": len(plan),
        "target_local_plan_info": plan_info.as_dict(),
        "predefined_gate": "bitwise exact reference and repeat over 5,237,760 FP32 elements",
        "metrics": result,
        "repeat_determinism": repeat,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
