#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualify the pure-CUDA DiT numeric plugins against PyTorch CUDA source semantics."""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
from pathlib import Path

import numpy as np
from tensorrt_model_connect.trt_compat import trt
import torch
import torch.nn.functional as functional


def _build_engine(plugin_name: str, inputs, fields=()):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    tensors = [network.add_input(name, dtype, shape) for name, dtype, shape in inputs]
    creator = trt.get_plugin_registry().get_creator(plugin_name, "1", "")
    if creator is None:
        raise RuntimeError(f"{plugin_name} creator is not registered")
    plugin = creator.create_plugin(plugin_name.lower(), trt.PluginFieldCollection(list(fields)))
    layer = network.add_plugin_v2(tensors, plugin)
    if layer is None:
        raise RuntimeError(f"Could not add {plugin_name}")
    output = layer.get_output(0)
    output.name = "output"
    network.mark_output(output)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"Could not build {plugin_name} qualification engine")
    engine = trt.Runtime(logger).deserialize_cuda_engine(plan)
    if engine is None:
        raise RuntimeError(f"Could not deserialize {plugin_name} qualification engine")
    return engine


def _execute(engine, tensors: dict[str, torch.Tensor], output: torch.Tensor) -> None:
    context = engine.create_execution_context()
    for name, tensor in {**tensors, "output": output}.items():
        if not context.set_tensor_address(name, tensor.data_ptr()):
            raise RuntimeError(f"Could not bind {name}")
    stream = torch.cuda.current_stream().cuda_stream
    if not context.execute_async_v3(stream_handle=stream):
        raise RuntimeError("TensorRT plugin qualification execution failed")
    torch.cuda.synchronize()


def _qualify_gelu(device: torch.device) -> dict:
    engine = _build_engine("Wan22DitGelu", [("input", trt.bfloat16, (65536,))])
    bit_patterns = torch.arange(65536, dtype=torch.int32).to(torch.uint16)
    values = bit_patterns.view(torch.bfloat16).to(device)
    actual = torch.empty_like(values)
    _execute(engine, {"input": values}, actual)
    reference = functional.gelu(values, approximate="tanh")
    expected_bits = reference.cpu().view(torch.uint16)
    actual_bits = actual.cpu().view(torch.uint16)
    mismatches = expected_bits != actual_bits
    return {
        "input_count": 65536,
        "exact": bool(not torch.any(mismatches)),
        "mismatch_count": int(torch.count_nonzero(mismatches)),
    }


def _qualify_rotary(device: torch.device) -> dict:
    rows, heads, head_dim = 97, 24, 128
    half_dim = head_dim // 2
    generator = torch.Generator(device=device).manual_seed(42)
    source = torch.randn(
        (rows, heads * head_dim), generator=generator, device=device, dtype=torch.float32
    )
    phase = np.outer(
        np.arange(rows, dtype=np.float64),
        np.power(10000.0, -np.arange(0, head_dim, 2, dtype=np.float64) / head_dim),
    )
    cosine = np.cos(phase)
    sine = np.sin(phase)
    cos_high = cosine.astype(np.float32)
    sin_high = sine.astype(np.float32)
    cos_low = (cosine - cos_high.astype(np.float64)).astype(np.float32)
    sin_low = (sine - sin_high.astype(np.float64)).astype(np.float32)
    constants = {
        "cos_high": torch.from_numpy(cos_high).to(device),
        "sin_high": torch.from_numpy(sin_high).to(device),
        "cos_low": torch.from_numpy(cos_low).to(device),
        "sin_low": torch.from_numpy(sin_low).to(device),
    }
    fields = [
        trt.PluginField("heads", np.array([heads], dtype=np.int32), trt.PluginFieldType.INT32),
        trt.PluginField(
            "head_dim", np.array([head_dim], dtype=np.int32), trt.PluginFieldType.INT32
        ),
    ]
    engine = _build_engine(
        "Wan22DitRotary",
        [
            ("input", trt.float32, tuple(source.shape)),
            ("cos_high", trt.float32, tuple(cos_high.shape)),
            ("sin_high", trt.float32, tuple(sin_high.shape)),
            ("cos_low", trt.float32, tuple(cos_low.shape)),
            ("sin_low", trt.float32, tuple(sin_low.shape)),
        ],
        fields,
    )
    actual = torch.empty_like(source, dtype=torch.bfloat16)
    _execute(engine, {"input": source, **constants}, actual)

    pairs = source.double().reshape(rows, heads, half_dim, 2)
    complex_source = torch.view_as_complex(pairs)
    frequencies = torch.complex(
        torch.from_numpy(cosine).to(device), torch.from_numpy(sine).to(device)
    ).unsqueeze(1)
    reference = (
        torch.view_as_real(complex_source * frequencies).flatten(2).float().to(torch.bfloat16)
    )
    reference = reference.reshape(rows, heads * head_dim)
    mismatch_count = int(torch.count_nonzero(reference != actual))
    delta = reference.float() - actual.float()
    return {
        "shape": list(source.shape),
        "exact": mismatch_count == 0,
        "mismatch_count": mismatch_count,
        "max_abs_error": float(delta.abs().max()),
        "mean_abs_error": float(delta.abs().mean()),
    }


class _PatchPlanInfo(ctypes.Structure):
    _fields_ = [
        ("heuristic_index", ctypes.c_int32),
        ("reserved", ctypes.c_int32),
        ("engine_id", ctypes.c_int64),
        ("workspace_bytes", ctypes.c_uint64),
        ("cudnn_version", ctypes.c_uint64),
    ]


class _TimeLinear1PlanInfo(ctypes.Structure):
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


def _qualify_time_linear1(plugin_path: Path, capture_dir: Path, device: torch.device) -> dict:
    def load(name: str) -> torch.Tensor:
        array = np.ascontiguousarray(np.load(capture_dir / f"{name}.npy", allow_pickle=False))
        return torch.from_numpy(array).to(device)

    x = load("time_features").reshape(27_280, 256)
    weight = load("time_linear1_weight")
    bias = load("time_linear1_bias")
    reference = load("time_linear1").reshape(27_280, 3_072)
    engine = _build_engine(
        "Wan22DitTimeLinear1",
        [
            ("x", trt.float32, tuple(x.shape)),
            ("weight", trt.float32, tuple(weight.shape)),
            ("bias", trt.float32, tuple(bias.shape)),
        ],
    )
    context = engine.create_execution_context()
    output = torch.empty_like(reference)
    for name, tensor in {"x": x, "weight": weight, "bias": bias, "output": output}.items():
        if not context.set_tensor_address(name, tensor.data_ptr()):
            raise RuntimeError(f"Could not bind time Linear1 tensor {name}")

    def execute() -> None:
        stream = torch.cuda.current_stream(device).cuda_stream
        if not context.execute_async_v3(stream_handle=stream):
            raise RuntimeError("TensorRT time Linear1 execution failed")

    for _ in range(3):
        execute()
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(10):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        execute()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    exact_elements = int(
        torch.count_nonzero(output.view(torch.int32) == reference.view(torch.int32)).item()
    )
    delta = output.double() - reference.double()

    library = ctypes.CDLL(str(plugin_path.resolve()), mode=ctypes.RTLD_GLOBAL)
    library.trtmc_wan22_dit_time_linear1_plan_info.argtypes = [ctypes.POINTER(_TimeLinear1PlanInfo)]
    library.trtmc_wan22_dit_time_linear1_plan_info.restype = ctypes.c_int
    info = _TimeLinear1PlanInfo()
    if library.trtmc_wan22_dit_time_linear1_plan_info(ctypes.byref(info)) != 0:
        raise RuntimeError("Could not query the production time Linear1 cuBLASLt plan")
    return {
        "shape": list(output.shape),
        "exact": exact_elements == reference.numel(),
        "exact_elements": exact_elements,
        "total_elements": reference.numel(),
        "max_abs_error": float(delta.abs().max()),
        "mean_abs_error": float(delta.abs().mean()),
        "rmse": float(delta.square().mean().sqrt()),
        "latency": {
            "samples_ms": samples,
            "min_ms": min(samples),
            "median_ms": statistics.median(samples),
            "mean_ms": statistics.mean(samples),
        },
        "plan": {name: getattr(info, name) for name, _ctype in _TimeLinear1PlanInfo._fields_},
        "tensorrt_device_memory_bytes": int(engine.device_memory_size_v2),
    }


def _qualify_patch_embedding(plugin_path: Path, capture_path: Path, device: torch.device) -> dict:
    capture = torch.load(capture_path, map_location="cpu", weights_only=True)
    latent = capture["latent"].to(device=device, dtype=torch.bfloat16).contiguous()
    weight = (
        capture["weight"]
        .reshape(3072, 48, 1, 2, 2)
        .to(device=device, dtype=torch.bfloat16)
        .contiguous()
    )
    bias = capture["bias"].to(device=device, dtype=torch.bfloat16).contiguous()
    reference = capture["reference"].to(device=device, dtype=torch.bfloat16).contiguous()
    engine = _build_engine(
        "Wan22DitPatchEmbedding",
        [
            ("latent", trt.bfloat16, tuple(latent.shape)),
            ("weight", trt.bfloat16, tuple(weight.shape)),
            ("bias", trt.bfloat16, tuple(bias.shape)),
        ],
    )
    context = engine.create_execution_context()
    output = torch.empty((1, 3072, 31, 22, 40), device=device, dtype=torch.bfloat16)
    for name, tensor in {
        "latent": latent,
        "weight": weight,
        "bias": bias,
        "output": output,
    }.items():
        if not context.set_tensor_address(name, tensor.data_ptr()):
            raise RuntimeError(f"Could not bind patch embedding tensor {name}")

    def execute() -> None:
        stream = torch.cuda.current_stream(device).cuda_stream
        if not context.execute_async_v3(stream_handle=stream):
            raise RuntimeError("TensorRT patch embedding execution failed")

    for _ in range(3):
        execute()
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(10):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        execute()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    actual = output.flatten(2).transpose(1, 2).reshape(reference.shape).contiguous()
    exact_elements = int(
        torch.count_nonzero(actual.view(torch.int16) == reference.view(torch.int16)).item()
    )
    delta = actual.float() - reference.float()

    library = ctypes.CDLL(str(plugin_path.resolve()), mode=ctypes.RTLD_GLOBAL)
    library.trtmc_wan22_dit_patch_plan_info.argtypes = [ctypes.POINTER(_PatchPlanInfo)]
    library.trtmc_wan22_dit_patch_plan_info.restype = ctypes.c_int
    info = _PatchPlanInfo()
    if library.trtmc_wan22_dit_patch_plan_info(ctypes.byref(info)) != 0:
        raise RuntimeError("Could not query the production patch cuDNN plan")
    return {
        "shape": list(actual.shape),
        "exact": exact_elements == reference.numel(),
        "exact_elements": exact_elements,
        "total_elements": reference.numel(),
        "max_abs_error": float(delta.abs().max()),
        "mean_abs_error": float(delta.abs().mean()),
        "rmse": float(delta.square().mean().sqrt()),
        "latency": {
            "samples_ms": samples,
            "min_ms": min(samples),
            "median_ms": statistics.median(samples),
            "mean_ms": statistics.mean(samples),
        },
        "plan": {
            "heuristic_index": int(info.heuristic_index),
            "engine_id": int(info.engine_id),
            "workspace_bytes": int(info.workspace_bytes),
            "cudnn_version": int(info.cudnn_version),
            "tensorrt_device_memory_bytes": int(engine.device_memory_size_v2),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patch-capture", type=Path)
    parser.add_argument("--time-capture-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    ctypes.CDLL(str(args.plugin.resolve()), mode=ctypes.RTLD_GLOBAL)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    report = {
        "kind": "wan2_2_ti2v_dit_cuda_plugin_qualification",
        "device": torch.cuda.get_device_name(device),
        "gelu": _qualify_gelu(device),
        "rotary": _qualify_rotary(device),
    }
    if args.patch_capture is not None:
        report["patch_embedding"] = _qualify_patch_embedding(
            args.plugin, args.patch_capture, device
        )
    if args.time_capture_dir is not None:
        report["time_linear1"] = _qualify_time_linear1(args.plugin, args.time_capture_dir, device)
    report["passed"] = bool(
        report["gelu"]["exact"]
        and report["rotary"]["exact"]
        and report.get("patch_embedding", {"exact": True})["exact"]
        and report.get("time_linear1", {"exact": True})["exact"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
