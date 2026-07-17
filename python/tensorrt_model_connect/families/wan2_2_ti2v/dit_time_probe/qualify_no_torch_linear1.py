#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No-Torch TensorRT/CUDA qualification for official time Linear1 semantics."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import tensorrt as trt

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart  # type: ignore[no-redef]


M = 27_280
N = 3_072
K = 256
WORKSPACE_MIB = 32


class AlgoInfo(ctypes.Structure):
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
        ("workspace_bytes", ctypes.c_uint64),
        ("waves_count", ctypes.c_float),
    ]

    def as_dict(self) -> dict[str, int | float]:
        return {name: getattr(self, name) for name, _ctype in self._fields_}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_cuda(result, operation: str):
    status = result[0] if isinstance(result, tuple) else result
    success = cudart.cudaError_t.cudaSuccess if hasattr(cudart, "cudaError_t") else 0
    if status != success:
        raise RuntimeError(f"{operation} failed with CUDA status {status}")
    if isinstance(result, tuple):
        if len(result) == 2:
            return result[1]
        return result[1:]
    return None


def load_array(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    array = np.ascontiguousarray(np.load(path, allow_pickle=False))
    if tuple(array.shape) != shape or array.dtype != np.float32:
        raise TypeError(f"{path} is {array.shape}/{array.dtype}, expected {shape}/float32")
    return array


def query_algorithms(library: ctypes.CDLL) -> list[dict[str, int | float]]:
    library.trtmc_wan22_linear_probe_query.argtypes = [
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(AlgoInfo),
        ctypes.c_int32,
    ]
    library.trtmc_wan22_linear_probe_query.restype = ctypes.c_int32
    storage = (AlgoInfo * 128)()
    count = library.trtmc_wan22_linear_probe_query(M, N, K, WORKSPACE_MIB, storage, len(storage))
    if count <= 0:
        raise RuntimeError(f"cuBLASLt returned {count} algorithms")
    return [storage[index].as_dict() for index in range(min(count, len(storage)))]


def build_plan(plugin_path: Path, plan_path: Path) -> bytes:
    ctypes.CDLL(str(plugin_path.resolve()), mode=ctypes.RTLD_GLOBAL)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 << 20)
    inputs = [
        network.add_input("x", trt.float32, (M, K)),
        network.add_input("weight", trt.float32, (N, K)),
        network.add_input("bias", trt.float32, (N,)),
    ]
    creator = trt.get_plugin_registry().get_creator("Wan22DitLinearProbe", "1", "")
    if creator is None:
        raise RuntimeError("Experimental FP32 linear plugin creator is not registered")
    field_values = {
        "m": np.array([M], dtype=np.int32),
        "n": np.array([N], dtype=np.int32),
        "k": np.array([K], dtype=np.int32),
        "heuristic_index": np.array([0], dtype=np.int32),
        "workspace_mib": np.array([WORKSPACE_MIB], dtype=np.int32),
    }
    fields = [
        trt.PluginField(name, value, trt.PluginFieldType.INT32)
        for name, value in field_values.items()
    ]
    plugin = creator.create_plugin(
        "wan22_time_linear1_exact_probe", trt.PluginFieldCollection(fields)
    )
    layer = network.add_plugin_v2(inputs, plugin)
    if layer is None:
        raise RuntimeError("Could not add experimental FP32 time-linear plugin")
    output = layer.get_output(0)
    output.name = "output"
    network.mark_output(output)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("Could not build experimental FP32 time-linear plan")
    payload = bytes(plan)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_bytes(payload)
    return payload


def array_metrics(actual: np.ndarray, reference: np.ndarray) -> dict:
    actual_flat = actual.reshape(-1)
    reference_flat = reference.reshape(-1)
    exact_elements = int(
        np.count_nonzero(actual_flat.view(np.uint32) == reference_flat.view(np.uint32))
    )
    if exact_elements == reference_flat.size:
        return {
            "bitwise_exact": True,
            "exact_elements": exact_elements,
            "total_elements": reference_flat.size,
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "rmse": 0.0,
            "cosine_similarity": 1.0,
        }
    maximum = 0.0
    absolute_sum = 0.0
    square_sum = 0.0
    dot = 0.0
    actual_square = 0.0
    reference_square = 0.0
    chunk_elements = 4 << 20
    for start in range(0, reference_flat.size, chunk_elements):
        stop = min(start + chunk_elements, reference_flat.size)
        got = actual_flat[start:stop].astype(np.float64)
        ref = reference_flat[start:stop].astype(np.float64)
        delta = got - ref
        maximum = max(maximum, float(np.max(np.abs(delta))))
        absolute_sum += float(np.sum(np.abs(delta), dtype=np.float64))
        square_sum += float(np.sum(delta * delta, dtype=np.float64))
        dot += float(np.sum(got * ref, dtype=np.float64))
        actual_square += float(np.sum(got * got, dtype=np.float64))
        reference_square += float(np.sum(ref * ref, dtype=np.float64))
    count = reference_flat.size
    return {
        "bitwise_exact": False,
        "exact_elements": exact_elements,
        "total_elements": count,
        "max_abs_error": maximum,
        "mean_abs_error": absolute_sum / count,
        "rmse": math.sqrt(square_sum / count),
        "cosine_similarity": dot / math.sqrt(actual_square * reference_square),
    }


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("Invalid benchmark iteration counts")
    if any(name == "torch" or name.startswith("torch.") for name in sys.modules):
        raise RuntimeError("No-Torch qualifier unexpectedly imported torch")
    check_cuda(cudart.cudaSetDevice(0), "cudaSetDevice")
    device_properties = check_cuda(cudart.cudaGetDeviceProperties(0), "cudaGetDeviceProperties")
    device_name = device_properties.name
    if isinstance(device_name, bytes):
        device_name = device_name.decode("utf-8", errors="replace")
    library = ctypes.CDLL(str(args.plugin.resolve()), mode=ctypes.RTLD_GLOBAL)
    algorithms = query_algorithms(library)
    plan = args.plan.read_bytes() if args.plan.is_file() else build_plan(args.plugin, args.plan)
    logger = trt.Logger(trt.Logger.WARNING)
    engine = trt.Runtime(logger).deserialize_cuda_engine(plan)
    if engine is None:
        raise RuntimeError("Could not deserialize experimental FP32 time-linear plan")
    context = engine.create_execution_context()
    x = load_array(args.capture_dir / "time_features.npy", (1, M, K)).reshape(M, K)
    weight = load_array(args.capture_dir / "time_linear1_weight.npy", (N, K))
    bias = load_array(args.capture_dir / "time_linear1_bias.npy", (N,))
    reference = load_array(args.capture_dir / "time_linear1.npy", (1, M, N)).reshape(M, N)
    actual = np.empty((M, N), dtype=np.float32)
    host = {"x": x, "weight": weight, "bias": bias, "output": actual}
    device = {}
    for name, array in host.items():
        device[name] = check_cuda(cudart.cudaMalloc(array.nbytes), f"cudaMalloc({name})")
        context.set_tensor_address(name, device[name])
    stream = check_cuda(cudart.cudaStreamCreate(), "cudaStreamCreate")
    h2d = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
    d2h = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
    for name in ("x", "weight", "bias"):
        array = host[name]
        check_cuda(
            cudart.cudaMemcpyAsync(device[name], array.ctypes.data, array.nbytes, h2d, stream),
            f"cudaMemcpyAsync({name}, H2D)",
        )

    def execute() -> None:
        if not context.execute_async_v3(stream_handle=stream):
            raise RuntimeError("Experimental FP32 time-linear TensorRT execution failed")

    for _ in range(args.warmup):
        execute()
    check_cuda(cudart.cudaStreamSynchronize(stream), "warmup synchronize")
    start = check_cuda(cudart.cudaEventCreate(), "cudaEventCreate(start)")
    end = check_cuda(cudart.cudaEventCreate(), "cudaEventCreate(end)")
    samples = []
    for _ in range(args.iterations):
        check_cuda(cudart.cudaEventRecord(start, stream), "cudaEventRecord(start)")
        execute()
        check_cuda(cudart.cudaEventRecord(end, stream), "cudaEventRecord(end)")
        check_cuda(cudart.cudaEventSynchronize(end), "cudaEventSynchronize(end)")
        samples.append(float(check_cuda(cudart.cudaEventElapsedTime(start, end), "elapsed")))
    check_cuda(
        cudart.cudaMemcpyAsync(actual.ctypes.data, device["output"], actual.nbytes, d2h, stream),
        "cudaMemcpyAsync(output, D2H)",
    )
    check_cuda(cudart.cudaStreamSynchronize(stream), "output synchronize")
    metrics = array_metrics(actual, reference)
    args.actual.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.actual, actual, allow_pickle=False)
    forbidden_modules = sorted(
        name for name in sys.modules if name == "torch" or name.startswith("torch.")
    )
    report = {
        "kind": "wan2_2_ti2v_no_torch_time_linear1_qualification",
        "status": "PASS" if metrics["bitwise_exact"] and not forbidden_modules else "FAIL",
        "device": str(device_name),
        "shape": {"m": M, "n": N, "k": K},
        "semantics": {
            "input_dtype": "float32",
            "weight_dtype": "float32",
            "bias_dtype": "float32",
            "output_dtype": "float32",
            "compute": "CUBLAS_COMPUTE_32F",
            "tf32": False,
            "epilogue": "CUBLASLT_EPILOGUE_BIAS",
            "workspace_limit_bytes": WORKSPACE_MIB << 20,
        },
        "selected_algorithm": algorithms[0],
        "returned_algorithm_count": len(algorithms),
        "metrics": metrics,
        "latency": {
            "samples_ms": samples,
            "min_ms": min(samples),
            "median_ms": statistics.median(samples),
            "mean_ms": statistics.mean(samples),
        },
        "plan": {
            "path": str(args.plan.resolve()),
            "bytes": args.plan.stat().st_size,
            "sha256": sha256_file(args.plan),
            "device_memory_bytes": int(engine.device_memory_size_v2),
        },
        "plugin": {
            "path": str(args.plugin.resolve()),
            "bytes": args.plugin.stat().st_size,
            "sha256": sha256_file(args.plugin),
        },
        "actual": {
            "path": str(args.actual.resolve()),
            "sha256": sha256_file(args.actual),
        },
        "capture_hashes": {
            name: sha256_file(args.capture_dir / name)
            for name in (
                "time_features.npy",
                "time_linear1_weight.npy",
                "time_linear1_bias.npy",
                "time_linear1.npy",
            )
        },
        "forbidden_python_modules": forbidden_modules,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

    for pointer in device.values():
        check_cuda(cudart.cudaFree(pointer), "cudaFree")
    check_cuda(cudart.cudaEventDestroy(start), "cudaEventDestroy(start)")
    check_cuda(cudart.cudaEventDestroy(end), "cudaEventDestroy(end)")
    check_cuda(cudart.cudaStreamDestroy(stream), "cudaStreamDestroy")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
