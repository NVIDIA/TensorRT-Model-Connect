#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualify a torch-free cuBLASLt plugin against official Wan2.2 block0 FFN2.

The qualification contract is deliberately narrow and reproducible:

* capture every one of the 27,280 rows entering block0 ``ffn[2]`` from the
  official Wan2.2 source forward;
* preserve the exact BF16 autocast weight, bias, and output tensors;
* enumerate cuBLASLt heuristic algorithms for 14336 -> 3072 linear+bias;
* compare BF16 bits, numerical metrics, and CUDA-event latency; and
* rebuild the best exact candidate as a real TensorRT plugin layer and verify
  it once more.

PyTorch is used only by this qualification program for the official reference,
tensor storage, and device allocation.  The plugin DSO itself links only CUDA,
cuBLASLt, TensorRT, and system libraries.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Callable

import numpy as np
import tensorrt as trt
import torch


M = 27_280
N = 3_072
K = 14_336


class AlgoInfo(ctypes.Structure):
    """Mirror of the stable POD returned by the experimental C ABI."""

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


class ProbeLibrary:
    """ctypes wrapper for direct candidate enumeration and timing."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.library = ctypes.CDLL(str(self.path), mode=ctypes.RTLD_GLOBAL)
        self.library.trtmc_wan22_linear_probe_query.argtypes = [
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(AlgoInfo),
            ctypes.c_int32,
        ]
        self.library.trtmc_wan22_linear_probe_query.restype = ctypes.c_int32
        self.library.trtmc_wan22_linear_probe_create.argtypes = [ctypes.c_int32] * 5
        self.library.trtmc_wan22_linear_probe_create.restype = ctypes.c_void_p
        self.library.trtmc_wan22_linear_probe_destroy.argtypes = [ctypes.c_void_p]
        self.library.trtmc_wan22_linear_probe_workspace_bytes.argtypes = [ctypes.c_void_p]
        self.library.trtmc_wan22_linear_probe_workspace_bytes.restype = ctypes.c_uint64
        self.library.trtmc_wan22_linear_probe_get_info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(AlgoInfo),
        ]
        self.library.trtmc_wan22_linear_probe_get_info.restype = ctypes.c_int32
        self.library.trtmc_wan22_linear_probe_run.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self.library.trtmc_wan22_linear_probe_run.restype = ctypes.c_int32

    def query(self, workspace_mib: int) -> list[dict[str, int | float]]:
        storage = (AlgoInfo * 128)()
        count = self.library.trtmc_wan22_linear_probe_query(
            M, N, K, workspace_mib, storage, len(storage)
        )
        if count < 0:
            raise RuntimeError("cuBLASLt heuristic query failed")
        return [storage[index].as_dict() for index in range(count)]

    def create(self, heuristic_index: int, workspace_mib: int) -> ctypes.c_void_p:
        context = self.library.trtmc_wan22_linear_probe_create(
            M, N, K, heuristic_index, workspace_mib
        )
        if not context:
            raise RuntimeError(f"Could not create candidate {heuristic_index}")
        return context

    def destroy(self, context: ctypes.c_void_p) -> None:
        self.library.trtmc_wan22_linear_probe_destroy(context)

    def workspace_bytes(self, context: ctypes.c_void_p) -> int:
        return int(self.library.trtmc_wan22_linear_probe_workspace_bytes(context))

    def run(
        self,
        context: ctypes.c_void_p,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        output: torch.Tensor,
        workspace: torch.Tensor | None,
    ) -> None:
        workspace_bytes = 0 if workspace is None else workspace.numel()
        status = self.library.trtmc_wan22_linear_probe_run(
            context,
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(weight.data_ptr()),
            ctypes.c_void_p(bias.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_void_p(0 if workspace is None else workspace.data_ptr()),
            workspace_bytes,
            ctypes.c_void_p(torch.cuda.current_stream(x.device).cuda_stream),
        )
        if status != 0:
            raise RuntimeError(f"cuBLASLt candidate execution failed with status {status}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--first-call", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--plugin", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workspace-mib", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--force-capture", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--capture-only", action="store_true")
    return parser.parse_args()


def build_plugin(source_dir: Path, build_dir: Path) -> Path:
    subprocess.run(
        [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            "-GNinja",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
    )
    subprocess.run(["cmake", "--build", str(build_dir), "-j2"], check=True)
    return build_dir / "libtrtmc_wan22_dit_linear_probe.so"


def import_official_model(official_source: Path) -> Any:
    official_root = official_source.resolve()
    sys.path.insert(0, str(official_root))
    wan_package = types.ModuleType("wan")
    wan_package.__path__ = [str(official_root / "wan")]
    modules_package = types.ModuleType("wan.modules")
    modules_package.__path__ = [str(official_root / "wan" / "modules")]
    sys.modules["wan"] = wan_package
    sys.modules["wan.modules"] = modules_package
    from wan.modules.model import WanModel  # pylint: disable=import-outside-toplevel

    return WanModel


def timed_cuda(
    function: Callable[[], Any], *, device: torch.device, warmup: int, iterations: int
) -> tuple[list[float], Any]:
    result = None
    for _ in range(warmup):
        result = function()
    torch.cuda.synchronize(device)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for index in range(iterations):
        starts[index].record()
        result = function()
        ends[index].record()
    ends[-1].synchronize()
    return [float(start.elapsed_time(end)) for start, end in zip(starts, ends)], result


def latency_summary(samples_ms: list[float]) -> dict[str, float | list[float]]:
    ordered = sorted(samples_ms)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        "samples_ms": samples_ms,
        "min_ms": min(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "mean_ms": statistics.mean(samples_ms),
        "p95_ms": ordered[p95_index],
    }


def tensor_metrics(got: torch.Tensor, reference: torch.Tensor) -> dict[str, int | float | bool]:
    if got.dtype != torch.bfloat16 or reference.dtype != torch.bfloat16:
        raise TypeError(f"Expected BF16 tensors, got {got.dtype} and {reference.dtype}")
    got_flat = got.reshape(-1)
    ref_flat = reference.reshape(-1)
    exact_count = int((got_flat.view(torch.int16) == ref_flat.view(torch.int16)).sum().item())
    count = got_flat.numel()
    if exact_count == count:
        return {
            "bit_exact": True,
            "exact_elements": count,
            "total_elements": count,
            "exact_rate": 1.0,
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "rmse": 0.0,
            "cosine_similarity": 1.0,
        }
    got_float = got_flat.float()
    ref_float = ref_flat.float()
    delta = got_float - ref_float
    cosine = torch.nn.functional.cosine_similarity(got_float, ref_float, dim=0)
    return {
        "bit_exact": False,
        "exact_elements": exact_count,
        "total_elements": count,
        "exact_rate": exact_count / count,
        "max_abs_error": float(delta.abs().max().item()),
        "mean_abs_error": float(delta.abs().mean().item()),
        "rmse": float(delta.square().mean().sqrt().item()),
        "cosine_similarity": float(cosine.item()),
    }


def capture_official(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    print("Loading official Wan2.2 checkpoint and retaining block0 only...", flush=True)
    WanModel = import_official_model(args.official_source)
    inputs = torch.load(args.first_call, map_location="cpu", weights_only=True)
    native = WanModel.from_pretrained(str(args.checkpoint)).eval().requires_grad_(False)
    native.blocks = torch.nn.ModuleList(list(native.blocks[:1]))
    native.to(device)
    ffn2 = native.blocks[0].ffn[2]
    captured: dict[str, torch.Tensor] = {}

    def pre_hook(_module: torch.nn.Module, module_inputs: tuple[torch.Tensor, ...]) -> None:
        captured["x"] = module_inputs[0].detach()

    def post_hook(
        _module: torch.nn.Module,
        _module_inputs: tuple[torch.Tensor, ...],
        value: torch.Tensor,
    ) -> None:
        captured["reference"] = value.detach()

    hooks = [ffn2.register_forward_pre_hook(pre_hook), ffn2.register_forward_hook(post_hook)]
    latent = inputs["latent"].unsqueeze(0).to(device=device, dtype=torch.float32)
    timestep = inputs["timestep"].to(device=device, dtype=torch.float32)
    context = inputs["context"].to(device=device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        native([latent[0]], timestep, [context], seq_len=int(inputs["seq_len"]))
    torch.cuda.synchronize(device)
    for hook in hooks:
        hook.remove()

    x = captured["x"].reshape(-1, captured["x"].shape[-1]).contiguous()
    reference = captured["reference"].reshape(-1, captured["reference"].shape[-1]).contiguous()
    weight = ffn2.weight.detach().to(dtype=torch.bfloat16).contiguous()
    bias = ffn2.bias.detach().to(dtype=torch.bfloat16).contiguous()
    if tuple(x.shape) != (M, K) or tuple(weight.shape) != (N, K) or tuple(bias.shape) != (N,):
        raise ValueError(
            f"Unexpected FFN2 shapes x={tuple(x.shape)}, weight={tuple(weight.shape)}, "
            f"bias={tuple(bias.shape)}"
        )
    if tuple(reference.shape) != (M, N):
        raise ValueError(f"Unexpected reference shape {tuple(reference.shape)}")
    if x.dtype != torch.bfloat16 or reference.dtype != torch.bfloat16:
        raise TypeError(f"Official autocast emitted {x.dtype} input and {reference.dtype} output")

    holder: dict[str, torch.Tensor] = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):

        def official_hot_call() -> torch.Tensor:
            holder["official"] = ffn2(x)
            return holder["official"]

        official_samples, official_replay = timed_cuda(
            official_hot_call,
            device=device,
            warmup=args.warmup,
            iterations=args.iterations,
        )
    official_replay_metrics = tensor_metrics(official_replay, reference)
    payload = {
        "x": x.cpu(),
        "weight": weight.cpu(),
        "bias": bias.cpu(),
        "reference": reference.cpu(),
        "metadata": {
            "kind": "wan2_2_ti2v_official_block0_ffn2_full_rows",
            "official_source": str(args.official_source.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "first_call": str(args.first_call.resolve()),
            "sequence_rows": M,
            "linear": f"{K}->{N}",
            "parameter_weight_dtype": str(ffn2.weight.dtype),
            "parameter_bias_dtype": str(ffn2.bias.dtype),
            "autocast_tensor_dtype": str(x.dtype),
            "official_hot_latency": latency_summary(official_samples),
            "official_hot_replay_metrics": official_replay_metrics,
        },
    }
    args.capture.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.capture)
    print(f"Saved {args.capture} ({args.capture.stat().st_size / 2**30:.3f} GiB)", flush=True)
    return payload


def load_capture(
    path: Path, device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    tensors = {
        name: payload[name].to(device=device, non_blocking=False).contiguous()
        for name in ("x", "weight", "bias", "reference")
    }
    if tuple(tensors["x"].shape) != (M, K) or tuple(tensors["reference"].shape) != (M, N):
        raise ValueError("Capture does not contain the official full-row FFN2 contract")
    return tensors, payload["metadata"]


def benchmark_bf16_pytorch(
    tensors: dict[str, torch.Tensor], args: argparse.Namespace, device: torch.device
) -> dict[str, Any]:
    holder: dict[str, torch.Tensor] = {}

    def call() -> torch.Tensor:
        holder["output"] = torch.nn.functional.linear(
            tensors["x"], tensors["weight"], tensors["bias"]
        )
        return holder["output"]

    with torch.inference_mode():
        samples, output = timed_cuda(
            call, device=device, warmup=args.warmup, iterations=args.iterations
        )
    return {
        "latency": latency_summary(samples),
        "metrics": tensor_metrics(output, tensors["reference"]),
    }


def benchmark_candidates(
    probe: ProbeLibrary,
    candidates: list[dict[str, int | float]],
    tensors: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    results = []
    for candidate in candidates:
        index = int(candidate["heuristic_index"])
        context = probe.create(index, args.workspace_mib)
        try:
            workspace_bytes = probe.workspace_bytes(context)
            workspace = (
                torch.empty(workspace_bytes, device=device, dtype=torch.uint8)
                if workspace_bytes
                else None
            )
            output = torch.empty((M, N), device=device, dtype=torch.bfloat16)

            def call() -> torch.Tensor:
                probe.run(
                    context,
                    tensors["x"],
                    tensors["weight"],
                    tensors["bias"],
                    output,
                    workspace,
                )
                return output

            samples, _ = timed_cuda(
                call, device=device, warmup=args.warmup, iterations=args.iterations
            )
            metrics = tensor_metrics(output, tensors["reference"])
            latency = latency_summary(samples)
            latency["effective_tflops_median"] = (
                2.0 * M * N * K / (float(latency["median_ms"]) * 1.0e9)
            )
            result = dict(candidate)
            result.update({"latency": latency, "metrics": metrics})
            results.append(result)
            print(
                f"candidate {index}: tile={candidate['tile_id']} split_k={candidate['split_k']} "
                f"median={latency['median_ms']:.4f} ms exact={metrics['exact_rate']:.9f}",
                flush=True,
            )
        finally:
            probe.destroy(context)
    return results


def plugin_fields(values: dict[str, int]) -> tuple[list[np.ndarray], trt.PluginFieldCollection]:
    storage = [np.asarray([value], dtype=np.int32) for value in values.values()]
    fields = [
        trt.PluginField(name, value, trt.PluginFieldType.INT32)
        for name, value in zip(values, storage)
    ]
    return storage, trt.PluginFieldCollection(fields)


def build_trt_plan(
    plugin_path: Path,
    selected_index: int,
    workspace_mib: int,
    plan_path: Path,
) -> bytes:
    # Loading the DSO registers Wan22DitLinearProbe through its static creator.
    ctypes.CDLL(str(plugin_path.resolve()), mode=ctypes.RTLD_GLOBAL)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    x = network.add_input("x", trt.bfloat16, (M, K))
    weight = network.add_input("weight", trt.bfloat16, (N, K))
    bias = network.add_input("bias", trt.bfloat16, (N,))
    creator = trt.get_plugin_registry().get_creator("Wan22DitLinearProbe", "1", "")
    if creator is None:
        raise RuntimeError("Wan22DitLinearProbe creator did not register")
    storage, fields = plugin_fields(
        {
            "m": M,
            "n": N,
            "k": K,
            "heuristic_index": selected_index,
            "workspace_mib": workspace_mib,
        }
    )
    plugin = creator.create_plugin("wan22_block0_ffn2_probe", fields)
    if plugin is None:
        raise RuntimeError("Could not create Wan22DitLinearProbe")
    layer = network.add_plugin_v2([x, weight, bias], plugin)
    del storage  # Plugin fields are consumed synchronously by create_plugin.
    output = layer.get_output(0)
    output.name = "output"
    network.mark_output(output)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, max(2 * 1024**3, workspace_mib * 1024**2)
    )
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build the linear probe plan")
    plan = bytes(serialized)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_bytes(plan)
    return plan


def benchmark_trt_plan(
    plan: bytes,
    tensors: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    if engine is None:
        raise RuntimeError("Could not deserialize linear probe plan")
    context = engine.create_execution_context()
    output = torch.empty((M, N), device=device, dtype=torch.bfloat16)
    for name, tensor in (
        ("x", tensors["x"]),
        ("weight", tensors["weight"]),
        ("bias", tensors["bias"]),
        ("output", output),
    ):
        if not context.set_tensor_address(name, tensor.data_ptr()):
            raise RuntimeError(f"Could not bind TensorRT tensor {name}")

    def call() -> torch.Tensor:
        if not context.execute_async_v3(torch.cuda.current_stream(device).cuda_stream):
            raise RuntimeError("TensorRT plugin execution failed")
        return output

    samples, _ = timed_cuda(call, device=device, warmup=args.warmup, iterations=args.iterations)
    latency = latency_summary(samples)
    latency["effective_tflops_median"] = 2.0 * M * N * K / (float(latency["median_ms"]) * 1.0e9)
    return {"latency": latency, "metrics": tensor_metrics(output, tensors["reference"])}


def dependency_audit(plugin: Path) -> dict[str, Any]:
    output = subprocess.check_output(["ldd", str(plugin.resolve())], text=True)
    dependencies = [line.strip() for line in output.splitlines() if line.strip()]
    prohibited = [
        line
        for line in dependencies
        if any(token in line.lower() for token in ("libtorch", "libaten", "libc10"))
    ]
    return {
        "ldd": dependencies,
        "prohibited_torch_aten_dependencies": prohibited,
        "torch_aten_free": not prohibited,
    }


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.iterations <= 0 or args.workspace_mib < 0:
        raise ValueError("Invalid benchmark iteration/workspace configuration")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    source_dir = Path(__file__).resolve().parent
    plugin_path = args.plugin
    if not args.skip_build:
        plugin_path = build_plugin(source_dir, args.build_dir)
    if plugin_path is None or not plugin_path.is_file():
        raise FileNotFoundError("Plugin DSO is missing; pass --plugin or allow the build")

    if args.force_capture or not args.capture.is_file():
        capture_official(args, device)
        torch.cuda.empty_cache()
    if args.capture_only:
        return 0

    tensors, capture_metadata = load_capture(args.capture, device)
    probe = ProbeLibrary(plugin_path)
    candidates = probe.query(args.workspace_mib)
    if not candidates:
        raise RuntimeError("cuBLASLt returned no algorithms")
    # A non-default stream avoids TensorRT's defensive default-stream
    # synchronization and keeps all three hot-path measurements comparable.
    benchmark_stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(benchmark_stream):
        pytorch_bf16 = benchmark_bf16_pytorch(tensors, args, device)
        candidate_results = benchmark_candidates(probe, candidates, tensors, args, device)
    exact_candidates = [item for item in candidate_results if item["metrics"]["bit_exact"]]
    if not exact_candidates:
        raise RuntimeError(
            "No bit-exact torch-free cuBLASLt candidate exists; do not add PyTorch to the plugin"
        )
    selected = min(exact_candidates, key=lambda item: item["latency"]["median_ms"])
    selected_index = int(selected["heuristic_index"])
    plan = build_trt_plan(plugin_path, selected_index, args.workspace_mib, args.plan)
    with torch.cuda.stream(benchmark_stream):
        trt_result = benchmark_trt_plan(plan, tensors, args, device)
    dependencies = dependency_audit(plugin_path)
    if not dependencies["torch_aten_free"]:
        raise RuntimeError("Experimental plugin has a prohibited torch/ATen runtime dependency")

    report = {
        "kind": "wan2_2_ti2v_block0_ffn2_cublaslt_qualification",
        "status": "PASS" if trt_result["metrics"]["bit_exact"] else "FAIL",
        "hardware": {
            "device": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "visible_device": str(device),
        },
        "software": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "tensorrt": trt.__version__,
        },
        "contract": {
            "rows": M,
            "in_features": K,
            "out_features": N,
            "operation": "BF16 D = X @ W^T + BF16 bias, FP32 accumulation",
            "cublaslt_memory_view": "column-major TNN view of row-major PyTorch buffers",
            "workspace_limit_mib": args.workspace_mib,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "latency_scope": "hot isolated CUDA-event kernel latency",
        },
        "artifacts": {
            "capture": str(args.capture.resolve()),
            "capture_bytes": args.capture.stat().st_size,
            "plugin": str(plugin_path.resolve()),
            "plugin_bytes": plugin_path.stat().st_size,
            "plan": str(args.plan.resolve()),
            "plan_bytes": args.plan.stat().st_size,
        },
        "capture_metadata": capture_metadata,
        "pytorch_bf16_replay": pytorch_bf16,
        "heuristic_candidates": candidate_results,
        "selected_candidate": selected,
        "tensorrt_plugin": trt_result,
        "dependencies": dependencies,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected_candidate": selected_index,
                "algorithm_id": selected["algorithm_id"],
                "tile_id": selected["tile_id"],
                "plugin_median_ms": trt_result["latency"]["median_ms"],
                "plugin_bit_exact": trt_result["metrics"]["bit_exact"],
                "report": str(args.report),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
