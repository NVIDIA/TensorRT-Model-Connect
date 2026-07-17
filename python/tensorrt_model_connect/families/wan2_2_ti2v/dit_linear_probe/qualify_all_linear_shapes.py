#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualify every unique fixed-profile Wan2.2 DiT BF16 linear shape.

All operands and references are captured from one real official block0 forward.
Synthetic data is not used for qualification.  The script deliberately keeps
target-local autotuning separate from production builders: it enumerates every
returned cuBLASLt heuristic, rejects split-K for the portable exactness policy,
times the remaining candidates, and verifies the winner through TensorRT.
"""

from __future__ import annotations

import argparse
import ctypes
import functools
import json
import sys
from pathlib import Path
from typing import Any, Callable

import tensorrt as trt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qualify_block0_ffn2 as base  # noqa: E402  pylint: disable=wrong-import-position


SHAPES: dict[str, dict[str, Any]] = {
    "m27280_k3072_n3072": {
        "module": "blocks.0.self_attn.q",
        "m": 27_280,
        "k": 3_072,
        "n": 3_072,
    },
    "m27280_k3072_n14336": {
        "module": "blocks.0.ffn.0",
        "m": 27_280,
        "k": 3_072,
        "n": 14_336,
    },
    "m27280_k14336_n3072": {
        "module": "blocks.0.ffn.2",
        "m": 27_280,
        "k": 14_336,
        "n": 3_072,
    },
    "m512_k4096_n3072": {
        "module": "text_embedding.0",
        "m": 512,
        "k": 4_096,
        "n": 3_072,
    },
    "m512_k3072_n3072": {
        "module": "text_embedding.2",
        "m": 512,
        "k": 3_072,
        "n": 3_072,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--first-call", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--plan-dir", type=Path, required=True)
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


def module_by_path(model: torch.nn.Module, path: str) -> torch.nn.Module:
    current: Any = model
    for component in path.split("."):
        current = current[int(component)] if component.isdigit() else getattr(current, component)
    if not isinstance(current, torch.nn.Module):
        raise TypeError(f"{path} did not resolve to a module")
    return current


def capture_path(capture_dir: Path, shape_name: str) -> Path:
    return capture_dir / f"{shape_name}.pt"


def manifest_path(capture_dir: Path) -> Path:
    return capture_dir / "manifest.json"


def all_captures_exist(capture_dir: Path) -> bool:
    return manifest_path(capture_dir).is_file() and all(
        capture_path(capture_dir, name).is_file() for name in SHAPES
    )


def capture_official_shapes(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    print("Loading official Wan2.2 checkpoint and retaining block0 only...", flush=True)
    WanModel = base.import_official_model(args.official_source)
    first_call = torch.load(args.first_call, map_location="cpu", weights_only=True)
    model = WanModel.from_pretrained(str(args.checkpoint)).eval().requires_grad_(False)
    model.blocks = torch.nn.ModuleList(list(model.blocks[:1]))
    model.to(device)

    captures: dict[str, dict[str, torch.Tensor]] = {name: {} for name in SHAPES}
    hooks = []
    modules: dict[str, torch.nn.Linear] = {}
    for name, specification in SHAPES.items():
        module = module_by_path(model, specification["module"])
        if not isinstance(module, torch.nn.Linear):
            raise TypeError(f"{specification['module']} is not nn.Linear")
        modules[name] = module

        def capture_input(
            _module: torch.nn.Module,
            module_inputs: tuple[torch.Tensor, ...],
            *,
            key: str = name,
        ) -> None:
            captures[key]["raw_input"] = module_inputs[0].detach()

        def capture_output(
            _module: torch.nn.Module,
            _module_inputs: tuple[torch.Tensor, ...],
            value: torch.Tensor,
            *,
            key: str = name,
        ) -> None:
            captures[key]["reference"] = value.detach()

        hooks.extend(
            [
                module.register_forward_pre_hook(capture_input),
                module.register_forward_hook(capture_output),
            ]
        )

    latent = first_call["latent"].unsqueeze(0).to(device=device, dtype=torch.float32)
    timestep = first_call["timestep"].to(device=device, dtype=torch.float32)
    context = first_call["context"].to(device=device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        model(
            [latent[0]],
            timestep,
            [context],
            seq_len=int(first_call["seq_len"]),
        )
    torch.cuda.synchronize(device)
    for hook in hooks:
        hook.remove()

    args.capture_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "kind": "wan2_2_ti2v_official_all_unique_bf16_linear_shapes",
        "official_source": str(args.official_source.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "first_call": str(args.first_call.resolve()),
        "shapes": {},
    }
    for name, specification in SHAPES.items():
        module = modules[name]
        raw_input = captures[name]["raw_input"]
        reference = captures[name]["reference"]
        m, k, n = specification["m"], specification["k"], specification["n"]
        raw_input = raw_input.reshape(-1, raw_input.shape[-1]).contiguous()
        x = raw_input.to(dtype=torch.bfloat16).contiguous()
        reference = reference.reshape(-1, reference.shape[-1]).contiguous()
        weight = module.weight.detach().to(dtype=torch.bfloat16).contiguous()
        bias = (
            module.bias.detach().to(dtype=torch.bfloat16).contiguous()
            if module.bias is not None
            else None
        )
        if tuple(x.shape) != (m, k) or tuple(weight.shape) != (n, k):
            raise ValueError(
                f"{name} shape mismatch: x={tuple(x.shape)}, weight={tuple(weight.shape)}"
            )
        if tuple(reference.shape) != (m, n):
            raise ValueError(f"{name} reference shape is {tuple(reference.shape)}")
        if (bias is None) != (module.bias is None):
            raise AssertionError("Bias capture changed module semantics")

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            official_call: Callable[[], torch.Tensor] = functools.partial(module, raw_input)
            samples, replay = base.timed_cuda(
                official_call,
                device=device,
                warmup=args.warmup,
                iterations=args.iterations,
            )
        replay_metrics = base.tensor_metrics(replay, reference)
        if not replay_metrics["bit_exact"]:
            raise RuntimeError(f"Official isolated replay is not exact for {name}")
        metadata = {
            **specification,
            "bias": bias is not None,
            "raw_module_input_dtype": str(raw_input.dtype),
            "autocast_gemm_input_dtype": str(x.dtype),
            "parameter_weight_dtype": str(module.weight.dtype),
            "parameter_bias_dtype": str(module.bias.dtype) if module.bias is not None else None,
            "official_hot_latency": base.latency_summary(samples),
            "official_hot_replay_metrics": replay_metrics,
        }
        payload: dict[str, Any] = {
            "x": x.cpu(),
            "weight": weight.cpu(),
            "reference": reference.cpu(),
            "metadata": metadata,
        }
        if bias is not None:
            payload["bias"] = bias.cpu()
        output_path = capture_path(args.capture_dir, name)
        torch.save(payload, output_path)
        metadata["capture"] = str(output_path.resolve())
        metadata["capture_bytes"] = output_path.stat().st_size
        manifest["shapes"][name] = metadata
        print(
            f"captured {name} from {specification['module']}: "
            f"{output_path.stat().st_size / 2**20:.1f} MiB bias={bias is not None}",
            flush=True,
        )
        del payload, x, weight, reference, raw_input, replay, official_call
        if bias is not None:
            del bias
        captures[name].clear()
        torch.cuda.empty_cache()

    manifest_path(args.capture_dir).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


class GenericProbe:
    """Generic-shape wrapper around the direct qualification C ABI."""

    def __init__(self, plugin_path: Path) -> None:
        self.library = ctypes.CDLL(str(plugin_path.resolve()), mode=ctypes.RTLD_GLOBAL)
        self.library.trtmc_wan22_linear_probe_query.argtypes = [
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(base.AlgoInfo),
            ctypes.c_int32,
        ]
        self.library.trtmc_wan22_linear_probe_query.restype = ctypes.c_int32
        self.library.trtmc_wan22_linear_probe_create.argtypes = [ctypes.c_int32] * 5
        self.library.trtmc_wan22_linear_probe_create.restype = ctypes.c_void_p
        self.library.trtmc_wan22_linear_probe_destroy.argtypes = [ctypes.c_void_p]
        self.library.trtmc_wan22_linear_probe_workspace_bytes.argtypes = [ctypes.c_void_p]
        self.library.trtmc_wan22_linear_probe_workspace_bytes.restype = ctypes.c_uint64
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

    def query(self, m: int, n: int, k: int, workspace_mib: int) -> list[dict[str, Any]]:
        storage = (base.AlgoInfo * 128)()
        count = self.library.trtmc_wan22_linear_probe_query(
            m, n, k, workspace_mib, storage, len(storage)
        )
        if count < 0:
            raise RuntimeError(f"cuBLASLt query failed for M={m},N={n},K={k}")
        return [storage[index].as_dict() for index in range(count)]

    def create(self, m: int, n: int, k: int, index: int, workspace_mib: int) -> ctypes.c_void_p:
        context = self.library.trtmc_wan22_linear_probe_create(m, n, k, index, workspace_mib)
        if not context:
            raise RuntimeError(f"Could not create heuristic {index} for M={m},N={n},K={k}")
        return context

    def destroy(self, context: ctypes.c_void_p) -> None:
        self.library.trtmc_wan22_linear_probe_destroy(context)

    def run(
        self,
        context: ctypes.c_void_p,
        tensors: dict[str, torch.Tensor],
        output: torch.Tensor,
        workspace: torch.Tensor | None,
    ) -> None:
        bias = tensors.get("bias")
        if bias is None:
            raise RuntimeError("Current official DiT capture unexpectedly has no bias")
        status = self.library.trtmc_wan22_linear_probe_run(
            context,
            ctypes.c_void_p(tensors["x"].data_ptr()),
            ctypes.c_void_p(tensors["weight"].data_ptr()),
            ctypes.c_void_p(bias.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_void_p(0 if workspace is None else workspace.data_ptr()),
            0 if workspace is None else workspace.numel(),
            ctypes.c_void_p(torch.cuda.current_stream(output.device).cuda_stream),
        )
        if status != 0:
            raise RuntimeError(f"cuBLASLt run failed with status {status}")


def load_shape_capture(
    path: Path, device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    tensors = {
        key: value.to(device=device).contiguous()
        for key, value in payload.items()
        if key in {"x", "weight", "bias", "reference"}
    }
    return tensors, payload["metadata"]


def benchmark_pytorch_bf16(
    tensors: dict[str, torch.Tensor], args: argparse.Namespace, device: torch.device
) -> dict[str, Any]:
    holder: dict[str, torch.Tensor] = {}

    def call() -> torch.Tensor:
        holder["output"] = torch.nn.functional.linear(
            tensors["x"], tensors["weight"], tensors.get("bias")
        )
        return holder["output"]

    with torch.inference_mode():
        samples, output = base.timed_cuda(
            call, device=device, warmup=args.warmup, iterations=args.iterations
        )
    return {
        "latency": base.latency_summary(samples),
        "metrics": base.tensor_metrics(output, tensors["reference"]),
    }


def benchmark_candidates(
    probe: GenericProbe,
    candidates: list[dict[str, Any]],
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    m, n, k = metadata["m"], metadata["n"], metadata["k"]
    results = []
    for candidate in candidates:
        index = int(candidate["heuristic_index"])
        context = probe.create(m, n, k, index, args.workspace_mib)
        try:
            workspace_bytes = int(probe.library.trtmc_wan22_linear_probe_workspace_bytes(context))
            workspace = (
                torch.empty(workspace_bytes, device=device, dtype=torch.uint8)
                if workspace_bytes
                else None
            )
            output = torch.empty((m, n), device=device, dtype=torch.bfloat16)

            def call() -> torch.Tensor:
                probe.run(context, tensors, output, workspace)
                return output

            samples, _ = base.timed_cuda(
                call, device=device, warmup=args.warmup, iterations=args.iterations
            )
            latency = base.latency_summary(samples)
            latency["effective_tflops_median"] = (
                2.0 * m * n * k / (float(latency["median_ms"]) * 1.0e9)
            )
            result = dict(candidate)
            result["portable_non_splitk_admissible"] = (
                int(candidate["split_k"]) == 1 and int(candidate["reduction_scheme"]) == 0
            )
            result["latency"] = latency
            result["metrics"] = base.tensor_metrics(output, tensors["reference"])
            results.append(result)
            print(
                f"  h{index}: tile={candidate['tile_id']} split={candidate['split_k']} "
                f"reduction={candidate['reduction_scheme']} "
                f"median={latency['median_ms']:.4f}ms "
                f"exact={result['metrics']['exact_rate']:.9f}",
                flush=True,
            )
        finally:
            probe.destroy(context)
    return results


def build_plan(
    plugin_path: Path,
    metadata: dict[str, Any],
    selected_index: int,
    workspace_mib: int,
    output_path: Path,
) -> bytes:
    ctypes.CDLL(str(plugin_path.resolve()), mode=ctypes.RTLD_GLOBAL)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    m, n, k = metadata["m"], metadata["n"], metadata["k"]
    x = network.add_input("x", trt.bfloat16, (m, k))
    weight = network.add_input("weight", trt.bfloat16, (n, k))
    inputs = [x, weight]
    if metadata["bias"]:
        inputs.append(network.add_input("bias", trt.bfloat16, (n,)))
    else:
        raise RuntimeError("No-bias plugin ABI is not needed by the official Wan2.2 DiT")
    creator = trt.get_plugin_registry().get_creator("Wan22DitLinearProbe", "1", "")
    if creator is None:
        raise RuntimeError("Wan22DitLinearProbe creator is not registered")
    storage, fields = base.plugin_fields(
        {
            "m": m,
            "n": n,
            "k": k,
            "heuristic_index": selected_index,
            "workspace_mib": workspace_mib,
        }
    )
    plugin = creator.create_plugin(f"wan22_{m}_{k}_{n}", fields)
    layer = network.add_plugin_v2(inputs, plugin)
    del storage
    output = layer.get_output(0)
    output.name = "output"
    network.mark_output(output)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, max(2 * 1024**3, workspace_mib * 1024**2)
    )
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"Could not build TensorRT plan for M={m},N={n},K={k}")
    plan = bytes(serialized)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(plan)
    return plan


def benchmark_plan(
    plan: bytes,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    output = torch.empty((metadata["m"], metadata["n"]), device=device, dtype=torch.bfloat16)
    bindings = {**tensors, "output": output}
    bindings.pop("reference")
    for name, tensor in bindings.items():
        if not context.set_tensor_address(name, tensor.data_ptr()):
            raise RuntimeError(f"Could not bind {name}")

    def call() -> torch.Tensor:
        if not context.execute_async_v3(torch.cuda.current_stream(device).cuda_stream):
            raise RuntimeError("TensorRT plugin execution failed")
        return output

    samples, _ = base.timed_cuda(
        call, device=device, warmup=args.warmup, iterations=args.iterations
    )
    latency = base.latency_summary(samples)
    latency["effective_tflops_median"] = (
        2.0 * metadata["m"] * metadata["n"] * metadata["k"] / (float(latency["median_ms"]) * 1.0e9)
    )
    return {
        "latency": latency,
        "metrics": base.tensor_metrics(output, tensors["reference"]),
    }


def qualify_shape(
    name: str,
    capture: Path,
    plan_path: Path,
    plugin_path: Path,
    probe: GenericProbe,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    tensors, metadata = load_shape_capture(capture, device)
    print(
        f"qualifying {name}: M={metadata['m']} K={metadata['k']} N={metadata['n']} "
        f"bias={metadata['bias']}",
        flush=True,
    )
    pytorch_bf16 = benchmark_pytorch_bf16(tensors, args, device)
    candidates = probe.query(metadata["m"], metadata["n"], metadata["k"], args.workspace_mib)
    results = benchmark_candidates(probe, candidates, tensors, metadata, args, device)
    admissible = [
        candidate
        for candidate in results
        if candidate["portable_non_splitk_admissible"] and candidate["metrics"]["bit_exact"]
    ]
    if not admissible:
        raise RuntimeError(f"No exact non-splitK candidate for {name}")
    heuristic_first = min(admissible, key=lambda candidate: candidate["heuristic_index"])
    measured_fastest = min(admissible, key=lambda candidate: candidate["latency"]["median_ms"])
    first_ms = float(heuristic_first["latency"]["median_ms"])
    fastest_ms = float(measured_fastest["latency"]["median_ms"])
    plan = build_plan(
        plugin_path,
        metadata,
        int(measured_fastest["heuristic_index"]),
        args.workspace_mib,
        plan_path,
    )
    trt_result = benchmark_plan(plan, tensors, metadata, args, device)
    result = {
        "metadata": metadata,
        "capture": str(capture.resolve()),
        "capture_bytes": capture.stat().st_size,
        "plan": str(plan_path.resolve()),
        "plan_bytes": plan_path.stat().st_size,
        "pytorch_bf16_replay": pytorch_bf16,
        "returned_candidate_count": len(results),
        "candidates": results,
        "portable_policy": {
            "filter": "split_k == 1 and reduction_scheme == 0 and qualified bit_exact",
            "admissible_count": len(admissible),
            "heuristic_first_non_splitk": heuristic_first,
            "empirical_fastest_non_splitk": measured_fastest,
            "heuristic_first_is_empirical_fastest": int(heuristic_first["heuristic_index"])
            == int(measured_fastest["heuristic_index"]),
            "empirical_tuning_speedup_over_heuristic_first": first_ms / fastest_ms,
        },
        "tensorrt_plugin": trt_result,
    }
    del tensors
    torch.cuda.empty_cache()
    return result


def main() -> int:
    args = parse_args()
    if args.iterations <= 0 or args.warmup < 0 or args.workspace_mib < 0:
        raise ValueError("Invalid benchmark configuration")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    source_dir = Path(__file__).resolve().parent
    plugin_path = args.plugin
    if not args.skip_build:
        plugin_path = base.build_plugin(source_dir, args.build_dir)
    if plugin_path is None or not plugin_path.is_file():
        raise FileNotFoundError("Plugin DSO is missing")

    if args.force_capture or not all_captures_exist(args.capture_dir):
        manifest = capture_official_shapes(args, device)
        torch.cuda.empty_cache()
    else:
        manifest = json.loads(manifest_path(args.capture_dir).read_text(encoding="utf-8"))
    if args.capture_only:
        return 0

    probe = GenericProbe(plugin_path)
    benchmark_stream = torch.cuda.Stream(device=device)
    shape_results: dict[str, Any] = {}
    with torch.cuda.stream(benchmark_stream):
        for name in SHAPES:
            shape_results[name] = qualify_shape(
                name,
                capture_path(args.capture_dir, name),
                args.plan_dir / f"{name}.plan",
                plugin_path,
                probe,
                args,
                device,
            )

    all_exact = all(
        result["tensorrt_plugin"]["metrics"]["bit_exact"] for result in shape_results.values()
    )
    all_non_split_candidates_exact = all(
        candidate["metrics"]["bit_exact"]
        for result in shape_results.values()
        for candidate in result["candidates"]
        if candidate["portable_non_splitk_admissible"]
    )
    first_is_fastest = {
        name: result["portable_policy"]["heuristic_first_is_empirical_fastest"]
        for name, result in shape_results.items()
    }
    actual_bias_cases = {name: result["metadata"]["bias"] for name, result in shape_results.items()}
    report = {
        "kind": "wan2_2_ti2v_all_unique_bf16_linear_shape_qualification",
        "status": "PASS" if all_exact and all_non_split_candidates_exact else "FAIL",
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
            "source": "official real module inputs/outputs from one block0 forward",
            "shape_count": len(SHAPES),
            "workspace_limit_mib": args.workspace_mib,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "latency_scope": "hot isolated CUDA-event latency on a non-default stream",
        },
        "bias_semantics": {
            "actual_cases": actual_bias_cases,
            "bias_case_count": sum(actual_bias_cases.values()),
            "no_bias_case_count": sum(not value for value in actual_bias_cases.values()),
            "finding": (
                "Every nn.Linear in the official Wan2.2 DiT has bias=True; there is no real "
                "no-bias DiT case to qualify. UMT5 no-bias linears are outside this DiT scope."
            ),
        },
        "selection_policy_assessment": {
            "portable_admissibility_rule": "split_k == 1 and reduction_scheme == 0",
            "all_admissible_candidates_bit_exact": all_non_split_candidates_exact,
            "heuristic_first_is_empirical_fastest_by_shape": first_is_fastest,
            "requires_target_local_empirical_timing_for_measured_fastest": not all(
                first_is_fastest.values()
            ),
            "conclusion": (
                "Do not hardcode a heuristic index. Requery on the target, reject split-K/nonzero "
                "reduction, time each admissible descriptor, and cache the fastest descriptor for "
                "that shape plus GPU/CUDA/cuBLASLt tuple. Heuristic order alone is only an estimate."
            ),
            "portability_limits": [
                "cuBLASLt heuristic order, algorithm IDs, tiles, and stages are GPU/library specific.",
                "TensorRT plans are target specific and must be rebuilt on Jetson Thor.",
                "SM110 is packaged in the DSO but exactness and timing still require Thor qualification.",
            ],
        },
        "manifest": manifest,
        "plugin": str(plugin_path.resolve()),
        "dependencies": base.dependency_audit(plugin_path),
        "shapes": shape_results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "shapes": {
                    name: {
                        "candidate_count": result["returned_candidate_count"],
                        "selected_index": result["portable_policy"]["empirical_fastest_non_splitk"][
                            "heuristic_index"
                        ],
                        "trt_median_ms": result["tensorrt_plugin"]["latency"]["median_ms"],
                        "bit_exact": result["tensorrt_plugin"]["metrics"]["bit_exact"],
                    }
                    for name, result in shape_results.items()
                },
                "report": str(args.report),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
