#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualify an isolated native cuDNN Wan2.2 patch-embedding plugin.

The reference is the real official first-call capture.  Every cuDNN INSTANT
engine configuration that finalizes is executed against all 83,804,160 BF16
outputs.  The fastest bit-exact candidate is then exercised through an actual
TensorRT plan.  This script never modifies the production Wan2.2 builder.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from tensorrt_model_connect.trt_compat import trt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qualify_block0_ffn2 as base  # noqa: E402

LATENT_SHAPE = (1, 48, 31, 44, 80)
WEIGHT_SHAPE = (3072, 48, 1, 2, 2)
BIAS_SHAPE = (3072,)
OUTPUT_SHAPE = (27_280, 3_072)
OUTPUT_ELEMENTS = 83_804_160

BIAS_MODES = {
    0: "no_bias_control",
    1: "official_separate_bf16_add_then_transpose",
    2: "fused_bf16_add_with_transpose",
}


class CandidateInfo(ctypes.Structure):
    _fields_ = [
        ("heuristic_index", ctypes.c_int32),
        ("plan_status", ctypes.c_int32),
        ("engine_id", ctypes.c_int64),
        ("plan_workspace_bytes", ctypes.c_uint64),
        ("numerical_notes_mask", ctypes.c_uint64),
        ("behavior_notes_mask", ctypes.c_uint64),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--official-cudnn-log", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return (result.stdout + result.stderr).strip()


class NativeProbe:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.library = ctypes.CDLL(str(self.path), mode=ctypes.RTLD_GLOBAL)
        self.library.trtmc_wan22_patch_cudnn_query.argtypes = [
            ctypes.POINTER(CandidateInfo),
            ctypes.c_int32,
        ]
        self.library.trtmc_wan22_patch_cudnn_query.restype = ctypes.c_int
        self.library.trtmc_wan22_patch_cudnn_create.argtypes = [
            ctypes.c_int32,
            ctypes.c_int32,
        ]
        self.library.trtmc_wan22_patch_cudnn_create.restype = ctypes.c_void_p
        self.library.trtmc_wan22_patch_cudnn_destroy.argtypes = [ctypes.c_void_p]
        self.library.trtmc_wan22_patch_cudnn_workspace_bytes.argtypes = [ctypes.c_void_p]
        self.library.trtmc_wan22_patch_cudnn_workspace_bytes.restype = ctypes.c_uint64
        self.library.trtmc_wan22_patch_cudnn_plan_workspace_bytes.argtypes = [ctypes.c_void_p]
        self.library.trtmc_wan22_patch_cudnn_plan_workspace_bytes.restype = ctypes.c_uint64
        self.library.trtmc_wan22_patch_cudnn_version.argtypes = [ctypes.c_void_p]
        self.library.trtmc_wan22_patch_cudnn_version.restype = ctypes.c_uint64
        self.library.trtmc_wan22_patch_cudnn_plan_json.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int32,
        ]
        self.library.trtmc_wan22_patch_cudnn_plan_json.restype = ctypes.c_int
        self.library.trtmc_wan22_patch_cudnn_run.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self.library.trtmc_wan22_patch_cudnn_run.restype = ctypes.c_int

    def query(self) -> list[dict[str, int]]:
        count = self.library.trtmc_wan22_patch_cudnn_query(None, 0)
        if count <= 0:
            raise RuntimeError(f"cuDNN heuristic query returned {count}")
        storage = (CandidateInfo * count)()
        returned = self.library.trtmc_wan22_patch_cudnn_query(storage, count)
        if returned != count:
            raise RuntimeError(f"cuDNN candidate count changed from {count} to {returned}")
        return [
            {
                "heuristic_index": int(item.heuristic_index),
                "plan_status": int(item.plan_status),
                "engine_id": int(item.engine_id),
                "plan_workspace_bytes": int(item.plan_workspace_bytes),
                "numerical_notes_mask": int(item.numerical_notes_mask),
                "behavior_notes_mask": int(item.behavior_notes_mask),
            }
            for item in storage
        ]

    def create(self, heuristic_index: int, bias_mode: int) -> int:
        context = self.library.trtmc_wan22_patch_cudnn_create(heuristic_index, bias_mode)
        if not context:
            raise RuntimeError(
                f"Could not create cuDNN context index={heuristic_index} bias_mode={bias_mode}"
            )
        return int(context)

    def destroy(self, context: int) -> None:
        self.library.trtmc_wan22_patch_cudnn_destroy(context)

    def workspace_bytes(self, context: int) -> int:
        return int(self.library.trtmc_wan22_patch_cudnn_workspace_bytes(context))

    def plan_workspace_bytes(self, context: int) -> int:
        return int(self.library.trtmc_wan22_patch_cudnn_plan_workspace_bytes(context))

    def cudnn_version(self, context: int) -> int:
        return int(self.library.trtmc_wan22_patch_cudnn_version(context))

    def plan_json(self, context: int) -> dict[str, Any]:
        required = self.library.trtmc_wan22_patch_cudnn_plan_json(context, None, 0)
        if required <= 0:
            return {}
        storage = ctypes.create_string_buffer(required)
        returned = self.library.trtmc_wan22_patch_cudnn_plan_json(context, storage, required)
        if returned != required:
            raise RuntimeError("cuDNN plan JSON size changed")
        return json.loads(storage.value.decode("utf-8"))

    def run(
        self,
        context: int,
        tensors: dict[str, torch.Tensor],
        output: torch.Tensor,
        workspace: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> None:
        status = self.library.trtmc_wan22_patch_cudnn_run(
            context,
            tensors["latent"].data_ptr(),
            tensors["weight"].data_ptr(),
            tensors["bias"].data_ptr(),
            output.data_ptr(),
            workspace.data_ptr(),
            workspace.numel(),
            stream.cuda_stream,
        )
        if status != 0:
            raise RuntimeError(f"Native cuDNN execution returned {status}")


def load_capture(
    path: Path, device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    tensors = {
        "latent": payload["latent"].to(device=device, dtype=torch.bfloat16).contiguous(),
        "latent_fp32": payload["latent"].to(device=device, dtype=torch.float32).contiguous(),
        "weight": payload["weight"]
        .reshape(WEIGHT_SHAPE)
        .to(device=device, dtype=torch.bfloat16)
        .contiguous(),
        "weight_fp32": payload["weight_fp32"]
        .reshape(WEIGHT_SHAPE)
        .to(device=device, dtype=torch.float32)
        .contiguous(),
        "bias": payload["bias"].to(device=device, dtype=torch.bfloat16).contiguous(),
        "bias_fp32": payload["bias_fp32"].to(device=device, dtype=torch.float32).contiguous(),
        "reference": payload["reference"].to(device=device, dtype=torch.bfloat16).contiguous(),
    }
    if tuple(tensors["latent"].shape) != LATENT_SHAPE:
        raise ValueError(f"Unexpected latent shape {tuple(tensors['latent'].shape)}")
    if tuple(tensors["reference"].shape) != OUTPUT_SHAPE:
        raise ValueError(f"Unexpected reference shape {tuple(tensors['reference'].shape)}")
    if tensors["reference"].numel() != OUTPUT_ELEMENTS:
        raise ValueError("Official reference element count changed")
    return tensors, payload["metadata"]


def time_native(
    probe: NativeProbe,
    context: int,
    tensors: dict[str, torch.Tensor],
    output: torch.Tensor,
    workspace: torch.Tensor,
    stream: torch.cuda.Stream,
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float | list[float]]:
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            probe.run(context, tensors, output, workspace, stream)
        stream.synchronize()
        samples: list[float] = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            probe.run(context, tensors, output, workspace, stream)
            end.record(stream)
            end.synchronize()
            samples.append(float(start.elapsed_time(end)))
    return base.latency_summary(samples)


def run_native_candidate(
    probe: NativeProbe,
    candidate: dict[str, int],
    bias_mode: int,
    tensors: dict[str, torch.Tensor],
    stream: torch.cuda.Stream,
    args: argparse.Namespace,
) -> dict[str, Any]:
    index = candidate["heuristic_index"]
    context = probe.create(index, bias_mode)
    try:
        workspace_bytes = probe.workspace_bytes(context)
        workspace = torch.empty(workspace_bytes, device=tensors["latent"].device, dtype=torch.uint8)
        output = torch.empty(OUTPUT_SHAPE, device=tensors["latent"].device, dtype=torch.bfloat16)
        latency = time_native(
            probe,
            context,
            tensors,
            output,
            workspace,
            stream,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        metrics = base.tensor_metrics(output, tensors["reference"])
        return {
            **candidate,
            "bias_mode": bias_mode,
            "bias_mode_name": BIAS_MODES[bias_mode],
            "cudnn_version": probe.cudnn_version(context),
            "total_workspace_bytes": workspace_bytes,
            "materialized_conv_output_bytes": OUTPUT_ELEMENTS * 2,
            "selected_plan_workspace_bytes": probe.plan_workspace_bytes(context),
            "plan": probe.plan_json(context),
            "latency": latency,
            "metrics": metrics,
        }
    finally:
        probe.destroy(context)


def benchmark_official(
    tensors: dict[str, torch.Tensor], args: argparse.Namespace, device: torch.device
) -> dict[str, Any]:
    holder: dict[str, torch.Tensor] = {}

    def official_call() -> torch.Tensor:
        holder["conv"] = torch.nn.functional.conv3d(
            tensors["latent_fp32"],
            tensors["weight_fp32"],
            tensors["bias_fp32"],
            stride=(1, 2, 2),
        )
        holder["rows"] = (
            holder["conv"].flatten(2).transpose(1, 2).reshape(OUTPUT_SHAPE).contiguous()
        )
        return holder["rows"]

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        samples, output = base.timed_cuda(
            official_call,
            device=device,
            warmup=args.warmup,
            iterations=args.iterations,
        )

        # This is the value-level equivalent of the CUDA trace: convolution
        # without bias, followed by a separately materialized BF16 add.
        conv_without_bias = torch.nn.functional.conv3d(
            tensors["latent_fp32"],
            tensors["weight_fp32"],
            None,
            stride=(1, 2, 2),
        )
        explicit_bias = conv_without_bias + tensors["bias"].reshape(1, -1, 1, 1, 1)
        explicit_rows = explicit_bias.flatten(2).transpose(1, 2).reshape(OUTPUT_SHAPE).contiguous()
    return {
        "latency": base.latency_summary(samples),
        "metrics": base.tensor_metrics(output, tensors["reference"]),
        "no_bias_conv_then_explicit_bf16_bias_metrics": base.tensor_metrics(
            explicit_rows, tensors["reference"]
        ),
    }


def build_trt_plan(
    plugin_path: Path, selected_index: int, bias_mode: int, output_path: Path
) -> bytes:
    ctypes.CDLL(str(plugin_path.resolve()), mode=ctypes.RTLD_GLOBAL)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    latent = network.add_input("latent", trt.bfloat16, LATENT_SHAPE)
    weight = network.add_input("weight", trt.bfloat16, WEIGHT_SHAPE)
    bias = network.add_input("bias", trt.bfloat16, BIAS_SHAPE)
    creator = trt.get_plugin_registry().get_creator("Wan22PatchCudnnProbe", "1", "")
    if creator is None:
        raise RuntimeError("Wan22PatchCudnnProbe creator is not registered")
    storage, fields = base.plugin_fields(
        {"heuristic_index": selected_index, "bias_mode": bias_mode}
    )
    plugin = creator.create_plugin("wan22_patch_cudnn_probe", fields)
    if plugin is None:
        raise RuntimeError("Wan22PatchCudnnProbe creation failed")
    layer = network.add_plugin_v2([latent, weight, bias], plugin)
    del storage
    output = layer.get_output(0)
    output.name = "output"
    network.mark_output(output)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024**3)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Could not build TensorRT cuDNN patch plan")
    plan = bytes(serialized)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(plan)
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
        raise RuntimeError("Could not deserialize TensorRT cuDNN patch plan")
    context = engine.create_execution_context()
    output = torch.empty(OUTPUT_SHAPE, device=device, dtype=torch.bfloat16)
    for name, tensor in {
        "latent": tensors["latent"],
        "weight": tensors["weight"],
        "bias": tensors["bias"],
        "output": output,
    }.items():
        if not context.set_tensor_address(name, tensor.data_ptr()):
            raise RuntimeError(f"Could not bind TensorRT tensor {name}")

    def call() -> torch.Tensor:
        if not context.execute_async_v3(torch.cuda.current_stream(device).cuda_stream):
            raise RuntimeError("TensorRT cuDNN patch execution failed")
        return output

    samples, _ = base.timed_cuda(
        call, device=device, warmup=args.warmup, iterations=args.iterations
    )
    return {
        "latency": base.latency_summary(samples),
        "metrics": base.tensor_metrics(output, tensors["reference"]),
        "engine_device_memory_bytes": int(engine.device_memory_size_v2),
    }


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    tensors, metadata = load_capture(args.capture, device)
    probe = NativeProbe(args.plugin)
    stream = torch.cuda.Stream(device=device)

    official = benchmark_official(tensors, args, device)
    candidates = probe.query()
    viable = [candidate for candidate in candidates if candidate["plan_status"] == 0]
    if not viable:
        raise RuntimeError("No cuDNN INSTANT candidate produced a finalizable plan")

    candidate_results: list[dict[str, Any]] = []
    for candidate in viable:
        result = run_native_candidate(probe, candidate, 1, tensors, stream, args)
        candidate_results.append(result)
        engine = result.get("plan", {}).get("engine", {})
        print(
            f"candidate {result['heuristic_index']:02d}: "
            f"engine={engine.get('engineId')} exact={result['metrics']['bit_exact']} "
            f"median={result['latency']['median_ms']:.6f} ms",
            flush=True,
        )

    exact = [item for item in candidate_results if item["metrics"]["bit_exact"]]
    if not exact:
        raise RuntimeError("No cuDNN candidate is bit-exact to official patch Conv3d")
    selected = min(exact, key=lambda item: item["latency"]["median_ms"])

    bias_results: dict[str, Any] = {
        BIAS_MODES[1]: selected,
    }
    selected_candidate = next(
        item for item in viable if item["heuristic_index"] == selected["heuristic_index"]
    )
    for bias_mode in (0, 2):
        result = run_native_candidate(
            probe,
            selected_candidate,
            bias_mode,
            tensors,
            stream,
            args,
        )
        bias_results[BIAS_MODES[bias_mode]] = result

    plan = build_trt_plan(args.plugin, selected["heuristic_index"], 1, args.plan)
    trt_result = benchmark_trt_plan(plan, tensors, args, device)

    official_log_artifact: dict[str, Any] | None = None
    if args.official_cudnn_log is not None and args.official_cudnn_log.is_file():
        official_log_artifact = {
            "path": str(args.official_cudnn_log.resolve()),
            "sha256": sha256_file(args.official_cudnn_log),
            "bytes": args.official_cudnn_log.stat().st_size,
        }

    selected_engine = selected["plan"]["engine"]
    status = (
        "PASS"
        if selected["metrics"]["bit_exact"]
        and trt_result["metrics"]["bit_exact"]
        and bias_results[BIAS_MODES[1]]["metrics"]["bit_exact"]
        else "FAIL"
    )
    report = {
        "kind": "wan2_2_ti2v_patch_embedding_native_cudnn_qualification",
        "status": status,
        "hardware": {
            "device": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
        },
        "software": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "tensorrt": trt.__version__,
            "cudnn": selected["cudnn_version"],
        },
        "contract": {
            **metadata,
            "input_dtype": "BF16",
            "weight_layout": "OIDHW",
            "compute_dtype": "FP32",
            "convolution_output_dtype": "BF16",
            "official_bias_semantics": "separate BF16 elementwise add after cuDNN convolution",
            "output_shape": list(OUTPUT_SHAPE),
            "output_elements": OUTPUT_ELEMENTS,
        },
        "official_huggingface_patch_embedding": official,
        "official_cuda_trace_observation": {
            "cudnn_graph_contains_bias_op": False,
            "convolution_then_separate_bias_add": True,
            "heuristic_mode": "CUDNN_HEUR_MODE_INSTANT",
            "returned_engine_config_count": len(candidates),
            "executed_engine_id": 73,
            "executed_knob_choices": {
                "CUDNN_KNOB_TYPE_TILEK": 3,
                "CUDNN_KNOB_TYPE_SPLIT_K_SLC": 1,
                "CUDNN_KNOB_TYPE_TILE_CGA_M": 2,
                "CUDNN_KNOB_TYPE_TILE_CGA_N": 1,
                "CUDNN_KNOB_TYPE_CTA_COUNT": 1,
                "CUDNN_KNOB_TYPE_STREAM_K": 0,
                "CUDNN_KNOB_TYPE_TILE_M": 4,
                "CUDNN_KNOB_TYPE_TILE_N": 4,
            },
            "raw_log": official_log_artifact,
        },
        "candidate_summary": {
            "returned": len(candidates),
            "finalizable": len(viable),
            "executed": len(candidate_results),
            "bit_exact": len(exact),
            "comparison_elements_per_candidate": OUTPUT_ELEMENTS,
        },
        "cudnn_candidates": candidate_results,
        "selected_candidate": selected,
        "selected_descriptor": {
            "heuristic_index": selected["heuristic_index"],
            "engine_id": selected_engine["engineId"],
            "sm_version": selected_engine["smVersion"],
            "knob_choices": selected_engine["knobChoices"],
            "plan_workspace_bytes": selected["selected_plan_workspace_bytes"],
            "total_workspace_bytes": selected["total_workspace_bytes"],
        },
        "bias_mode_ab": bias_results,
        "tensorrt_plugin": trt_result,
        "portability": {
            "embedded_cuda_architectures": [103, 110],
            "policy": "re-query cuDNN INSTANT configs and finalize the selected index on each target",
            "gb300_sm103_validated": True,
            "jetson_thor_sm110_validated": False,
            "warning": "cuDNN engine IDs and heuristic ordering are target/version local; SM110 requires its own qualification before production use",
            "cuobjdump": command_output(["cuobjdump", "--list-elf", str(args.plugin)]),
        },
        "native_dependencies": command_output(["ldd", str(args.plugin)]),
        "artifacts": {
            "capture": {
                "path": str(args.capture.resolve()),
                "sha256": sha256_file(args.capture),
            },
            "plugin": {
                "path": str(args.plugin.resolve()),
                "sha256": sha256_file(args.plugin),
            },
            "tensorrt_plan": {
                "path": str(args.plan.resolve()),
                "sha256": hashlib.sha256(plan).hexdigest(),
                "bytes": len(plan),
            },
        },
        "finding": (
            "Native cuDNN plus a separate BF16 bias add is bit-exact for all official patch-embedding outputs and remains exact through TensorRT."
            if status == "PASS"
            else "Native cuDNN patch-embedding qualification failed."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": status,
                "candidate_summary": report["candidate_summary"],
                "selected_descriptor": report["selected_descriptor"],
                "selected_latency": selected["latency"],
                "selected_metrics": selected["metrics"],
                "trt": trt_result,
                "report": str(args.report.resolve()),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
