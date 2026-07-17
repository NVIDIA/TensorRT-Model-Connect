#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualify strict-FP32 cuBLASLt tactics for the fixed Wan2.2 output head."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import statistics
from pathlib import Path

import torch
from safetensors import safe_open


M = 27_280
N = 192
K = 3_072
BASELINE_MISMATCHES = 5_087_265
BASELINE_MAX_ABS = 2.7179718017578125e-05
BASELINE_MAE = 1.1086557378803263e-06
BASELINE_RMSE = 1.5790527641001972e-06
FALLBACK_COSINE = 0.998


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
        "exact_rate": exact / total,
        "max_abs_error": float(delta.abs().max()),
        "mean_abs_error": float(delta.abs().mean()),
        "rmse": float(delta.square().mean().sqrt()),
        "cosine_similarity": float(
            torch.nn.functional.cosine_similarity(
                reference.flatten().double(), candidate.flatten().double(), dim=0
            )
        ),
    }


def fallback_pass(result: dict[str, float | int | bool]) -> bool:
    return bool(
        result["cosine_similarity"] >= FALLBACK_COSINE
        and result["mismatched_elements"] <= BASELINE_MISMATCHES // 10
        and result["max_abs_error"] <= BASELINE_MAX_ABS / 4.0
        and result["mean_abs_error"] <= BASELINE_MAE / 4.0
        and result["rmse"] <= BASELINE_RMSE / 4.0
    )


def load_head(checkpoint: Path) -> tuple[torch.Tensor, torch.Tensor]:
    index_path = checkpoint / "diffusion_pytorch_model.safetensors.index.json"
    shard = checkpoint / "diffusion_pytorch_model-00003-of-00003.safetensors"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        shard = checkpoint / index["weight_map"]["head.head.weight"]
    with safe_open(shard, framework="pt", device="cpu") as tensors:
        weight = tensors.get_tensor("head.head.weight").float()
        bias = tensors.get_tensor("head.head.bias").float()
    return weight.reshape(N, K), bias.reshape(N)


class Probe:
    def __init__(self, path: Path) -> None:
        self.library = ctypes.CDLL(str(path.resolve()), mode=ctypes.RTLD_GLOBAL)
        self.library.trtmc_wan22_final_projection_probe_query.argtypes = [
            ctypes.c_int32,
            ctypes.POINTER(AlgoInfo),
            ctypes.c_int32,
        ]
        self.library.trtmc_wan22_final_projection_probe_query.restype = ctypes.c_int32
        self.library.trtmc_wan22_final_projection_probe_create.argtypes = [
            ctypes.c_int32,
            ctypes.c_int32,
        ]
        self.library.trtmc_wan22_final_projection_probe_create.restype = ctypes.c_void_p
        self.library.trtmc_wan22_final_projection_probe_destroy.argtypes = [ctypes.c_void_p]
        self.library.trtmc_wan22_final_projection_probe_workspace_bytes.argtypes = [ctypes.c_void_p]
        self.library.trtmc_wan22_final_projection_probe_workspace_bytes.restype = ctypes.c_uint64
        self.library.trtmc_wan22_final_projection_probe_run.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self.library.trtmc_wan22_final_projection_probe_run.restype = ctypes.c_int32

    def query(self, workspace_mib: int) -> list[dict[str, int | float]]:
        storage = (AlgoInfo * 128)()
        count = self.library.trtmc_wan22_final_projection_probe_query(
            workspace_mib, storage, len(storage)
        )
        if count < 0:
            raise RuntimeError("strict-FP32 cuBLASLt heuristic query failed")
        return [storage[index].as_dict() for index in range(count)]

    def create(self, heuristic_index: int, workspace_mib: int) -> ctypes.c_void_p:
        context = self.library.trtmc_wan22_final_projection_probe_create(
            heuristic_index, workspace_mib
        )
        if not context:
            raise RuntimeError(f"Could not create tactic {heuristic_index}")
        return context

    def destroy(self, context: ctypes.c_void_p) -> None:
        self.library.trtmc_wan22_final_projection_probe_destroy(context)

    def workspace_bytes(self, context: ctypes.c_void_p) -> int:
        return int(self.library.trtmc_wan22_final_projection_probe_workspace_bytes(context))

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
        status = self.library.trtmc_wan22_final_projection_probe_run(
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
            raise RuntimeError(f"cuBLASLt tactic failed with status {status}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workspace-mib", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    if args.workspace_mib != 32:
        raise ValueError("Final projection qualification is fixed to the source 32 MiB pool")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("Invalid warmup/iteration count")

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
    weight = weight.to(device).contiguous()
    bias = bias.to(device).contiguous()
    source_replay = torch.nn.functional.linear(x, weight, bias)
    source_replay_metrics = metrics(reference, source_replay)
    if not source_replay_metrics["bitwise_exact"]:
        raise RuntimeError("Saved source final projection is not reproducible")

    probe = Probe(args.plugin)
    candidates = probe.query(args.workspace_mib)
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
            output = torch.empty((M, N), device=device, dtype=torch.float32)
            for _ in range(args.warmup):
                probe.run(context, x, weight, bias, output, workspace)
            torch.cuda.synchronize(device)
            probe.run(context, x, weight, bias, output, workspace)
            torch.cuda.synchronize(device)
            first_output = output.clone()
            samples = []
            stream = torch.cuda.current_stream(device)
            for _ in range(args.iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(stream)
                probe.run(context, x, weight, bias, output, workspace)
                end.record(stream)
                end.synchronize()
                samples.append(float(start.elapsed_time(end)))
            result_metrics = metrics(reference, output)
            repeat_metrics = metrics(first_output, output)
            result = {
                **candidate,
                "latency_ms": {
                    "samples": samples,
                    "min": min(samples),
                    "median": statistics.median(samples),
                    "mean": statistics.mean(samples),
                },
                "metrics": result_metrics,
                "repeat_determinism": repeat_metrics,
                "primary_pass": bool(result_metrics["bitwise_exact"]),
                "fallback_pass": fallback_pass(result_metrics),
            }
            results.append(result)
            print(
                f"tactic {index}: algo={candidate['algorithm_id']} tile={candidate['tile_id']} "
                f"split_k={candidate['split_k']} exact={result_metrics['bitwise_exact']} "
                f"rmse={result_metrics['rmse']:.9g} median={result['latency_ms']['median']:.4f} ms",
                flush=True,
            )
        finally:
            probe.destroy(context)

    exact = [result for result in results if result["primary_pass"]]
    fallback = [result for result in results if result["fallback_pass"]]
    eligible = exact or fallback
    selected = (
        min(eligible, key=lambda result: result["latency_ms"]["median"]) if eligible else None
    )
    status = "PASS_EXACT" if exact else "PASS_FALLBACK" if fallback else "FAIL"
    report = {
        "kind": "wan2_2_ti2v_final_projection_fp32_tactic_qualification",
        "status": status,
        "hardware": {
            "device": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
        },
        "plugin": str(args.plugin.resolve()),
        "plugin_sha256": hashlib.sha256(args.plugin.read_bytes()).hexdigest(),
        "contract": {
            "shape": {"m": M, "n": N, "k": K},
            "dtype": "FP32 input, weight, bias, accumulation, output",
            "compute": "CUBLAS_COMPUTE_32F with TF32 disabled",
            "workspace_mib": args.workspace_mib,
            "source_replay": source_replay_metrics,
        },
        "predeclared_gate": {
            "primary": "bitwise exact over all 5,237,760 FP32 elements",
            "fallback": {
                "cosine_similarity_min": FALLBACK_COSINE,
                "mismatched_elements_max": BASELINE_MISMATCHES // 10,
                "max_abs_error_max": BASELINE_MAX_ABS / 4.0,
                "mean_abs_error_max": BASELINE_MAE / 4.0,
                "rmse_max": BASELINE_RMSE / 4.0,
            },
        },
        "candidate_count": len(candidates),
        "exact_candidate_count": len(exact),
        "fallback_candidate_count": len(fallback),
        "selected": selected,
        "candidates": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": status, "selected": selected}, indent=2))
    return 0 if selected is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
