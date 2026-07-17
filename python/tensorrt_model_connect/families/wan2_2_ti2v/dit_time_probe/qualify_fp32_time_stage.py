#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualify one official FP32 Wan2.2 time-path linear with cuBLASLt.

This diagnostic replays a captured official operator boundary.  It deliberately
feeds the *official* source input to every candidate, so an upstream SiLU drift
cannot be mistaken for a linear-kernel mismatch.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import torch


STAGES: dict[str, dict[str, Any]] = {
    "time_embed": {
        "m": 27_280,
        "n": 3_072,
        "k": 3_072,
        "x": "time_silu.npy",
        "weight": "time_linear2_weight.npy",
        "bias": "time_linear2_bias.npy",
        "reference": "time_embed.npy",
    },
    "time_proj": {
        "m": 27_280,
        "n": 18_432,
        "k": 3_072,
        "x": "projection_silu.npy",
        "weight": "projection_linear_weight.npy",
        "bias": "projection_linear_bias.npy",
        "reference": "time_projection_flat.npy",
    },
}


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
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workspace-mib", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--candidate",
        type=int,
        action="append",
        help="Only run this heuristic index; repeat to select several. Default: all.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tensor(path: Path, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if tuple(array.shape) != shape or array.dtype != np.float32:
        raise TypeError(f"{path} is {array.shape}/{array.dtype}, expected {shape}/float32")
    # The mmap is read-only, but the CPU view is only copied to the GPU.
    return torch.from_numpy(array).to(device=device).contiguous()


def tensor_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    actual_flat = actual.reshape(-1)
    reference_flat = reference.reshape(-1)
    exact_elements = int(
        torch.count_nonzero(
            actual_flat.view(torch.int32) == reference_flat.view(torch.int32)
        ).item()
    )
    count = reference_flat.numel()
    if exact_elements == count:
        return {
            "bitwise_exact": True,
            "exact_elements": count,
            "total_elements": count,
            "exact_rate": 1.0,
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "rmse": 0.0,
            "cosine_similarity": 1.0,
        }
    delta = actual_flat.double() - reference_flat.double()
    cosine = torch.nn.functional.cosine_similarity(
        actual_flat.double(), reference_flat.double(), dim=0
    )
    return {
        "bitwise_exact": False,
        "exact_elements": exact_elements,
        "total_elements": count,
        "exact_rate": exact_elements / count,
        "max_abs_error": float(delta.abs().max().item()),
        "mean_abs_error": float(delta.abs().mean().item()),
        "rmse": float(delta.square().mean().sqrt().item()),
        "cosine_similarity": float(cosine.item()),
    }


class Probe:
    def __init__(self, path: Path) -> None:
        self.library = ctypes.CDLL(str(path.resolve()), mode=ctypes.RTLD_GLOBAL)
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

    def query(self, m: int, n: int, k: int, workspace_mib: int) -> list[AlgoInfo]:
        storage = (AlgoInfo * 128)()
        count = self.library.trtmc_wan22_linear_probe_query(
            m, n, k, workspace_mib, storage, len(storage)
        )
        if count <= 0:
            raise RuntimeError(f"cuBLASLt returned {count} candidates")
        return [storage[index] for index in range(min(count, len(storage)))]


def latency_summary(samples: list[float]) -> dict[str, Any]:
    return {
        "samples_ms": samples,
        "min_ms": min(samples),
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
    }


def main() -> int:
    args = parse_args()
    if args.workspace_mib < 0 or args.warmup < 0 or args.iterations <= 0:
        raise ValueError("Invalid benchmark configuration")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    specification = STAGES[args.stage]
    m, n, k = specification["m"], specification["n"], specification["k"]
    x = load_tensor(args.capture_dir / specification["x"], (1, m, k), device).reshape(m, k)
    weight = load_tensor(args.capture_dir / specification["weight"], (n, k), device)
    bias = load_tensor(args.capture_dir / specification["bias"], (n,), device)
    reference = load_tensor(
        args.capture_dir / specification["reference"], (1, m, n), device
    ).reshape(m, n)
    output = torch.empty_like(reference)

    probe = Probe(args.plugin)
    candidates = probe.query(m, n, k, args.workspace_mib)
    selected = set(args.candidate) if args.candidate else None
    candidates = [
        candidate
        for candidate in candidates
        if selected is None or candidate.heuristic_index in selected
    ]
    if not candidates:
        raise ValueError("No queried candidate matched --candidate")

    results = []
    stream = torch.cuda.current_stream(device)
    for candidate in candidates:
        context = probe.library.trtmc_wan22_linear_probe_create(
            m, n, k, candidate.heuristic_index, args.workspace_mib
        )
        if not context:
            raise RuntimeError(f"Could not create candidate {candidate.heuristic_index}")
        try:
            workspace_bytes = int(probe.library.trtmc_wan22_linear_probe_workspace_bytes(context))
            workspace = (
                torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
                if workspace_bytes
                else None
            )

            def execute() -> None:
                status = probe.library.trtmc_wan22_linear_probe_run(
                    context,
                    ctypes.c_void_p(x.data_ptr()),
                    ctypes.c_void_p(weight.data_ptr()),
                    ctypes.c_void_p(bias.data_ptr()),
                    ctypes.c_void_p(output.data_ptr()),
                    ctypes.c_void_p(0 if workspace is None else workspace.data_ptr()),
                    workspace_bytes,
                    ctypes.c_void_p(stream.cuda_stream),
                )
                if status != 0:
                    raise RuntimeError(f"candidate {candidate.heuristic_index} execution failed")

            for _ in range(args.warmup):
                execute()
            torch.cuda.synchronize(device)
            samples = []
            for _ in range(args.iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(stream)
                execute()
                end.record(stream)
                end.synchronize()
                samples.append(float(start.elapsed_time(end)))
            result = candidate.as_dict()
            result["latency"] = latency_summary(samples)
            result["metrics"] = tensor_metrics(output, reference)
            results.append(result)
            print(
                f"h{candidate.heuristic_index}: {result['latency']['median_ms']:.4f} ms "
                f"exact={result['metrics']['bitwise_exact']} "
                f"rate={result['metrics']['exact_rate']:.9f}",
                flush=True,
            )
        finally:
            probe.library.trtmc_wan22_linear_probe_destroy(context)

    exact = [result for result in results if result["metrics"]["bitwise_exact"]]
    report = {
        "kind": "wan2_2_ti2v_official_call92_fp32_time_linear_qualification",
        "status": "PASS" if exact else "FAIL",
        "stage": args.stage,
        "device": torch.cuda.get_device_name(device),
        "shape": {"m": m, "n": n, "k": k},
        "contract": {
            "input": specification["x"],
            "weight": specification["weight"],
            "bias": specification["bias"],
            "reference": specification["reference"],
            "dtype": "float32",
            "compute": "CUBLAS_COMPUTE_32F",
            "tf32": False,
            "epilogue": "CUBLASLT_EPILOGUE_BIAS",
            "workspace_limit_mib": args.workspace_mib,
        },
        "capture_manifest_sha256": sha256_file(args.capture_dir / "manifest.json"),
        "plugin": {
            "path": str(args.plugin.resolve()),
            "sha256": sha256_file(args.plugin),
        },
        "candidate_count": len(results),
        "candidates": results,
        "fastest_exact": (
            min(exact, key=lambda item: item["latency"]["median_ms"]) if exact else None
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(args.report)}, indent=2))
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
