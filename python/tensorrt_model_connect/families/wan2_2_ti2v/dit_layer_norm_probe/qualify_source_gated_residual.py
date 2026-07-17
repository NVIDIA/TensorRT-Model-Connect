#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualify the non-FMA CUDA Wan2.2 self-attention gated residual."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path

import torch


ROWS = 27_280
COLUMNS = 3_072
TOTAL_ELEMENTS = ROWS * COLUMNS
BASELINES = {
    "self_attention": {
        "mismatches": 14_659_535,
        "rmse": 1.4739489628823326e-08,
        "max_abs": 3.814697265625e-06,
    },
    "ffn": {
        "mismatches": 25_622_579,
        "rmse": 1.5425548838265968e-08,
        "max_abs": 3.814697265625e-06,
    },
}


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
        "cosine_similarity": float(
            torch.nn.functional.cosine_similarity(
                reference.flatten().double(), candidate.flatten().double(), dim=0
            )
        ),
    }


def load_rows(path: Path, device: torch.device) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=True).reshape(ROWS, COLUMNS).to(device)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--residual", type=Path, required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--modulation", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage", choices=tuple(BASELINES), default="self_attention")
    parser.add_argument("--gate-chunk", type=int, choices=range(6), default=2)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    residual = load_rows(args.residual, device)
    projection = load_rows(args.projection, device)
    reference = load_rows(args.reference, device)
    modulation = (
        torch.load(args.modulation, map_location="cpu", weights_only=True)
        .reshape(-1, 18_432)[:1]
        .to(device)
    )
    gate_offset = args.gate_chunk * COLUMNS
    gate = modulation[:, gate_offset : gate_offset + COLUMNS].contiguous()
    candidate = torch.empty_like(reference)

    for name, value in {
        "residual": residual,
        "projection": projection,
        "gate": gate,
        "reference": reference,
    }.items():
        if value.dtype != torch.float32:
            raise TypeError(f"{name} must be FP32, got {value.dtype}")

    library = ctypes.CDLL(str(args.plugin.resolve()), mode=ctypes.RTLD_GLOBAL)
    launch = library.trtmc_wan22_dit_gated_residual_fp32_launch
    launch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_void_p,
    ]
    launch.restype = ctypes.c_int
    stream = torch.cuda.current_stream(device)
    status = launch(
        ctypes.c_void_p(residual.data_ptr()),
        ctypes.c_void_p(projection.data_ptr()),
        ctypes.c_void_p(gate.data_ptr()),
        ctypes.c_void_p(candidate.data_ptr()),
        ROWS,
        COLUMNS,
        ctypes.c_void_p(stream.cuda_stream),
    )
    if status != 0:
        raise RuntimeError(f"Wan22 gated residual launch failed with status {status}")
    torch.cuda.synchronize(device)

    result = metrics(reference, candidate)
    baseline = BASELINES[args.stage]
    fallback_pass = (
        result["mismatched_elements"] <= baseline["mismatches"] // 10
        and result["rmse"] <= baseline["rmse"] / 4.0
        and result["max_abs_error"] <= baseline["max_abs"]
    )
    report = {
        "kind": f"wan2_2_ti2v_source_exact_{args.stage}_gated_residual_qualification",
        "device": torch.cuda.get_device_name(device),
        "shape": [ROWS, COLUMNS],
        "gate_chunk": args.gate_chunk,
        "source_semantics": "separately materialized FP32 projection * gate then residual + update",
        "predefined_gate": {
            "primary": f"bitwise exact over all {TOTAL_ELEMENTS:,} elements",
            "fallback": {
                "mismatched_elements_max": baseline["mismatches"] // 10,
                "rmse_max": baseline["rmse"] / 4.0,
                "max_abs_error_max": baseline["max_abs"],
            },
        },
        "metrics": result,
        "gate": {
            "primary_pass": bool(result["bitwise_exact"]),
            "fallback_pass": fallback_pass,
            "pass": bool(result["bitwise_exact"] or fallback_pass),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["gate"]["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
