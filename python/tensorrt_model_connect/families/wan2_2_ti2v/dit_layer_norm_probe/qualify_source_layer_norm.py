#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualify the CUDA-only Wan2.2 block-0 LayerNorm against a saved source call."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path

import torch


BASELINE_MISMATCHES = 1_551
BASELINE_RMSE = 1.8264307072968222e-05
BASELINE_MAX_ABS = 0.0625


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    input_tensor = torch.load(args.input, map_location="cpu", weights_only=True).reshape(
        27_280, 3_072
    )
    reference = torch.load(args.reference, map_location="cpu", weights_only=True).reshape(
        27_280, 3_072
    )
    input_tensor = input_tensor.to(device=device, dtype=torch.float32)
    reference = reference.to(device=device, dtype=torch.float32)
    output_fp32 = torch.empty_like(input_tensor)

    library = ctypes.CDLL(str(args.plugin.resolve()), mode=ctypes.RTLD_GLOBAL)
    launch = library.trtmc_wan22_dit_layer_norm_fp32_launch
    launch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_float,
        ctypes.c_void_p,
    ]
    launch.restype = ctypes.c_int
    stream = torch.cuda.current_stream(device)
    status = launch(
        ctypes.c_void_p(input_tensor.data_ptr()),
        ctypes.c_void_p(output_fp32.data_ptr()),
        27_280,
        3_072,
        ctypes.c_float(1.0e-6),
        ctypes.c_void_p(stream.cuda_stream),
    )
    if status != 0:
        raise RuntimeError(f"Wan22 LayerNorm launch failed with status {status}")
    torch.cuda.synchronize(device)

    # Upstream WanLayerNorm casts the FP32 native_layer_norm result back to the
    # BF16 dtype of block-0 input.  Compare those materialized values directly.
    candidate = output_fp32.to(torch.bfloat16).float()
    result = metrics(reference, candidate)
    fallback_pass = (
        result["mismatched_elements"] <= BASELINE_MISMATCHES // 10
        and result["rmse"] <= BASELINE_RMSE / 4.0
        and result["max_abs_error"] <= BASELINE_MAX_ABS
    )
    report = {
        "kind": "wan2_2_ti2v_source_exact_layer_norm_qualification",
        "device": torch.cuda.get_device_name(device),
        "shape": [27_280, 3_072],
        "epsilon": 1.0e-6,
        "source_semantics": "FP32 vectorized LayerNorm followed by BF16 materialization",
        "predefined_gate": {
            "primary": "bitwise exact over all 83,804,160 elements",
            "fallback": {
                "mismatched_elements_max": BASELINE_MISMATCHES // 10,
                "rmse_max": BASELINE_RMSE / 4.0,
                "max_abs_error_max": BASELINE_MAX_ABS,
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
