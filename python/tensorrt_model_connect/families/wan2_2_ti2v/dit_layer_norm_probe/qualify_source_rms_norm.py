#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualify the CUDA Wan2.2 RMSNorm against a saved source call."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path

import torch
from safetensors import safe_open


BASELINE_MISMATCHES = 750
BASELINE_RMSE = 2.1559540982707404e-05
BASELINE_MAX_ABS = 0.04245758056640625


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
    parser.add_argument("--q-linear", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gamma", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    input_fp32 = torch.load(args.q_linear, map_location="cpu", weights_only=True).reshape(-1, 3_072)
    rows = int(input_fp32.shape[0])
    if rows not in {27_280, 512}:
        raise ValueError(f"RMSNorm qualification rows must be 27,280 or 512, got {rows}")
    input_bf16 = input_fp32.to(device=device, dtype=torch.bfloat16)
    reference = (
        torch.load(args.reference, map_location="cpu", weights_only=True)
        .reshape(rows, 3_072)
        .to(device)
    )
    if args.gamma is not None:
        gamma = torch.load(args.gamma, map_location="cpu", weights_only=True)
    else:
        if args.checkpoint is None:
            raise ValueError("Either --gamma or --checkpoint is required")
        shard = args.checkpoint / "diffusion_pytorch_model-00001-of-00003.safetensors"
        with safe_open(shard, framework="pt", device="cpu") as checkpoint:
            gamma = checkpoint.get_tensor("blocks.0.self_attn.norm_q.weight")
    gamma = gamma.reshape(1, 3_072).to(device=device, dtype=torch.float32)
    source_means = input_bf16.float().pow(2).mean(dim=-1)
    candidate = torch.empty_like(reference)
    candidate_means = torch.empty(rows, device=device, dtype=torch.float32)

    library = ctypes.CDLL(str(args.plugin.resolve()), mode=ctypes.RTLD_GLOBAL)
    launch = library.trtmc_wan22_dit_rms_norm_fp32_launch
    launch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
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
        ctypes.c_void_p(input_bf16.data_ptr()),
        ctypes.c_void_p(gamma.data_ptr()),
        ctypes.c_void_p(candidate.data_ptr()),
        ctypes.c_void_p(candidate_means.data_ptr()),
        rows,
        3_072,
        ctypes.c_float(1.0e-6),
        ctypes.c_void_p(stream.cuda_stream),
    )
    if status != 0:
        raise RuntimeError(f"Wan22 RMSNorm launch failed with status {status}")
    torch.cuda.synchronize(device)

    mean_result = metrics(source_means, candidate_means)
    output_result = metrics(reference, candidate)
    fallback_pass = rows == 27_280 and (
        output_result["mismatched_elements"] <= BASELINE_MISMATCHES // 10
        and output_result["rmse"] <= BASELINE_RMSE / 4.0
        and output_result["max_abs_error"] <= BASELINE_MAX_ABS
        and mean_result["rmse"] < 4.011356509181496e-07
    )
    primary_pass = bool(mean_result["bitwise_exact"] and output_result["bitwise_exact"])
    report = {
        "kind": "wan2_2_ti2v_source_exact_rms_norm_qualification",
        "device": torch.cuda.get_device_name(device),
        "shape": [rows, 3_072],
        "epsilon": 1.0e-6,
        "source_semantics": "PyTorch 2.12 Reduce.cuh FP32 mean then BF16 normalized values and FP32 gamma",
        "predefined_gate": {
            "primary": f"bitwise exact row means and all {rows * 3_072:,} output elements",
            "fallback": {
                "mismatched_elements_max": BASELINE_MISMATCHES // 10,
                "rmse_max": BASELINE_RMSE / 4.0,
                "max_abs_error_max": BASELINE_MAX_ABS,
                "mean_rmse_must_improve": True,
            },
        },
        "mean_metrics": mean_result,
        "output_metrics": output_result,
        "gate": {
            "primary_pass": primary_pass,
            "fallback_pass": fallback_pass,
            "pass": primary_pass or fallback_pass,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["gate"]["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
