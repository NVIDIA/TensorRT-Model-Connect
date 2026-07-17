#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diagnose the first post-LayerNorm FP32 boundary in Wan2.2 block 0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


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
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalized", type=Path, required=True)
    parser.add_argument("--modulation", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--trt-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    normalized = (
        torch.load(args.normalized, map_location="cpu", weights_only=True)
        .reshape(27_280, 3_072)
        .to(device)
    )
    modulation = (
        torch.load(args.modulation, map_location="cpu", weights_only=True)
        .reshape(-1, 18_432)[:1]
        .to(device)
    )
    source_output = (
        torch.load(args.source_output, map_location="cpu", weights_only=True)
        .reshape(27_280, 3_072)
        .to(device)
    )
    trt_output = (
        torch.load(args.trt_output, map_location="cpu", weights_only=True)
        .reshape(27_280, 3_072)
        .to(device)
    )
    shift = modulation[:, :3_072]
    scale = modulation[:, 3_072:6_144]

    # These assignments intentionally force the three upstream eager FP32
    # materialization boundaries instead of leaving one fusable expression.
    scale_plus_one = scale + 1.0
    multiplied = normalized * scale_plus_one
    separate = multiplied + shift
    fused_multiply_add = torch.addcmul(shift, normalized, scale_plus_one)
    torch.cuda.synchronize(device)

    report = {
        "kind": "wan2_2_ti2v_adaptive_norm_boundary_diagnosis",
        "device": torch.cuda.get_device_name(device),
        "shape": [27_280, 3_072],
        "source_expression": "normalized * (1 + scale) + shift",
        "comparisons_to_source": {
            "three_materialized_fp32_boundaries": metrics(source_output, separate),
            "fused_multiply_add": metrics(source_output, fused_multiply_add),
            "current_tensorrt": metrics(source_output, trt_output),
        },
        "comparisons_to_current_tensorrt": {
            "three_materialized_fp32_boundaries": metrics(trt_output, separate),
            "fused_multiply_add": metrics(trt_output, fused_multiply_add),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
