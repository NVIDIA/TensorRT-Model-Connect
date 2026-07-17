#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diagnose FMA contraction at a Wan2.2 gated residual boundary."""

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


def load(path: Path, device: torch.device) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=True).reshape(27_280, 3_072).to(device)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual", type=Path, required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--trt-residual", type=Path)
    parser.add_argument("--trt-projection", type=Path)
    parser.add_argument("--modulation", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--trt-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage", choices=("self_attention", "ffn"), default="self_attention")
    parser.add_argument("--gate-chunk", type=int, choices=range(6), default=2)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    residual = load(args.residual, device)
    projection = load(args.projection, device)
    trt_residual = load(args.trt_residual or args.residual, device)
    trt_projection = load(args.trt_projection or args.projection, device)
    source_output = load(args.source_output, device)
    trt_output = load(args.trt_output, device)
    modulation = (
        torch.load(args.modulation, map_location="cpu", weights_only=True)
        .reshape(-1, 18_432)[:1]
        .to(device)
    )
    gate_offset = args.gate_chunk * 3_072
    gate = modulation[:, gate_offset : gate_offset + 3_072]

    gated = projection * gate
    separate = residual + gated
    fused = torch.addcmul(residual, projection, gate)
    torch.cuda.synchronize(device)
    report = {
        "kind": f"wan2_2_ti2v_{args.stage}_residual_boundary_diagnosis",
        "device": torch.cuda.get_device_name(device),
        "shape": [27_280, 3_072],
        "gate_chunk": args.gate_chunk,
        "source_expression": "residual + projection * gate",
        "carrier_comparisons": {
            "residual_source_vs_tensorrt": metrics(residual, trt_residual),
            "projection_source_vs_tensorrt": metrics(projection, trt_projection),
        },
        "comparisons_to_source": {
            "separate_multiply_then_add": metrics(source_output, separate),
            "fused_multiply_add": metrics(source_output, fused),
            "current_tensorrt": metrics(source_output, trt_output),
        },
        "comparisons_to_current_tensorrt": {
            "separate_multiply_then_add": metrics(trt_output, separate),
            "fused_multiply_add": metrics(trt_output, fused),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
