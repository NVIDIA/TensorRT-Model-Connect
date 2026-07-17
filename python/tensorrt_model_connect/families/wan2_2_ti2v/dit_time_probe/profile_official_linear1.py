#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay and profile the first official time linear in isolation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    def load(name: str) -> torch.Tensor:
        return torch.from_numpy(
            np.ascontiguousarray(np.load(args.capture_dir / f"{name}.npy", allow_pickle=False))
        ).to(device)

    x = load("time_features").reshape(-1, 256)
    weight = load("time_linear1_weight")
    bias = load("time_linear1_bias")
    reference = load("time_linear1").reshape(-1, 3072)
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float32):
        replay = torch.nn.functional.linear(x, weight, bias)
    torch.cuda.synchronize(device)
    exact = replay.view(torch.int32) == reference.view(torch.int32)
    delta = replay.double() - reference.double()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as profile:
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float32):
            profiled = torch.nn.functional.linear(x, weight, bias)
        torch.cuda.synchronize(device)
    if not torch.equal(profiled.view(torch.int32), reference.view(torch.int32)):
        raise RuntimeError("Profiled official linear replay changed output bits")
    events = []
    for event in profile.events():
        if event.device_type != torch.autograd.DeviceType.CPU or event.name.startswith("aten::"):
            events.append(
                {
                    "name": event.name,
                    "device_type": str(event.device_type),
                    "cpu_time_us": float(event.cpu_time_total),
                    "device_time_us": float(event.device_time_total),
                    "input_shapes": event.input_shapes,
                }
            )
    report = {
        "kind": "wan2_2_ti2v_official_time_linear1_isolated_replay",
        "device": torch.cuda.get_device_name(device),
        "shape": {"m": 27280, "n": 3072, "k": 256},
        "dtypes": {
            "x": str(x.dtype),
            "weight": str(weight.dtype),
            "bias": str(bias.dtype),
            "output": str(replay.dtype),
        },
        "torch_backends": {
            "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "allow_fp16_reduced_precision_reduction": bool(
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
            ),
            "allow_bf16_reduced_precision_reduction": bool(
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            ),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "metrics": {
            "bitwise_exact": bool(torch.all(exact).item()),
            "exact_elements": int(torch.count_nonzero(exact).item()),
            "total_elements": reference.numel(),
            "max_abs_error": float(delta.abs().max().item()),
            "mean_abs_error": float(delta.abs().mean().item()),
            "rmse": float(delta.square().mean().sqrt().item()),
        },
        "profiler_events": events,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["metrics"]["bitwise_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
