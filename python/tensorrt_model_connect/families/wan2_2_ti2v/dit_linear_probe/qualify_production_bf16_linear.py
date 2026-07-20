#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualify the production Wan2.2 BF16 linear plugin on all five real shapes."""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tensorrt_model_connect.trt_compat import trt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qualify_all_linear_shapes as shapes  # noqa: E402
import qualify_block0_ffn2 as base  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    return parser.parse_args()


def build_plan(
    plugin_path: Path,
    metadata: dict[str, Any],
    output_path: Path,
) -> bytes:
    ctypes.CDLL(str(plugin_path.resolve()), mode=ctypes.RTLD_GLOBAL)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    m, n, k = metadata["m"], metadata["n"], metadata["k"]
    x = network.add_input("x", trt.bfloat16, (m, k))
    weight = network.add_input("weight", trt.bfloat16, (n, k))
    bias = network.add_input("bias", trt.bfloat16, (n,))
    creator = trt.get_plugin_registry().get_creator("Wan22DitBf16Linear", "1", "")
    if creator is None:
        raise RuntimeError("Wan22DitBf16Linear creator is not registered")
    storage, fields = base.plugin_fields({"m": m, "n": n, "k": k})
    plugin = creator.create_plugin(f"wan22_production_{m}_{k}_{n}", fields)
    if plugin is None:
        raise RuntimeError(f"Could not create production plugin for {(m, k, n)}")
    layer = network.add_plugin_v2([x, weight, bias], plugin)
    del storage
    if layer is None:
        raise RuntimeError(f"Could not add production plugin for {(m, k, n)}")
    output = layer.get_output(0)
    output.name = "output"
    network.mark_output(output)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024**3)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"Could not build production plugin plan for {(m, k, n)}")
    plan = bytes(serialized)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(plan)
    return plan


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("Invalid benchmark iterations")
    if not args.plugin.is_file():
        raise FileNotFoundError(args.plugin)
    if not shapes.all_captures_exist(args.capture_dir):
        raise FileNotFoundError(f"Missing five-shape captures in {args.capture_dir}")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    ctypes.CDLL(str(args.plugin.resolve()), mode=ctypes.RTLD_GLOBAL)

    results: dict[str, Any] = {}
    benchmark_args = SimpleNamespace(warmup=args.warmup, iterations=args.iterations)
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        for name in shapes.SHAPES:
            capture = shapes.capture_path(args.capture_dir, name)
            tensors, metadata = shapes.load_shape_capture(capture, device)
            plan_path = args.plan_dir / f"{name}.plan"
            plan = build_plan(args.plugin, metadata, plan_path)
            qualification = shapes.benchmark_plan(plan, tensors, metadata, benchmark_args, device)
            results[name] = {
                "metadata": metadata,
                "capture": str(capture.resolve()),
                "plan": str(plan_path.resolve()),
                "plan_bytes": len(plan),
                **qualification,
            }
            print(
                f"{name}: exact={qualification['metrics']['bit_exact']} "
                f"median={qualification['latency']['median_ms']:.6f} ms",
                flush=True,
            )
            del tensors
            torch.cuda.empty_cache()

    passed = all(result["metrics"]["bit_exact"] for result in results.values())
    report = {
        "kind": "wan2_2_ti2v_production_bf16_linear_qualification",
        "status": "PASS" if passed else "FAIL",
        "hardware": {
            "device": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
        },
        "plugin": str(args.plugin.resolve()),
        "dependencies": base.dependency_audit(args.plugin),
        "contract": {
            "source": "five captured official real-module BF16 operands and outputs",
            "selection": "target-local first zero-workspace split_k=1 reduction=0 candidate",
            "warmup": args.warmup,
            "iterations": args.iterations,
        },
        "shapes": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(args.report)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
