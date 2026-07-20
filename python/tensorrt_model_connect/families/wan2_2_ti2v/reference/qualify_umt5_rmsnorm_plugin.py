#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Microqualify the fixed-shape Wan2.2 UMT5 source RMSNorm plugin."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import time
from pathlib import Path

from tensorrt_model_connect.trt_compat import trt
import torch

from tensorrt_model_connect.families.wan2_2_ti2v.reference.qualify_umt5_block_stages import (
    _metrics,
    _sha256_file,
    _tensor_bf16_sha256,
)
from tensorrt_model_connect.families.wan2_2_ti2v.umt5_encoder_builder import (
    _mark_fp32_debug_output,
)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _build_plan() -> bytes:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 0
    hidden = network.add_input("hidden", trt.bfloat16, (512, 4096))
    gamma = network.add_input("gamma", trt.bfloat16, (4096,))
    creator = trt.get_plugin_registry().get_creator("Wan22Umt5SourceRmsNorm", "1", "")
    if creator is None:
        raise RuntimeError("Wan22Umt5SourceRmsNorm plugin creator is not registered")
    plugin = creator.create_plugin("wan22_umt5_source_rmsnorm_micro", trt.PluginFieldCollection([]))
    layer = network.add_plugin_v2([hidden, gamma], plugin)
    if layer is None:
        raise RuntimeError("Could not add Wan22Umt5SourceRmsNorm plugin")
    _mark_fp32_debug_output(network, layer.get_output(0), "normalized", (1, 512, 4096), trt)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT failed to build the RMSNorm microplan")
    return bytes(plan)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()
    for name in ("checkpoint", "plugin", "cases", "manifest"):
        value = getattr(args, name).resolve()
        if not value.is_file():
            raise FileNotFoundError(value)
        setattr(args, name, value)
    args.plan = args.plan.resolve()
    args.report = args.report.resolve()
    if args.repetitions < 1:
        raise ValueError("--repetitions must be positive")

    manifest = json.loads(args.manifest.read_text())
    if _sha256_file(args.cases) != manifest["saved_cases_sha256"]:
        raise RuntimeError("Saved RMSNorm cases do not match their manifest")
    cases = torch.load(args.cases, map_location="cpu", weights_only=True)
    required_cases = manifest["acceptance"]["required_cases"]
    if sorted(cases) != sorted(required_cases):
        raise RuntimeError(f"Expected saved cases {required_cases}, got {sorted(cases)}")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    plugin_library = ctypes.CDLL(str(args.plugin), mode=ctypes.RTLD_GLOBAL)
    plan = _build_plan()
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    args.plan.write_bytes(plan)
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(plan)
    if engine is None:
        raise RuntimeError("Could not deserialize RMSNorm microplan")
    context = engine.create_execution_context()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)

    case_reports = {}
    all_passed = True
    for name in required_cases:
        case = cases[name]
        layer = int(case["layer"])
        token_count = int(case["token_count"])
        hidden = case["hidden"].to(device=device, dtype=torch.bfloat16).squeeze(0)
        expected = case["expected"].to(device=device, dtype=torch.bfloat16)
        gamma = state[f"blocks.{layer}.norm2.weight"].to(device=device)
        actual = torch.empty((1, 512, 4096), device=device, dtype=torch.float32)
        for binding_name, value in {
            "hidden": hidden,
            "gamma": gamma,
            "normalized": actual,
        }.items():
            if not context.set_tensor_address(binding_name, value.data_ptr()):
                raise RuntimeError(f"Could not bind {binding_name}")
        stream = torch.cuda.current_stream(device=device)
        hashes = []
        timings = []
        for _ in range(args.repetitions):
            begin = time.perf_counter()
            if not context.execute_async_v3(stream_handle=stream.cuda_stream):
                raise RuntimeError(f"RMSNorm microplan failed for {name}")
            torch.cuda.synchronize(device)
            timings.append(time.perf_counter() - begin)
            hashes.append(_tensor_bf16_sha256(actual))
        full = _metrics(expected, actual)
        real = _metrics(expected[:, :token_count], actual[:, :token_count])
        expected_manifest = manifest["cases"][name]
        input_hash_matches = (
            _tensor_bf16_sha256(case["hidden"]) == expected_manifest["input_full_bf16_sha256"]
        )
        expected_hash_matches = (
            _tensor_bf16_sha256(expected) == expected_manifest["expected_full_bf16_sha256"]
        )
        deterministic = len(set(hashes)) == 1
        passed = (
            input_hash_matches
            and expected_hash_matches
            and deterministic
            and full["bf16_exact"]
            and real["bf16_exact"]
            and full["max_abs_error"] == 0.0
            and full["mean_abs_error"] == 0.0
            and full["rmse"] == 0.0
        )
        all_passed = all_passed and passed
        case_reports[name] = {
            "status": "passed" if passed else "failed",
            "prompt": case["prompt"],
            "layer": layer,
            "token_count": token_count,
            "input_hash_matches_manifest": input_hash_matches,
            "expected_hash_matches_manifest": expected_hash_matches,
            "deterministic": deterministic,
            "repetitions": args.repetitions,
            "output_bf16_sha256": hashes[-1],
            "mean_seconds": sum(timings) / len(timings),
            "full_512_rows": full,
            "real_token_rows": real,
        }

    report = {
        "kind": "wan2_2_ti2v_umt5_source_rmsnorm_microqualification",
        "completed_at": _now(),
        "status": "passed" if all_passed else "failed",
        "device": torch.cuda.get_device_name(device),
        "device_index": device.index,
        "torch_version": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "checkpoint": str(args.checkpoint),
        "plugin": str(args.plugin),
        "plugin_sha256": _sha256_file(args.plugin),
        "cases": str(args.cases),
        "cases_sha256": _sha256_file(args.cases),
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256_file(args.manifest),
        "plan": str(args.plan),
        "plan_size_bytes": len(plan),
        "plan_sha256": hashlib.sha256(plan).hexdigest(),
        "acceptance": manifest["acceptance"],
        "case_reports": case_reports,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    assert plugin_library is not None
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
