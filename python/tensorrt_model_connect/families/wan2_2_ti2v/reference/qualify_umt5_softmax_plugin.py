#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualify the Wan2.2 exact-softmax plugin on saved BF16 logits.

This diagnostic builds only a one-layer TensorRT network around
``Wan22Umt5SourceSoftmax``.  It never loads UMT5 weights and is not a runtime
dependency.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any

from tensorrt_model_connect.trt_compat import trt
import torch
import torch.nn.functional as functional


_SHAPE = (1, 64, 512, 512)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_bf16_sha256(value: torch.Tensor) -> str:
    raw = (
        value.detach()
        .to(device="cpu", dtype=torch.bfloat16)
        .contiguous()
        .view(torch.uint16)
        .numpy()
        .astype("<u2", copy=False)
        .tobytes()
    )
    return hashlib.sha256(raw).hexdigest()


def _metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    delta = actual.float() - reference.float()
    mismatch_count = int(torch.count_nonzero(reference != actual))
    return {
        "shape": list(reference.shape),
        "bf16_exact": mismatch_count == 0,
        "bf16_mismatch_count": mismatch_count,
        "max_abs_error": float(delta.abs().max()),
        "mean_abs_error": float(delta.abs().mean()),
        "rmse": float(delta.square().mean().sqrt()),
        "reference_bf16_sha256": _tensor_bf16_sha256(reference),
        "actual_bf16_sha256": _tensor_bf16_sha256(actual),
    }


def _build_plan() -> bytes:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    logits = network.add_input("biased_logits", trt.bfloat16, _SHAPE)
    creator = trt.get_plugin_registry().get_creator("Wan22Umt5SourceSoftmax", "1", "")
    if creator is None:
        raise RuntimeError("Wan22Umt5SourceSoftmax plugin creator is not registered")
    plugin = creator.create_plugin("wan22_umt5_softmax_microprobe", trt.PluginFieldCollection([]))
    layer = network.add_plugin_v2([logits], plugin)
    if layer is None:
        raise RuntimeError("Could not add Wan22Umt5SourceSoftmax")
    probabilities = layer.get_output(0)
    probabilities.name = "probabilities"
    network.mark_output(probabilities)
    config = builder.create_builder_config()
    config.builder_optimization_level = 0
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT failed to build the UMT5 softmax microplan")
    return bytes(plan)


def _load_cases(path: Path) -> dict[str, dict[str, Any]]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError("Saved logits must be a dictionary")
    cases = {}
    for name in ("positive", "negative"):
        entry = value.get(name)
        if not isinstance(entry, dict):
            raise TypeError(f"Saved logits entry {name!r} must be a dictionary")
        logits = entry.get("biased_logits")
        token_count = entry.get("token_count")
        if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != _SHAPE:
            raise ValueError(f"Saved {name} logits must have shape {_SHAPE}")
        if not isinstance(token_count, int) or not 0 < token_count <= 512:
            raise ValueError(f"Saved {name} token_count is invalid: {token_count!r}")
        cases[name] = {
            "biased_logits": logits.to(dtype=torch.bfloat16).contiguous(),
            "token_count": token_count,
        }
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()

    args.plugin = args.plugin.resolve()
    args.logits = args.logits.resolve()
    args.plan = args.plan.resolve()
    args.report = args.report.resolve()
    if not args.plugin.is_file():
        raise FileNotFoundError(args.plugin)
    if not args.logits.is_file():
        raise FileNotFoundError(args.logits)
    if args.iterations < 1:
        parser.error("--iterations must be positive")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    plugin_library = ctypes.CDLL(str(args.plugin), mode=ctypes.RTLD_GLOBAL)
    cases = _load_cases(args.logits)

    begin = time.perf_counter()
    plan = _build_plan()
    build_seconds = time.perf_counter() - begin
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    args.plan.write_bytes(plan)

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(plan)
    if engine is None:
        raise RuntimeError("Could not deserialize the UMT5 softmax microplan")
    context = engine.create_execution_context()
    stream = torch.cuda.current_stream(device=device)
    case_reports = {}
    all_exact = True
    for name, entry in cases.items():
        logits = entry["biased_logits"].to(device=device)
        probabilities = torch.empty_like(logits)
        if not context.set_tensor_address("biased_logits", logits.data_ptr()):
            raise RuntimeError("Could not bind biased_logits")
        if not context.set_tensor_address("probabilities", probabilities.data_ptr()):
            raise RuntimeError("Could not bind probabilities")

        if not context.execute_async_v3(stream_handle=stream.cuda_stream):
            raise RuntimeError(f"TensorRT softmax failed for {name}")
        torch.cuda.synchronize(device)
        begin = time.perf_counter()
        for _ in range(args.iterations):
            if not context.execute_async_v3(stream_handle=stream.cuda_stream):
                raise RuntimeError(f"TensorRT softmax failed for {name}")
        torch.cuda.synchronize(device)
        trt_seconds = time.perf_counter() - begin

        begin = time.perf_counter()
        reference = None
        for _ in range(args.iterations):
            reference = functional.softmax(logits.float(), dim=-1).to(dtype=torch.bfloat16)
        torch.cuda.synchronize(device)
        torch_seconds = time.perf_counter() - begin
        assert reference is not None

        token_count = entry["token_count"]
        full = _metrics(reference, probabilities)
        real = _metrics(
            reference[:, :, :token_count, :token_count],
            probabilities[:, :, :token_count, :token_count],
        )
        all_exact = all_exact and full["bf16_exact"]
        case_reports[name] = {
            "token_count": token_count,
            "input_bf16_sha256": _tensor_bf16_sha256(logits),
            "full": full,
            "real_tokens": real,
            "trt_mean_seconds": trt_seconds / args.iterations,
            "torch_mean_seconds": torch_seconds / args.iterations,
        }

    report = {
        "kind": "wan2_2_ti2v_umt5_source_softmax_microqualification",
        "completed_at": _now(),
        "status": "passed" if all_exact else "failed",
        "device": torch.cuda.get_device_name(device),
        "device_index": device.index,
        "torch_version": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "plugin": str(args.plugin),
        "plugin_sha256": _sha256_file(args.plugin),
        "saved_logits": str(args.logits),
        "saved_logits_sha256": _sha256_file(args.logits),
        "plan": str(args.plan),
        "plan_size_bytes": len(plan),
        "plan_sha256": hashlib.sha256(plan).hexdigest(),
        "build_seconds": build_seconds,
        "iterations": args.iterations,
        "cases": case_reports,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    assert plugin_library is not None
    return 0 if all_exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
