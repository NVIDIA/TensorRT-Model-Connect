#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Localize Wan2.2 UMT5 FFN accuracy on a saved exact block input.

The diagnostic builds one isolated FFN with the production TensorRT helpers
and compares every materialized boundary against the official PyTorch formula.
It never builds or loads another full UMT5 engine.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import resource
import time
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np
import tensorrt as trt
import torch
import torch.nn.functional as functional

from tensorrt_model_connect.families.wan2_2_ti2v.umt5_encoder_builder import (
    WAN22_UMT5_XXL,
    _bf16_barrier,
    _gelu_bf16,
    _linear_bf16,
    _mark_fp32_debug_output,
    _rms_norm_bf16,
    _source_rms_norm_bf16,
)


_STAGE_SHAPES = {
    "norm2": (1, 512, 4096),
    "fc1": (1, 512, 10240),
    "gate_input": (1, 512, 10240),
    "gelu": (1, 512, 10240),
    "gated_product": (1, 512, 10240),
    "fc2": (1, 512, 4096),
    "residual": (1, 512, 4096),
}


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
    reference_fp32 = reference.detach().float().cpu()
    actual_fp32 = actual.detach().float().cpu()
    delta = actual_fp32 - reference_fp32
    reference_bf16 = reference_fp32.to(torch.bfloat16)
    actual_bf16 = actual_fp32.to(torch.bfloat16)
    mismatch_count = int(torch.count_nonzero(reference_bf16 != actual_bf16))
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


def _native_bf16_numpy(value: torch.Tensor) -> np.ndarray:
    value = value.detach().cpu().contiguous()
    if value.dtype != torch.bfloat16:
        raise TypeError(f"Expected BF16 checkpoint tensor, got {value.dtype}")
    return value.view(torch.uint16).numpy().view(ml_dtypes.bfloat16).reshape(value.shape)


def _layer_weights(
    checkpoint: Path, layer: int
) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]:
    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    prefix = f"blocks.{layer}"
    names = {
        "norm2": f"{prefix}.norm2.weight",
        "fc1": f"{prefix}.ffn.fc1.weight",
        "gate": f"{prefix}.ffn.gate.0.weight",
        "fc2": f"{prefix}.ffn.fc2.weight",
    }
    tensors = {name: state[key].detach().cpu() for name, key in names.items()}
    arrays = {name: _native_bf16_numpy(value) for name, value in tensors.items()}
    return tensors, arrays


def _build_plan(weights: dict[str, np.ndarray], *, source_rmsnorm: bool = False) -> bytes:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 0
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    hidden = network.add_input("hidden", trt.bfloat16, (512, 4096))
    refs: list[np.ndarray] = []

    rms_norm = _source_rms_norm_bf16 if source_rmsnorm else _rms_norm_bf16
    normed = rms_norm(
        network,
        hidden,
        weights["norm2"],
        WAN22_UMT5_XXL.hidden_size,
        WAN22_UMT5_XXL.epsilon,
        refs,
        trt,
    )
    _mark_fp32_debug_output(network, normed, "norm2", _STAGE_SHAPES["norm2"], trt)

    fc1 = _linear_bf16(network, normed, weights["fc1"], refs, trt)
    fc1 = _bf16_barrier(network, fc1, "wan22_umt5_ffn_probe_fc1", trt)
    _mark_fp32_debug_output(network, fc1, "fc1", _STAGE_SHAPES["fc1"], trt)

    gate_input = _linear_bf16(network, normed, weights["gate"], refs, trt)
    _mark_fp32_debug_output(
        network,
        gate_input,
        "gate_input",
        _STAGE_SHAPES["gate_input"],
        trt,
    )
    gelu = _gelu_bf16(
        network,
        gate_input,
        refs,
        trt,
        cuda_plugin_name="wan22_umt5_ffn_probe_source_gelu",
    )
    _mark_fp32_debug_output(network, gelu, "gelu", _STAGE_SHAPES["gelu"], trt)

    gated = network.add_elementwise(fc1, gelu, trt.ElementWiseOperation.PROD).get_output(0)
    gated = _bf16_barrier(network, gated, "wan22_umt5_ffn_probe_gated_product", trt)
    _mark_fp32_debug_output(
        network,
        gated,
        "gated_product",
        _STAGE_SHAPES["gated_product"],
        trt,
    )

    fc2 = _linear_bf16(network, gated, weights["fc2"], refs, trt)
    fc2 = _bf16_barrier(network, fc2, "wan22_umt5_ffn_probe_fc2", trt)
    _mark_fp32_debug_output(network, fc2, "fc2", _STAGE_SHAPES["fc2"], trt)
    residual = network.add_elementwise(hidden, fc2, trt.ElementWiseOperation.SUM).get_output(0)
    residual = _bf16_barrier(network, residual, "wan22_umt5_ffn_probe_residual", trt)
    _mark_fp32_debug_output(
        network,
        residual,
        "residual",
        _STAGE_SHAPES["residual"],
        trt,
    )

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT failed to build the UMT5 FFN probe")
    return bytes(plan)


def _official_stages(
    hidden: torch.Tensor,
    weights: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    x = hidden.to(device=device, dtype=torch.bfloat16)
    norm_weight = weights["norm2"].to(device=device)
    normalized = x * torch.rsqrt(
        x.float().pow(2).mean(dim=-1, keepdim=True) + WAN22_UMT5_XXL.epsilon
    )
    normalized = normalized.type_as(norm_weight)
    norm2 = norm_weight * normalized
    fc1 = functional.linear(norm2, weights["fc1"].to(device=device))
    gate_input = functional.linear(norm2, weights["gate"].to(device=device))
    gelu = (
        0.5
        * gate_input
        * (
            1.0
            + torch.tanh(
                math.sqrt(2.0 / math.pi) * (gate_input + 0.044715 * torch.pow(gate_input, 3.0))
            )
        )
    )
    gated = fc1 * gelu
    fc2 = functional.linear(gated, weights["fc2"].to(device=device))
    residual = x + fc2
    return {
        "norm2": norm2,
        "fc1": fc1,
        "gate_input": gate_input,
        "gelu": gelu,
        "gated_product": gated,
        "fc2": fc2,
        "residual": residual,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--input-key", required=True)
    parser.add_argument("--official-report", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-rmsnorm", action="store_true")
    args = parser.parse_args()
    for name in ("checkpoint", "plugin", "inputs", "official_report"):
        path = getattr(args, name).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        setattr(args, name, path)
    args.plan = args.plan.resolve()
    args.report = args.report.resolve()

    saved = torch.load(args.inputs, map_location="cpu", weights_only=True)
    if args.input_key not in saved:
        raise KeyError(args.input_key)
    case = saved[args.input_key]
    hidden = case["hidden"].to(torch.bfloat16).contiguous()
    if tuple(hidden.shape) != (1, 512, 4096):
        raise ValueError(f"Expected hidden [1,512,4096], got {tuple(hidden.shape)}")
    layer = int(case["layer"])
    token_count = int(case["token_count"])
    prompt = str(case["prompt"])
    official_report = json.loads(args.official_report.read_text())

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    plugin_library = ctypes.CDLL(str(args.plugin), mode=ctypes.RTLD_GLOBAL)
    torch_weights, trt_weights = _layer_weights(args.checkpoint, layer)
    reference = _official_stages(hidden, torch_weights, device)

    begin = time.perf_counter()
    plan = _build_plan(trt_weights, source_rmsnorm=args.source_rmsnorm)
    build_seconds = time.perf_counter() - begin
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    args.plan.write_bytes(plan)
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(plan)
    if engine is None:
        raise RuntimeError("Could not deserialize the UMT5 FFN probe")
    context = engine.create_execution_context()
    input_device = hidden.squeeze(0).to(device=device)
    outputs = {
        name: torch.empty(shape, device=device, dtype=torch.float32)
        for name, shape in _STAGE_SHAPES.items()
    }
    if not context.set_tensor_address("hidden", input_device.data_ptr()):
        raise RuntimeError("Could not bind hidden")
    for name, value in outputs.items():
        if not context.set_tensor_address(name, value.data_ptr()):
            raise RuntimeError(f"Could not bind {name}")
    stream = torch.cuda.current_stream(device=device)
    if not context.execute_async_v3(stream_handle=stream.cuda_stream):
        raise RuntimeError("TensorRT UMT5 FFN probe failed")
    torch.cuda.synchronize(device)

    stages = {}
    first_non_exact = None
    for name in _STAGE_SHAPES:
        full = _metrics(reference[name], outputs[name])
        real = _metrics(
            reference[name][:, :token_count],
            outputs[name][:, :token_count],
        )
        if first_non_exact is None and not real["bf16_exact"]:
            first_non_exact = name
        stages[name] = {"full": full, "real_tokens": real}

    official_layer = official_report["prompts"][prompt]["layers"][str(layer)]
    reference_residual_matches_official = (
        stages["residual"]["real_tokens"]["reference_bf16_sha256"]
        == official_layer["real_token_rows"]["reference_bf16_sha256"]
    )
    actual_residual_matches_full_plan = (
        stages["residual"]["real_tokens"]["actual_bf16_sha256"]
        == official_layer["real_token_rows"]["actual_bf16_sha256"]
    )
    actual_residual_matches_reference = stages["residual"]["real_tokens"]["bf16_exact"]
    if args.source_rmsnorm:
        passed = (
            reference_residual_matches_official
            and actual_residual_matches_reference
            and first_non_exact is None
        )
    else:
        passed = (
            reference_residual_matches_official
            and actual_residual_matches_full_plan
            and first_non_exact is not None
        )
    status = "passed" if passed else "failed"
    report = {
        "kind": "wan2_2_ti2v_umt5_ffn_stage_localization",
        "completed_at": _now(),
        "status": status,
        "device": torch.cuda.get_device_name(device),
        "device_index": device.index,
        "torch_version": torch.__version__,
        "input_key": args.input_key,
        "prompt": prompt,
        "layer": layer,
        "token_count": token_count,
        "input_real_bf16_sha256": _tensor_bf16_sha256(hidden[:, :token_count]),
        "checkpoint": str(args.checkpoint),
        "plugin": str(args.plugin),
        "plugin_sha256": _sha256_file(args.plugin),
        "source_rmsnorm": args.source_rmsnorm,
        "plan": str(args.plan),
        "plan_size_bytes": len(plan),
        "plan_sha256": hashlib.sha256(plan).hexdigest(),
        "build_seconds": build_seconds,
        "first_non_exact_real_boundary": first_non_exact,
        "reference_residual_matches_official": reference_residual_matches_official,
        "actual_residual_matches_full_plan": actual_residual_matches_full_plan,
        "actual_residual_matches_reference": actual_residual_matches_reference,
        "stages": stages,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    assert plugin_library is not None
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
