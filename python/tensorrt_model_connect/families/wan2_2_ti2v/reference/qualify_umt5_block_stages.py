#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Localize one Wan2.2 UMT5 block from a saved exact block input.

This diagnostic is intentionally separate from the production qualification
harness.  It builds only one attention + FFN block, exposes already-materialized
boundaries, and accepts its localization result only when both block-output
hashes reproduce the official and full-plan qualification reports.
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
    _int32_constant,
    _linear_bf16,
    _mark_fp32_debug_output,
    _mask_attention_bias,
    _relative_attention_bias,
    _rms_norm_bf16,
    _self_attention_bf16,
    _source_rms_norm_bf16,
    relative_position_buckets,
)


_STAGE_SHAPES = {
    "attention_norm": (1, 512, 4096),
    "q": (1, 512, 4096),
    "k": (1, 512, 4096),
    "v": (1, 512, 4096),
    "qk_logits": (1, 64, 512, 512),
    "biased_logits": (1, 64, 512, 512),
    "probabilities": (1, 64, 512, 512),
    "pv_context": (1, 64, 512, 64),
    "attention_output": (1, 512, 4096),
    "attention_residual": (1, 512, 4096),
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


def _block_weights(
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
        "attention_norm": f"{prefix}.norm1.weight",
        "q": f"{prefix}.attn.q.weight",
        "k": f"{prefix}.attn.k.weight",
        "v": f"{prefix}.attn.v.weight",
        "o": f"{prefix}.attn.o.weight",
        "relative_attention_bias": f"{prefix}.pos_embedding.embedding.weight",
        "norm2": f"{prefix}.norm2.weight",
        "fc1": f"{prefix}.ffn.fc1.weight",
        "gate": f"{prefix}.ffn.gate.0.weight",
        "fc2": f"{prefix}.ffn.fc2.weight",
    }
    tensors = {name: state[key].detach().cpu() for name, key in names.items()}
    arrays = {name: _native_bf16_numpy(value) for name, value in tensors.items()}
    return tensors, arrays


def _mark(network: Any, value: Any, name: str) -> None:
    _mark_fp32_debug_output(network, value, name, _STAGE_SHAPES[name], trt)


def _build_plan(
    weights: dict[str, np.ndarray], *, layer: int, source_rmsnorm: bool = False
) -> bytes:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 0
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    hidden = network.add_input("hidden", trt.bfloat16, (512, 4096))
    attention_mask = network.add_input("attention_mask", trt.int32, (1, 512))
    refs: list[np.ndarray] = []

    rms_norm = _source_rms_norm_bf16 if source_rmsnorm else _rms_norm_bf16
    attention_norm = rms_norm(
        network,
        hidden,
        weights["attention_norm"],
        WAN22_UMT5_XXL.hidden_size,
        WAN22_UMT5_XXL.epsilon,
        refs,
        trt,
    )
    _mark(network, attention_norm, "attention_norm")
    buckets = relative_position_buckets(
        WAN22_UMT5_XXL.sequence_length,
        WAN22_UMT5_XXL.sequence_length,
        num_buckets=WAN22_UMT5_XXL.num_buckets,
        max_distance=WAN22_UMT5_XXL.relative_attention_max_distance,
    )
    position_bias = _relative_attention_bias(
        network,
        weights["relative_attention_bias"],
        _int32_constant(network, buckets),
        WAN22_UMT5_XXL,
        refs,
        trt,
    )
    position_bias = _mask_attention_bias(
        network,
        position_bias,
        attention_mask,
        WAN22_UMT5_XXL,
        refs,
        trt,
    )
    canonical_attention = {
        f"attention.{name}.weight": weights[name] for name in ("q", "k", "v", "o")
    }
    attention_output = _self_attention_bf16(
        network,
        attention_norm,
        position_bias,
        canonical_attention,
        "attention",
        WAN22_UMT5_XXL,
        refs,
        trt,
        cuda_barrier_prefix=f"wan22_umt5_layer_{layer}",
        debug_output_prefix="",
        source_softmax=True,
    )
    # _self_attention_bf16 names an empty-prefix debug tensor as "_q" etc.
    # Rename those output tensors so the execution bindings remain explicit.
    for index in range(network.num_outputs):
        output = network.get_output(index)
        if output.name.startswith("_"):
            output.name = output.name[1:]
    attention_residual = network.add_elementwise(
        hidden, attention_output, trt.ElementWiseOperation.SUM
    ).get_output(0)
    attention_residual = _bf16_barrier(
        network,
        attention_residual,
        f"wan22_umt5_layer_{layer}_attention_residual",
        trt,
    )
    _mark(network, attention_residual, "attention_residual")

    norm2 = rms_norm(
        network,
        attention_residual,
        weights["norm2"],
        WAN22_UMT5_XXL.hidden_size,
        WAN22_UMT5_XXL.epsilon,
        refs,
        trt,
    )
    _mark(network, norm2, "norm2")
    fc1 = _linear_bf16(network, norm2, weights["fc1"], refs, trt)
    fc1 = _bf16_barrier(network, fc1, f"wan22_umt5_layer_{layer}_fc1", trt)
    _mark(network, fc1, "fc1")
    gate_input = _linear_bf16(network, norm2, weights["gate"], refs, trt)
    _mark(network, gate_input, "gate_input")
    gelu = _gelu_bf16(
        network,
        gate_input,
        refs,
        trt,
        cuda_plugin_name=f"wan22_umt5_layer_{layer}_source_gelu",
    )
    _mark(network, gelu, "gelu")
    gated = network.add_elementwise(fc1, gelu, trt.ElementWiseOperation.PROD).get_output(0)
    gated = _bf16_barrier(network, gated, f"wan22_umt5_layer_{layer}_gated_product", trt)
    _mark(network, gated, "gated_product")
    fc2 = _linear_bf16(network, gated, weights["fc2"], refs, trt)
    fc2 = _bf16_barrier(network, fc2, f"wan22_umt5_layer_{layer}_fc2", trt)
    _mark(network, fc2, "fc2")
    residual = network.add_elementwise(
        attention_residual, fc2, trt.ElementWiseOperation.SUM
    ).get_output(0)
    residual = _bf16_barrier(network, residual, f"wan22_umt5_layer_{layer}_residual", trt)
    _mark(network, residual, "residual")

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT failed to build the UMT5 block probe")
    return bytes(plan)


def _rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    normalized = x * torch.rsqrt(
        x.float().pow(2).mean(dim=-1, keepdim=True) + WAN22_UMT5_XXL.epsilon
    )
    normalized = normalized.type_as(weight)
    return weight * normalized


def _official_stages(
    hidden: torch.Tensor,
    mask: torch.Tensor,
    weights: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    x = hidden.to(device=device, dtype=torch.bfloat16)
    w = {name: value.to(device=device) for name, value in weights.items()}
    attention_norm = _rms_norm(x, w["attention_norm"])
    q = functional.linear(attention_norm, w["q"])
    k = functional.linear(attention_norm, w["k"])
    v = functional.linear(attention_norm, w["v"])
    q_rows = q.view(1, 512, 64, 64)
    k_rows = k.view(1, 512, 64, 64)
    v_rows = v.view(1, 512, 64, 64)
    qk_logits = torch.einsum("binc,bjnc->bnij", q_rows, k_rows)
    buckets = torch.from_numpy(
        relative_position_buckets(
            512,
            512,
            num_buckets=WAN22_UMT5_XXL.num_buckets,
            max_distance=WAN22_UMT5_XXL.relative_attention_max_distance,
        )
    ).to(device=device)
    raw_position_bias = w["relative_attention_bias"][buckets]
    raw_position_bias = raw_position_bias.permute(2, 0, 1).unsqueeze(0).contiguous()
    attention_bias = q.new_zeros(1, 64, 512, 512)
    attention_bias += raw_position_bias
    attention_bias.masked_fill_(
        mask.view(1, 1, 1, 512) == 0,
        torch.finfo(q.dtype).min,
    )
    biased_logits = qk_logits + attention_bias
    probabilities = functional.softmax(biased_logits.float(), dim=-1).type_as(biased_logits)
    context_rows = torch.einsum("bnij,bjnc->binc", probabilities, v_rows)
    pv_context = context_rows.permute(0, 2, 1, 3)
    attention_output = functional.linear(context_rows.reshape(1, 512, 4096), w["o"])
    attention_residual = x + attention_output
    norm2 = _rms_norm(attention_residual, w["norm2"])
    fc1 = functional.linear(norm2, w["fc1"])
    gate_input = functional.linear(norm2, w["gate"])
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
    fc2 = functional.linear(gated, w["fc2"])
    residual = attention_residual + fc2
    return {
        "attention_norm": attention_norm,
        "q": q,
        "k": k,
        "v": v,
        "qk_logits": qk_logits,
        "biased_logits": biased_logits,
        "probabilities": probabilities,
        "pv_context": pv_context,
        "attention_output": attention_output,
        "attention_residual": attention_residual,
        "norm2": norm2,
        "fc1": fc1,
        "gate_input": gate_input,
        "gelu": gelu,
        "gated_product": gated,
        "fc2": fc2,
        "residual": residual,
    }


def _real_tokens(name: str, value: torch.Tensor, token_count: int) -> torch.Tensor:
    if name in {"qk_logits", "biased_logits", "probabilities"}:
        return value[:, :, :token_count, :token_count]
    if name == "pv_context":
        return value[:, :, :token_count]
    return value[:, :token_count]


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
    mask = torch.zeros((1, 512), dtype=torch.int32)
    mask[:, :token_count] = 1
    official_report = json.loads(args.official_report.read_text())

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    plugin_library = ctypes.CDLL(str(args.plugin), mode=ctypes.RTLD_GLOBAL)
    torch_weights, trt_weights = _block_weights(args.checkpoint, layer)
    reference = _official_stages(hidden, mask.to(device), torch_weights, device)

    begin = time.perf_counter()
    plan = _build_plan(trt_weights, layer=layer, source_rmsnorm=args.source_rmsnorm)
    build_seconds = time.perf_counter() - begin
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    args.plan.write_bytes(plan)
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(plan)
    if engine is None:
        raise RuntimeError("Could not deserialize the UMT5 block probe")
    context = engine.create_execution_context()
    input_device = hidden.squeeze(0).to(device=device)
    mask_device = mask.to(device=device)
    outputs = {
        name: torch.empty(shape, device=device, dtype=torch.float32)
        for name, shape in _STAGE_SHAPES.items()
    }
    for name, value in {"hidden": input_device, "attention_mask": mask_device}.items():
        if not context.set_tensor_address(name, value.data_ptr()):
            raise RuntimeError(f"Could not bind {name}")
    for name, value in outputs.items():
        if not context.set_tensor_address(name, value.data_ptr()):
            raise RuntimeError(f"Could not bind {name}")
    stream = torch.cuda.current_stream(device=device)
    if not context.execute_async_v3(stream_handle=stream.cuda_stream):
        raise RuntimeError("TensorRT UMT5 block probe failed")
    torch.cuda.synchronize(device)

    stages = {}
    first_non_exact = None
    for name in _STAGE_SHAPES:
        full = _metrics(reference[name], outputs[name])
        real = _metrics(
            _real_tokens(name, reference[name], token_count),
            _real_tokens(name, outputs[name], token_count),
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
        "kind": "wan2_2_ti2v_umt5_block_stage_localization",
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
