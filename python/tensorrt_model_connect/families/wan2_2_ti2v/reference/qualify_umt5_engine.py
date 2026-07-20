#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and qualify the native Wan2.2 UMT5 TensorRT engine.

This is a qualification utility, not a runtime dependency.  It uses the
official Wan Python implementation only to generate the A/B reference after
the pure TensorRT engine has been serialized.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from tensorrt_model_connect.trt_compat import trt
import torch
import torch.nn.functional as functional

from tensorrt_model_connect.families.wan2_2_ti2v.model_config import (
    OFFICIAL_NEGATIVE_PROMPT,
)
from tensorrt_model_connect.families.wan2_2_ti2v.umt5_encoder_builder import (
    NATIVE_UMT5_CHECKPOINT,
    build_umt5_encoder_engine,
    load_native_umt5_weights,
)


OFFICIAL_POSITIVE_PROMPT = (
    "Two anthropomorphic cats in comfy boxing gear and bright gloves fight "
    "intensely on a spotlighted stage"
)

EXPECTED_TOKEN_HASHES = {
    "positive": "5e2323bafbdc8edea3c82a9ceaa36a0cc95ba6887fd3b86bf9182a12e0831793",
    "negative": "d7217a4e13a6dde04a32abbf22446dda68e20db55481cb5facf8e7371b41e496",
}

EXPECTED_REFERENCE_HASHES = {
    "positive": "4f946c34736fd31bd1e09727b0bbe8da537275b382cae660aafd2be35f40261e",
    "negative": "b03635e7c693eefbdb48dad449be7eda7b18d181b63217e7bfe736537db2c2fd",
}

DEFAULT_MAX_RELATIVE_L2_ERROR = 0.01

ATTENTION_DEBUG_SHAPES = {
    "attention_norm": (1, 512, 4096),
    "attention_bias": (1, 64, 512, 512),
    "q": (1, 512, 4096),
    "k": (1, 512, 4096),
    "v": (1, 512, 4096),
    "qk_logits": (1, 64, 512, 512),
    "biased_logits": (1, 64, 512, 512),
    "probabilities": (1, 64, 512, 512),
    "pv_context": (1, 64, 512, 64),
    "attention_output": (1, 512, 4096),
    "attention_residual": (1, 512, 4096),
}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tensor_bf16_sha256(value: torch.Tensor) -> str:
    bf16 = value.detach().to(dtype=torch.bfloat16, device="cpu").contiguous()
    raw = bf16.view(torch.uint16).numpy().astype("<u2", copy=False).tobytes()
    return hashlib.sha256(raw).hexdigest()


def _token_sha256(ids: torch.Tensor, length: int) -> str:
    raw = ids[0, :length].detach().cpu().numpy().astype("<i4", copy=False).tobytes()
    return hashlib.sha256(raw).hexdigest()


def _metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    reference_fp32 = reference.detach().float().cpu()
    actual_fp32 = actual.detach().float().cpu()
    delta = actual_fp32 - reference_fp32
    reference_l2_norm = float(torch.linalg.vector_norm(reference_fp32.flatten().double()))
    actual_l2_norm = float(torch.linalg.vector_norm(actual_fp32.flatten().double()))
    delta_l2_norm = float(torch.linalg.vector_norm(delta.flatten().double()))
    relative_l2_error = (
        delta_l2_norm / reference_l2_norm
        if reference_l2_norm > 0.0
        else (0.0 if delta_l2_norm == 0.0 else math.inf)
    )
    actual_bf16 = actual_fp32.to(torch.bfloat16)
    reference_bf16 = reference_fp32.to(torch.bfloat16)
    return {
        "shape": list(reference.shape),
        "max_abs_error": float(delta.abs().max()),
        "mean_abs_error": float(delta.abs().mean()),
        "rmse": float(delta.square().mean().sqrt()),
        "reference_l2_norm": reference_l2_norm,
        "actual_l2_norm": actual_l2_norm,
        "delta_l2_norm": delta_l2_norm,
        "relative_l2_error": relative_l2_error,
        "cosine_similarity": float(
            functional.cosine_similarity(
                reference_fp32.flatten().double(),
                actual_fp32.flatten().double(),
                dim=0,
            )
        ),
        "bf16_exact": bool(torch.equal(reference_bf16, actual_bf16)),
        "bf16_mismatch_count": int(torch.count_nonzero(reference_bf16 != actual_bf16)),
        "reference_bf16_sha256": _tensor_bf16_sha256(reference),
        "actual_bf16_sha256": _tensor_bf16_sha256(actual),
    }


def _qualification(
    prompt_reports: dict[str, dict[str, Any]],
    min_cosine: float,
    max_relative_l2_error: float = DEFAULT_MAX_RELATIVE_L2_ERROR,
    build_report_engine_sha256_matches: bool | None = None,
) -> dict[str, Any]:
    comparisons: list[tuple[str, dict[str, Any]]] = []
    for prompt_name, prompt_report in prompt_reports.items():
        comparisons.extend(
            (
                (f"{prompt_name}.full_512_rows", prompt_report["full_512_rows"]),
                (f"{prompt_name}.real_token_rows", prompt_report["real_token_rows"]),
            )
        )
        for layer_name, layer_report in prompt_report["layers"].items():
            comparisons.extend(
                (
                    (
                        f"{prompt_name}.layers.{layer_name}.full_512_rows",
                        layer_report["full_512_rows"],
                    ),
                    (
                        f"{prompt_name}.layers.{layer_name}.real_token_rows",
                        layer_report["real_token_rows"],
                    ),
                )
            )
        for layer_name, attention_report in prompt_report["attention_layers"].items():
            for output_name, output_report in attention_report.items():
                comparisons.extend(
                    (
                        (
                            f"{prompt_name}.attention_layers.{layer_name}.{output_name}.full",
                            output_report["full"],
                        ),
                        (
                            f"{prompt_name}.attention_layers.{layer_name}.{output_name}.real_tokens",
                            output_report["real_tokens"],
                        ),
                    )
                )

    non_finite_cosine_comparisons = [
        name for name, metrics in comparisons if not math.isfinite(metrics["cosine_similarity"])
    ]
    non_finite_relative_l2_comparisons = [
        name for name, metrics in comparisons if not math.isfinite(metrics["relative_l2_error"])
    ]
    non_finite_comparisons = sorted(
        set(non_finite_cosine_comparisons) | set(non_finite_relative_l2_comparisons)
    )
    worst_name = None
    worst_cosine = None
    if comparisons and not non_finite_cosine_comparisons:
        worst_name, worst_metrics = min(
            comparisons,
            key=lambda item: item[1]["cosine_similarity"],
        )
        worst_cosine = worst_metrics["cosine_similarity"]
    worst_relative_l2_name = None
    worst_relative_l2_error = None
    if comparisons and not non_finite_relative_l2_comparisons:
        worst_relative_l2_name, worst_relative_l2_metrics = max(
            comparisons,
            key=lambda item: item[1]["relative_l2_error"],
        )
        worst_relative_l2_error = worst_relative_l2_metrics["relative_l2_error"]
    token_hashes_match_expected = all(
        prompt_report["token_hash_matches_expected"] for prompt_report in prompt_reports.values()
    )
    reference_hashes_match_expected = all(
        prompt_report["reference_real_hash_matches_expected"]
        for prompt_report in prompt_reports.values()
    )
    return {
        "min_cosine": min_cosine,
        "max_relative_l2_error": max_relative_l2_error,
        "comparisons_checked": len(comparisons),
        "non_finite_comparisons": non_finite_comparisons,
        "non_finite_cosine_comparisons": non_finite_cosine_comparisons,
        "non_finite_relative_l2_comparisons": non_finite_relative_l2_comparisons,
        "worst_comparison": worst_name,
        "worst_cosine_similarity": worst_cosine,
        "worst_relative_l2_comparison": worst_relative_l2_name,
        "worst_relative_l2_error": worst_relative_l2_error,
        "all_bf16_exact": all(metrics["bf16_exact"] for _, metrics in comparisons),
        "token_hashes_match_expected": token_hashes_match_expected,
        "reference_hashes_match_expected": reference_hashes_match_expected,
        "build_report_engine_sha256_matches": build_report_engine_sha256_matches,
        "passed": (
            bool(comparisons)
            and not non_finite_comparisons
            and worst_cosine is not None
            and worst_cosine >= min_cosine
            and worst_relative_l2_error is not None
            and worst_relative_l2_error <= max_relative_l2_error
            and token_hashes_match_expected
            and reference_hashes_match_expected
            and build_report_engine_sha256_matches is not False
        ),
    }


def _validate_official_positive_prompt(prompt: str) -> None:
    if prompt != OFFICIAL_POSITIVE_PROMPT:
        raise ValueError(
            "--positive-prompt is fixed for official Wan2.2 qualification; "
            "custom prompts are not source-bound to the expected token and reference hashes"
        )


def _attention_real_tokens(
    name: str,
    value: torch.Tensor,
    token_count: int,
) -> torch.Tensor:
    if name in {"attention_bias", "qk_logits", "biased_logits", "probabilities"}:
        return value[:, :, :token_count, :token_count]
    if name == "pv_context":
        return value[:, :, :token_count]
    return value[:, :token_count]


def _build_engine(args: argparse.Namespace, build_report_path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "kind": "wan2_2_ti2v_umt5_tensorrt_build",
        "started_at": _now(),
        "checkpoint": str(args.checkpoint.resolve()),
        "engine": str(args.engine.resolve()),
        "builder_optimization_level": args.builder_optimization_level,
        "workspace_gib": args.workspace_gib,
        "debug_layers": list(args.debug_layers),
        "debug_attention_layers": list(args.debug_attention_layers),
        "source_gelu_plugin": (
            str(args.gelu_plugin.resolve()) if args.gelu_plugin is not None else None
        ),
        "source_softmax": args.source_softmax,
        "source_rmsnorm": args.source_rmsnorm,
        "argv": sys.argv,
        "status": "running",
    }
    _write_json(build_report_path, report)
    try:
        print(f"[{_now()}] loading native UMT5 checkpoint", flush=True)
        begin = time.perf_counter()
        weights = load_native_umt5_weights(args.checkpoint)
        load_seconds = time.perf_counter() - begin
        print(
            f"[{_now()}] loaded {len(weights)} tensors in {load_seconds:.3f}s; "
            "starting TensorRT build",
            flush=True,
        )
        begin = time.perf_counter()
        plan = build_umt5_encoder_engine(
            weights,
            workspace_size=args.workspace_gib << 30,
            builder_optimization_level=args.builder_optimization_level,
            source_gelu_plugin=args.gelu_plugin,
            source_softmax=args.source_softmax,
            source_rmsnorm=args.source_rmsnorm,
            debug_layer_outputs=args.debug_layers,
            debug_attention_outputs=args.debug_attention_layers,
            verbose=args.verbose,
        )
        build_seconds = time.perf_counter() - begin
        print(
            f"[{_now()}] build completed in {build_seconds:.3f}s; writing plan",
            flush=True,
        )
        args.engine.parent.mkdir(parents=True, exist_ok=True)
        begin = time.perf_counter()
        args.engine.write_bytes(plan)
        write_seconds = time.perf_counter() - begin
        report.update(
            {
                "status": "complete",
                "completed_at": _now(),
                "load_checkpoint_seconds": load_seconds,
                "build_engine_seconds": build_seconds,
                "write_engine_seconds": write_seconds,
                "total_build_seconds": load_seconds + build_seconds + write_seconds,
                "engine_size_bytes": len(plan),
                "engine_sha256": _sha256_bytes(plan),
                "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            }
        )
        _write_json(build_report_path, report)
        print(f"[{_now()}] wrote {args.engine} ({len(plan)} bytes)", flush=True)
        return report
    except BaseException as error:
        report.update(
            {
                "status": "failed",
                "completed_at": _now(),
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            }
        )
        _write_json(build_report_path, report)
        raise


def _run_trt(
    context: Any,
    ids: torch.Tensor,
    mask: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    *,
    warmup: int,
    iterations: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    bindings = {"input_ids": ids, "attention_mask": mask, **outputs}
    for name, tensor in bindings.items():
        if not context.set_tensor_address(name, tensor.data_ptr()):
            raise RuntimeError(f"Could not bind TensorRT tensor {name!r}")
    stream = torch.cuda.current_stream().cuda_stream
    for _ in range(warmup):
        if not context.execute_async_v3(stream_handle=stream):
            raise RuntimeError("TensorRT UMT5 warmup failed")
    torch.cuda.synchronize()
    begin = time.perf_counter()
    for _ in range(iterations):
        if not context.execute_async_v3(stream_handle=stream):
            raise RuntimeError("TensorRT UMT5 inference failed")
    torch.cuda.synchronize()
    seconds = time.perf_counter() - begin
    return {name: value.clone() for name, value in outputs.items()}, {
        "iterations": iterations,
        "total_seconds": seconds,
        "mean_seconds": seconds / iterations,
    }


def _run_reference(
    model: Any,
    ids: torch.Tensor,
    mask: torch.Tensor,
    *,
    warmup: int,
    iterations: int,
    debug_layers: tuple[int, ...],
    debug_attention_layers: tuple[int, ...],
) -> tuple[
    torch.Tensor,
    dict[int, torch.Tensor],
    dict[int, dict[str, torch.Tensor]],
    dict[str, float],
]:
    captured: dict[int, torch.Tensor] = {}
    attention_captured: dict[int, dict[str, torch.Tensor]] = {
        index: {} for index in debug_attention_layers
    }
    active_attention: list[int | None] = [None]

    def capture(index: int):
        def hook(_module, _inputs, output):
            captured[index] = output

        return hook

    def capture_attention(index: int, name: str):
        def hook(_module, _inputs, output):
            attention_captured[index][name] = output

        return hook

    def capture_attention_input(index: int, name: str):
        def hook(_module, inputs):
            attention_captured[index][name] = inputs[0]

        return hook

    def enter_attention(index: int):
        def hook(_module, _inputs):
            active_attention[0] = index

        return hook

    def leave_attention(_index: int):
        def hook(_module, _inputs, _output):
            active_attention[0] = None

        return hook

    hooks = [model.blocks[index].register_forward_hook(capture(index)) for index in debug_layers]
    for index in debug_attention_layers:
        block = model.blocks[index]
        hooks.extend(
            [
                block.norm1.register_forward_hook(capture_attention(index, "attention_norm")),
                block.pos_embedding.register_forward_hook(
                    capture_attention(index, "raw_position_bias")
                ),
                block.attn.q.register_forward_hook(capture_attention(index, "q")),
                block.attn.k.register_forward_hook(capture_attention(index, "k")),
                block.attn.v.register_forward_hook(capture_attention(index, "v")),
                block.attn.o.register_forward_hook(capture_attention(index, "attention_output")),
                block.norm2.register_forward_pre_hook(
                    capture_attention_input(index, "attention_residual")
                ),
                block.attn.register_forward_pre_hook(enter_attention(index)),
                block.attn.register_forward_hook(leave_attention(index)),
            ]
        )

    original_einsum = torch.einsum
    original_softmax = functional.softmax

    def traced_einsum(equation, *operands, **kwargs):
        output = original_einsum(equation, *operands, **kwargs)
        index = active_attention[0]
        if index in attention_captured:
            if equation == "binc,bjnc->bnij":
                attention_captured[index]["qk_logits"] = output
            elif equation == "bnij,bjnc->binc":
                attention_captured[index]["pv_context_rows"] = output
        return output

    def traced_softmax(input, *args, **kwargs):
        output = original_softmax(input, *args, **kwargs)
        index = active_attention[0]
        if index in attention_captured:
            attention_captured[index]["probabilities_fp32"] = output
        return output

    torch.einsum = traced_einsum
    functional.softmax = traced_softmax
    try:
        with torch.inference_mode():
            for _ in range(warmup):
                model(ids, mask)
            torch.cuda.synchronize()
            begin = time.perf_counter()
            output = None
            for _ in range(iterations):
                output = model(ids, mask)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - begin
    finally:
        torch.einsum = original_einsum
        functional.softmax = original_softmax
        for hook in hooks:
            hook.remove()
    assert output is not None
    missing = sorted(set(debug_layers) - set(captured))
    if missing:
        raise RuntimeError(f"Official UMT5 did not capture layers {missing}")
    finalized_attention = {}
    for index, values in attention_captured.items():
        expected = {
            "attention_norm",
            "raw_position_bias",
            "q",
            "k",
            "v",
            "qk_logits",
            "probabilities_fp32",
            "pv_context_rows",
            "attention_output",
            "attention_residual",
        }
        missing_values = sorted(expected - set(values))
        if missing_values:
            raise RuntimeError(f"Official UMT5 layer {index} did not capture {missing_values}")
        q = values["q"]
        attention_bias = q.new_zeros(1, 64, 512, 512)
        attention_bias += values["raw_position_bias"]
        attention_bias.masked_fill_(
            mask.view(1, 1, 1, 512) == 0,
            torch.finfo(q.dtype).min,
        )
        biased_logits = values["qk_logits"] + attention_bias
        finalized_attention[index] = {
            "attention_norm": values["attention_norm"].clone(),
            "attention_bias": attention_bias.clone(),
            "q": values["q"].clone(),
            "k": values["k"].clone(),
            "v": values["v"].clone(),
            "qk_logits": values["qk_logits"].clone(),
            "biased_logits": biased_logits.clone(),
            "probabilities": values["probabilities_fp32"].to(q.dtype).clone(),
            "pv_context": values["pv_context_rows"].permute(0, 2, 1, 3).clone(),
            "attention_output": values["attention_output"].clone(),
            "attention_residual": values["attention_residual"].clone(),
        }
    return (
        output,
        {index: captured[index].clone() for index in debug_layers},
        finalized_attention,
        {
            "iterations": iterations,
            "total_seconds": seconds,
            "mean_seconds": seconds / iterations,
        },
    )


def _qualify(args: argparse.Namespace, build_report: dict[str, Any]) -> dict[str, Any]:
    sys.path.insert(0, str(args.official_source.resolve()))
    from wan.modules.t5 import T5EncoderModel

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    prompts = {
        "positive": args.positive_prompt,
        "negative": OFFICIAL_NEGATIVE_PROMPT,
    }

    # The official UMT5 model and its 11-GiB TensorRT engine do not coexist in
    # Thor's 32-GiB CUDA-visible arena. Capture the official outputs on CPU,
    # release that stage, and only then deserialize TensorRT. This preserves
    # the exact A/B inputs and metrics while matching the staged runtime.
    print(f"[{_now()}] loading official BF16 UMT5 reference", flush=True)
    torch.cuda.reset_peak_memory_stats(device)
    begin = time.perf_counter()
    reference = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=device,
        checkpoint_path=str(args.checkpoint / NATIVE_UMT5_CHECKPOINT),
        tokenizer_path=str(args.checkpoint / "google" / "umt5-xxl"),
    )
    reference_load_seconds = time.perf_counter() - begin
    reference_fixtures: dict[str, dict[str, Any]] = {}
    for name, prompt in prompts.items():
        ids_cpu, mask_cpu = reference.tokenizer([prompt], return_mask=True, add_special_tokens=True)
        token_count = int(mask_cpu.gt(0).sum())
        token_hash = _token_sha256(ids_cpu, token_count)
        cleaned = reference.tokenizer._clean(prompt)
        print(
            f"[{_now()}] {name}: {token_count} tokens, token_sha256={token_hash}",
            flush=True,
        )
        (
            official,
            official_layers,
            official_attention,
            reference_timing,
        ) = _run_reference(
            reference.model,
            ids_cpu.to(device),
            mask_cpu.to(device),
            warmup=args.warmup,
            iterations=args.iterations,
            debug_layers=args.debug_layers,
            debug_attention_layers=args.debug_attention_layers,
        )
        reference_fixtures[name] = {
            "prompt": prompt,
            "cleaned_prompt": cleaned,
            "token_count": token_count,
            "token_hash": token_hash,
            "ids_cpu": ids_cpu.contiguous(),
            "mask_cpu": mask_cpu.contiguous(),
            "official": official.detach().cpu(),
            "official_layers": {
                index: value.detach().cpu() for index, value in official_layers.items()
            },
            "official_attention": {
                index: {output_name: value.detach().cpu() for output_name, value in values.items()}
                for index, values in official_attention.items()
            },
            "reference_timing": reference_timing,
        }
        del official, official_layers, official_attention
    torch.cuda.synchronize(device)
    reference_torch_peak_cuda_bytes = torch.cuda.max_memory_allocated(device)
    del reference
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    reference_torch_released_cuda_bytes = torch.cuda.memory_allocated(device)

    print(f"[{_now()}] reading and deserializing {args.engine}", flush=True)
    torch.cuda.reset_peak_memory_stats(device)
    begin = time.perf_counter()
    plan = args.engine.read_bytes()
    read_seconds = time.perf_counter() - begin
    engine_sha256 = _sha256_bytes(plan)
    build_report_engine_sha256 = build_report.get("engine_sha256")
    build_report_engine_sha256_matches = (
        None if build_report_engine_sha256 is None else build_report_engine_sha256 == engine_sha256
    )
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    begin = time.perf_counter()
    engine = runtime.deserialize_cuda_engine(plan)
    deserialize_seconds = time.perf_counter() - begin
    if engine is None:
        raise RuntimeError(f"Could not deserialize {args.engine}")
    context = engine.create_execution_context()
    del plan

    prompt_reports = {}
    for name, fixture in reference_fixtures.items():
        prompt = fixture["prompt"]
        cleaned = fixture["cleaned_prompt"]
        token_count = fixture["token_count"]
        token_hash = fixture["token_hash"]
        ids_cpu = fixture["ids_cpu"]
        mask_cpu = fixture["mask_cpu"]
        official = fixture["official"]
        official_layers = fixture["official_layers"]
        official_attention = fixture["official_attention"]
        reference_timing = fixture["reference_timing"]
        ids_trt = ids_cpu.to(device=device, dtype=torch.int32)
        mask_trt = mask_cpu.to(device=device, dtype=torch.int32)
        trt_outputs = {
            "text_embeddings": torch.empty((1, 512, 4096), device=device, dtype=torch.float32),
            **{
                f"layer_{index}_hidden": torch.empty(
                    (1, 512, 4096), device=device, dtype=torch.float32
                )
                for index in args.debug_layers
            },
            **{
                f"layer_{index}_{output_name}": torch.empty(
                    output_shape,
                    device=device,
                    dtype=torch.float32,
                )
                for index in args.debug_attention_layers
                for output_name, output_shape in ATTENTION_DEBUG_SHAPES.items()
            },
        }
        actual_outputs, trt_timing = _run_trt(
            context,
            ids_trt,
            mask_trt,
            trt_outputs,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        actual = actual_outputs["text_embeddings"]
        full_metrics = _metrics(official, actual)
        real_metrics = _metrics(official[:, :token_count], actual[:, :token_count])
        prompt_reports[name] = {
            "prompt": prompt,
            "cleaned_prompt": cleaned,
            "cleaned_utf8_sha256": hashlib.sha256(cleaned.encode()).hexdigest(),
            "token_count": token_count,
            "token_ids": ids_cpu[0, :token_count].tolist(),
            "token_ids_int32_le_sha256": token_hash,
            "expected_token_hash": EXPECTED_TOKEN_HASHES[name],
            "token_hash_matches_expected": token_hash == EXPECTED_TOKEN_HASHES[name],
            "trt_timing": trt_timing,
            "official_timing": reference_timing,
            "full_512_rows": full_metrics,
            "real_token_rows": real_metrics,
            "layers": {
                str(index): {
                    "full_512_rows": _metrics(
                        official_layers[index],
                        actual_outputs[f"layer_{index}_hidden"],
                    ),
                    "real_token_rows": _metrics(
                        official_layers[index][:, :token_count],
                        actual_outputs[f"layer_{index}_hidden"][:, :token_count],
                    ),
                }
                for index in args.debug_layers
            },
            "attention_layers": {
                str(index): {
                    output_name: {
                        "full": _metrics(
                            official_attention[index][output_name],
                            actual_outputs[f"layer_{index}_{output_name}"],
                        ),
                        "real_tokens": _metrics(
                            _attention_real_tokens(
                                output_name,
                                official_attention[index][output_name],
                                token_count,
                            ),
                            _attention_real_tokens(
                                output_name,
                                actual_outputs[f"layer_{index}_{output_name}"],
                                token_count,
                            ),
                        ),
                    }
                    for output_name in ATTENTION_DEBUG_SHAPES
                }
                for index in args.debug_attention_layers
            },
            "expected_reference_real_bf16_sha256": EXPECTED_REFERENCE_HASHES[name],
            "reference_real_hash_matches_expected": (
                real_metrics["reference_bf16_sha256"] == EXPECTED_REFERENCE_HASHES[name]
            ),
        }
        print(
            f"[{_now()}] {name}: real cosine="
            f"{real_metrics['cosine_similarity']:.10f}, max="
            f"{real_metrics['max_abs_error']:.10f}, mean="
            f"{real_metrics['mean_abs_error']:.10f}, exact="
            f"{real_metrics['bf16_exact']}",
            flush=True,
        )
    torch.cuda.synchronize(device)
    trt_torch_peak_cuda_bytes = torch.cuda.max_memory_allocated(device)

    qualification = _qualification(
        prompt_reports,
        args.min_cosine,
        max_relative_l2_error=args.max_relative_l2_error,
        build_report_engine_sha256_matches=build_report_engine_sha256_matches,
    )
    return {
        "kind": "wan2_2_ti2v_umt5_tensorrt_official_ab",
        "completed_at": _now(),
        "device": torch.cuda.get_device_name(device),
        "device_index": device.index,
        "checkpoint": str(args.checkpoint.resolve()),
        "official_source": str(args.official_source.resolve()),
        "engine": str(args.engine.resolve()),
        "engine_size_bytes": args.engine.stat().st_size,
        "engine_sha256": engine_sha256,
        "build_report_engine_sha256": build_report_engine_sha256,
        "build_report_engine_sha256_matches": build_report_engine_sha256_matches,
        "source_gelu_plugin": (
            str(args.gelu_plugin.resolve()) if args.gelu_plugin is not None else None
        ),
        "source_softmax": args.source_softmax,
        "source_rmsnorm": args.source_rmsnorm,
        "qualification_execution": "staged_official_then_tensorrt",
        "reference_torch_peak_cuda_bytes": reference_torch_peak_cuda_bytes,
        "reference_torch_released_cuda_bytes": reference_torch_released_cuda_bytes,
        "trt_torch_peak_cuda_bytes": trt_torch_peak_cuda_bytes,
        "read_engine_seconds": read_seconds,
        "deserialize_engine_seconds": deserialize_seconds,
        "reference_load_seconds": reference_load_seconds,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "debug_layers": list(args.debug_layers),
        "debug_attention_layers": list(args.debug_attention_layers),
        "argv": sys.argv,
        "prompts": prompt_reports,
        "qualification": qualification,
        "build": build_report,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--build-report", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workspace-gib", type=int, default=32)
    parser.add_argument("--builder-optimization-level", type=int, default=0)
    parser.add_argument("--gelu-plugin", type=Path)
    parser.add_argument(
        "--source-softmax",
        action="store_true",
        help="Use the opt-in PyTorch-compatible fixed-512 softmax plugin",
    )
    parser.add_argument(
        "--source-rmsnorm",
        action="store_true",
        help="Use the opt-in PyTorch-compatible fixed UMT5-XXL RMSNorm plugin",
    )
    parser.add_argument(
        "--debug-layers",
        default="",
        help="Comma-separated zero-based block outputs to compare, or 'all'",
    )
    parser.add_argument(
        "--debug-attention-layers",
        default="",
        help="Comma-separated layers whose attention substages are compared",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--min-cosine", type=float, default=0.998)
    parser.add_argument(
        "--max-relative-l2-error",
        type=float,
        default=DEFAULT_MAX_RELATIVE_L2_ERROR,
        help="Maximum relative L2 error for every compared output (default: %(default)s)",
    )
    parser.add_argument("--positive-prompt", default=OFFICIAL_POSITIVE_PROMPT)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.official_source = args.official_source.resolve()
    args.engine = args.engine.resolve()
    args.report = args.report.resolve()
    if not math.isfinite(args.min_cosine) or not 0.0 <= args.min_cosine <= 1.0:
        parser.error("--min-cosine must be finite and in [0, 1]")
    if (
        not math.isfinite(args.max_relative_l2_error)
        or not 0.0 <= args.max_relative_l2_error <= 1.0
    ):
        parser.error("--max-relative-l2-error must be finite and in [0, 1]")
    try:
        _validate_official_positive_prompt(args.positive_prompt)
    except ValueError as error:
        parser.error(str(error))
    if (args.source_softmax or args.source_rmsnorm) and args.gelu_plugin is None:
        parser.error("--source-softmax/--source-rmsnorm require --gelu-plugin")
    for argument in ("debug_layers", "debug_attention_layers"):
        value = getattr(args, argument)
        parsed = (
            tuple(range(24))
            if value.strip().lower() == "all"
            else tuple(int(item.strip()) for item in value.split(",") if item.strip())
        )
        setattr(args, argument, parsed)
    build_report_path = (
        args.build_report.resolve()
        if args.build_report is not None
        else args.engine.with_suffix(args.engine.suffix + ".build.json")
    )
    torch.cuda.set_device(torch.device(args.device))
    plugin_library = None
    if args.gelu_plugin is not None:
        args.gelu_plugin = args.gelu_plugin.resolve()
        if not args.gelu_plugin.is_file():
            raise FileNotFoundError(args.gelu_plugin)
        plugin_library = ctypes.CDLL(str(args.gelu_plugin), mode=ctypes.RTLD_GLOBAL)

    if args.rebuild or not args.engine.is_file():
        build_report = _build_engine(args, build_report_path)
    elif build_report_path.is_file():
        build_report = json.loads(build_report_path.read_text())
        print(f"[{_now()}] reusing existing {args.engine}", flush=True)
    else:
        build_report = {
            "kind": "wan2_2_ti2v_umt5_tensorrt_build",
            "status": "preexisting",
            "engine": str(args.engine),
            "engine_size_bytes": args.engine.stat().st_size,
        }
    report = _qualify(args, build_report)
    assert plugin_library is not None or args.gelu_plugin is None
    _write_json(args.report, report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if not report["qualification"]["passed"]:
        raise SystemExit(
            "Wan2.2 UMT5 qualification failed: "
            + json.dumps(report["qualification"], sort_keys=True)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
