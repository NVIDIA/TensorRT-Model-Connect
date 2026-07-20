#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Locate Wan2.2's first divergence at the official patch Conv3d boundary.

This is an isolated qualification only.  It compares the official autocast
Conv3d rows with both the current TensorRT unfold+MM+bias graph and every
cuBLASLt TNN candidate for the equivalent M=27280, K=192, N=3072 problem.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tensorrt_model_connect.trt_compat import trt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qualify_all_linear_shapes as all_linear  # noqa: E402
import qualify_block0_ffn2 as base  # noqa: E402

M = 27_280
K = 192
N = 3_072
LATENT_SHAPE = (1, 48, 31, 44, 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--first-call", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--current-trt-plan", type=Path, required=True)
    parser.add_argument("--best-lt-plan", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workspace-mib", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--force-capture", action="store_true")
    return parser.parse_args()


def unfold_rows(latent: torch.Tensor) -> torch.Tensor:
    """Exactly mirror production `_patchify` for patch_size=(1,2,2)."""

    return latent.reshape(1, 48, 31, 1, 22, 2, 40, 2).permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(M, K)


def capture_official(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    WanModel = base.import_official_model(args.official_source)
    first_call = torch.load(args.first_call, map_location="cpu", weights_only=True)
    model = WanModel.from_pretrained(str(args.checkpoint)).eval().requires_grad_(False)
    patch = model.patch_embedding.to(device)
    del model
    latent = first_call["latent"].unsqueeze(0).to(device=device, dtype=torch.float32)
    if tuple(latent.shape) != LATENT_SHAPE:
        raise ValueError(f"Official latent shape is {tuple(latent.shape)}, expected {LATENT_SHAPE}")

    holder: dict[str, torch.Tensor] = {}

    def official_call() -> torch.Tensor:
        holder["output"] = patch(latent)
        return holder["output"]

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        samples, conv_output = base.timed_cuda(
            official_call,
            device=device,
            warmup=args.warmup,
            iterations=args.iterations,
        )
    reference = conv_output.flatten(2).transpose(1, 2).reshape(M, N).contiguous()
    rows = unfold_rows(latent).to(dtype=torch.bfloat16).contiguous()
    weight_fp32 = patch.weight.detach().float().contiguous()
    bias_fp32 = patch.bias.detach().float().contiguous()
    weight = weight_fp32.to(dtype=torch.bfloat16).reshape(N, K).contiguous()
    bias = bias_fp32.to(dtype=torch.bfloat16).contiguous()
    with torch.inference_mode():
        linear_reference = torch.nn.functional.linear(rows, weight, bias)
    payload = {
        "latent": latent.cpu(),
        "x": rows.cpu(),
        "weight": weight.cpu(),
        "bias": bias.cpu(),
        "weight_fp32": weight_fp32.cpu(),
        "bias_fp32": bias_fp32.cpu(),
        "reference": reference.cpu(),
        "metadata": {
            "kind": "wan2_2_ti2v_official_patch_embedding_conv3d",
            "official_source": str(args.official_source.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "first_call": str(args.first_call.resolve()),
            "latent_shape": list(LATENT_SHAPE),
            "conv_kernel": [1, 2, 2],
            "conv_stride": [1, 2, 2],
            "m": M,
            "k": K,
            "n": N,
            "bias": True,
            "official_conv_hot_latency": base.latency_summary(samples),
            "pytorch_bf16_unfolded_linear_metrics": base.tensor_metrics(
                linear_reference, reference
            ),
        },
    }
    args.capture.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.capture)
    print(
        f"captured official patch embedding: {args.capture} "
        f"({args.capture.stat().st_size / 2**20:.1f} MiB)",
        flush=True,
    )
    return payload


def load_capture(
    path: Path, device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    tensors = {
        name: payload[name].to(device=device).contiguous()
        for name in (
            "latent",
            "x",
            "weight",
            "bias",
            "weight_fp32",
            "bias_fp32",
            "reference",
        )
    }
    return tensors, payload["metadata"]


def build_current_trt_plan(
    weight_fp32: torch.Tensor,
    bias_fp32: torch.Tensor,
    output_path: Path,
) -> bytes:
    # Import the actual production graph helper, but keep all optional plugins
    # disabled so this is exactly the current unfold+MM+bias fallback.
    repo_python = Path(__file__).resolve().parents[4]
    if str(repo_python) not in sys.path:
        sys.path.insert(0, str(repo_python))
    from tensorrt_model_connect.families.wan2_2_ti2v import (  # noqa: E402
        trt_ops as op,
    )
    from tensorrt_model_connect.families.wan2_2_ti2v.dit_builder import (  # noqa: E402
        _patchify,
    )
    from tensorrt_model_connect.families.wan2_2_ti2v.model_config import (  # noqa: E402
        WAN22_TI2V_5B,
    )

    op.set_bf16_gemm_emulation(False)
    op.set_source_attention_plugin(False)
    op.set_cuda_bf16_barriers(False)
    op.set_dit_cuda_numerics(False)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    latent = network.add_input("latent", trt.float32, LATENT_SHAPE)
    output = _patchify(
        network,
        latent,
        weight_fp32.cpu().numpy(),
        bias_fp32.cpu().numpy(),
        WAN22_TI2V_5B,
    )
    output.name = "output"
    network.mark_output(output)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 * 1024**3)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Could not build current TensorRT patch embedding graph")
    plan = bytes(serialized)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(plan)
    return plan


def benchmark_current_trt(
    plan: bytes,
    tensors: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    output = torch.empty((M, N), device=device, dtype=torch.bfloat16)
    context.set_tensor_address("latent", tensors["latent"].data_ptr())
    context.set_tensor_address("output", output.data_ptr())

    def call() -> torch.Tensor:
        if not context.execute_async_v3(torch.cuda.current_stream(device).cuda_stream):
            raise RuntimeError("Current TensorRT patch embedding execution failed")
        return output

    samples, _ = base.timed_cuda(
        call, device=device, warmup=args.warmup, iterations=args.iterations
    )
    return {
        "latency": base.latency_summary(samples),
        "metrics": base.tensor_metrics(output, tensors["reference"]),
    }


def benchmark_pytorch_unfolded_linear(
    tensors: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    holder: dict[str, torch.Tensor] = {}

    def call() -> torch.Tensor:
        holder["output"] = torch.nn.functional.linear(
            tensors["x"], tensors["weight"], tensors["bias"]
        )
        return holder["output"]

    with torch.inference_mode():
        samples, output = base.timed_cuda(
            call, device=device, warmup=args.warmup, iterations=args.iterations
        )
    return {
        "latency": base.latency_summary(samples),
        "metrics": base.tensor_metrics(output, tensors["reference"]),
    }


def execute_plan_once(
    plan: bytes,
    inputs: dict[str, torch.Tensor],
    *,
    device: torch.device,
) -> torch.Tensor:
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    output = torch.empty((M, N), device=device, dtype=torch.bfloat16)
    for name, tensor in {**inputs, "output": output}.items():
        if not context.set_tensor_address(name, tensor.data_ptr()):
            raise RuntimeError(f"Could not bind {name} for cross-path comparison")
    if not context.execute_async_v3(torch.cuda.current_stream(device).cuda_stream):
        raise RuntimeError("Cross-path TensorRT execution failed")
    torch.cuda.synchronize(device)
    return output


def candidate_rank(candidate: dict[str, Any]) -> tuple[float, float, float]:
    metrics = candidate["metrics"]
    return (
        -float(metrics["exact_rate"]),
        -float(metrics["cosine_similarity"]),
        float(metrics["mean_abs_error"]),
    )


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if args.force_capture or not args.capture.is_file():
        capture_official(args, device)
        torch.cuda.empty_cache()
    tensors, metadata = load_capture(args.capture, device)
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        pytorch_linear = benchmark_pytorch_unfolded_linear(tensors, args, device)
        current_plan = build_current_trt_plan(
            tensors["weight_fp32"], tensors["bias_fp32"], args.current_trt_plan
        )
        current_trt = benchmark_current_trt(current_plan, tensors, args, device)

        probe = all_linear.GenericProbe(args.plugin)
        candidates = probe.query(M, N, K, args.workspace_mib)
        candidate_results = all_linear.benchmark_candidates(
            probe,
            candidates,
            tensors,
            metadata,
            args,
            device,
        )
        non_split = [
            candidate
            for candidate in candidate_results
            if candidate["portable_non_splitk_admissible"]
        ]
        if not non_split:
            raise RuntimeError("No non-splitK cuBLASLt patch candidates")
        exact = [candidate for candidate in non_split if candidate["metrics"]["bit_exact"]]
        selected = (
            min(exact, key=lambda item: item["latency"]["median_ms"])
            if exact
            else min(non_split, key=candidate_rank)
        )
        lt_tensors = {name: tensors[name] for name in ("x", "weight", "bias", "reference")}
        best_plan = all_linear.build_plan(
            args.plugin,
            metadata,
            int(selected["heuristic_index"]),
            args.workspace_mib,
            args.best_lt_plan,
        )
        best_trt = all_linear.benchmark_plan(best_plan, lt_tensors, metadata, args, device)
        current_output = execute_plan_once(
            current_plan, {"latent": tensors["latent"]}, device=device
        )
        lt_output = execute_plan_once(
            best_plan,
            {
                "x": tensors["x"],
                "weight": tensors["weight"],
                "bias": tensors["bias"],
            },
            device=device,
        )
        pytorch_linear_output = torch.nn.functional.linear(
            tensors["x"], tensors["weight"], tensors["bias"]
        )
        cross_path_metrics = {
            "current_trt_vs_pytorch_unfolded_linear": base.tensor_metrics(
                current_output, pytorch_linear_output
            ),
            "selected_lt_vs_pytorch_unfolded_linear": base.tensor_metrics(
                lt_output, pytorch_linear_output
            ),
            "selected_lt_vs_current_trt": base.tensor_metrics(lt_output, current_output),
        }

    report = {
        "kind": "wan2_2_ti2v_patch_embedding_first_divergence",
        "status": "PASS",
        "hardware": {
            "device": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
        },
        "software": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "tensorrt": trt.__version__,
        },
        "contract": metadata,
        "pytorch_bf16_unfolded_linear": pytorch_linear,
        "current_trt_unfold_mm_bias": current_trt,
        "cublaslt_candidates": candidate_results,
        "cublaslt_exact_candidate_count": len(exact),
        "cublaslt_any_bit_exact": bool(exact),
        "selected_cublaslt_candidate": selected,
        "selected_cublaslt_tensorrt_plugin": best_trt,
        "cross_path_metrics": cross_path_metrics,
        "artifacts": {
            "capture": str(args.capture.resolve()),
            "current_trt_plan": str(args.current_trt_plan.resolve()),
            "best_lt_plan": str(args.best_lt_plan.resolve()),
            "plugin": str(args.plugin.resolve()),
        },
        "finding": (
            "At least one pure cuBLASLt TNN candidate is bit-exact to official Conv3d."
            if exact
            else "No returned pure cuBLASLt TNN candidate is bit-exact to official Conv3d."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "current_trt": current_trt,
                "cublaslt_candidate_count": len(candidate_results),
                "cublaslt_exact_candidate_count": len(exact),
                "selected": {
                    "heuristic_index": selected["heuristic_index"],
                    "algorithm_id": selected["algorithm_id"],
                    "tile_id": selected["tile_id"],
                    "metrics": selected["metrics"],
                    "latency": selected["latency"],
                },
                "report": str(args.report),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
