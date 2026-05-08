#!/usr/bin/env python3
"""Minimal TensorRT-only repro for multiple FP8-normalized IAttention layers.

This intentionally avoids tensorrt_model_connect, torch, diffusers, and model weights. It
constructs a tiny strongly typed TensorRT network with repeated IAttention
layers, FP8 Q/DQ on Q/K/V, and IAttention softmax normalization quantization.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorrt as trt


def _constant(
    network: trt.INetworkDefinition,
    shape: tuple[int, ...],
    value: float,
    dtype: np.dtype = np.float32,
    name: str | None = None,
) -> trt.ITensor:
    arr = np.full(shape if shape else (), value, dtype=dtype)
    layer = network.add_constant(shape, trt.Weights(arr))
    if name:
        layer.name = name
    return layer.get_output(0)


def _scale_input(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    factor: float,
) -> trt.ITensor:
    scale_dtype = np.float16 if tensor.dtype == trt.float16 else np.float32
    scale = _constant(network, (1, 1, 1, 1), factor, scale_dtype)
    if tensor.dtype == trt.bfloat16:
        scale = network.add_cast(scale, trt.bfloat16).get_output(0)
    return network.add_elementwise(
        tensor, scale, trt.ElementWiseOperation.PROD).get_output(0)


def _fp8_qdq(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    scale: trt.ITensor,
    output_dtype: trt.DataType,
) -> trt.ITensor:
    q = network.add_quantize(tensor, scale, trt.DataType.FP8)
    dq = network.add_dequantize(q.get_output(0), scale, output_dtype)
    return dq.get_output(0)


def build_plan(args: argparse.Namespace) -> bytes | None:
    logger = trt.Logger(trt.Logger.VERBOSE if args.verbose else trt.Logger.INFO)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mib << 20)
    if hasattr(trt, "ProfilingVerbosity"):
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    dtype = trt.float16 if args.dtype == "fp16" else trt.bfloat16
    shape = (args.batch, args.heads, args.seq, args.head_dim)

    q_in = network.add_input("q", dtype, shape)
    k_in = network.add_input("k", dtype, shape)
    v_in = network.add_input("v", dtype, shape)

    q_scale = _constant(network, (), args.q_scale, np.float32, "q_bmm_scale")
    k_scale = _constant(network, (), args.k_scale, np.float32, "k_bmm_scale")
    v_scale = _constant(network, (), args.v_scale, np.float32, "v_bmm_scale")
    out_scale = _constant(network, (), args.out_scale, np.float32, "bmm2_output_scale")
    softmax_scale = _constant(
        network, (), args.softmax_scale, np.float32, "softmax_scale")

    outputs: list[trt.ITensor] = []
    current_q = q_in
    score_scale = 1.0 / np.sqrt(args.head_dim)
    sqrt_score_scale = float(np.sqrt(score_scale))

    for layer_idx in range(args.layers):
        q_source = current_q if args.chain and layer_idx > 0 else q_in
        q = _scale_input(network, q_source, sqrt_score_scale)
        k = _scale_input(network, k_in, sqrt_score_scale)
        v = v_in

        q = _fp8_qdq(network, q, q_scale, dtype)
        k = _fp8_qdq(network, k, k_scale, dtype)
        v = _fp8_qdq(network, v, v_scale, dtype)

        attn = network.add_attention(
            q, k, v, trt.AttentionNormalizationOp.SOFTMAX, False)
        attn.name = f"fp8_mha_{layer_idx}"
        attn.decomposable = args.decomposable
        if args.normalization_quant:
            attn.normalization_quantize_scale = softmax_scale
            attn.normalization_quantize_to_type = trt.DataType.FP8

        out = attn.get_output(0)
        if args.bmm2_output_qdq:
            out = _fp8_qdq(network, out, out_scale, dtype)
        outputs.append(out)
        current_q = out
        if args.mark_each_attention:
            network.mark_output(out)

    if len(outputs) == 1:
        final = outputs[0]
    else:
        cat = network.add_concatenation(outputs)
        cat.axis = 2
        final = cat.get_output(0)
    network.mark_output(final)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        return None
    return bytes(serialized)


def dump_layer_info(plan: bytes, output_path: Path) -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    if engine is None:
        raise RuntimeError("failed to deserialize the just-built plan")
    inspector = engine.create_engine_inspector()
    info = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    output_path.write_text(info, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--seq", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--workspace-mib", type=int, default=1024)
    parser.add_argument("--q-scale", type=float, default=0.01)
    parser.add_argument("--k-scale", type=float, default=0.01)
    parser.add_argument("--v-scale", type=float, default=0.01)
    parser.add_argument("--out-scale", type=float, default=0.01)
    parser.add_argument("--softmax-scale", type=float, default=1.0 / 448.0)
    parser.add_argument("--no-normalization-quant", dest="normalization_quant",
                        action="store_false")
    parser.add_argument("--no-bmm2-output-qdq", dest="bmm2_output_qdq",
                        action="store_false")
    parser.add_argument("--decomposable", action="store_true")
    parser.add_argument("--chain", action="store_true")
    parser.add_argument("--mark-each-attention", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-plan", type=Path)
    parser.add_argument("--dump-layer-info", type=Path)
    parser.set_defaults(normalization_quant=True, bmm2_output_qdq=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"TensorRT version: {trt.__version__}", flush=True)
    plan = build_plan(args)
    if plan is None:
        print("BUILD_FAILED", flush=True)
        return 2
    print(f"BUILD_OK plan_bytes={len(plan)}", flush=True)
    if args.output_plan:
        args.output_plan.write_bytes(plan)
        print(f"Wrote plan: {args.output_plan}", flush=True)
    if args.dump_layer_info:
        dump_layer_info(plan, args.dump_layer_info)
        print(f"Wrote layer info: {args.dump_layer_info}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
