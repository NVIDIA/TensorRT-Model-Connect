#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a source-input attention probe for Wan2.2 GB300 parity analysis."""

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path

import numpy as np

from tensorrt_model_connect import trt_compat
from tensorrt_model_connect.families.wan2_2_ti2v import trt_ops as op
from tensorrt_model_connect.families.wan2_2_ti2v.model_config import WAN22_TI2V_5B


trt = trt_compat.get_trt()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("self", "cross"), required=True)
    parser.add_argument("--fp32-accumulation", action="store_true")
    parser.add_argument("--source-attention-plugin", type=Path)
    args = parser.parse_args()
    if args.source_attention_plugin is not None:
        ctypes.CDLL(str(args.source_attention_plugin.resolve()), mode=ctypes.RTLD_GLOBAL)
    cfg = WAN22_TI2V_5B
    q_seq = cfg.num_patches
    kv_seq = cfg.num_patches if args.kind == "self" else cfg.text_seq_len
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    build_config = builder.create_builder_config()
    build_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 160 << 30)
    q = network.add_input("q", trt.bfloat16, (1, cfg.num_heads, q_seq, cfg.head_dim))
    k = network.add_input("k", trt.bfloat16, (1, cfg.num_heads, kv_seq, cfg.head_dim))
    v = network.add_input("v", trt.bfloat16, (1, cfg.num_heads, kv_seq, cfg.head_dim))
    output_dtype = q.dtype
    if args.source_attention_plugin is not None:
        creator = trt.get_plugin_registry().get_creator("Wan22SourceAttention", "1", "")
        if creator is None:
            raise RuntimeError("Wan22SourceAttention creator is not registered")
        plugin = creator.create_plugin("wan2_2_source_attention", trt.PluginFieldCollection([]))
        layer = network.add_plugin_v2([q, k, v], plugin)
        output = op.cast(network, layer.get_output(0), trt.float32)
        output.name = "context"
        network.mark_output(output)
        plan = builder.build_serialized_network(network, build_config)
        if plan is None:
            raise RuntimeError("TensorRT failed to build the Wan2.2 attention probe")
        serialized = bytes(plan)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(serialized)
        print(f"wrote {args.output} ({len(serialized)} bytes)")
        return 0
    if args.fp32_accumulation:
        q = op.cast(network, q, trt.float32)
        k = op.cast(network, k, trt.float32)
        v = op.cast(network, v, trt.float32)
    scale = op.constant(
        network,
        np.array([[[[1.0 / np.sqrt(cfg.head_dim)]]]], dtype=np.float32),
    )
    scale = op.cast(network, scale, q.dtype)
    q = network.add_elementwise(q, scale, trt.ElementWiseOperation.PROD).get_output(0)
    attention = network.add_attention(q, k, v, trt.AttentionNormalizationOp.SOFTMAX, False)
    attention.decomposable = True
    output = op.cast(network, attention.get_output(0), output_dtype)
    output = op.cast(network, output, trt.float32)
    output.name = "context"
    network.mark_output(output)
    plan = builder.build_serialized_network(network, build_config)
    if plan is None:
        raise RuntimeError("TensorRT failed to build the Wan2.2 attention probe")
    serialized = bytes(plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(serialized)
    print(f"wrote {args.output} ({len(serialized)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
