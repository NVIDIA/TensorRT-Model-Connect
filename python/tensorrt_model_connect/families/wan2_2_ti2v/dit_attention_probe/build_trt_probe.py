#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the isolated Wan2.2 cuDNN SDPA TensorRT probe plan."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np
import tensorrt as trt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--q-sequence", type=int, default=27280)
    parser.add_argument("--kv-sequence", type=int, default=27280)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--engine-id", type=int, default=10)
    parser.add_argument("--kernel-config", type=int, default=36)
    args = parser.parse_args()

    ctypes.CDLL(str(args.plugin.resolve()), mode=ctypes.RTLD_GLOBAL)
    creator = trt.get_plugin_registry().get_creator("Wan22CudnnSdpaProbe", "1", "")
    if creator is None:
        raise RuntimeError("Wan22CudnnSdpaProbe creator is not registered")

    values = {
        "batch": np.array([args.batch], dtype=np.int32),
        "heads": np.array([args.heads], dtype=np.int32),
        "q_sequence": np.array([args.q_sequence], dtype=np.int32),
        "kv_sequence": np.array([args.kv_sequence], dtype=np.int32),
        "dimension": np.array([args.dimension], dtype=np.int32),
        "engine_id": np.array([args.engine_id], dtype=np.int32),
        "kernel_config": np.array([args.kernel_config], dtype=np.int32),
    }
    fields = trt.PluginFieldCollection(
        [trt.PluginField(name, value, trt.PluginFieldType.INT32) for name, value in values.items()]
    )
    plugin = creator.create_plugin("wan2_2_cudnn_sdpa_probe", fields)
    if plugin is None:
        raise RuntimeError("Wan22CudnnSdpaProbe creation failed")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    q_physical_shape = (args.batch, args.q_sequence, args.heads, args.dimension)
    kv_physical_shape = (args.batch, args.kv_sequence, args.heads, args.dimension)
    q = network.add_input("q", trt.bfloat16, q_physical_shape)
    k = network.add_input("k", trt.bfloat16, kv_physical_shape)
    v = network.add_input("v", trt.bfloat16, kv_physical_shape)
    layer = network.add_plugin_v2([q, k, v], plugin)
    if layer is None:
        raise RuntimeError("TensorRT rejected the Wan22CudnnSdpaProbe layer")
    output = layer.get_output(0)
    output.name = "o"
    network.mark_output(output)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build the Wan2.2 SDPA probe plan")
    payload = bytes(serialized)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    report = {
        "kind": "wan2_2_ti2v_cudnn_sdpa_trt_probe_build",
        "plugin": str(args.plugin.resolve()),
        "tensorrt_version": trt.__version__,
        "q_physical_shape_bshd": list(q_physical_shape),
        "kv_physical_shape_bshd": list(kv_physical_shape),
        "q_logical_shape_bhsd": [args.batch, args.heads, args.q_sequence, args.dimension],
        "kv_logical_shape_bhsd": [args.batch, args.heads, args.kv_sequence, args.dimension],
        "q_logical_stride_bhsd": [
            args.heads * args.q_sequence * args.dimension,
            args.dimension,
            args.heads * args.dimension,
            1,
        ],
        "kv_logical_stride_bhsd": [
            args.heads * args.kv_sequence * args.dimension,
            args.dimension,
            args.heads * args.dimension,
            1,
        ],
        "engine_id": args.engine_id,
        "kernel_config": args.kernel_config,
        "plan": {
            "path": str(args.output),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
