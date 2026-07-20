#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and deserialize a fixed-profile production Wan2.2 SDPA plugin probe."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np
from tensorrt_model_connect.trt_compat import trt


_PLUGIN_NAME = "Wan22DitCudnnSdpa"
_BATCH = 1
_HEADS = 24
_Q_SEQUENCE = 27_280
_HEAD_DIMENSION = 128
_KV_SEQUENCES = {"self": 27_280, "cross": 512}
_ATTENTION_KINDS = {"self": 0, "cross": 1}


def _plugin_fields(
    attention: str,
) -> tuple[trt.PluginFieldCollection, dict[str, np.ndarray]]:
    values = {
        "attention_kind": _ATTENTION_KINDS[attention],
        "batch": _BATCH,
        "heads": _HEADS,
        "q_sequence": _Q_SEQUENCE,
        "kv_sequence": _KV_SEQUENCES[attention],
        "head_dimension": _HEAD_DIMENSION,
    }
    arrays = {name: np.array([value], dtype=np.int32) for name, value in values.items()}
    fields = trt.PluginFieldCollection(
        [trt.PluginField(name, value, trt.PluginFieldType.INT32) for name, value in arrays.items()]
    )
    return fields, arrays


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--attention", choices=sorted(_KV_SEQUENCES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    plugin_path = args.plugin.resolve()
    ctypes.CDLL(str(plugin_path), mode=ctypes.RTLD_GLOBAL)
    creator = trt.get_plugin_registry().get_creator(_PLUGIN_NAME, "1", "")
    if creator is None:
        raise RuntimeError(f"{_PLUGIN_NAME} creator is not registered")
    fields, field_storage = _plugin_fields(args.attention)
    plugin = creator.create_plugin(f"wan2_2_{args.attention}_cudnn_sdpa", fields)
    del field_storage
    if plugin is None:
        raise RuntimeError(f"{_PLUGIN_NAME} creation failed for {args.attention} attention")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    q_shape = (_BATCH, _Q_SEQUENCE, _HEADS, _HEAD_DIMENSION)
    kv_shape = (_BATCH, _KV_SEQUENCES[args.attention], _HEADS, _HEAD_DIMENSION)
    q = network.add_input("q", trt.bfloat16, q_shape)
    k = network.add_input("k", trt.bfloat16, kv_shape)
    v = network.add_input("v", trt.bfloat16, kv_shape)
    layer = network.add_plugin_v2([q, k, v], plugin)
    if layer is None:
        raise RuntimeError(f"TensorRT rejected the {_PLUGIN_NAME} layer")
    output = layer.get_output(0)
    output.name = "o"
    network.mark_output(output)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT failed to build {args.attention} SDPA probe plan")
    payload = bytes(serialized)

    # A fresh TensorRT runtime exercises the plugin's versioned deserializer;
    # its Context rebuilds the target-local HeurMode-A plan from shape fields.
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(payload)
    if engine is None:
        raise RuntimeError("fresh-runtime deserialization of the SDPA plan failed")
    observed_shapes = {name: list(engine.get_tensor_shape(name)) for name in ("q", "k", "v", "o")}
    expected_shapes = {
        "q": list(q_shape),
        "k": list(kv_shape),
        "v": list(kv_shape),
        "o": list(q_shape),
    }
    if observed_shapes != expected_shapes:
        raise RuntimeError(
            f"deserialized SDPA shapes differ: observed={observed_shapes}, "
            f"expected={expected_shapes}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    report = {
        "kind": "wan2_2_ti2v_production_cudnn_sdpa_plugin_build",
        "attention": args.attention,
        "plugin": str(plugin_path),
        "plugin_name": _PLUGIN_NAME,
        "plugin_version": "1",
        "plan_selection": "target_local_cudnn_heur_mode_a_first_supported",
        "tensorrt_version": trt.__version__,
        "physical_shapes_bshd": observed_shapes,
        "fresh_runtime_deserialization": True,
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
