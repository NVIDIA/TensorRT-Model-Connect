# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.6/3.8 native head and SwiGLU tensor partitioning.

Embedding, norms and the vocabulary projection are replicated. Attention heads,
DeltaNet recurrent heads and MLP intermediate channels are partitioned. Each
row-parallel output is reduced before the residual and subsequent norm.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import tensorrt as trt

from .config import ModelConfig


def shard_weights(config: ModelConfig, weights: dict, rank: int, size: int = 2):
    """Return an owning rank's graph geometry and weights without changing input."""
    if size != 2 or type(rank) is not int or not 0 <= rank < size:
        raise ValueError("Qwen native tensor parallel requires two ranks")
    if weights.get("_fp8_kv_cache"):
        raise ValueError("Qwen native TP2 requires original unquantized KV storage")
    dimensions = {
        "attention heads": config.num_attention_heads,
        "KV heads": config.num_key_value_heads,
        "MLP channels": int(weights["_mlp_size"]),
        "DeltaNet value heads": int(weights["_deltanet_num_heads"]),
        "DeltaNet key heads": int(weights["_deltanet_num_kv_heads"]),
    }
    if any(value <= 0 or value % size for value in dimensions.values()):
        raise ValueError(f"Qwen TP2 requires divisible head/channel dimensions: {dimensions}")
    if dimensions["attention heads"] % dimensions["KV heads"]:
        raise ValueError("Qwen attention Q heads must be divisible by KV heads")
    if dimensions["DeltaNet value heads"] % dimensions["DeltaNet key heads"]:
        raise ValueError("Qwen DeltaNet value heads must be divisible by key heads")
    head = int(weights["_deltanet_head_dim"])
    qk = dimensions["DeltaNet key heads"] * head
    value = dimensions["DeltaNet value heads"] * head
    if weights["_conv_dim"] != 2 * qk + value or weights["_d_inner"] != value:
        raise ValueError("Qwen DeltaNet state geometry disagrees with packed QKV")

    def part(array, axis):
        if array.shape[axis] % size:
            raise ValueError("Qwen TP2 weight axis is not divisible by rank count")
        return np.ascontiguousarray(np.split(array, size, axis=axis)[rank])

    def qkv_part(array, axis):
        # The projection and convolution are Q | K | V, not interleaved heads.
        # A contiguous half of the complete matrix would cross those boundaries.
        if array.shape[axis] != 2 * qk + value:
            raise ValueError("Qwen TP2 packed QKV weight has invalid width")
        return np.ascontiguousarray(np.concatenate(
            [part(block, axis) for block in np.split(array, [qk, 2 * qk], axis=axis)],
            axis=axis))

    result = type(weights)()
    column = (".w_q", ".w_k", ".w_v", ".w_gate_attn", ".w_gate", ".w_up",
              ".deltanet_z_proj", ".deltanet_a_proj", ".deltanet_b_proj")
    row = (".w_o", ".w_down", ".deltanet_out_proj")
    vectors = (".q_norm", ".k_norm", ".A", ".dt_bias", ".deltanet_norm")
    for key, weight in weights.items():
        if not isinstance(weight, np.ndarray):
            result[key] = weight
        elif key.endswith(".deltanet_in_proj_qkv"):
            result[key] = qkv_part(weight, -1)
        elif key.endswith((".conv1d_weight", ".conv1d_bias")):
            result[key] = qkv_part(weight, 0)
        elif key.endswith(column):
            result[key] = part(weight, -1)
        elif key.endswith(row):
            result[key] = part(weight, 0)
        elif key.endswith(vectors):
            result[key] = part(weight, -1)
        else:
            result[key] = weight
    for name in ("_d_inner", "_conv_dim", "_deltanet_num_heads",
                 "_deltanet_num_kv_heads", "_attn_size", "_mlp_size"):
        result[name] = int(weights[name]) // size
    result["_tensor_parallel_size"] = size
    result["_tensor_parallel_rank"] = rank
    local = replace(config, num_attention_heads=config.num_attention_heads // size,
                    num_key_value_heads=config.num_key_value_heads // size,
                    intermediate_size=config.intermediate_size // size,
                    _head_dim=config.head_dim)
    return local, result


def add_all_reduce_sum(network, tensor, weights):
    """TensorRT owns collective lowering; no family kernel or hardware policy."""
    size = int(weights.get("_tensor_parallel_size", 1))
    if size == 1:
        return tensor
    layer = network.add_dist_collective(tensor, trt.CollectiveOperation.ALL_REDUCE,
                                        trt.ReduceOperation.SUM, -1, [])
    if layer is None:
        raise RuntimeError("TensorRT failed to create Qwen TP ALL_REDUCE")
    layer.num_ranks = size
    return layer.get_output(0)


def broadcast_from_rank0(network, tensor, weights):
    """Identical media embeddings and sampled logits on every rank/device."""
    size = int(weights.get("_tensor_parallel_size", 1))
    if size == 1:
        return tensor
    layer = network.add_dist_collective(tensor, trt.CollectiveOperation.BROADCAST,
                                        trt.ReduceOperation.NONE, 0, [])
    if layer is None:
        raise RuntimeError("TensorRT failed to create Qwen TP BROADCAST")
    layer.num_ranks = size
    return layer.get_output(0)
