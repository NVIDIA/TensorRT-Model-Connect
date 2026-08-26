# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correctness-first attention graph-building primitives for fixed linear KV caches.

The helpers in this module are intentionally limited to the native decoder
contract shared by Qwen, Llama, and LFM2: batch one, a fixed-capacity LINEAR
cache, contiguous appends, and a runtime length that includes the rows written
by the current enqueue.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from tensorrt_model_connect import trt_compat


trt = trt_compat.get_trt()


# Explicit attention materializes O(Sq * K) scores. Keep one enqueue bounded;
# the Qwen and Llama runtimes transparently split longer prompts into chunks.
EXPLICIT_ATTENTION_PREFILL_CHUNK_TOKENS = 64


class NativeKvMasks(NamedTuple):
    """Masks shared by every attention layer in one decoder engine."""

    attention: trt.ITensor
    active_prefix: trt.ITensor


def _constant(
    network: trt.INetworkDefinition,
    shape: tuple[int, ...],
    values: np.ndarray,
    *,
    dtype: np.dtype = np.dtype(np.float32),
) -> trt.ITensor:
    weights = trt.Weights(np.ascontiguousarray(values, dtype=dtype))
    return network.add_constant(shape, weights).get_output(0)


def _cast(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    dtype: trt.DataType,
) -> trt.ITensor:
    if tensor.dtype == dtype:
        return tensor
    return network.add_cast(tensor, dtype).get_output(0)


def add_active_prefix_causal_masks(
    network: trt.INetworkDefinition,
    query_rows: trt.ITensor,
    cache_write_indices: trt.ITensor,
    key_value_lengths: trt.ITensor,
    cache_capacity: int,
) -> NativeKvMasks:
    """Build BOOL attention and active-prefix masks for a cache append.

    Row ``r`` may attend exactly when both ``k < key_value_lengths[0]`` and
    ``k < cache_write_indices[0] + r + 1``. A BOOL mask avoids finite additive
    sentinels and prevents an inactive, uninitialized cache suffix from
    influencing the softmax. The separate ``[1, 1, 1, K]`` active-prefix
    mask also sanitizes inactive K/V rows before matrix multiplication, so
    uninitialized non-finite cache data cannot leak through ``0 * NaN``.
    """
    if cache_capacity <= 0:
        raise ValueError("native KV cache capacity must be positive")

    query_shape = network.add_shape(query_rows).get_output(0)
    query_length = network.add_slice(
        query_shape, start=(0,), shape=(1,), stride=(1,)
    ).get_output(0)
    zero_i32 = _constant(
        network, (), np.array(0, dtype=np.int32), dtype=np.dtype(np.int32)
    )
    one_i32 = _constant(
        network,
        (1,),
        np.array([1], dtype=np.int32),
        dtype=np.dtype(np.int32),
    )
    query_iota = network.add_fill((1,), trt.FillOperation.LINSPACE, trt.int32)
    query_iota.set_input(0, query_length)
    query_iota.set_input(1, zero_i32)
    query_iota.set_input(2, one_i32)
    query_positions = network.add_elementwise(
        query_iota.get_output(0),
        cache_write_indices,
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    query_limits = network.add_elementwise(
        query_positions,
        one_i32,
        trt.ElementWiseOperation.SUM,
    ).get_output(0)

    one_i64 = _constant(
        network,
        (1,),
        np.array([1], dtype=np.int64),
        dtype=np.dtype(np.int64),
    )
    query_limit_shape = network.add_concatenation([query_length, one_i64])
    query_limit_shape.axis = 0
    query_limits_2d = network.add_shuffle(query_limits)
    query_limits_2d.set_input(1, query_limit_shape.get_output(0))

    key_positions = _constant(
        network,
        (1, cache_capacity),
        np.arange(cache_capacity, dtype=np.int32).reshape(1, cache_capacity),
        dtype=np.dtype(np.int32),
    )
    active_length_2d = network.add_shuffle(key_value_lengths)
    active_length_2d.reshape_dims = (1, 1)
    causal = network.add_elementwise(
        key_positions,
        query_limits_2d.get_output(0),
        trt.ElementWiseOperation.LESS,
    ).get_output(0)
    active = network.add_elementwise(
        key_positions,
        active_length_2d.get_output(0),
        trt.ElementWiseOperation.LESS,
    ).get_output(0)
    valid = network.add_elementwise(
        causal,
        active,
        trt.ElementWiseOperation.AND,
    ).get_output(0)

    mask_shape = network.add_shape(valid).get_output(0)
    leading_ones = _constant(
        network,
        (2,),
        np.array([1, 1], dtype=np.int64),
        dtype=np.dtype(np.int64),
    )
    target_shape = network.add_concatenation([leading_ones, mask_shape])
    target_shape.axis = 0
    mask_4d = network.add_shuffle(valid)
    mask_4d.set_input(1, target_shape.get_output(0))

    active_4d = network.add_shuffle(active)
    active_4d.reshape_dims = (1, 1, 1, cache_capacity)
    return NativeKvMasks(
        attention=mask_4d.get_output(0),
        active_prefix=active_4d.get_output(0),
    )


def _blocked_score(
    network: trt.INetworkDefinition,
    dtype: trt.DataType,
) -> trt.ITensor:
    if dtype == trt.float16:
        return _constant(
            network,
            (1, 1, 1, 1, 1),
            np.array([-1.0e4], dtype=np.float16),
            dtype=np.dtype(np.float16),
        )
    blocked = _constant(
        network,
        (1, 1, 1, 1, 1),
        np.array([-1.0e30], dtype=np.float32),
    )
    return _cast(network, blocked, dtype)


def add_explicit_masked_grouped_query_attention(
    network: trt.INetworkDefinition,
    query: trt.ITensor,
    key: trt.ITensor,
    value: trt.ITensor,
    masks: NativeKvMasks,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float | None = None,
    tag: str | None = None,
) -> trt.ITensor:
    """Compute masked GQA explicitly as QK^T, softmax, and PV.

    Inputs use TensorRT's padded attention layout: query is
    ``[1, Hq, Sq, D]`` and key/value are ``[1, Hkv, K, D]``. Query heads are
    grouped as ``[Hkv, Hq/Hkv]`` so K/V broadcast across each group without
    materializing repeated full-capacity caches.

    This primitive graph is deliberate. TensorRT 11.1 can fail its FP16
    four-input IAttention wrapper even when decomposition is enabled, while
    the basic matrix, select, and softmax operations work for both FP16 and
    BF16 on the same graph contract.
    """
    if num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads={num_heads} must be divisible by "
            f"num_kv_heads={num_kv_heads}"
        )
    if head_dim <= 0:
        raise ValueError("native KV attention head_dim must be positive")
    if query.dtype not in {trt.float16, trt.bfloat16, trt.float32}:
        raise ValueError("native KV attention requires FP16, BF16, or FP32")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("native KV query, key, and value dtypes must match")
    if masks.attention.dtype != trt.bool or masks.active_prefix.dtype != trt.bool:
        raise ValueError("native KV attention requires BOOL explicit masks")

    groups = num_heads // num_kv_heads
    query_grouped = network.add_shuffle(query)
    query_grouped.reshape_dims = (
        1,
        num_kv_heads,
        groups,
        -1,
        head_dim,
    )
    key_grouped = network.add_shuffle(key)
    key_grouped.reshape_dims = (1, num_kv_heads, 1, -1, head_dim)
    value_grouped = network.add_shuffle(value)
    value_grouped.reshape_dims = (1, num_kv_heads, 1, -1, head_dim)
    mask_grouped = network.add_shuffle(masks.attention)
    mask_grouped.reshape_dims = (1, 1, 1, -1, int(key.shape[-2]))
    active_grouped = network.add_shuffle(masks.active_prefix)
    active_grouped.reshape_dims = (1, 1, 1, int(key.shape[-2]), 1)

    zero = _constant(
        network,
        (1, 1, 1, 1, 1),
        np.array([0.0], dtype=np.float32),
    )
    zero = _cast(network, zero, query.dtype)
    safe_key = network.add_select(
        active_grouped.get_output(0),
        key_grouped.get_output(0),
        zero,
    )
    safe_value = network.add_select(
        active_grouped.get_output(0),
        value_grouped.get_output(0),
        zero,
    )
    if safe_key is None or safe_value is None:
        raise RuntimeError("TensorRT failed to sanitize inactive native KV rows")

    if scale is None:
        scale = float(1.0 / np.sqrt(head_dim))
    query_fp32 = network.add_cast(
        query_grouped.get_output(0), trt.float32
    ).get_output(0)
    scale_tensor = _constant(
        network,
        (1, 1, 1, 1, 1),
        np.array([scale], dtype=np.float32),
    )
    scaled_query = network.add_elementwise(
        query_fp32,
        scale_tensor,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    scaled_query = _cast(network, scaled_query, query.dtype)

    # The current native-KV ABI profiles KVL only as an execution input. Using
    # it as a dynamic ISlice size would also make it a shape input and require
    # shape-value profile/runtime plumbing. This correctness-first graph keeps
    # the existing ABI and deliberately masks the complete configured capacity.
    scores = network.add_matrix_multiply(
        scaled_query,
        trt.MatrixOperation.NONE,
        safe_key.get_output(0),
        trt.MatrixOperation.TRANSPOSE,
    )
    if scores is None:
        raise RuntimeError("TensorRT failed to create native KV score matmul")
    if tag:
        scores.name = tag + ".scores"
    masked_scores = network.add_select(
        mask_grouped.get_output(0),
        scores.get_output(0),
        _blocked_score(network, query.dtype),
    )
    if masked_scores is None:
        raise RuntimeError("TensorRT failed to apply native KV attention mask")
    if tag:
        masked_scores.name = tag + ".mask"

    probabilities = network.add_softmax(masked_scores.get_output(0))
    if probabilities is None:
        raise RuntimeError("TensorRT failed to create native KV softmax")
    probabilities.axes = 1 << 4
    if tag:
        probabilities.name = tag + ".softmax"
    context_grouped = network.add_matrix_multiply(
        probabilities.get_output(0),
        trt.MatrixOperation.NONE,
        safe_value.get_output(0),
        trt.MatrixOperation.NONE,
    )
    if context_grouped is None:
        raise RuntimeError("TensorRT failed to create native KV context matmul")
    if tag:
        context_grouped.name = tag + ".context"

    context = network.add_shuffle(context_grouped.get_output(0))
    context.reshape_dims = (1, num_heads, -1, head_dim)
    return context.get_output(0)
