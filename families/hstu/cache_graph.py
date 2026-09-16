# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HSTU-owned native TensorRT cache and attention graph operations."""

from __future__ import annotations

import math

import numpy as np


def _output(layer, name):
    if layer is None:
        raise RuntimeError(f"TensorRT rejected HSTU layer {name}")
    layer.name = name
    return layer.get_output(0)


def _constant(trt, network, value, dtype, host_weights, name):
    array = np.full((1, 1, 1, 1), value, np.float32)
    host_weights.append(array)
    tensor = _output(network.add_constant(array.shape, trt.Weights(array)), name)
    if tensor.dtype != dtype:
        tensor = _output(network.add_cast(tensor, dtype), f"{name}.cast")
    return tensor


def add_cached_kv(
    trt, network, k_update, v_update, capacity, write_indices, active_mask,
    prefix, host_weights, *, packed_rows=None, update_lengths=None,
):
    """Append K/V in place and return sanitized tensors for attention.

    ``k_update`` and ``v_update`` have shape [batch, heads, tokens, head_dim].
    With ``packed_rows`` and ``update_lengths``, only the selected flattened
    [batch, tokens] rows are written. The INT32 ``packed_rows`` vector lists
    rows in sequence order; INT32 ``update_lengths`` contains cumulative row
    counts [batch + 1], beginning with zero and ending at len(packed_rows).
    This lets a batch include sequences with different update lengths,
    including zero, without writing padded rows into their cache buffers.

    Updates may include candidate or decode rows only when the bound cache
    is request-private workspace. The runtime must exclude those ephemeral
    rows when it commits the reusable historical prefix after execution.
    ``active_mask`` is BOOL [batch, 1, 1, capacity]. It describes the valid
    prefix after this append; uninitialized rows are replaced with
    zero before attention so masked NaNs cannot enter either matrix multiply.

    The caller binds each ``cache_{prefix}_{k,v}`` input and corresponding
    ``present_{prefix}_{k,v}`` output to the same device address. Cache bounds,
    prefix validity, and the lifetime of those buffers belong to the runtime.
    The marked outputs are raw in-place cache updates; the returned tensors
    are sanitized views used only by downstream graph operations.
    """
    if type(capacity) is not int or capacity <= 0:
        raise ValueError("HSTU KV cache capacity must be a positive integer")
    if len(k_update.shape) != 4 or tuple(k_update.shape) != tuple(v_update.shape):
        raise ValueError("HSTU K/V updates must have identical rank-four shapes")
    if k_update.dtype != v_update.dtype or k_update.dtype not in {
        trt.float32, trt.float16, trt.bfloat16,
    }:
        raise ValueError("HSTU K/V updates must share FP32, FP16, or BF16 dtype")
    batch, heads, _, head_dim = k_update.shape
    if heads <= 0 or head_dim <= 0:
        raise ValueError("HSTU KV cache heads and head dimension must be static")
    if len(write_indices.shape) != 1 or write_indices.dtype not in {trt.int32, trt.int64}:
        raise ValueError("HSTU cache write indices must be a rank-one integer tensor")
    if active_mask.dtype != trt.bool or tuple(active_mask.shape[1:]) != (1, 1, capacity):
        raise ValueError("HSTU cache active mask must be BOOL [batch, 1, 1, capacity]")
    if (packed_rows is None) != (update_lengths is None):
        raise ValueError("HSTU packed cache updates require both rows and cumulative lengths")
    if packed_rows is not None:
        if packed_rows.dtype != trt.int32 or len(packed_rows.shape) != 1:
            raise ValueError("HSTU packed cache rows must be a rank-one INT32 tensor")
        if update_lengths.dtype != trt.int32 or len(update_lengths.shape) != 1:
            raise ValueError("HSTU packed cache lengths must be a rank-one INT32 tensor")

    transpose = network.add_shuffle(active_mask)
    if transpose is None:
        raise RuntimeError("TensorRT rejected HSTU cache active-mask transpose")
    transpose.first_transpose = (0, 1, 3, 2)
    valid = _output(transpose, f"cache.{prefix}.active_rows")
    zero = _constant(
        trt, network, 0.0, k_update.dtype, host_weights, f"cache.{prefix}.zero",
    )
    sanitized = []
    for kind, update in (("k", k_update), ("v", v_update)):
        cache = network.add_input(
            f"cache_{prefix}_{kind}", update.dtype, (batch, heads, capacity, head_dim),
        )
        if cache is None:
            raise RuntimeError(f"TensorRT rejected HSTU cache input {prefix}.{kind}")
        if packed_rows is not None:
            packed = network.add_shuffle(update)
            if packed is None:
                raise RuntimeError("TensorRT rejected HSTU packed cache transpose")
            packed.first_transpose = (0, 2, 1, 3)
            packed.reshape_dims = (-1, heads, head_dim)
            flattened = _output(packed, f"cache.{prefix}.{kind}.flatten_rows")
            update = _output(
                network.add_gather(flattened, packed_rows, 0),
                f"cache.{prefix}.{kind}.packed_rows",
            )
        cache_update = network.add_kv_cache_update(
            cache, update, write_indices, trt.KVCacheMode.LINEAR,
        )
        if cache_update is None:
            raise RuntimeError(f"TensorRT rejected HSTU cache update {prefix}.{kind}")
        if packed_rows is not None:
            cache_update.update_form = trt.AttentionIOForm.PACKED_NHD
            cache_update.update_lengths = update_lengths
        present = _output(cache_update, f"cache.{prefix}.{kind}.update")
        present.name = f"present_{prefix}_{kind}"
        network.mark_output(present)
        sanitized.append(_output(
            network.add_select(valid, present, zero), f"cache.{prefix}.{kind}.active",
        ))
    return tuple(sanitized)


def native_sdpa(
    trt, network, query, key, value, attention_mask, scale, host_weights, prefix,
):
    """Build scaled dot-product attention with an explicit BOOL visibility mask.

    Inputs use [batch, heads, sequence, head_dim]. ``scale`` is applied once
    to Q before TensorRT attention; no additional causal mask is introduced.
    This operation implements softmax SDPA, not HSTU's SiLU attention.
    """
    if any(len(tensor.shape) != 4 for tensor in (query, key, value, attention_mask)):
        raise ValueError("native SDPA expects rank-four Q/K/V and attention mask")
    if query.dtype not in {trt.float32, trt.float16, trt.bfloat16} or any(
        tensor.dtype != query.dtype for tensor in (key, value)
    ):
        raise ValueError("native SDPA Q/K/V must share FP32, FP16, or BF16 dtype")
    if attention_mask.dtype != trt.bool:
        raise ValueError("native SDPA visibility mask must have BOOL dtype")
    if not math.isfinite(scale):
        raise ValueError("native SDPA scale must be finite")
    multiplier = _constant(
        trt, network, scale, query.dtype, host_weights, f"{prefix}.q_scale",
    )
    scaled_query = _output(
        network.add_elementwise(query, multiplier, trt.ElementWiseOperation.PROD),
        f"{prefix}.scaled_query",
    )
    attention = network.add_attention_v2(
        scaled_query, key, value, trt.AttentionNormalizationOp.SOFTMAX,
        trt.CausalMaskKind.NONE,
    )
    if attention is None:
        raise RuntimeError(f"TensorRT rejected native SDPA {prefix}")
    attention.mask = attention_mask
    attention.decomposable = True
    result = _output(attention, prefix)
    # SDPA returns zero for an entirely masked query. A decomposed softmax
    # over only -inf scores can produce NaN; Select preserves that contract
    # without depending on which attention implementation TensorRT chooses.
    visible = _output(
        network.add_cast(attention_mask, trt.float32), f"{prefix}.visible_keys",
    )
    any_visible = _output(
        network.add_reduce(visible, trt.ReduceOperation.MAX, 1 << 3, True),
        f"{prefix}.any_visible_key",
    )
    valid_query = _output(
        network.add_cast(any_visible, trt.bool), f"{prefix}.valid_query",
    )
    zero = _constant(trt, network, 0.0, query.dtype, host_weights, f"{prefix}.zero")
    return _output(
        network.add_select(valid_query, result, zero), f"{prefix}.masked_queries",
    )
