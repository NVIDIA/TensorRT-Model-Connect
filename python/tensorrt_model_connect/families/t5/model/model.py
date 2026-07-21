# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph helpers used by T5 encoder and decoder builders."""

from __future__ import annotations

import numpy as np

from tensorrt_model_connect import trt_compat


trt = trt_compat.get_trt()


def _cast(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    dtype: trt.DataType,
) -> trt.ITensor:
    if tensor.dtype == dtype:
        return tensor
    return network.add_cast(tensor, dtype).get_output(0)


def layer_tensor_name(stem: str, layer: int) -> str:
    return f"{stem}_{layer}"


def add_constant(
    network: trt.INetworkDefinition,
    shape: tuple[int, ...],
    values: np.ndarray,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    weights = trt.Weights(np.ascontiguousarray(values, dtype=dtype))
    return network.add_constant(shape, weights).get_output(0)


def add_matmul_rhs_constant(
    network: trt.INetworkDefinition,
    lhs: trt.ITensor,
    lhs_width: int,
    rhs_width: int,
    rhs_weights: np.ndarray,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Multiply row-major activations by a family-owned constant projection."""
    rank = len(tuple(lhs.shape))
    rhs_shape = (
        (lhs_width, rhs_width)
        if rank <= 2
        else (1,) * (rank - 2) + (lhs_width, rhs_width)
    )
    rhs = add_constant(
        network,
        rhs_shape,
        np.asarray(rhs_weights).reshape(rhs_shape),
        dtype=dtype,
    )
    rhs = _cast(network, rhs, lhs.dtype)
    result = network.add_matrix_multiply(
        lhs,
        trt.MatrixOperation.NONE,
        rhs,
        trt.MatrixOperation.NONE,
    ).get_output(0)
    return _cast(network, result, lhs.dtype)


def add_rms_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    gamma: np.ndarray,
    eps_tensor: trt.ITensor,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Apply T5 RMSNorm, using FP32 arithmetic for reduced-precision builds."""
    output_dtype = inp.dtype
    reduced_precision = dtype != np.float32
    if reduced_precision:
        inp = _cast(network, inp, trt.float32)
        eps_tensor = _cast(network, eps_tensor, trt.float32)

    squared = network.add_elementwise(inp, inp, trt.ElementWiseOperation.PROD)
    mean = network.add_reduce(
        squared.get_output(0),
        trt.ReduceOperation.AVG,
        1 << 1,
        keep_dims=True,
    )
    variance = network.add_elementwise(
        mean.get_output(0), eps_tensor, trt.ElementWiseOperation.SUM
    )
    root = network.add_unary(variance.get_output(0), trt.UnaryOperation.SQRT)
    reciprocal = network.add_unary(root.get_output(0), trt.UnaryOperation.RECIP)
    normalized = network.add_elementwise(
        inp, reciprocal.get_output(0), trt.ElementWiseOperation.PROD
    )
    gamma_tensor = add_constant(
        network, (1, hidden_size), gamma, dtype=np.float32
    )
    result = network.add_elementwise(
        normalized.get_output(0), gamma_tensor, trt.ElementWiseOperation.PROD
    ).get_output(0)
    return _cast(network, result, output_dtype) if reduced_precision else result


def make_t5_relative_position_bias(
    num_heads: int,
    max_seq_len: int,
    num_buckets: int = 32,
    max_distance: int = 128,
) -> np.ndarray:
    """Return bidirectional T5 relative-position bucket indices."""
    del num_heads
    context = np.arange(max_seq_len, dtype=np.int32)[:, None]
    memory = np.arange(max_seq_len, dtype=np.int32)[None, :]
    distance = -(memory - context)
    half = num_buckets // 2
    buckets = (distance < 0).astype(np.int32) * half
    distance = np.abs(distance)
    exact = half // 2
    clamped = np.maximum(distance.astype(np.float32), 1)
    logarithmic = exact + (
        np.log(clamped / exact)
        / np.log(max_distance / exact)
        * (half - exact)
    ).astype(np.int32)
    buckets += np.where(
        distance < exact,
        distance,
        np.minimum(logarithmic, half - 1),
    )
    return buckets.astype(np.int32)


def _rows_to_heads(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    num_heads: int,
    head_dim: int,
    sequence_length: int | None,
) -> trt.ITensor:
    sequence = -1 if sequence_length is None else sequence_length
    split = network.add_shuffle(tensor)
    split.reshape_dims = (sequence, num_heads, head_dim)
    split.second_transpose = trt.Permutation([1, 0, 2])
    batched = network.add_shuffle(split.get_output(0))
    batched.reshape_dims = (1, num_heads, sequence, head_dim)
    return batched.get_output(0)


def _heads_to_rows(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    attention_size: int,
    sequence_length: int | None,
) -> trt.ITensor:
    sequence = -1 if sequence_length is None else sequence_length
    rows = network.add_shuffle(tensor)
    rows.first_transpose = trt.Permutation([0, 2, 1, 3])
    rows.reshape_dims = (sequence, attention_size)
    return rows.get_output(0)


def add_2d_mask_to_4d(
    network: trt.INetworkDefinition,
    mask_2d: trt.ITensor,
) -> trt.ITensor:
    """Reshape an additive ``[query, key]`` mask to IAttention layout."""
    shape = network.add_shape(mask_2d).get_output(0)
    ones = add_constant(
        network, (2,), np.array([1, 1], dtype=np.int64), dtype=np.int64
    )
    target = network.add_concatenation([ones, shape])
    target.axis = 0
    mask = network.add_shuffle(mask_2d)
    mask.set_input(1, target.get_output(0))
    return mask.get_output(0)


def add_attention_from_rows(
    network: trt.INetworkDefinition,
    q: trt.ITensor,
    k: trt.ITensor,
    v: trt.ITensor,
    *,
    num_heads: int,
    head_dim: int,
    q_seq: int | None,
    kv_seq: int | None,
    mask: trt.ITensor | None = None,
    scale: float = 1.0,
) -> trt.ITensor:
    """Apply T5 multi-head attention to row-major Q/K/V tensors."""
    q_heads = _rows_to_heads(network, q, num_heads, head_dim, q_seq)
    k_heads = _rows_to_heads(network, k, num_heads, head_dim, kv_seq)
    v_heads = _rows_to_heads(network, v, num_heads, head_dim, kv_seq)

    scale_dtype = np.float16 if q_heads.dtype == trt.float16 else np.float32
    scale_tensor = add_constant(
        network,
        (1, 1, 1, 1),
        np.array([[[[scale]]]], dtype=scale_dtype),
        dtype=scale_dtype,
    )
    if q_heads.dtype == trt.bfloat16:
        scale_tensor = _cast(network, scale_tensor, trt.bfloat16)
    scaled_q = network.add_elementwise(
        q_heads, scale_tensor, trt.ElementWiseOperation.PROD
    )
    attention = network.add_attention(
        scaled_q.get_output(0),
        k_heads,
        v_heads,
        trt.AttentionNormalizationOp.SOFTMAX,
        False,
    )
    attention.decomposable = True
    if mask is not None:
        attention.mask = mask
    context = _cast(network, attention.get_output(0), q_heads.dtype)
    return _heads_to_rows(
        network,
        context,
        num_heads * head_dim,
        q_seq,
    )
