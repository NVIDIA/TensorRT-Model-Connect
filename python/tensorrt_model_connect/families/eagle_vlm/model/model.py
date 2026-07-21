# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph operations used by Eagle VLM's text and vision encoders."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tensorrt_model_connect import trt_compat

if TYPE_CHECKING:
    from ..weights import WeightDict


trt = trt_compat.get_trt()


def _cast_to(network, tensor, dtype):
    return tensor if tensor.dtype == dtype else network.add_cast(tensor, dtype).get_output(0)


def add_constant(
    network: trt.INetworkDefinition,
    shape: tuple[int, ...],
    values: np.ndarray,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    layer = network.add_constant(shape, trt.Weights(np.ascontiguousarray(values, dtype=dtype)))
    return layer.get_output(0)


def add_matmul_rhs_constant(
    network: trt.INetworkDefinition,
    lhs: trt.ITensor,
    lhs_width: int,
    rhs_width: int,
    rhs_weights: np.ndarray,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Return ``lhs @ rhs_weights`` with family precision boundaries."""
    rank = len(tuple(lhs.shape))
    shape = (1,) * max(rank - 2, 0) + (lhs_width, rhs_width)
    rhs = add_constant(network, shape, np.asarray(rhs_weights).reshape(shape), dtype=dtype)
    rhs = _cast_to(network, rhs, lhs.dtype)
    layer = network.add_matrix_multiply(
        lhs, trt.MatrixOperation.NONE, rhs, trt.MatrixOperation.NONE
    )
    return _cast_to(network, layer.get_output(0), lhs.dtype)


def add_bias_sum(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    width: int,
    bias: np.ndarray,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    rank = len(tuple(inp.shape))
    shape = (1,) * max(rank - 1, 0) + (width,)
    bias_tensor = add_constant(network, shape, np.asarray(bias).reshape(shape), dtype=dtype)
    layer = network.add_elementwise(
        inp, _cast_to(network, bias_tensor, inp.dtype), trt.ElementWiseOperation.SUM
    )
    return _cast_to(network, layer.get_output(0), inp.dtype)


def add_rms_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    gamma: np.ndarray,
    eps_tensor: trt.ITensor,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Apply Eagle's FP32-compute RMSNorm and restore the input dtype."""
    output_dtype = inp.dtype
    if dtype != np.float32:
        inp = _cast_to(network, inp, trt.float32)
        eps_tensor = _cast_to(network, eps_tensor, trt.float32)
    squared = network.add_elementwise(inp, inp, trt.ElementWiseOperation.PROD)
    mean = network.add_reduce(
        squared.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True
    )
    variance = network.add_elementwise(
        mean.get_output(0), eps_tensor, trt.ElementWiseOperation.SUM
    )
    root = network.add_unary(variance.get_output(0), trt.UnaryOperation.SQRT)
    reciprocal = network.add_unary(root.get_output(0), trt.UnaryOperation.RECIP)
    normalized = network.add_elementwise(
        inp, reciprocal.get_output(0), trt.ElementWiseOperation.PROD
    )
    scale = add_constant(network, (1, hidden_size), gamma, dtype=np.float32)
    result = network.add_elementwise(
        normalized.get_output(0), scale, trt.ElementWiseOperation.PROD
    ).get_output(0)
    return _cast_to(network, result, output_dtype)


def validate_native_rope_dim(rotary_embedding_dim: int, *, field_name: str = "head_dim") -> int:
    rotary_embedding_dim = int(rotary_embedding_dim)
    if rotary_embedding_dim < 2 or rotary_embedding_dim % 2:
        raise ValueError(
            f"TRT native RoPE requires {field_name} to be an even value >= 2; "
            f"got {rotary_embedding_dim}"
        )
    return rotary_embedding_dim


def make_rope_table_half_dim(
    max_cache_length: int,
    head_dim: int,
    rope_theta: float,
    cosine: bool,
) -> np.ndarray:
    """Build the non-interleaved Llama RoPE half table used by Eagle."""
    head_dim = validate_native_rope_dim(head_dim)
    if max_cache_length <= 0 or rope_theta <= 0:
        value = 1.0 if cosine else 0.0
        return np.full((max(max_cache_length, 1), head_dim // 2), value, dtype=np.float32)
    positions = np.arange(max_cache_length, dtype=np.float64)[:, None]
    dimensions = np.arange(0, head_dim, 2, dtype=np.float64)[None, :]
    angles = positions * np.power(rope_theta, -dimensions / head_dim)
    return np.asarray(np.cos(angles) if cosine else np.sin(angles), dtype=np.float32)


def _rows_to_heads(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    num_heads: int,
    head_dim: int,
    sequence_length: int,
) -> trt.ITensor:
    split = network.add_shuffle(tensor)
    split.reshape_dims = (sequence_length, num_heads, head_dim)
    split.second_transpose = trt.Permutation([1, 0, 2])
    batched = network.add_shuffle(split.get_output(0))
    batched.reshape_dims = (1, num_heads, sequence_length, head_dim)
    return batched.get_output(0)


def _heads_to_rows(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    attention_size: int,
    sequence_length: int,
) -> trt.ITensor:
    rows = network.add_shuffle(tensor)
    rows.first_transpose = trt.Permutation([0, 2, 1, 3])
    rows.reshape_dims = (sequence_length, attention_size)
    return rows.get_output(0)


def add_2d_mask_to_4d(network: trt.INetworkDefinition, mask_2d: trt.ITensor) -> trt.ITensor:
    """Reshape an additive ``[query, key]`` mask to ``[1, 1, query, key]``."""
    shape = network.add_shape(mask_2d).get_output(0)
    ones = add_constant(network, (2,), np.array([1, 1], dtype=np.int64), dtype=np.int64)
    target = network.add_concatenation([ones, shape])
    target.axis = 0
    mask = network.add_shuffle(mask_2d)
    mask.set_input(1, target.get_output(0))
    return mask.get_output(0)


def add_apply_rope_native(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    num_heads: int,
    head_dim: int,
    cos_cache_2d: trt.ITensor,
    sin_cache_2d: trt.ITensor,
    position_id: trt.ITensor,
    rotary_embedding_dim: int,
    *,
    sequence_length: int,
) -> trt.ITensor:
    """Apply Eagle's full-dimension, non-interleaved Llama RoPE."""
    rotary_embedding_dim = validate_native_rope_dim(rotary_embedding_dim)
    inp_4d = _rows_to_heads(network, inp, num_heads, head_dim, sequence_length)
    positions = network.add_shuffle(position_id)
    positions.reshape_dims = (1, sequence_length)
    rope = network.add_rotary_embedding(
        inp_4d, cos_cache_2d, sin_cache_2d, False, rotary_embedding_dim
    )
    rope.set_input(3, positions.get_output(0))
    return _heads_to_rows(
        network, rope.get_output(0), num_heads * head_dim, sequence_length
    )


def _attention(
    network: trt.INetworkDefinition,
    q_4d: trt.ITensor,
    k_4d: trt.ITensor,
    v_4d: trt.ITensor,
    *,
    mask: trt.ITensor | None,
    scale: float,
) -> trt.ITensor:
    scale_dtype = np.float16 if q_4d.dtype == trt.float16 else np.float32
    scale_tensor = add_constant(
        network, (1, 1, 1, 1), np.array([scale]), dtype=scale_dtype
    )
    scale_tensor = _cast_to(network, scale_tensor, q_4d.dtype)
    scaled_q = network.add_elementwise(q_4d, scale_tensor, trt.ElementWiseOperation.PROD)
    attention = network.add_attention(
        scaled_q.get_output(0),
        k_4d,
        v_4d,
        trt.AttentionNormalizationOp.SOFTMAX,
        False,
    )
    attention.decomposable = True
    if mask is not None:
        attention.mask = mask
    return attention.get_output(0)


def add_attention_from_rows(
    network: trt.INetworkDefinition,
    q: trt.ITensor,
    k: trt.ITensor,
    v: trt.ITensor,
    *,
    num_heads: int,
    head_dim: int,
    q_seq: int,
    kv_seq: int,
    num_kv_heads: int | None = None,
    mask: trt.ITensor | None = None,
    scale: float | None = None,
) -> trt.ITensor:
    """Apply bidirectional attention to Eagle's row-major Q/K/V tensors."""
    kv_heads = num_heads if num_kv_heads is None else num_kv_heads
    q_4d = _rows_to_heads(network, q, num_heads, head_dim, q_seq)
    k_4d = _rows_to_heads(network, k, kv_heads, head_dim, kv_seq)
    v_4d = _rows_to_heads(network, v, kv_heads, head_dim, kv_seq)
    if scale is None:
        scale = float(1.0 / np.sqrt(head_dim)) if head_dim > 0 else 1.0
    context = _attention(network, q_4d, k_4d, v_4d, mask=mask, scale=scale)
    return _heads_to_rows(network, context, num_heads * head_dim, q_seq)


def infer_kv_attention_size(
    weights: dict,
    *,
    prefix: str = "layer.0",
    num_kv_heads: int,
    head_dim: int,
) -> int:
    """Validate and return Eagle's compact GQA K/V projection width."""
    expected = int(num_kv_heads * head_dim)
    explicit = weights.get("_kv_attention_size")
    if explicit is not None and int(explicit) != expected:
        raise ValueError(
            f"Compact K/V width must be {expected}, got _kv_attention_size={int(explicit)}"
        )
    weight = weights.get(f"{prefix}.w_k")
    if isinstance(weight, np.ndarray) and weight.ndim == 2 and int(weight.shape[1]) != expected:
        raise ValueError(f"{prefix}.w_k must use compact K/V width {expected}")
    return expected


def add_swiglu_mlp(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    mlp_size: int,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Apply Eagle's bias-free SiLU-gated MLP."""
    gate = add_matmul_rhs_constant(
        network, inp, hidden_size, mlp_size, weights[f"{prefix}.w_gate"], dtype=dtype
    )
    up = add_matmul_rhs_constant(
        network, inp, hidden_size, mlp_size, weights[f"{prefix}.w_up"], dtype=dtype
    )
    sigmoid = network.add_activation(gate, trt.ActivationType.SIGMOID)
    swish = network.add_elementwise(gate, sigmoid.get_output(0), trt.ElementWiseOperation.PROD)
    gated = network.add_elementwise(swish.get_output(0), up, trt.ElementWiseOperation.PROD)
    return add_matmul_rhs_constant(
        network,
        gated.get_output(0),
        mlp_size,
        hidden_size,
        weights[f"{prefix}.w_down"],
        dtype=dtype,
    )
