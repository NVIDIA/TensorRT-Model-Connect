# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph operations exercised by M2M-100 and NLLB."""

from __future__ import annotations

import numpy as np

from tensorrt_model_connect import trt_compat


trt = trt_compat.get_trt()


def _cast_back_to_trt_dtype(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    target_dtype: trt.DataType,
) -> trt.ITensor:
    if tensor.dtype == target_dtype:
        return tensor
    return network.add_cast(tensor, target_dtype).get_output(0)


def layer_tensor_name(stem: str, layer: int) -> str:
    return f"{stem}_{layer}"


def add_constant(
    network: trt.INetworkDefinition,
    shape: tuple[int, ...],
    values: np.ndarray,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    layer = network.add_constant(
        shape,
        trt.Weights(np.ascontiguousarray(values, dtype=dtype)),
    )
    return layer.get_output(0)


def add_matmul_rhs_constant(
    network: trt.INetworkDefinition,
    lhs: trt.ITensor,
    lhs_width: int,
    rhs_width: int,
    rhs_weights: np.ndarray,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
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
    rhs = _cast_back_to_trt_dtype(network, rhs, lhs.dtype)
    output = network.add_matrix_multiply(
        lhs,
        trt.MatrixOperation.NONE,
        rhs,
        trt.MatrixOperation.NONE,
    ).get_output(0)
    return _cast_back_to_trt_dtype(network, output, lhs.dtype)


def add_bias_sum(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    width: int,
    bias: np.ndarray,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    rank = len(tuple(inp.shape))
    bias_shape = (width,) if rank <= 1 else (1,) * (rank - 1) + (width,)
    bias_tensor = add_constant(
        network,
        bias_shape,
        np.asarray(bias).reshape(bias_shape),
        dtype=dtype,
    )
    bias_tensor = _cast_back_to_trt_dtype(network, bias_tensor, inp.dtype)
    output = network.add_elementwise(inp, bias_tensor, trt.ElementWiseOperation.SUM).get_output(0)
    return _cast_back_to_trt_dtype(network, output, inp.dtype)


def add_relu(network: trt.INetworkDefinition, inp: trt.ITensor) -> trt.ITensor:
    return network.add_activation(inp, trt.ActivationType.RELU).get_output(0)


def add_layer_norm_native(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    rank = len(tuple(inp.shape))
    param_shape = (hidden_size,) if rank <= 1 else (1,) * (rank - 1) + (hidden_size,)
    gamma_tensor = add_constant(
        network,
        param_shape,
        np.asarray(gamma).reshape(param_shape),
        dtype=dtype,
    )
    beta_tensor = add_constant(
        network,
        param_shape,
        np.asarray(beta).reshape(param_shape),
        dtype=dtype,
    )
    gamma_tensor = _cast_back_to_trt_dtype(network, gamma_tensor, inp.dtype)
    beta_tensor = _cast_back_to_trt_dtype(network, beta_tensor, inp.dtype)
    norm = network.add_normalization_v2(
        inp,
        gamma_tensor,
        beta_tensor,
        1 << (rank - 1),
    )
    norm.epsilon = eps
    if hasattr(norm, "compute_precision"):
        norm.compute_precision = trt.float32
    return norm.get_output(0)


def _reshape_rows_to_heads(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    num_heads: int,
    head_dim: int,
    sequence_length: int,
) -> trt.ITensor:
    rows = network.add_shuffle(tensor)
    rows.reshape_dims = (sequence_length, num_heads, head_dim)
    rows.second_transpose = trt.Permutation([1, 0, 2])
    heads = network.add_shuffle(rows.get_output(0))
    heads.reshape_dims = (1, num_heads, sequence_length, head_dim)
    return heads.get_output(0)


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
    mask: trt.ITensor,
) -> trt.ITensor:
    q_heads = _reshape_rows_to_heads(network, q, num_heads, head_dim, q_seq)
    k_heads = _reshape_rows_to_heads(network, k, num_heads, head_dim, kv_seq)
    v_heads = _reshape_rows_to_heads(network, v, num_heads, head_dim, kv_seq)

    scale_dtype = np.float16 if q_heads.dtype == trt.float16 else np.float32
    scale = add_constant(
        network,
        (1, 1, 1, 1),
        np.array([[[[1.0 / np.sqrt(head_dim)]]]], dtype=scale_dtype),
        dtype=scale_dtype,
    )
    if q_heads.dtype == trt.bfloat16:
        scale = network.add_cast(scale, trt.bfloat16).get_output(0)
    scaled_q = network.add_elementwise(
        q_heads,
        scale,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    attention = network.add_attention(
        scaled_q,
        k_heads,
        v_heads,
        trt.AttentionNormalizationOp.SOFTMAX,
        False,
    )
    attention.decomposable = True
    attention.mask = mask
    context = _cast_back_to_trt_dtype(network, attention.get_output(0), q_heads.dtype)

    rows = network.add_shuffle(context)
    rows.first_transpose = trt.Permutation([0, 2, 1, 3])
    rows.reshape_dims = (q_seq, num_heads * head_dim)
    return rows.get_output(0)


def add_self_attention_block(
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    hidden_size: int,
    num_heads: int,
    seq_length: int,
    q_bias: np.ndarray,
    k_bias: np.ndarray,
    v_bias: np.ndarray,
    o_bias: np.ndarray,
    mask: trt.ITensor,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    q = add_bias_sum(
        network,
        add_matmul_rhs_constant(network, hidden, hidden_size, hidden_size, w_q, dtype=dtype),
        hidden_size,
        q_bias,
        dtype=dtype,
    )
    k = add_bias_sum(
        network,
        add_matmul_rhs_constant(network, hidden, hidden_size, hidden_size, w_k, dtype=dtype),
        hidden_size,
        k_bias,
        dtype=dtype,
    )
    v = add_bias_sum(
        network,
        add_matmul_rhs_constant(network, hidden, hidden_size, hidden_size, w_v, dtype=dtype),
        hidden_size,
        v_bias,
        dtype=dtype,
    )
    context = add_attention_from_rows(
        network,
        q,
        k,
        v,
        num_heads=num_heads,
        head_dim=hidden_size // num_heads,
        q_seq=seq_length,
        kv_seq=seq_length,
        mask=mask,
    )
    return add_bias_sum(
        network,
        add_matmul_rhs_constant(
            network,
            context,
            hidden_size,
            hidden_size,
            w_o,
            dtype=dtype,
        ),
        hidden_size,
        o_bias,
        dtype=dtype,
    )


def add_relu_mlp(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    weights: dict,
    prefix: str,
    hidden_size: int,
    mlp_size: int,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    fc1 = add_bias_sum(
        network,
        add_matmul_rhs_constant(
            network,
            inp,
            hidden_size,
            mlp_size,
            weights[f"{prefix}.w_fc1"],
            dtype=dtype,
        ),
        mlp_size,
        weights[f"{prefix}.fc1_bias"],
        dtype=dtype,
    )
    fc2 = add_matmul_rhs_constant(
        network,
        add_relu(network, fc1),
        mlp_size,
        hidden_size,
        weights[f"{prefix}.w_fc2"],
        dtype=dtype,
    )
    return add_bias_sum(
        network,
        fc2,
        hidden_size,
        weights[f"{prefix}.fc2_bias"],
        dtype=dtype,
    )
