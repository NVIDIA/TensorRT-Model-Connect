# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned TensorRT graph operations for Python engine builds.

Tensor names and shapes must stay compatible with the C++ bundle runtime.
"""

from __future__ import annotations

import numpy as np
from tensorrt_model_connect import trt_compat


trt = trt_compat.get_trt()


def layer_tensor_name(stem: str, layer: int) -> str:
    return f"{stem}_{layer}"


def add_constant(
    network: trt.INetworkDefinition,
    shape: tuple[int, ...],
    values: np.ndarray,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Add a constant tensor in the given *dtype* (default float32)."""
    weights = trt.Weights(np.ascontiguousarray(values, dtype=dtype))
    layer = network.add_constant(shape, weights)
    return layer.get_output(0)


def add_matmul_rhs_constant(
    network: trt.INetworkDefinition,
    lhs: trt.ITensor,
    lhs_width: int,
    rhs_width: int,
    rhs_weights: np.ndarray,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Matrix multiply: lhs @ rhs_constant.  rhs is [lhs_width, rhs_width]."""
    rank = len(tuple(lhs.shape))
    rhs_shape = (lhs_width, rhs_width) if rank <= 2 else (1,) * (rank - 2) + (lhs_width, rhs_width)
    rhs = add_constant(
        network,
        rhs_shape,
        np.asarray(rhs_weights).reshape(rhs_shape),
        dtype=dtype,
    )
    return network.add_matrix_multiply(
        lhs,
        trt.MatrixOperation.NONE,
        rhs,
        trt.MatrixOperation.NONE,
    ).get_output(0)


def add_bias_sum(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    width: int,
    bias: np.ndarray,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Element-wise add a bias broadcast over all non-feature axes."""
    rank = len(tuple(inp.shape))
    bias_shape = (width,) if rank <= 1 else (1,) * (rank - 1) + (width,)
    bias_t = add_constant(network, bias_shape, np.asarray(bias).reshape(bias_shape), dtype=dtype)
    return network.add_elementwise(inp, bias_t, trt.ElementWiseOperation.SUM).get_output(0)


def add_rms_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    gamma: np.ndarray,
    eps_tensor: trt.ITensor,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """RMSNorm: gamma * (x / sqrt(mean(x^2) + eps))."""
    sq = network.add_elementwise(inp, inp, trt.ElementWiseOperation.PROD)
    mean = network.add_reduce(sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    denom_in = network.add_elementwise(mean.get_output(0), eps_tensor, trt.ElementWiseOperation.SUM)
    sqrt_l = network.add_unary(denom_in.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
    normalized = network.add_elementwise(inp, recip.get_output(0), trt.ElementWiseOperation.PROD)
    gamma_t = add_constant(network, (1, hidden_size), gamma, dtype=np.float32)
    scaled = network.add_elementwise(
        normalized.get_output(0), gamma_t, trt.ElementWiseOperation.PROD
    )
    return scaled.get_output(0)


def add_layer_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps_tensor: trt.ITensor,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """LayerNorm: gamma * ((x - mean) / sqrt(var + eps)) + beta."""
    # mean = reduce_mean(x)
    mean = network.add_reduce(inp, trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    # x - mean
    centered = network.add_elementwise(inp, mean.get_output(0), trt.ElementWiseOperation.SUB)
    # variance = mean((x - mean)^2)
    sq = network.add_elementwise(
        centered.get_output(0), centered.get_output(0), trt.ElementWiseOperation.PROD
    )
    var = network.add_reduce(sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    # sqrt(var + eps)
    denom_in = network.add_elementwise(var.get_output(0), eps_tensor, trt.ElementWiseOperation.SUM)
    sqrt_l = network.add_unary(denom_in.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
    # normalized = (x - mean) / sqrt(var + eps)
    normalized = network.add_elementwise(
        centered.get_output(0), recip.get_output(0), trt.ElementWiseOperation.PROD
    )
    # gamma * normalized + beta
    gamma_t = add_constant(network, (1, hidden_size), gamma, dtype=np.float32)
    scaled = network.add_elementwise(
        normalized.get_output(0), gamma_t, trt.ElementWiseOperation.PROD
    )
    beta_t = add_constant(network, (1, hidden_size), beta, dtype=np.float32)
    return network.add_elementwise(
        scaled.get_output(0), beta_t, trt.ElementWiseOperation.SUM
    ).get_output(0)


def add_activation(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    activation_type: str,
) -> trt.ITensor:
    """Dispatch the activations used by VoiceChat graphs."""
    if activation_type == "relu":
        act = network.add_activation(inp, trt.ActivationType.RELU)
        return act.get_output(0)
    elif activation_type in ("relu2", "squared_relu"):
        relu = network.add_activation(inp, trt.ActivationType.RELU)
        sq = network.add_elementwise(
            relu.get_output(0), relu.get_output(0), trt.ElementWiseOperation.PROD
        )
        return sq.get_output(0)
    elif activation_type == "silu":
        sigmoid = network.add_activation(inp, trt.ActivationType.SIGMOID)
        swish = network.add_elementwise(inp, sigmoid.get_output(0), trt.ElementWiseOperation.PROD)
        return swish.get_output(0)
    else:
        raise ValueError(f"Unsupported activation: {activation_type}")


def reshape_rows_to_heads_4d(
    network: trt.INetworkDefinition,
    x: trt.ITensor,
    num_heads: int,
    head_dim: int,
    sequence_length: int,
) -> trt.ITensor:
    """Reshape fixed [S, H * D] rows into [1, H, S, D]."""
    r1 = network.add_shuffle(x)
    r1.reshape_dims = (sequence_length, num_heads, head_dim)
    r1.second_transpose = trt.Permutation([1, 0, 2])

    r2 = network.add_shuffle(r1.get_output(0))
    r2.reshape_dims = (1, num_heads, sequence_length, head_dim)
    return r2.get_output(0)


def reshape_heads_4d_to_rows(
    network: trt.INetworkDefinition,
    x_4d: trt.ITensor,
    attention_size: int,
    sequence_length: int,
) -> trt.ITensor:
    """Reshape [1, H, S, D] back to [S, H * D]."""
    out = network.add_shuffle(x_4d)
    out.first_transpose = trt.Permutation([0, 2, 1, 3])
    out.reshape_dims = (sequence_length, attention_size)
    return out.get_output(0)


def add_2d_mask_to_4d(
    network: trt.INetworkDefinition,
    mask_2d: trt.ITensor,
) -> trt.ITensor:
    """Reshape additive attention mask [Sq, K] to [1, 1, Sq, K]."""
    mask_4d = network.add_shuffle(mask_2d)
    mask_4d.reshape_dims = (1, 1, *tuple(mask_2d.shape))
    return mask_4d.get_output(0)


def add_attention_core(
    network: trt.INetworkDefinition,
    q_4d: trt.ITensor,
    k_4d: trt.ITensor,
    v_4d: trt.ITensor,
    *,
    mask: trt.ITensor,
    scale: float,
) -> trt.ITensor:
    """Scaled FP32 GQA via TensorRT native IAttention."""
    scale_t = add_constant(network, (1, 1, 1, 1), np.array([[[[scale]]]], dtype=np.float32))
    q_scaled = network.add_elementwise(q_4d, scale_t, trt.ElementWiseOperation.PROD)
    attn = network.add_attention(
        q_scaled.get_output(0),
        k_4d,
        v_4d,
        trt.AttentionNormalizationOp.SOFTMAX,
        False,
    )
    attn.decomposable = True
    attn.mask = mask
    return attn.get_output(0)


def add_attention_from_rows(
    network: trt.INetworkDefinition,
    q: trt.ITensor,
    k: trt.ITensor,
    v: trt.ITensor,
    *,
    num_heads: int,
    head_dim: int,
    num_kv_heads: int,
    q_seq: int,
    kv_seq: int,
    mask: trt.ITensor,
) -> trt.ITensor:
    """Native IAttention for the thinker's fixed row-major GQA tensors."""
    attention_size = num_heads * head_dim
    q_4d = reshape_rows_to_heads_4d(network, q, num_heads, head_dim, sequence_length=q_seq)
    k_4d = reshape_rows_to_heads_4d(network, k, num_kv_heads, head_dim, sequence_length=kv_seq)
    v_4d = reshape_rows_to_heads_4d(network, v, num_kv_heads, head_dim, sequence_length=kv_seq)
    ctx_4d = add_attention_core(
        network,
        q_4d,
        k_4d,
        v_4d,
        mask=mask,
        scale=float(1.0 / np.sqrt(head_dim)),
    )
    return reshape_heads_4d_to_rows(network, ctx_4d, attention_size, sequence_length=q_seq)


# VoiceChat owns both the Nemotron-H recurrent thinker and a FastConformer
# perception front end. Keep the convolution primitives local to this owner.
def add_conv2d(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weight: np.ndarray,
    bias: np.ndarray | None,
    out_channels: int,
    kernel_size: tuple[int, int],
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    groups: int = 1,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Add a strongly typed 2-D convolution."""
    conv_w = trt.Weights(np.ascontiguousarray(weight, dtype=dtype))
    conv_b = (
        trt.Weights(np.ascontiguousarray(bias, dtype=dtype)) if bias is not None else trt.Weights()
    )
    layer = network.add_convolution_nd(
        inp,
        num_output_maps=out_channels,
        kernel_shape=kernel_size,
        kernel=conv_w,
        bias=conv_b,
    )
    layer.stride_nd = stride
    layer.padding_nd = padding
    layer.num_groups = groups
    return layer.get_output(0)


def add_conv1d(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weight: np.ndarray,
    bias: np.ndarray | None,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    groups: int = 1,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Add a 1-D convolution through TensorRT's 2-D convolution layer."""
    n, c_in, length = inp.shape
    reshape_in = network.add_shuffle(inp)
    reshape_in.reshape_dims = (n, c_in, 1, length)
    result = add_conv2d(
        network,
        reshape_in.get_output(0),
        weight.reshape(out_channels, -1, 1, kernel_size),
        bias,
        out_channels,
        kernel_size=(1, kernel_size),
        stride=(1, stride),
        padding=(0, padding),
        groups=groups,
        dtype=dtype,
    )
    reshape_out = network.add_shuffle(result)
    reshape_out.reshape_dims = (n, out_channels, result.shape[3])
    return reshape_out.get_output(0)
