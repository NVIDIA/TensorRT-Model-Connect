# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned TensorRT graph operations for DETR engine builds.

DETR = ResNet-50 backbone + sine spatial position embeddings + a 6/6
encoder/decoder transformer + class/bbox heads.  Tensor names and shapes must
stay compatible with the C++ bundle runtime.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt


def _cast_back_to_trt_dtype(network, tensor, target_dtype):
    if tensor.dtype == target_dtype:
        return tensor
    return network.add_cast(tensor, target_dtype).get_output(0)


def add_constant(network, shape: tuple[int, ...], values: np.ndarray,
                 dtype: np.dtype = np.float32):
    weights = trt.Weights(np.ascontiguousarray(values, dtype=dtype))
    return network.add_constant(shape, weights).get_output(0)


def add_matmul_rhs_constant(network, lhs, lhs_width: int, rhs_width: int,
                            rhs_weights: np.ndarray, dtype: np.dtype = np.float32):
    """Matrix multiply: lhs @ rhs_constant.  rhs is [lhs_width, rhs_width]."""
    rank = len(tuple(lhs.shape))
    rhs_shape = ((lhs_width, rhs_width) if rank <= 2
                 else (1,) * (rank - 2) + (lhs_width, rhs_width))
    rhs = add_constant(network, rhs_shape, np.asarray(rhs_weights).reshape(rhs_shape), dtype=dtype)
    rhs = _cast_back_to_trt_dtype(network, rhs, lhs.dtype)
    mm = network.add_matrix_multiply(lhs, trt.MatrixOperation.NONE,
                                     rhs, trt.MatrixOperation.NONE)
    return _cast_back_to_trt_dtype(network, mm.get_output(0), lhs.dtype)


def add_bias_sum(network, inp, width: int, bias: np.ndarray,
                 dtype: np.dtype = np.float32):
    """Element-wise add a bias broadcast over all non-feature axes."""
    rank = len(tuple(inp.shape))
    bias_shape = (width,) if rank <= 1 else (1,) * (rank - 1) + (width,)
    bias_t = add_constant(network, bias_shape, np.asarray(bias).reshape(bias_shape), dtype=dtype)
    bias_t = _cast_back_to_trt_dtype(network, bias_t, inp.dtype)
    out = network.add_elementwise(inp, bias_t, trt.ElementWiseOperation.SUM)
    return _cast_back_to_trt_dtype(network, out.get_output(0), inp.dtype)


def add_linear(network, inp, weight: np.ndarray, bias: np.ndarray | None,
               out_features: int, dtype: np.dtype = np.float32):
    """Linear layer for [*, in_features] tensors.  Weight is [out, in]."""
    in_features = int(weight.shape[1])
    x = add_matmul_rhs_constant(
        network, inp, in_features, out_features, weight.T, dtype=dtype)
    if bias is not None:
        x = add_bias_sum(network, x, out_features, np.asarray(bias), dtype=dtype)
    return x


def add_conv2d(network, inp, weight: np.ndarray, bias: np.ndarray | None,
               out_channels: int, kernel_size: tuple[int, int],
               stride: tuple[int, int] = (1, 1), padding: tuple[int, int] = (0, 0),
               groups: int = 1, dtype: np.dtype = np.float32):
    """2D convolution wrapper.  Weight is [C_out, C_in/groups, kH, kW]."""
    conv_w = trt.Weights(np.ascontiguousarray(weight, dtype=dtype))
    conv_b = trt.Weights()
    if bias is not None:
        conv_b = trt.Weights(np.ascontiguousarray(bias, dtype=dtype))
    conv = network.add_convolution_nd(inp, out_channels, kernel_size, conv_w, conv_b)
    conv.stride_nd = stride
    conv.padding_nd = padding
    conv.num_groups = groups
    return conv.get_output(0)


def add_bn_folded(network, x, gamma: np.ndarray, beta: np.ndarray,
                  running_mean: np.ndarray, running_var: np.ndarray, eps: float,
                  dtype: np.dtype = np.float32):
    """Inference-time batch norm folded into per-channel scale + shift."""
    scale = (gamma / np.sqrt(running_var + eps)).astype(np.float32)
    shift = (beta - running_mean * scale).astype(np.float32)
    layer = network.add_scale(
        x, trt.ScaleMode.CHANNEL,
        shift=trt.Weights(np.ascontiguousarray(shift, dtype=dtype)),
        scale=trt.Weights(np.ascontiguousarray(scale, dtype=dtype)))
    return layer.get_output(0)


def add_sum(network, a, b):
    return network.add_elementwise(a, b, trt.ElementWiseOperation.SUM).get_output(0)


def add_relu(network, x):
    return network.add_activation(x, trt.ActivationType.RELU).get_output(0)


def add_max_pool2d(network, x, kernel: int, stride: int, padding: int):
    pool = network.add_pooling_nd(x, trt.PoolingType.MAX, (kernel, kernel))
    pool.stride_nd = (stride, stride)
    pool.padding_nd = (padding, padding)
    return pool.get_output(0)


def add_layer_norm_v2(network, inp, hidden_size: int, gamma: np.ndarray,
                      beta: np.ndarray, eps: float, dtype: np.dtype = np.float32):
    """LayerNorm via TRT native add_normalization_v2."""
    rank = len(tuple(inp.shape))
    param_shape = (hidden_size,) if rank <= 1 else (1,) * (rank - 1) + (hidden_size,)
    gamma_t = add_constant(network, param_shape, np.asarray(gamma).reshape(param_shape), dtype=dtype)
    beta_t = add_constant(network, param_shape, np.asarray(beta).reshape(param_shape), dtype=dtype)
    gamma_t = _cast_back_to_trt_dtype(network, gamma_t, inp.dtype)
    beta_t = _cast_back_to_trt_dtype(network, beta_t, inp.dtype)
    norm = network.add_normalization_v2(inp, gamma_t, beta_t, 1 << (rank - 1))
    norm.epsilon = eps
    if hasattr(norm, "compute_precision"):
        norm.compute_precision = trt.float32
    return norm.get_output(0)


def reshape_rows_to_heads_4d(network, x, num_heads: int, head_dim: int,
                             sequence_length: int | None = None, tag: str | None = None):
    """Reshape [S, H * D] rows into [1, H, S, D]."""
    seq_dim = -1 if sequence_length is None else sequence_length
    r1 = network.add_shuffle(x)
    if tag:
        r1.name = tag + "_s_h_d"
    r1.reshape_dims = (seq_dim, num_heads, head_dim)
    r1.second_transpose = trt.Permutation([1, 0, 2])
    r2 = network.add_shuffle(r1.get_output(0))
    if tag:
        r2.name = tag + "_1_h_s_d"
    r2.reshape_dims = (1, num_heads, seq_dim, head_dim)
    return r2.get_output(0)


def reshape_heads_4d_to_rows(network, x_4d, attention_size: int,
                             sequence_length: int | None = None, tag: str | None = None):
    """Reshape [1, H, S, D] back to [S, H * D]."""
    seq_dim = -1 if sequence_length is None else sequence_length
    out = network.add_shuffle(x_4d)
    if tag:
        out.name = tag + "_s_h_d"
    out.first_transpose = trt.Permutation([0, 2, 1, 3])
    out.reshape_dims = (seq_dim, attention_size)
    return out.get_output(0)


def add_attention_core(network, q_4d, k_4d, v_4d, scale: float | None = None,
                       mask=None):
    """Scaled dot-product attention via TRT native IAttention.

    TRT IAttention does not apply 1/sqrt(D); pre-scale Q.
    """
    output_dtype = q_4d.dtype
    if scale is None:
        head_dim = int(q_4d.shape[-1])
        scale = float(1.0 / np.sqrt(head_dim)) if head_dim > 0 else 1.0
    scale_np_dtype = np.float16 if q_4d.dtype == trt.float16 else np.float32
    scale_t = add_constant(network, (1, 1, 1, 1), np.array([[[[scale]]]]), dtype=scale_np_dtype)
    if q_4d.dtype == trt.bfloat16:
        scale_t = network.add_cast(scale_t, trt.bfloat16).get_output(0)
    q_scaled = network.add_elementwise(q_4d, scale_t, trt.ElementWiseOperation.PROD).get_output(0)
    attn = network.add_attention(q_scaled, k_4d, v_4d,
                                 trt.AttentionNormalizationOp.SOFTMAX, False)
    attn.decomposable = True
    if mask is not None:
        attn.mask = mask
    return _cast_back_to_trt_dtype(network, attn.get_output(0), output_dtype)


def add_sine_position_embedding(network, feature_map, hidden_size: int,
                                temperature: int = 10000):
    """DETR positions for the actual, unpadded feature grid; output [H*W, C]."""
    spatial_shape = network.add_slice(
        network.add_shape(feature_map).get_output(0), (2,), (2,), (1,)).get_output(0)
    num_pos_feats = hidden_size // 2
    dim_t = np.arange(num_pos_feats, dtype=np.float32)
    dim_t = temperature ** (2.0 * np.floor(dim_t / 2.0) / num_pos_feats)
    frequencies = add_constant(network, (1, 1, num_pos_feats), dim_t)
    even = add_constant(network, (1, 1, num_pos_feats),
                        np.arange(num_pos_feats) % 2 == 0, dtype=np.bool_)
    start = add_constant(network, (), np.array(1.0))
    epsilon = add_constant(network, (1,), np.array([1e-6]))
    scale = add_constant(network, (1, 1), np.array([[2.0 * np.pi]]))
    positions = []
    for axis, step in ((0, [1.0, 0.0]), (1, [0.0, 1.0])):
        grid = network.add_fill((1, 1), trt.FillOperation.LINSPACE, trt.float32)
        grid.set_input(0, spatial_shape)
        grid.set_input(1, start)
        grid.set_input(2, add_constant(network, (2,), np.array(step)))
        length = network.add_slice(spatial_shape, (axis,), (1,), (1,)).get_output(0)
        length = network.add_cast(length, trt.float32).get_output(0)
        denominator = add_sum(network, length, epsilon)
        scalar = network.add_shuffle(denominator)
        scalar.reshape_dims = (1, 1)
        normalized = network.add_elementwise(
            grid.get_output(0), scalar.get_output(0), trt.ElementWiseOperation.DIV).get_output(0)
        normalized = network.add_elementwise(
            normalized, scale, trt.ElementWiseOperation.PROD).get_output(0)
        expanded = network.add_shuffle(normalized)
        expanded.reshape_dims = (0, 0, 1)
        angles = network.add_elementwise(
            expanded.get_output(0), frequencies, trt.ElementWiseOperation.DIV).get_output(0)
        sine = network.add_unary(angles, trt.UnaryOperation.SIN).get_output(0)
        cosine = network.add_unary(angles, trt.UnaryOperation.COS).get_output(0)
        positions.append(network.add_select(even, sine, cosine).get_output(0))
    pos = network.add_concatenation(positions)
    pos.axis = 2
    rows = network.add_shuffle(pos.get_output(0))
    rows.reshape_dims = (-1, hidden_size)
    return rows.get_output(0)
