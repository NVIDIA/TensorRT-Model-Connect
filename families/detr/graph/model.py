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


def make_sine_position_embedding(height: int, width: int, hidden_size: int,
                                 temperature: int = 10000, normalize: bool = True,
                                 scale: float | None = None):
    """Return [1, height*width, hidden_size] DETR sine position embeddings."""
    if scale is None:
        scale = 2.0 * np.pi
    y_embed = np.arange(1, height + 1, dtype=np.float32)[:, None].repeat(width, axis=1)
    x_embed = np.arange(1, width + 1, dtype=np.float32)[None, :].repeat(height, axis=0)
    if normalize:
        eps = 1e-6
        y_embed = y_embed / (y_embed[-1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, -1:] + eps) * scale
    num_pos_feats = hidden_size // 2
    dim_t = np.arange(num_pos_feats, dtype=np.float32)
    dim_t = temperature ** (2.0 * np.floor(dim_t / 2.0) / num_pos_feats)
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = np.stack((np.sin(pos_x[:, :, 0::2]), np.cos(pos_x[:, :, 1::2])), axis=3).reshape(height, width, num_pos_feats)
    pos_y = np.stack((np.sin(pos_y[:, :, 0::2]), np.cos(pos_y[:, :, 1::2])), axis=3).reshape(height, width, num_pos_feats)
    pos = np.concatenate((pos_y, pos_x), axis=2)  # [H, W, hidden_size]
    return np.ascontiguousarray(pos.reshape(1, height * width, hidden_size))
