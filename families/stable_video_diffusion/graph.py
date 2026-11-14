# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph helpers for the Stable Video Diffusion components.

Every helper takes the network's working dtype explicitly. Tensors between
layers are kept in that dtype; normalization statistics and activation functions
are computed in FP32 whatever the working dtype is, which is what the reference's
FP16 kernels do internally.

In an FP16 network every linear and convolution layer accumulates in FP32: it
receives FP32 copies of its FP16 operands (weights rounded to FP16 first) and the
engine is built with TF32 enabled. FP16 values are exact in TF32, so this is FP16
arithmetic with FP32 accumulation; FP16 tactics may otherwise accumulate in FP16.
"""

from __future__ import annotations

import math

import numpy as np
import tensorrt as trt


def numpy_dtype(work: trt.DataType):
    if work == trt.float16:
        return np.float16
    if work == trt.float32:
        return np.float32
    raise ValueError(f"unsupported working dtype {work}")


def shape_of(tensor) -> tuple[int, ...]:
    return tuple(int(value) for value in tensor.shape)


def constant(network, values, dtype):
    """A constant tensor holding ``values`` stored as numpy ``dtype``."""
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float32).astype(dtype))
    if array.ndim == 0:
        raise ValueError("graph constants must have rank >= 1")
    return network.add_constant(array.shape, trt.Weights(array)).get_output(0)


def cast(network, tensor, dtype: trt.DataType):
    if tensor.dtype == dtype:
        return tensor
    return network.add_cast(tensor, dtype).get_output(0)


def reshape(network, tensor, dims):
    layer = network.add_shuffle(tensor)
    layer.reshape_dims = tuple(int(value) for value in dims)
    return layer.get_output(0)


def transpose(network, tensor, permutation):
    layer = network.add_shuffle(tensor)
    layer.first_transpose = tuple(int(value) for value in permutation)
    return layer.get_output(0)


def add(network, left, right):
    return network.add_elementwise(left, right, trt.ElementWiseOperation.SUM).get_output(0)


def mul(network, left, right):
    return network.add_elementwise(left, right, trt.ElementWiseOperation.PROD).get_output(0)


def slice_axis(network, tensor, axis: int, start: int, size: int):
    shape = shape_of(tensor)
    starts = [0] * len(shape)
    sizes = list(shape)
    starts[axis] = start
    sizes[axis] = size
    return network.add_slice(tensor, tuple(starts), tuple(sizes), (1,) * len(shape)).get_output(0)


def concat(network, tensors, axis: int):
    layer = network.add_concatenation(list(tensors))
    layer.axis = axis
    return layer.get_output(0)


def silu(network, tensor):
    """x * sigmoid(x), evaluated in FP32 like the reference's FP16 kernels."""
    values = cast(network, tensor, trt.float32)
    gate = network.add_activation(values, trt.ActivationType.SIGMOID).get_output(0)
    return cast(network, mul(network, values, gate), tensor.dtype)


def gelu(network, tensor):
    """The exact erf GELU of torch.nn.functional.gelu, evaluated in FP32."""
    values = cast(network, tensor, trt.float32)
    return cast(network, network.add_activation(values, trt.ActivationType.GELU_ERF).get_output(0),
                tensor.dtype)


def _accumulating(values, work: trt.DataType) -> np.ndarray:
    """Layer parameters as stored: FP16-rounded in FP32 for an FP16 network, else FP32."""
    values = np.asarray(values, dtype=np.float32)
    if work == trt.float16:
        values = values.astype(np.float16).astype(np.float32)
    return values


def linear(network, tensor, weight, bias, work: trt.DataType, output_scale: float | None = None):
    """``(tensor @ weight.T + bias) * output_scale`` over the last axis; ``weight`` is [out, in].

    The scale multiplies the FP32 result before it is rounded to the working dtype.
    """
    weight = _accumulating(weight, work)
    out_features, in_features = (int(value) for value in weight.shape)
    rank = len(shape_of(tensor))
    rhs_shape = (1,) * (rank - 2) + (in_features, out_features)
    rhs = constant(network, weight.T.reshape(rhs_shape), np.float32)
    product = network.add_matrix_multiply(
        cast(network, tensor, trt.float32), trt.MatrixOperation.NONE, rhs,
        trt.MatrixOperation.NONE).get_output(0)
    if bias is not None:
        bias_shape = (1,) * (rank - 1) + (out_features,)
        product = add(network, product, constant(
            network, _accumulating(bias, work).reshape(bias_shape), np.float32))
    if output_scale is not None:
        product = mul(network, product, constant(network, np.full((1,) * rank, output_scale),
                                                 np.float32))
    return cast(network, product, work)


def conv(network, tensor, weight, bias, work: trt.DataType, stride=None, padding=None,
         post_padding=None):
    """Convolution over the trailing 2 or 3 axes; ``weight`` is [out, in, *kernel].

    ``padding`` pads both sides; with ``post_padding`` it is the leading padding only.
    """
    weight = _accumulating(weight, work)
    kernel = tuple(int(value) for value in weight.shape[2:])
    layer = network.add_convolution_nd(
        cast(network, tensor, trt.float32),
        int(weight.shape[0]),
        kernel,
        trt.Weights(np.ascontiguousarray(weight)),
        trt.Weights(np.ascontiguousarray(_accumulating(bias, work)))
        if bias is not None
        else trt.Weights(),
    )
    layer.stride_nd = tuple(stride or (1,) * len(kernel))
    if post_padding is None:
        layer.padding_nd = tuple(padding or (0,) * len(kernel))
    else:
        layer.pre_padding = tuple(padding or (0,) * len(kernel))
        layer.post_padding = tuple(post_padding)
    return cast(network, layer.get_output(0), work)


def _affine(network, tensor, gamma, beta, axis: int, work: trt.DataType):
    """Per-channel ``tensor * gamma + beta`` with the parameters stored in ``work``.

    The parameters are rounded to the working dtype first, as the reference's
    FP16 module stores them, and applied in the dtype of ``tensor``.
    """
    rank = len(shape_of(tensor))
    shape = [1] * rank
    shape[axis] = int(np.asarray(gamma).size)
    stored = numpy_dtype(work)
    target = numpy_dtype(tensor.dtype)
    gamma = np.asarray(gamma, dtype=np.float32).astype(stored).astype(target).reshape(shape)
    beta = np.asarray(beta, dtype=np.float32).astype(stored).astype(target).reshape(shape)
    scaled = mul(network, tensor, constant(network, gamma, target))
    return add(network, scaled, constant(network, beta, target))


def group_norm(network, tensor, gamma, beta, groups: int, eps: float, work: trt.DataType):
    """GroupNorm over [N, C, *spatial]: statistics per sample and channel group.

    A rank-5 [B, C, F, H, W] input normalizes each group across frames as well,
    which is what torch.nn.GroupNorm does for the temporal blocks. Mean and
    variance are explicit FP32 reductions over the [N, groups, elements] view:
    the normalization layer on that view returns wrong statistics when a group
    holds few elements (one channel of a small feature map).
    """
    shape = shape_of(tensor)
    channels = shape[1]
    if channels % groups:
        raise ValueError(f"group norm needs {groups} to divide {channels} channels")
    values = cast(network, tensor, trt.float32)
    per_group = (channels // groups) * int(np.prod(shape[2:]))
    grouped = reshape(network, values, (shape[0], groups, per_group))
    mean = network.add_reduce(grouped, trt.ReduceOperation.AVG, 1 << 2, True).get_output(0)
    centered = network.add_elementwise(grouped, mean, trt.ElementWiseOperation.SUB).get_output(0)
    variance = network.add_reduce(mul(network, centered, centered), trt.ReduceOperation.AVG,
                                  1 << 2, True).get_output(0)
    deviation = network.add_unary(
        add(network, variance, constant(network, np.full((1, 1, 1), eps), np.float32)),
        trt.UnaryOperation.SQRT).get_output(0)
    normalized = network.add_elementwise(
        centered, deviation, trt.ElementWiseOperation.DIV).get_output(0)
    normalized = reshape(network, normalized, shape)
    return cast(network, _affine(network, normalized, gamma, beta, 1, work), work)


def layer_norm(network, tensor, gamma, beta, eps: float, work: trt.DataType):
    """LayerNorm over the last axis, computed in FP32."""
    shape = shape_of(tensor)
    values = cast(network, tensor, trt.float32)
    width = shape[-1]
    rank = len(shape)
    stored = numpy_dtype(work)
    weight = np.asarray(gamma, dtype=np.float32).astype(stored).astype(np.float32)
    shift = np.asarray(beta, dtype=np.float32).astype(stored).astype(np.float32)
    affine_shape = (1,) * (rank - 1) + (width,)
    layer = network.add_normalization(
        values,
        constant(network, weight.reshape(affine_shape), np.float32),
        constant(network, shift.reshape(affine_shape), np.float32),
        1 << (rank - 1),
    )
    layer.epsilon = float(eps)
    return cast(network, layer.get_output(0), work)


def split_heads(network, tensor, heads: int):
    """[N, T, heads * D] -> [N, heads, T, D]."""
    batch, tokens, width = shape_of(tensor)
    layer = network.add_shuffle(tensor)
    layer.reshape_dims = (batch, tokens, heads, width // heads)
    layer.second_transpose = (0, 2, 1, 3)
    return layer.get_output(0)


def merge_heads(network, tensor):
    """[N, heads, T, D] -> [N, T, heads * D]."""
    batch, heads, tokens, width = shape_of(tensor)
    layer = network.add_shuffle(tensor)
    layer.first_transpose = (0, 2, 1, 3)
    layer.reshape_dims = (batch, tokens, heads * width)
    return layer.get_output(0)


def attention(network, query, key, value):
    """softmax(query @ key^T) @ value with the softmax scale already in ``query``."""
    layer = network.add_attention(query, key, value, trt.AttentionNormalizationOp.SOFTMAX, False)
    if layer is None:
        raise RuntimeError("TensorRT rejected the attention operands")
    layer.decomposable = True
    return layer.get_output(0)


def self_attention(network, tensor, weights, prefix: str, heads: int, work: trt.DataType,
                   *, bias: bool):
    """Multi-head self-attention with diffusers' to_q/to_k/to_v/to_out.0 projections.

    The softmax scale 1/sqrt(head_dim) multiplies the query projection's FP32 result.
    """
    width = int(np.asarray(weights[f"{prefix}.to_q.weight"]).shape[0])
    query = linear(network, tensor, weights[f"{prefix}.to_q.weight"],
                   weights[f"{prefix}.to_q.bias"] if bias else None, work,
                   output_scale=1.0 / math.sqrt(width // heads))
    key = linear(network, tensor, weights[f"{prefix}.to_k.weight"],
                 weights[f"{prefix}.to_k.bias"] if bias else None, work)
    value = linear(network, tensor, weights[f"{prefix}.to_v.weight"],
                   weights[f"{prefix}.to_v.bias"] if bias else None, work)
    attended = attention(network, split_heads(network, query, heads),
                         split_heads(network, key, heads), split_heads(network, value, heads))
    return linear(network, merge_heads(network, attended), weights[f"{prefix}.to_out.0.weight"],
                  weights[f"{prefix}.to_out.0.bias"], work)


def geglu_feed_forward(network, tensor, weights, prefix: str, work: trt.DataType):
    """diffusers FeedForward(activation_fn="geglu"): proj to 2x, value * gelu(gate), project."""
    projected = linear(network, tensor, weights[f"{prefix}.net.0.proj.weight"],
                       weights[f"{prefix}.net.0.proj.bias"], work)
    width = shape_of(projected)[-1] // 2
    axis = len(shape_of(projected)) - 1
    hidden = slice_axis(network, projected, axis, 0, width)
    gate = slice_axis(network, projected, axis, width, width)
    return linear(network, mul(network, hidden, gelu(network, gate)),
                  weights[f"{prefix}.net.2.weight"], weights[f"{prefix}.net.2.bias"], work)


def sinusoid_frequencies(channels: int, max_period: float = 10000.0) -> np.ndarray:
    """diffusers get_timestep_embedding frequencies, evaluated in float32 like the reference."""
    half = channels // 2
    exponent = np.float32(-math.log(max_period)) * np.arange(half, dtype=np.float32)
    exponent = exponent / np.float32(half)
    return np.exp(exponent).astype(np.float32)


def sinusoid_embedding(network, values, channels: int):
    """[rows, 1] FP32 values -> [rows, channels] FP32 [cos | sin], diffusers flip_sin_to_cos."""
    if channels % 2:
        raise ValueError("odd sinusoid widths are not used by this family")
    frequencies = constant(network, sinusoid_frequencies(channels).reshape(1, -1), np.float32)
    angles = mul(network, values, frequencies)
    cosine = network.add_unary(angles, trt.UnaryOperation.COS).get_output(0)
    sine = network.add_unary(angles, trt.UnaryOperation.SIN).get_output(0)
    return concat(network, [cosine, sine], axis=1)


def sinusoid_table(positions, channels: int) -> np.ndarray:
    """Host-side [len(positions), channels] [cos | sin] table for fixed integer positions."""
    frequencies = sinusoid_frequencies(channels).astype(np.float64)
    angles = (np.asarray(positions, dtype=np.float32).astype(np.float64)[:, None]
              * frequencies[None, :]).astype(np.float32).astype(np.float64)
    return np.concatenate([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)


def blend_constants(mix_factor, work: trt.DataType, *, spatial_is_sigmoid: bool):
    """AlphaBlender weights (spatial, temporal) as the reference evaluates them.

    "learned_with_images" (the UNet) weights the spatial branch by sigmoid(mix);
    "learned" with switch_spatial_to_temporal_mix (the VAE decoder) weights it by
    1 - sigmoid(mix). Values follow the parameter dtype's rounding.
    """
    stored = numpy_dtype(work)
    mix = np.asarray(mix_factor, dtype=np.float32).reshape(-1).astype(stored)
    alpha = (1.0 / (1.0 + np.exp(-mix.astype(np.float64)))).astype(np.float32).astype(stored)
    if not spatial_is_sigmoid:
        alpha = (np.float32(1.0) - alpha.astype(np.float32)).astype(stored)
    other = (np.float32(1.0) - alpha.astype(np.float32)).astype(stored)
    return alpha.astype(np.float32), other.astype(np.float32)


def blend(network, spatial, temporal, weights, work: trt.DataType):
    """``alpha * spatial + other * temporal`` with (alpha, other) from blend_constants."""
    alpha, other = weights
    rank = len(shape_of(spatial))
    shape = (1,) * rank
    dtype = numpy_dtype(work)
    left = mul(network, spatial, constant(network, np.full(shape, alpha[0]), dtype))
    right = mul(network, temporal, constant(network, np.full(shape, other[0]), dtype))
    return add(network, left, right)
