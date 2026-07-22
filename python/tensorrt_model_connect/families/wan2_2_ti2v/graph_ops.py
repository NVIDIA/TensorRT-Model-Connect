# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT operations used by the Wan2.2 recurrent VAE decoder."""

from __future__ import annotations

import numpy as np

from tensorrt_model_connect import trt_compat


trt = trt_compat.get_trt()


def add_constant(network, shape: tuple[int, ...], values):
    weights = trt.Weights(np.ascontiguousarray(values, dtype=np.float32))
    return network.add_constant(shape, weights).get_output(0)


def add_matmul_rhs_constant(
    network,
    lhs,
    lhs_width: int,
    rhs_width: int,
    rhs_weights,
):
    rank = len(tuple(lhs.shape))
    rhs_shape = (lhs_width, rhs_width) if rank <= 2 else (1,) * (rank - 2) + (lhs_width, rhs_width)
    rhs = add_constant(
        network,
        rhs_shape,
        np.asarray(rhs_weights).reshape(rhs_shape),
    )
    return network.add_matrix_multiply(
        lhs,
        trt.MatrixOperation.NONE,
        rhs,
        trt.MatrixOperation.NONE,
    ).get_output(0)


def add_bias_sum(network, tensor, width: int, bias):
    rank = len(tuple(tensor.shape))
    bias_shape = (width,) if rank <= 1 else (1,) * (rank - 1) + (width,)
    bias_tensor = add_constant(
        network,
        bias_shape,
        np.asarray(bias).reshape(bias_shape),
    )
    return network.add_elementwise(tensor, bias_tensor, trt.ElementWiseOperation.SUM).get_output(0)


def add_silu(network, tensor):
    sigmoid = network.add_activation(tensor, trt.ActivationType.SIGMOID).get_output(0)
    return network.add_elementwise(tensor, sigmoid, trt.ElementWiseOperation.PROD).get_output(0)


def add_conv3d_as_conv2d(
    network,
    tensor,
    weight,
    bias,
    out_channels: int,
    kernel_size: tuple[int, int, int],
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int] = (0, 0, 0),
):
    """Apply a temporal-kernel-one Conv3D as per-frame Conv2D."""

    batch, input_channels, frames, height, width = tuple(tensor.shape)
    kt, kh, kw = kernel_size
    st, sh, sw = stride
    pt, ph, pw = padding
    if kt != 1 or st != 1 or pt != 0:
        raise ValueError(
            "Wan2.2 Conv3D-as-Conv2D requires temporal kernel/stride 1 and no temporal padding"
        )

    reshape_input = network.add_shuffle(tensor)
    reshape_input.first_transpose = trt.Permutation([0, 2, 1, 3, 4])
    reshape_input.reshape_dims = (batch * frames, input_channels, height, width)
    weights = trt.Weights(
        np.ascontiguousarray(
            np.asarray(weight).reshape(out_channels, input_channels, kh, kw),
            dtype=np.float32,
        )
    )
    biases = (
        trt.Weights() if bias is None else trt.Weights(np.ascontiguousarray(bias, dtype=np.float32))
    )
    convolution = network.add_convolution_nd(
        reshape_input.get_output(0),
        num_output_maps=out_channels,
        kernel_shape=(kh, kw),
        kernel=weights,
        bias=biases,
    )
    convolution.stride_nd = (sh, sw)
    convolution.padding_nd = (ph, pw)
    output_height = (height + 2 * ph - kh) // sh + 1
    output_width = (width + 2 * pw - kw) // sw + 1
    reshape_output = network.add_shuffle(convolution.get_output(0))
    reshape_output.reshape_dims = (
        batch,
        frames,
        out_channels,
        output_height,
        output_width,
    )
    reshape_output.second_transpose = trt.Permutation([0, 2, 1, 3, 4])
    return reshape_output.get_output(0)


def add_causal_conv3d(
    network,
    tensor,
    cache,
    weight,
    bias,
    out_channels: int,
    kernel_size: tuple[int, int, int],
    stride: tuple[int, int, int] = (1, 1, 1),
    padding_hw: tuple[int, int] = (0, 0),
):
    """Apply a causal Conv3D and return its two-frame updated cache."""

    batch, input_channels, input_frames, height, width = tuple(tensor.shape)
    kt, kh, kw = kernel_size
    ph, pw = padding_hw
    if kt != 3:
        raise ValueError(f"Wan2.2 causal Conv3D requires temporal kernel 3, got {kt}")

    concatenation = network.add_concatenation([cache, tensor])
    concatenation.axis = 2
    temporal = concatenation.get_output(0)
    if input_frames == 1:
        reshape_input = network.add_shuffle(temporal)
        reshape_input.reshape_dims = (batch, input_channels * kt, height, width)
        weights = trt.Weights(
            np.ascontiguousarray(
                np.asarray(weight).reshape(out_channels, input_channels * kt, kh, kw),
                dtype=np.float32,
            )
        )
        biases = (
            trt.Weights()
            if bias is None
            else trt.Weights(np.ascontiguousarray(bias, dtype=np.float32))
        )
        convolution = network.add_convolution_nd(
            reshape_input.get_output(0),
            num_output_maps=out_channels,
            kernel_shape=(kh, kw),
            kernel=weights,
            bias=biases,
        )
        convolution.stride_nd = (stride[1], stride[2])
        convolution.padding_nd = (ph, pw)
        output_height = (height + 2 * ph - kh) // stride[1] + 1
        output_width = (width + 2 * pw - kw) // stride[2] + 1
        reshape_output = network.add_shuffle(convolution.get_output(0))
        reshape_output.reshape_dims = (
            batch,
            out_channels,
            1,
            output_height,
            output_width,
        )
        output = reshape_output.get_output(0)
    else:
        weights = trt.Weights(
            np.ascontiguousarray(
                np.asarray(weight).reshape(out_channels, input_channels, kt, kh, kw),
                dtype=np.float32,
            )
        )
        biases = (
            trt.Weights()
            if bias is None
            else trt.Weights(np.ascontiguousarray(bias, dtype=np.float32))
        )
        convolution = network.add_convolution_nd(
            temporal,
            num_output_maps=out_channels,
            kernel_shape=(kt, kh, kw),
            kernel=weights,
            bias=biases,
        )
        convolution.stride_nd = stride
        convolution.padding_nd = (0, ph, pw)
        output = convolution.get_output(0)

    updated_cache = network.add_slice(
        temporal,
        start=(0, 0, input_frames, 0, 0),
        shape=(batch, input_channels, kt - 1, height, width),
        stride=(1, 1, 1, 1, 1),
    ).get_output(0)
    return output, updated_cache


def add_spatial_upsample(network, tensor, scale_factor: int = 2):
    batch, channels, frames, height, width = tuple(tensor.shape)
    resize = network.add_resize(tensor)
    resize.resize_mode = trt.InterpolationMode.NEAREST
    resize.shape = (
        batch,
        channels,
        frames,
        height * scale_factor,
        width * scale_factor,
    )
    return resize.get_output(0)


def add_spatial_upsample_with_conv(
    network,
    tensor,
    weight,
    bias,
    scale: int = 2,
):
    upsampled = add_spatial_upsample(network, tensor, scale)
    return add_conv3d_as_conv2d(
        network,
        upsampled,
        weight=weight,
        bias=bias,
        out_channels=int(weight.shape[0]),
        kernel_size=(1, 3, 3),
        padding=(0, 1, 1),
    )


def add_l2_channel_norm(
    network,
    tensor,
    num_channels: int,
    gamma,
    eps: float = 1.0e-6,
):
    squared = network.add_elementwise(tensor, tensor, trt.ElementWiseOperation.PROD).get_output(0)
    summed = network.add_reduce(squared, trt.ReduceOperation.SUM, 1 << 1, True).get_output(0)
    epsilon = add_constant(
        network,
        (1, 1, 1, 1, 1),
        np.array([eps], dtype=np.float32),
    )
    denominator = network.add_elementwise(summed, epsilon, trt.ElementWiseOperation.SUM).get_output(
        0
    )
    denominator = network.add_unary(denominator, trt.UnaryOperation.SQRT).get_output(0)
    inverse = network.add_unary(denominator, trt.UnaryOperation.RECIP).get_output(0)
    normalized = network.add_elementwise(tensor, inverse, trt.ElementWiseOperation.PROD).get_output(
        0
    )
    scale = np.sqrt(num_channels) * np.asarray(gamma).flatten()[:num_channels]
    scale_tensor = add_constant(
        network,
        (1, num_channels, 1, 1, 1),
        scale.reshape(1, num_channels, 1, 1, 1),
    )
    return network.add_elementwise(
        normalized, scale_tensor, trt.ElementWiseOperation.PROD
    ).get_output(0)


def add_temporal_pixel_shuffle(network, tensor, factor: int = 2):
    batch, total_channels, frames, height, width = tuple(tensor.shape)
    channels = total_channels // factor
    reshape = network.add_shuffle(tensor)
    reshape.reshape_dims = (batch, factor, channels, frames, height, width)
    transpose = network.add_shuffle(reshape.get_output(0))
    transpose.first_transpose = trt.Permutation([0, 2, 3, 1, 4, 5])
    output = network.add_shuffle(transpose.get_output(0))
    output.reshape_dims = (batch, channels, factor * frames, height, width)
    return output.get_output(0)


def add_attention_core(network, q, k, v, *, scale: float):
    scale_tensor = add_constant(
        network,
        (1, 1, 1, 1),
        np.array([[[[scale]]]], dtype=np.float32),
    )
    q_scaled = network.add_elementwise(q, scale_tensor, trt.ElementWiseOperation.PROD).get_output(0)
    attention = network.add_attention(
        q_scaled,
        k,
        v,
        trt.AttentionNormalizationOp.SOFTMAX,
        False,
    )
    if attention is None:
        raise RuntimeError("Could not add native TensorRT VAE attention")
    attention.decomposable = True
    return attention.get_output(0)
