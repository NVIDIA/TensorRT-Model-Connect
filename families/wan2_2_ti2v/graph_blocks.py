# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT block used by the Wan2.2 recurrent VAE decoder."""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from . import graph_ops


def add_vae_spatial_attention(
    network,
    tensor,
    *,
    weights: dict,
    prefix: str,
    channels: int,
    eps: float = 1.0e-6,
):
    """Add the single-head spatial attention in the Wan2.2 VAE mid-block."""

    batch, tensor_channels, frames, height, width = tuple(tensor.shape)
    if tensor_channels != channels:
        raise ValueError(
            f"Wan2.2 VAE attention expected {channels} channels, got {tensor_channels}"
        )
    batch_frames = batch * frames
    spatial = height * width
    identity = tensor

    normalized = graph_ops.add_l2_channel_norm(
        network,
        tensor,
        channels,
        weights[f"{prefix}.norm.gamma"],
        eps,
    )
    flatten = network.add_shuffle(normalized)
    flatten.first_transpose = trt.Permutation([0, 2, 3, 4, 1])
    flatten.reshape_dims = (batch_frames * spatial, channels)

    qkv_weight = weights[f"{prefix}.to_qkv.weight"].reshape(3 * channels, channels).T.copy()
    qkv = graph_ops.add_matmul_rhs_constant(
        network,
        flatten.get_output(0),
        channels,
        3 * channels,
        qkv_weight,
    )
    qkv_bias = weights.get(f"{prefix}.to_qkv.bias")
    if qkv_bias is not None:
        qkv = graph_ops.add_bias_sum(network, qkv, 3 * channels, qkv_bias)

    qkv_shape = network.add_shuffle(qkv)
    qkv_shape.reshape_dims = (batch_frames, spatial, 3 * channels)
    q = network.add_slice(
        qkv_shape.get_output(0),
        start=(0, 0, 0),
        shape=(batch_frames, spatial, channels),
        stride=(1, 1, 1),
    ).get_output(0)
    k = network.add_slice(
        qkv_shape.get_output(0),
        start=(0, 0, channels),
        shape=(batch_frames, spatial, channels),
        stride=(1, 1, 1),
    ).get_output(0)
    v = network.add_slice(
        qkv_shape.get_output(0),
        start=(0, 0, 2 * channels),
        shape=(batch_frames, spatial, channels),
        stride=(1, 1, 1),
    ).get_output(0)

    q4 = network.add_shuffle(q)
    q4.reshape_dims = (batch_frames, 1, spatial, channels)
    k4 = network.add_shuffle(k)
    k4.reshape_dims = (batch_frames, 1, spatial, channels)
    v4 = network.add_shuffle(v)
    v4.reshape_dims = (batch_frames, 1, spatial, channels)
    context = graph_ops.add_attention_core(
        network,
        q4.get_output(0),
        k4.get_output(0),
        v4.get_output(0),
        scale=1.0 / np.sqrt(max(channels, 1)),
    )

    context_rows = network.add_shuffle(context)
    context_rows.reshape_dims = (batch_frames * spatial, channels)
    projection_weight = weights[f"{prefix}.proj.weight"].reshape(channels, channels).T.copy()
    projection = graph_ops.add_matmul_rhs_constant(
        network,
        context_rows.get_output(0),
        channels,
        channels,
        projection_weight,
    )
    projection_bias = weights.get(f"{prefix}.proj.bias")
    if projection_bias is not None:
        projection = graph_ops.add_bias_sum(network, projection, channels, projection_bias)

    output_shape = network.add_shuffle(projection)
    output_shape.reshape_dims = (batch, frames, height, width, channels)
    output_shape.second_transpose = trt.Permutation([0, 4, 1, 2, 3])
    return network.add_elementwise(
        output_shape.get_output(0), identity, trt.ElementWiseOperation.SUM
    ).get_output(0)
