# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned strongly-typed TensorRT graph helpers for DINOv3."""

from __future__ import annotations

import numpy as np

from tensorrt_model_connect import trt_compat


trt = trt_compat.get_trt()


def cast(network, tensor, dtype):
    if tensor.dtype == dtype:
        return tensor
    return network.add_cast(tensor, dtype).get_output(0)


def constant(network, values: np.ndarray, shape: tuple[int, ...], dtype: np.dtype):
    array = np.ascontiguousarray(values, dtype=dtype).reshape(shape)
    return network.add_constant(shape, trt.Weights(array)).get_output(0)


def linear(network, tensor, weight: np.ndarray, dtype: np.dtype):
    """Apply a logical [in_features, out_features] matrix."""
    in_features, out_features = weight.shape
    rank = len(tuple(tensor.shape))
    shape = (1,) * max(0, rank - 2) + (in_features, out_features)
    rhs = constant(network, weight, shape, dtype)
    rhs = cast(network, rhs, tensor.dtype)
    return network.add_matrix_multiply(
        tensor, trt.MatrixOperation.NONE, rhs, trt.MatrixOperation.NONE
    ).get_output(0)


def add_bias(network, tensor, bias: np.ndarray | None, dtype: np.dtype):
    if bias is None:
        return tensor
    rank = len(tuple(tensor.shape))
    shape = (1,) * (rank - 1) + (int(bias.shape[0]),)
    bias_tensor = cast(network, constant(network, bias, shape, dtype), tensor.dtype)
    return network.add_elementwise(
        tensor, bias_tensor, trt.ElementWiseOperation.SUM
    ).get_output(0)


def layer_norm(
    network,
    tensor,
    hidden_size: int,
    weight: np.ndarray,
    bias: np.ndarray,
    eps: float,
    dtype: np.dtype,
):
    rank = len(tuple(tensor.shape))
    shape = (1,) * (rank - 1) + (hidden_size,)
    gamma = cast(network, constant(network, weight, shape, dtype), tensor.dtype)
    beta = cast(network, constant(network, bias, shape, dtype), tensor.dtype)
    norm = network.add_normalization_v2(tensor, gamma, beta, 1 << (rank - 1))
    norm.epsilon = eps
    if hasattr(norm, "compute_precision"):
        norm.compute_precision = trt.float32
    return norm.get_output(0)


def gelu(network, tensor, dtype: np.dtype):
    """Exact PyTorch GELU used by released DINOv3 checkpoints."""
    rank = len(tuple(tensor.shape))
    scalar_shape = (1,) * rank

    def scalar(value: float):
        return cast(
            network,
            constant(network, np.asarray(value), scalar_shape, dtype),
            tensor.dtype,
        )

    scaled = network.add_elementwise(
        tensor, scalar(1.0 / np.sqrt(2.0)), trt.ElementWiseOperation.PROD
    ).get_output(0)
    erf = network.add_unary(scaled, trt.UnaryOperation.ERF).get_output(0)
    one_plus = network.add_elementwise(
        erf, scalar(1.0), trt.ElementWiseOperation.SUM
    ).get_output(0)
    half_x = network.add_elementwise(
        tensor, scalar(0.5), trt.ElementWiseOperation.PROD
    ).get_output(0)
    return network.add_elementwise(
        half_x, one_plus, trt.ElementWiseOperation.PROD
    ).get_output(0)


def silu(network, tensor):
    sigmoid = network.add_activation(tensor, trt.ActivationType.SIGMOID).get_output(0)
    return network.add_elementwise(
        tensor, sigmoid, trt.ElementWiseOperation.PROD
    ).get_output(0)


def activation(network, tensor, name: str, dtype: np.dtype):
    normalized = name.lower()
    if normalized in {"gelu", "gelu_erf"}:
        return gelu(network, tensor, dtype)
    if normalized in {"silu", "swish"}:
        return silu(network, tensor)
    raise ValueError(f"Unsupported DINOv3 activation: {name}")


def multiply_last_dim(network, tensor, scale: np.ndarray, dtype: np.dtype):
    rank = len(tuple(tensor.shape))
    shape = (1,) * (rank - 1) + (int(scale.shape[0]),)
    scale_tensor = cast(network, constant(network, scale, shape, dtype), tensor.dtype)
    return network.add_elementwise(
        tensor, scale_tensor, trt.ElementWiseOperation.PROD
    ).get_output(0)


def _rotate_half(network, tensor, batch: int, heads: int, tokens: int, head_dim: int):
    half = head_dim // 2
    first = network.add_slice(
        tensor, (0, 0, 0, 0), (batch, heads, tokens, half), (1, 1, 1, 1)
    ).get_output(0)
    second = network.add_slice(
        tensor, (0, 0, 0, half), (batch, heads, tokens, half), (1, 1, 1, 1)
    ).get_output(0)
    negative_second = network.add_unary(second, trt.UnaryOperation.NEG).get_output(0)
    concat = network.add_concatenation([negative_second, first])
    concat.axis = 3
    return concat.get_output(0)


def apply_patch_rope(
    network,
    tensor,
    *,
    num_heads: int,
    num_prefix_tokens: int,
    grid_h: int,
    grid_w: int,
    head_dim: int,
    theta: float,
    dtype: np.dtype,
):
    """Apply HF DINOv3's axial 2D RoPE only to the patch-token suffix."""
    num_patches = grid_h * grid_w
    prefix = network.add_slice(
        tensor,
        (0, 0, 0, 0),
        (1, num_heads, num_prefix_tokens, head_dim),
        (1, 1, 1, 1),
    ).get_output(0)
    patches = network.add_slice(
        tensor,
        (0, 0, num_prefix_tokens, 0),
        (1, num_heads, num_patches, head_dim),
        (1, 1, 1, 1),
    ).get_output(0)

    coords_h = (np.arange(grid_h, dtype=np.float32) + 0.5) / float(grid_h)
    coords_w = (np.arange(grid_w, dtype=np.float32) + 0.5) / float(grid_w)
    yy, xx = np.meshgrid(coords_h, coords_w, indexing="ij")
    coords = np.stack([yy, xx], axis=-1).reshape(num_patches, 2)
    coords = 2.0 * coords - 1.0
    inv_freq = 1.0 / np.power(
        np.float32(theta),
        np.arange(0.0, 1.0, 4.0 / float(head_dim), dtype=np.float32),
    )
    angles = 2.0 * np.pi * coords[:, :, None] * inv_freq[None, None, :]
    angles = np.tile(angles.reshape(num_patches, head_dim // 2), (1, 2))
    cos_values = np.cos(angles).astype(np.float32)
    sin_values = np.sin(angles).astype(np.float32)
    rope_shape = (1, 1, num_patches, head_dim)
    cos_tensor = cast(network, constant(network, cos_values, rope_shape, dtype), patches.dtype)
    sin_tensor = cast(network, constant(network, sin_values, rope_shape, dtype), patches.dtype)
    direct = network.add_elementwise(
        patches, cos_tensor, trt.ElementWiseOperation.PROD
    ).get_output(0)
    rotated = network.add_elementwise(
        _rotate_half(network, patches, 1, num_heads, num_patches, head_dim),
        sin_tensor,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    patches = network.add_elementwise(
        direct, rotated, trt.ElementWiseOperation.SUM
    ).get_output(0)
    concat = network.add_concatenation([prefix, patches])
    concat.axis = 2
    return concat.get_output(0)


def attention(network, q, k, v, head_dim: int, dtype: np.dtype):
    scalar = constant(
        network,
        np.asarray(1.0 / np.sqrt(float(head_dim))),
        (1, 1, 1, 1),
        dtype,
    )
    scalar = cast(network, scalar, q.dtype)
    q = network.add_elementwise(q, scalar, trt.ElementWiseOperation.PROD).get_output(0)
    scores = network.add_matrix_multiply(
        q, trt.MatrixOperation.NONE, k, trt.MatrixOperation.TRANSPOSE
    ).get_output(0)
    probs_layer = network.add_softmax(scores)
    probs_layer.axes = 1 << 3
    return network.add_matrix_multiply(
        probs_layer.get_output(0),
        trt.MatrixOperation.NONE,
        v,
        trt.MatrixOperation.NONE,
    ).get_output(0)
