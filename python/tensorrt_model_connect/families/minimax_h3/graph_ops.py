# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin-free TensorRT graph vocabulary for MiniMax-H3.

All math in this module uses TensorRT native layers. Distributed collectives
use TensorRT 11 ``IDistCollectiveLayer`` rather than a model plugin.
"""

from __future__ import annotations

import math

import numpy as np

from tensorrt_model_connect import trt_compat


trt = trt_compat.get_trt()


def constant(network, value, *, dtype=np.float32):
    array = np.ascontiguousarray(value, dtype=dtype)
    return network.add_constant(tuple(array.shape), array).get_output(0)


def cast(network, tensor, dtype):
    if tensor.dtype == dtype:
        return tensor
    return network.add_cast(tensor, dtype).get_output(0)


def linear(
    network,
    tensor,
    weight,
    bias=None,
    *,
    bf16: bool = True,
    compute_dtype=None,
):
    """PyTorch ``[out, in]`` linear expressed as native TensorRT GEMM."""

    tensor_rank = len(tuple(tensor.shape))
    rhs_value = np.asarray(weight, dtype=np.float32).T
    if tensor_rank > 2:
        # TensorRT MatrixMultiply applies NumPy-style broadcast across the
        # leading dimensions only when both operands expose those dimensions.
        # VAE layers are [batch, rows, width], so make the shared weight
        # explicitly [1, in, out] instead of relying on implicit rank lift.
        rhs_value = rhs_value.reshape((1,) * (tensor_rank - 2) + rhs_value.shape)
    rhs = constant(network, rhs_value)
    if compute_dtype is None and bf16:
        compute_dtype = trt.bfloat16
    if compute_dtype is not None:
        tensor = cast(network, tensor, compute_dtype)
        rhs = cast(network, rhs, compute_dtype)
    output = network.add_matrix_multiply(
        tensor, trt.MatrixOperation.NONE, rhs, trt.MatrixOperation.NONE
    ).get_output(0)
    if bias is not None:
        shape = (1,) * (len(tuple(output.shape)) - 1) + (-1,)
        bias_tensor = constant(network, np.asarray(bias, dtype=np.float32).reshape(shape))
        bias_tensor = cast(network, bias_tensor, output.dtype)
        output = network.add_elementwise(
            output, bias_tensor, trt.ElementWiseOperation.SUM
        ).get_output(0)
    return output


def silu(network, tensor):
    sigmoid = network.add_activation(tensor, trt.ActivationType.SIGMOID).get_output(0)
    return network.add_elementwise(tensor, sigmoid, trt.ElementWiseOperation.PROD).get_output(0)


def rms_norm(network, tensor, weight, width: int, eps: float):
    """PyTorch RMSNorm using native reduce, unary and elementwise layers."""

    source_dtype = tensor.dtype
    value = cast(network, tensor, trt.float32)
    square = network.add_elementwise(value, value, trt.ElementWiseOperation.PROD).get_output(0)
    axis = 1 << (len(tuple(value.shape)) - 1)
    mean = network.add_reduce(square, trt.ReduceOperation.AVG, axis, True).get_output(0)
    broadcast_shape = (1,) * len(tuple(value.shape))
    eps_tensor = constant(network, np.full(broadcast_shape, eps, dtype=np.float32))
    variance = network.add_elementwise(mean, eps_tensor, trt.ElementWiseOperation.SUM).get_output(0)
    root = network.add_unary(variance, trt.UnaryOperation.SQRT).get_output(0)
    inverse = network.add_unary(root, trt.UnaryOperation.RECIP).get_output(0)
    normalized = network.add_elementwise(value, inverse, trt.ElementWiseOperation.PROD).get_output(
        0
    )
    gamma_shape = (1,) * (len(tuple(value.shape)) - 1) + (width,)
    gamma = constant(network, np.asarray(weight, dtype=np.float32).reshape(gamma_shape))
    normalized = network.add_elementwise(
        normalized, gamma, trt.ElementWiseOperation.PROD
    ).get_output(0)
    return cast(network, normalized, source_dtype)


def gather_rows(network, table, indices):
    return network.add_gather(table, indices, 0).get_output(0)


def modulate(network, normalized, shift, scale):
    one = constant(
        network,
        np.ones((1,) * len(tuple(normalized.shape)), dtype=np.float32),
    )
    one = cast(network, one, normalized.dtype)
    scale = cast(network, scale, normalized.dtype)
    shift = cast(network, shift, normalized.dtype)
    scale = network.add_elementwise(scale, one, trt.ElementWiseOperation.SUM).get_output(0)
    value = network.add_elementwise(normalized, scale, trt.ElementWiseOperation.PROD).get_output(0)
    return network.add_elementwise(value, shift, trt.ElementWiseOperation.SUM).get_output(0)


def gated_residual(network, residual, update, gate):
    gate = cast(network, gate, update.dtype)
    update = network.add_elementwise(update, gate, trt.ElementWiseOperation.PROD).get_output(0)
    update = cast(network, update, residual.dtype)
    return network.add_elementwise(residual, update, trt.ElementWiseOperation.SUM).get_output(0)


def swiglu(network, tensor, weight_in, weight_out, ffn_dim: int):
    projected = linear(network, tensor, weight_in)
    rows = int(projected.shape[0])
    value = network.add_slice(projected, (0, 0), (rows, ffn_dim), (1, 1)).get_output(0)
    gate = network.add_slice(projected, (0, ffn_dim), (rows, ffn_dim), (1, 1)).get_output(0)
    activated = silu(network, gate)
    hidden = network.add_elementwise(value, activated, trt.ElementWiseOperation.PROD).get_output(0)
    return linear(network, hidden, weight_out)


def fused_qkv(network, tensor, weights: dict, prefix: str):
    """Pack Q/K/V into one TensorRT GEMM, matching Sol-Engine's lossless path."""

    packed_weight = np.concatenate(
        [weights[f"{prefix}.to_{name}.weight"] for name in ("q", "k", "v")], axis=0
    )
    packed = linear(network, tensor, packed_weight)
    rows = int(packed.shape[0])
    width = int(packed.shape[1]) // 3
    return tuple(
        network.add_slice(packed, (0, index * width), (rows, width), (1, 1)).get_output(0)
        for index in range(3)
    )


def rows_to_heads(network, tensor, rows: int, heads: int, head_dim: int):
    reshape = network.add_shuffle(tensor)
    reshape.reshape_dims = (rows, heads, head_dim)
    reshape.second_transpose = trt.Permutation([1, 0, 2])
    batch = network.add_shuffle(reshape.get_output(0))
    batch.reshape_dims = (1, heads, rows, head_dim)
    return batch.get_output(0)


def heads_to_rows(network, tensor, rows: int, width: int):
    reshape = network.add_shuffle(tensor)
    reshape.first_transpose = trt.Permutation([0, 2, 1, 3])
    reshape.reshape_dims = (rows, width)
    return reshape.get_output(0)


def partial_rope(
    network,
    tensor,
    cos_half,
    sin_half,
    *,
    rows: int,
    heads: int,
    head_dim: int,
    rotary_dim: int,
    interleaved: bool = False,
):
    """Apply H3's 96-channel MM-RoPE with native ``IRotaryLayer``."""

    value = rows_to_heads(network, tensor, rows, heads, head_dim)
    cos_half = cast(network, cos_half, value.dtype)
    sin_half = cast(network, sin_half, value.dtype)
    layer = network.add_rotary_embedding(value, cos_half, sin_half, interleaved, rotary_dim)
    if layer is None:
        raise RuntimeError("TensorRT failed to add MiniMax-H3 partial rotary embedding")
    return heads_to_rows(network, layer.get_output(0), rows, heads * head_dim)


def add_collective(network, tensor, operation, world_size: int):
    add = getattr(network, "add_dist_collective", None)
    if add is None:
        raise RuntimeError("MiniMax-H3 context parallelism requires TensorRT 11 collectives")
    reduce_none = getattr(trt.ReduceOperation, "NONE", trt.ReduceOperation.SUM)
    layer = add(tensor, operation, reduce_none, -1, [])
    if layer is None or not hasattr(layer, "num_ranks"):
        raise RuntimeError("TensorRT failed to add a distributed collective")
    layer.num_ranks = world_size
    return layer.get_output(0)


def slice_replicated_for_rank(network, tensor, rank, local_rows: int):
    """Select a rank's rows from an identical full input without a collective."""

    width = int(tensor.shape[1])
    world_size = int(tensor.shape[0]) // local_rows
    sharded = network.add_shuffle(tensor)
    sharded.reshape_dims = (world_size, local_rows, width)
    selected = network.add_gather(sharded.get_output(0), rank, 0)
    flattened = network.add_shuffle(selected.get_output(0))
    flattened.reshape_dims = (local_rows, width)
    return flattened.get_output(0)


def all_gather(network, tensor, world_size: int):
    return add_collective(network, tensor, trt.CollectiveOperation.ALL_GATHER, world_size)


def ulysses_seq_to_head(network, tensor, *, local_rows: int, heads: int, head_dim: int, cp: int):
    local_heads = heads // cp
    routed = network.add_shuffle(tensor)
    routed.reshape_dims = (local_rows, cp, local_heads, head_dim)
    routed.second_transpose = trt.Permutation([1, 0, 2, 3])
    exchanged = add_collective(
        network, routed.get_output(0), trt.CollectiveOperation.ALL_TO_ALL, cp
    )
    full = network.add_shuffle(exchanged)
    full.first_transpose = trt.Permutation([2, 0, 1, 3])
    full.reshape_dims = (1, local_heads, local_rows * cp, head_dim)
    return full.get_output(0)


def ulysses_head_to_seq(network, tensor, *, local_rows: int, heads: int, head_dim: int, cp: int):
    local_heads = heads // cp
    routed = network.add_shuffle(tensor)
    routed.reshape_dims = (local_heads, cp, local_rows, head_dim)
    routed.second_transpose = trt.Permutation([1, 0, 2, 3])
    exchanged = add_collective(
        network, routed.get_output(0), trt.CollectiveOperation.ALL_TO_ALL, cp
    )
    local = network.add_shuffle(exchanged)
    local.first_transpose = trt.Permutation([2, 0, 1, 3])
    local.reshape_dims = (local_rows, heads * head_dim)
    return local.get_output(0)


def ulysses_attention(
    network, q, k, v, key_mask, *, local_rows: int, heads: int, head_dim: int, cp: int
):
    """Ulysses exchange around native fused TensorRT attention."""

    q = ulysses_seq_to_head(
        network, q, local_rows=local_rows, heads=heads, head_dim=head_dim, cp=cp
    )
    k = ulysses_seq_to_head(
        network, k, local_rows=local_rows, heads=heads, head_dim=head_dim, cp=cp
    )
    v = ulysses_seq_to_head(
        network, v, local_rows=local_rows, heads=heads, head_dim=head_dim, cp=cp
    )
    # TensorRT's fused FP16 attention has three additional mantissa bits over
    # BF16. Keep the surrounding block and collectives in checkpoint-native
    # BF16 while reducing the recurrent numerical drift across 49 denoising
    # evaluations.
    q = cast(network, q, trt.float16)
    k = cast(network, k, trt.float16)
    v = cast(network, v, trt.float16)
    scale = constant(
        network,
        np.full((1, 1, 1, 1), 1.0 / math.sqrt(head_dim), dtype=np.float32),
    )
    scale = cast(network, scale, q.dtype)
    q = network.add_elementwise(q, scale, trt.ElementWiseOperation.PROD).get_output(0)
    layer = network.add_attention(q, k, v, trt.AttentionNormalizationOp.SOFTMAX, False)
    if layer is None:
        raise RuntimeError("TensorRT failed to add MiniMax-H3 native attention")
    layer.decomposable = False
    if key_mask is not None:
        layer.mask = cast(network, key_mask, trt.float16)
    context = cast(network, layer.get_output(0), trt.bfloat16)
    return ulysses_head_to_seq(
        network, context, local_rows=local_rows, heads=heads, head_dim=head_dim, cp=cp
    )
