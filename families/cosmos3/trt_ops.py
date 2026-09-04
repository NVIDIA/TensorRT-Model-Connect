# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small plugin-free TensorRT graph vocabulary for Cosmos3-Nano."""

from __future__ import annotations

import math

import numpy as np

import tensorrt as trt


def cast(network, tensor, dtype):
    if tensor.dtype == dtype:
        return tensor
    return network.add_cast(tensor, dtype).get_output(0)


def constant(network, value, *, dtype=np.float32):
    array = np.ascontiguousarray(value, dtype=dtype)
    return network.add_constant(tuple(array.shape), array).get_output(0)


def linear(network, x, weight, bias=None, *, bf16: bool = True):
    """Linear projection from a PyTorch-layout ``[out, in]`` weight."""

    rhs = constant(network, np.asarray(weight, dtype=np.float32).T)
    if bf16:
        x = cast(network, x, trt.bfloat16)
        rhs = cast(network, rhs, trt.bfloat16)
    result = network.add_matrix_multiply(
        x,
        trt.MatrixOperation.NONE,
        rhs,
        trt.MatrixOperation.NONE,
    ).get_output(0)
    if bias is not None:
        bias_tensor = constant(network, np.asarray(bias, dtype=np.float32).reshape(1, -1))
        bias_tensor = cast(network, bias_tensor, result.dtype)
        result = network.add_elementwise(
            result, bias_tensor, trt.ElementWiseOperation.SUM
        ).get_output(0)
    return cast(network, result, trt.bfloat16) if bf16 else result


def rms_norm(network, x, weight, hidden_size: int, eps: float):
    """RMSNorm in FP32 with a BF16 output boundary."""

    x_fp32 = cast(network, x, trt.float32)
    squared = network.add_elementwise(x_fp32, x_fp32, trt.ElementWiseOperation.PROD).get_output(0)
    rank = len(tuple(x.shape))
    mean = network.add_reduce(squared, trt.ReduceOperation.AVG, 1 << (rank - 1), True).get_output(0)
    epsilon = constant(network, np.array([eps], dtype=np.float32).reshape((1,) * rank))
    variance = network.add_elementwise(mean, epsilon, trt.ElementWiseOperation.SUM).get_output(0)
    inverse = network.add_unary(variance, trt.UnaryOperation.SQRT).get_output(0)
    inverse = network.add_unary(inverse, trt.UnaryOperation.RECIP).get_output(0)
    normalized = network.add_elementwise(x_fp32, inverse, trt.ElementWiseOperation.PROD).get_output(
        0
    )
    gamma_shape = (1,) * (rank - 1) + (hidden_size,)
    gamma = constant(network, np.asarray(weight, dtype=np.float32).reshape(gamma_shape))
    result = network.add_elementwise(normalized, gamma, trt.ElementWiseOperation.PROD).get_output(0)
    return cast(network, result, trt.bfloat16)


def rms_norm_per_head(
    network,
    x,
    weight,
    *,
    sequence_length: int,
    num_heads: int,
    head_dim: int,
    eps: float,
):
    rows = network.add_shuffle(x)
    rows.reshape_dims = (sequence_length, num_heads, head_dim)
    normalized = rms_norm(network, rows.get_output(0), weight, head_dim, eps)
    result = network.add_shuffle(normalized)
    result.reshape_dims = (sequence_length, num_heads * head_dim)
    return result.get_output(0)


def silu(network, x):
    sigmoid = network.add_activation(x, trt.ActivationType.SIGMOID).get_output(0)
    return network.add_elementwise(x, sigmoid, trt.ElementWiseOperation.PROD).get_output(0)


def swiglu_mlp(network, x, gate_weight, up_weight, down_weight):
    gate = silu(network, linear(network, x, gate_weight))
    up = linear(network, x, up_weight)
    gated = network.add_elementwise(gate, up, trt.ElementWiseOperation.PROD).get_output(0)
    return linear(network, gated, down_weight)


def residual(network, x, update):
    x = cast(network, x, trt.float32)
    update = cast(network, update, trt.float32)
    return network.add_elementwise(x, update, trt.ElementWiseOperation.SUM).get_output(0)


def apply_rotate_half_rope(
    network,
    rows,
    cos,
    sin,
    *,
    sequence_length: int,
    num_heads: int,
    head_dim: int,
):
    """Apply the checkpoint's rotate-half mRoPE values to row-major heads."""

    heads = network.add_shuffle(rows)
    heads.reshape_dims = (sequence_length, num_heads, head_dim)
    typed = cast(network, heads.get_output(0), trt.bfloat16)

    cos_view = network.add_shuffle(cast(network, cos, trt.bfloat16))
    cos_view.reshape_dims = (sequence_length, 1, head_dim)
    sin_view = network.add_shuffle(cast(network, sin, trt.bfloat16))
    sin_view.reshape_dims = (sequence_length, 1, head_dim)

    half = head_dim // 2
    first = network.add_slice(
        typed, (0, 0, 0), (sequence_length, num_heads, half), (1, 1, 1)
    ).get_output(0)
    second = network.add_slice(
        typed, (0, 0, half), (sequence_length, num_heads, half), (1, 1, 1)
    ).get_output(0)
    minus_one = cast(
        network,
        constant(network, np.array([[[-1.0]]], dtype=np.float32)),
        trt.bfloat16,
    )
    negative_second = network.add_elementwise(
        second, minus_one, trt.ElementWiseOperation.PROD
    ).get_output(0)
    rotated = network.add_concatenation([negative_second, first])
    rotated.axis = 2

    direct = network.add_elementwise(
        typed, cos_view.get_output(0), trt.ElementWiseOperation.PROD
    ).get_output(0)
    crossed = network.add_elementwise(
        rotated.get_output(0), sin_view.get_output(0), trt.ElementWiseOperation.PROD
    ).get_output(0)
    output = network.add_elementwise(direct, crossed, trt.ElementWiseOperation.SUM)
    output_rows = network.add_shuffle(output.get_output(0))
    output_rows.reshape_dims = (sequence_length, num_heads * head_dim)
    return output_rows.get_output(0)


def rows_to_heads(network, x, sequence_length: int, heads: int, head_dim: int):
    rows = network.add_shuffle(x)
    rows.reshape_dims = (sequence_length, heads, head_dim)
    rows.second_transpose = trt.Permutation([1, 0, 2])
    batched = network.add_shuffle(rows.get_output(0))
    batched.reshape_dims = (1, heads, sequence_length, head_dim)
    return batched.get_output(0)


def heads_to_rows(network, x, sequence_length: int, hidden_size: int):
    rows = network.add_shuffle(x)
    rows.first_transpose = trt.Permutation([0, 2, 1, 3])
    rows.reshape_dims = (sequence_length, hidden_size)
    return rows.get_output(0)


def attention(
    network,
    q,
    k,
    v,
    *,
    q_sequence_length: int,
    kv_sequence_length: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    mask=None,
):
    """Native TensorRT GQA using TensorRT's decomposable attention layer."""

    q4 = rows_to_heads(
        network, cast(network, q, trt.bfloat16), q_sequence_length, num_heads, head_dim
    )
    k4 = rows_to_heads(
        network, cast(network, k, trt.bfloat16), kv_sequence_length, num_kv_heads, head_dim
    )
    v4 = rows_to_heads(
        network, cast(network, v, trt.bfloat16), kv_sequence_length, num_kv_heads, head_dim
    )
    scale = cast(
        network,
        constant(network, np.array([[[[1.0 / math.sqrt(head_dim)]]]], dtype=np.float32)),
        trt.bfloat16,
    )
    q4 = network.add_elementwise(q4, scale, trt.ElementWiseOperation.PROD).get_output(0)
    layer = network.add_attention(
        q4,
        k4,
        v4,
        trt.AttentionNormalizationOp.SOFTMAX,
        causal,
    )
    if layer is None:
        raise RuntimeError("TensorRT failed to add Cosmos3 fused attention")
    layer.decomposable = True
    if mask is not None:
        layer.mask = cast(network, mask, trt.bfloat16)
    return heads_to_rows(
        network,
        cast(network, layer.get_output(0), trt.bfloat16),
        q_sequence_length,
        num_heads * head_dim,
    )


def add_collective(network, tensor, operation, world_size: int, *, reduce_operation=None):
    if reduce_operation is None:
        reduce_operation = trt.ReduceOperation.NONE
    layer = network.add_dist_collective(tensor, operation, reduce_operation, -1, [])
    if layer is None:
        raise RuntimeError(f"TensorRT failed to add Cosmos3 collective {operation}")
    layer.name = f"cosmos3_collective_{network.num_layers}_{operation}"
    layer.num_ranks = int(world_size)
    return layer.get_output(0)


def reduce_scatter_replicated(network, tensor, world_size: int):
    """Shard identical full rows while cancelling REDUCE_SCATTER's sum."""

    scale = cast(
        network,
        constant(network, np.array([[1.0 / world_size]], dtype=np.float32)),
        tensor.dtype,
    )
    scaled = network.add_elementwise(tensor, scale, trt.ElementWiseOperation.PROD).get_output(0)
    return add_collective(
        network,
        scaled,
        trt.CollectiveOperation.REDUCE_SCATTER,
        world_size,
        reduce_operation=trt.ReduceOperation.SUM,
    )


def ulysses_seq_to_heads(
    network,
    tensor,
    *,
    local_sequence_length: int,
    total_heads: int,
    head_dim: int,
    world_size: int,
):
    """Exchange sequence shards for full-sequence head shards."""

    local_heads = total_heads // world_size
    routed = network.add_shuffle(tensor)
    routed.reshape_dims = (local_sequence_length, world_size, local_heads, head_dim)
    routed.second_transpose = trt.Permutation([1, 0, 2, 3])
    # TensorRT 11.1 has native FP16/FP32 ALL_TO_ALL tactics, but its BF16
    # collective is routed through Myelin and rejected at engine-build time.
    # Keep the transformer math in BF16 and use FP16 only on the wire, matching
    # the precision used by the existing Wan Ulysses implementation.
    routed_fp16 = cast(network, routed.get_output(0), trt.float16)
    exchanged = add_collective(network, routed_fp16, trt.CollectiveOperation.ALL_TO_ALL, world_size)
    full = network.add_shuffle(cast(network, exchanged, trt.bfloat16))
    full.first_transpose = trt.Permutation([2, 0, 1, 3])
    full.reshape_dims = (1, local_heads, local_sequence_length * world_size, head_dim)
    return full.get_output(0)


def repeat_kv_heads(network, tensor, *, kv_heads: int, query_heads: int):
    """Expand each local GQA KV head into its consecutive query-head group."""

    if query_heads % kv_heads:
        raise ValueError("Cosmos3 local query heads must be a multiple of local KV heads")
    repeats = query_heads // kv_heads
    if repeats == 1:
        return tensor
    _, _, sequence_length, head_dim = tuple(tensor.shape)
    parts = []
    for index in range(kv_heads):
        head = network.add_slice(
            tensor,
            (0, index, 0, 0),
            (1, 1, sequence_length, head_dim),
            (1, 1, 1, 1),
        ).get_output(0)
        parts.extend([head] * repeats)
    expanded = network.add_concatenation(parts)
    expanded.axis = 1
    return expanded.get_output(0)


def ulysses_heads_to_seq(
    network,
    tensor,
    *,
    local_sequence_length: int,
    total_heads: int,
    head_dim: int,
    world_size: int,
):
    """Invert Ulysses head sharding back to local row-major tokens."""

    local_heads = total_heads // world_size
    routed = network.add_shuffle(tensor)
    routed.reshape_dims = (local_heads, world_size, local_sequence_length, head_dim)
    routed.second_transpose = trt.Permutation([1, 0, 2, 3])
    routed_fp16 = cast(network, routed.get_output(0), trt.float16)
    exchanged = add_collective(network, routed_fp16, trt.CollectiveOperation.ALL_TO_ALL, world_size)
    rows = network.add_shuffle(cast(network, exchanged, trt.bfloat16))
    rows.first_transpose = trt.Permutation([2, 0, 1, 3])
    rows.reshape_dims = (local_sequence_length, total_heads * head_dim)
    return rows.get_output(0)


def ulysses_dual_attention(
    network,
    q_text,
    k_text,
    v_text,
    q_vision,
    k_vision,
    v_vision,
    *,
    local_text_length: int,
    local_vision_length: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    world_size: int,
    generation_mask,
    text_causal_mask,
):
    """Official two-pathway Ulysses exchange with local GQA expansion.

    TensorRT 11.1 cannot compile two collective-adjacent ``IAttention``
    regions in this graph.  Keep the dominant joint video attention in
    ``IAttention`` so TensorRT can fuse supported shapes or decompose that
    layer, and manually decompose the much shorter causal text attention.
    """

    q_text_full = ulysses_seq_to_heads(
        network,
        q_text,
        local_sequence_length=local_text_length,
        total_heads=num_heads,
        head_dim=head_dim,
        world_size=world_size,
    )
    q_vision_full = ulysses_seq_to_heads(
        network,
        q_vision,
        local_sequence_length=local_vision_length,
        total_heads=num_heads,
        head_dim=head_dim,
        world_size=world_size,
    )
    kv_inputs = (k_text, v_text, k_vision, v_vision)
    k_text_full, v_text_full, k_vision_full, v_vision_full = (
        ulysses_seq_to_heads(
            network,
            value,
            local_sequence_length=(local_text_length if index < 2 else local_vision_length),
            total_heads=num_kv_heads,
            head_dim=head_dim,
            world_size=world_size,
        )
        for index, value in enumerate(kv_inputs)
    )
    local_query_heads = num_heads // world_size
    local_kv_heads = num_kv_heads // world_size
    k_text_full = repeat_kv_heads(
        network, k_text_full, kv_heads=local_kv_heads, query_heads=local_query_heads
    )
    v_text_full = repeat_kv_heads(
        network, v_text_full, kv_heads=local_kv_heads, query_heads=local_query_heads
    )
    k_vision_full = repeat_kv_heads(
        network, k_vision_full, kv_heads=local_kv_heads, query_heads=local_query_heads
    )
    v_vision_full = repeat_kv_heads(
        network, v_vision_full, kv_heads=local_kv_heads, query_heads=local_query_heads
    )

    scale = cast(
        network,
        constant(network, np.array([[[[1.0 / math.sqrt(head_dim)]]]], dtype=np.float32)),
        trt.bfloat16,
    )

    def _fused(q4, k4, v4, *, causal: bool, mask=None):
        q4 = network.add_elementwise(
            cast(network, q4, trt.bfloat16), scale, trt.ElementWiseOperation.PROD
        ).get_output(0)
        layer = network.add_attention(
            q4,
            cast(network, k4, trt.bfloat16),
            cast(network, v4, trt.bfloat16),
            trt.AttentionNormalizationOp.SOFTMAX,
            causal,
        )
        if layer is None:
            raise RuntimeError("TensorRT failed to add Cosmos3 CP fused attention")
        layer.name = f"cosmos3_cp_attention_{network.num_layers}_joint"
        layer.decomposable = True
        if mask is not None:
            layer.mask = cast(network, mask, trt.bfloat16)
        return cast(network, layer.get_output(0), trt.bfloat16)

    def _decomposed_causal(q4, k4, v4):
        k_transpose = network.add_shuffle(k4)
        k_transpose.second_transpose = trt.Permutation([0, 1, 3, 2])
        scores = network.add_matrix_multiply(
            q4,
            trt.MatrixOperation.NONE,
            k_transpose.get_output(0),
            trt.MatrixOperation.NONE,
        ).get_output(0)
        scores = cast(network, scores, trt.float32)
        score_scale = constant(
            network,
            np.array([[[[1.0 / math.sqrt(head_dim)]]]], dtype=np.float32),
        )
        scores = network.add_elementwise(scores, score_scale, trt.ElementWiseOperation.PROD)
        scores = network.add_elementwise(
            scores.get_output(0), text_causal_mask, trt.ElementWiseOperation.SUM
        ).get_output(0)
        probabilities = network.add_softmax(scores)
        probabilities.axes = 1 << 3
        return network.add_matrix_multiply(
            cast(network, probabilities.get_output(0), trt.bfloat16),
            trt.MatrixOperation.NONE,
            v4,
            trt.MatrixOperation.NONE,
        ).get_output(0)

    text_context = _decomposed_causal(q_text_full, k_text_full, v_text_full)
    all_k = network.add_concatenation([k_text_full, k_vision_full])
    all_k.axis = 2
    all_v = network.add_concatenation([v_text_full, v_vision_full])
    all_v.axis = 2
    vision_context = _fused(
        q_vision_full,
        all_k.get_output(0),
        all_v.get_output(0),
        causal=False,
        mask=generation_mask,
    )
    return (
        ulysses_heads_to_seq(
            network,
            text_context,
            local_sequence_length=local_text_length,
            total_heads=num_heads,
            head_dim=head_dim,
            world_size=world_size,
        ),
        ulysses_heads_to_seq(
            network,
            vision_context,
            local_sequence_length=local_vision_length,
            total_heads=num_heads,
            head_dim=head_dim,
            world_size=world_size,
        ),
    )
