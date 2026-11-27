# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen4-Exp sparse attention for an ordinary contiguous generation request."""

import math

import numpy as np
import tensorrt as trt


def sparse_attention(network, hidden, positions, rotary, past, weights, epsilon):
    """Append one request's current tokens to its K/V and raw index-key caches.

    The capacity is padded to four rows. The caller initializes the caches to
    zero, supplies consecutive valid physical positions, and commits the
    returned state only on success. Rotary coefficients are token-aligned:
    media positions need not equal physical cache indices.
    """
    n = network
    old_key, old_value, old_index = past
    capacity = int(old_key.shape[0])
    if capacity <= 0 or capacity % 4 or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Invalid Flash-Next sparse attention capacity or normalization epsilon")
    groups = capacity // 4
    block_budget = min(groups, 512)
    ew, mat = trt.ElementWiseOperation, trt.MatrixOperation

    def const(v):
        return n.add_constant(v.shape, np.ascontiguousarray(v)).get_output(0)

    def elt(a, b, op):
        return n.add_elementwise(a, b, op).get_output(0)

    def cast(x, dtype):
        return n.add_cast(x, dtype).get_output(0)

    def reshape(x, shape):
        sh = n.add_shuffle(x)
        sh.reshape_dims = shape
        return sh.get_output(0)

    def permute(x, order):
        sh = n.add_shuffle(x)
        sh.first_transpose = order
        return sh.get_output(0)

    def gather(x, ids, axis):
        return n.add_gather(x, ids, axis).get_output(0)

    def columns(x, first, last):
        return gather(x, const(np.arange(first, last, dtype=np.int32)), len(x.shape) - 1)

    def expected(name, shape):
        v = weights[name]
        if v.shape != shape:
            raise ValueError(f"Invalid Flash-Next sparse attention weight {name}: {v.shape}")
        return v

    def linear(x, name, features, width):
        w = expected(name + ".weight", (features, width)).astype(np.float16)
        return n.add_matrix_multiply(x, mat.NONE, const(w), mat.TRANSPOSE).get_output(0)

    def norm(x, name, width):
        fp = cast(x, trt.float32)
        sq = elt(fp, fp, ew.PROD)
        mean = n.add_reduce(sq, trt.ReduceOperation.AVG, 1 << 2, True).get_output(0)
        den = n.add_unary(
            elt(mean, const(np.full((1, 1, 1), epsilon, np.float32)), ew.SUM),
            trt.UnaryOperation.SQRT,
        ).get_output(0)
        gamma = 1 + expected(name + ".weight", (width,)).astype(np.float32)
        return cast(
            elt(elt(fp, den, ew.DIV), const(gamma.reshape(1, 1, width)), ew.PROD), trt.float16
        )

    def rotate(x, table, width):
        # Neox pairs across the two 32-wide halves of the rotary prefix.
        table = cast(table, trt.float16)
        c = reshape(columns(table, 0, 32), (0, 1, 32))
        s = reshape(columns(table, 32, 64), (0, 1, 32))
        a, b = columns(x, 0, 32), columns(x, 32, 64)
        left = elt(elt(a, c, ew.PROD), elt(b, s, ew.PROD), ew.SUB)
        right = elt(elt(b, c, ew.PROD), elt(a, s, ew.PROD), ew.SUM)
        concat = n.add_concatenation([left, right, columns(x, 64, width)])
        concat.axis = 2
        return concat.get_output(0)

    physical = reshape(positions, (-1, 1))
    current_rope = gather(rotary, positions, 0)

    def update(old, current):
        return n.add_scatter(old, physical, current, trt.ScatterMode.ND).get_output(0)

    proj = reshape(linear(hidden, "q_proj", 12288, 2560), (0, 24, 512))
    q = rotate(norm(columns(proj, 0, 256), "q_norm", 256), current_rope, 256)
    gate = reshape(columns(proj, 256, 512), (0, 6144))
    k = reshape(linear(hidden, "k_proj", 512, 2560), (0, 2, 256))
    k = update(old_key, rotate(norm(k, "k_norm", 256), current_rope, 256))
    v = update(old_value, reshape(linear(hidden, "v_proj", 512, 2560), (0, 2, 256)))

    idx = linear(hidden, "indexer.index_qk_proj", 640, 2560)
    iq = reshape(columns(idx, 0, 512), (0, 4, 128))
    iq = rotate(norm(iq, "indexer.q_layernorm", 128), current_rope, 128)
    ik = update(old_index, columns(idx, 512, 640))
    blocks = reshape(cast(ik, trt.float32), (groups, 4, 128))
    pooled = cast(
        n.add_reduce(blocks, trt.ReduceOperation.AVG, 1 << 1, True).get_output(0), trt.float16
    )
    block_rope = gather(rotary, const(np.arange(0, capacity, 4, dtype=np.int32)), 0)
    bk = rotate(norm(pooled, "indexer.k_layernorm", 128), block_rope, 128)

    # Score each complete block: ReLU per head before the sum.
    qidx = reshape(cast(iq, trt.float32), (-1, 128))
    kidx = reshape(cast(bk, trt.float32), (groups, 128))
    dots = n.add_matrix_multiply(qidx, mat.NONE, kidx, mat.TRANSPOSE).get_output(0)
    dots = n.add_activation(reshape(dots, (-1, 4, groups)), trt.ActivationType.RELU).get_output(0)
    scores = n.add_reduce(dots, trt.ReduceOperation.SUM, 1 << 1, False).get_output(0)
    scores = elt(scores, const(np.full((1, 1), math.sqrt(128), np.float32)), ew.DIV)
    visible = elt(
        elt(physical, const(np.array([[1]], np.int32)), ew.SUM),
        const(np.array([[4]], np.int32)),
        ew.FLOOR_DIV,
    )
    block = const(np.arange(groups, dtype=np.int32)[None, :])
    complete = elt(block, visible, ew.LESS)
    if groups > block_budget:
        ranked = n.add_select(
            complete, scores, const(np.array([[-np.inf]], np.float32))
        ).get_output(0)
        selected = n.add_topk(ranked, trt.TopKOperation.MAX, block_budget, 1 << 1).get_output(1)
        zeros = cast(elt(complete, complete, ew.XOR), trt.int32)
        ones = cast(elt(selected, selected, ew.EQUAL), trt.int32)
        scatter = n.add_scatter(zeros, selected, ones, trt.ScatterMode.ELEMENT)
        scatter.axis = 1
        # Top-k may include invisible slots on short prefixes. Never admit them.
        complete = elt(complete, cast(scatter.get_output(0), trt.bool), ew.AND)
    tokens = np.arange(capacity, dtype=np.int32)
    by_token = gather(complete, const(tokens // 4), 1)
    partial = elt(const((tokens // 4)[None, :]), visible, ew.EQUAL)
    after = elt(const(tokens[None, :]), physical, ew.GREATER)
    mask = elt(
        by_token,
        elt(partial, n.add_unary(after, trt.UnaryOperation.NOT).get_output(0), ew.AND),
        ew.OR,
    )

    # GQA shares each K/V head across 12 query heads; do not duplicate the KV
    # cache. The scoring workspace scales with the current chunk, not its square.
    query = reshape(permute(reshape(q, (0, 2, 12, 256)), (1, 0, 2, 3)), (2, -1, 256))
    logits = n.add_matrix_multiply(
        query, mat.NONE, permute(k, (1, 0, 2)), mat.TRANSPOSE
    ).get_output(0)
    logits = elt(
        reshape(logits, (2, -1, 12, capacity)),
        const(np.full((1, 1, 1, 1), 1 / 16, np.float16)),
        ew.PROD,
    )
    masked = n.add_select(
        reshape(mask, (1, -1, 1, capacity)),
        logits,
        const(np.full((1, 1, 1, 1), -np.inf, np.float16)),
    ).get_output(0)
    softmax = n.add_softmax(cast(masked, trt.float32))
    softmax.axes = 1 << 3
    prob = reshape(cast(softmax.get_output(0), trt.float16), (2, -1, capacity))
    context = n.add_matrix_multiply(prob, mat.NONE, permute(v, (1, 0, 2)), mat.NONE).get_output(0)
    context = reshape(permute(reshape(context, (2, -1, 12, 256)), (1, 0, 2, 3)), (-1, 6144))
    gated = elt(context, n.add_activation(gate, trt.ActivationType.SIGMOID).get_output(0), ew.PROD)
    return linear(gated, "o_proj", 2560, 6144), (k, v, ik), mask


def sparse_attention_sublayer(network, hyper, positions, rotary, past, checkpoint, layer, epsilon):
    """Apply both parts of the attention sublayer; preserve its K/V and index state."""
    from .residual import gated_sublayer

    prefix = f"model.language_model.layers.{layer}"

    class Projections:
        # sparse_attention checks the exact shape at each use. Normalize the
        # BF16 checkpoint to the dtype required by the projection or scale.
        def __getitem__(self, suffix):
            key = f"{prefix}.self_attn.{suffix}"
            shape = checkpoint.array(key).shape
            return checkpoint.bf16(key, shape, fp32=suffix.endswith("norm.weight"))

    def block(hidden):
        output, cache, admitted = sparse_attention(
            network, hidden, positions, rotary, past, Projections(), epsilon
        )
        return output, (cache, admitted)

    return gated_sublayer(
        network, hyper, checkpoint, prefix + ".attn_hyper_connection", epsilon, block
    )
