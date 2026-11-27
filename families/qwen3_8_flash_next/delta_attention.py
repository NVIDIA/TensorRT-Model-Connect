# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Flash-Next gated-delta attention with native TensorRT token recurrence."""

import math

import numpy as np
import tensorrt as trt

from .residual import gated_sublayer


def delta_attention(network, hidden, history, state, checkpoint, prefix, eps, max_chunk_tokens):
    """Process consecutive tokens from one request, returning both updated caches.

    History contains three *unconvolved* QKV rows; the FP32 state is indexed
    [value head, key channel, value channel]. Neither cache is shared between requests.
    """
    n = network
    ew, un, mat = trt.ElementWiseOperation, trt.UnaryOperation, trt.MatrixOperation
    heads, kheads, width = 48, 16, 128
    kd, vd = kheads * width, heads * width
    channels = 2 * kd + vd
    if (
        hidden.dtype != trt.float16
        or tuple(hidden.shape)[1:] != (2560,)
        or history.dtype != trt.float16
        or tuple(history.shape) != (3, channels)
        or state.dtype != trt.float32
        or tuple(state.shape) != (heads, width, width)
        or type(max_chunk_tokens) is not int
        or max_chunk_tokens < 1
        or not math.isfinite(eps)
        or eps <= 0
    ):
        raise ValueError("Invalid Flash-Next gated-delta tensors or epsilon")

    def const(v):
        return n.add_constant(v.shape, np.ascontiguousarray(v)).get_output(0)

    def cast(x, dtype):
        return n.add_cast(x, dtype).get_output(0)

    def op(x, y, kind):
        return n.add_elementwise(x, y, kind).get_output(0)

    def reshape(x, dims):
        sh = n.add_shuffle(x)
        sh.reshape_dims = dims
        return sh.get_output(0)

    def sigmoid(x):
        return n.add_activation(x, trt.ActivationType.SIGMOID).get_output(0)

    def weight(suffix, shape, *, fp32=False):
        return checkpoint.bf16(f"{prefix}.{suffix}", shape, fp32=fp32)

    def linear(x, suffix, rows, columns):
        w = const(weight(suffix + ".weight", (rows, columns)))
        return n.add_matrix_multiply(x, mat.NONE, w, mat.TRANSPOSE).get_output(0)

    qkv = linear(hidden, "in_proj_qkv", channels, 2560)
    size = n.add_shape(qkv).get_output(0)
    tokens = n.add_slice(size, (0,), (1,), (1,)).get_output(0)
    coords = np.int64 if size.dtype == trt.int64 else np.int32
    seq = n.add_concatenation([history, qkv])
    seq.axis = 0
    seq = seq.get_output(0)
    kernel = weight("conv1d.weight", (channels, 1, 4), fp32=True)
    # Prepending the carried history and convolving with no padding produces
    # exactly one row per current token. The weights are depthwise, not dense.
    layout = n.add_shuffle(cast(seq, trt.float32))
    layout.first_transpose = (1, 0)
    layout.reshape_dims = (1, channels, -1, 1)
    layer = n.add_convolution_nd(
        layout.get_output(0),
        channels,
        (4, 1),
        np.ascontiguousarray(kernel[..., None]),
        trt.Weights(trt.float32),
    )
    layer.num_groups = channels
    flatten = n.add_shuffle(layer.get_output(0))
    flatten.reshape_dims = (channels, -1)
    flatten.second_transpose = (1, 0)
    indices = op(tokens, const(np.arange(3, dtype=coords)), ew.SUM)
    tail = n.add_gather(seq, indices, 0)

    # HF's causal depthwise convolution rounds to FP16 before SiLU.
    conv = cast(flatten.get_output(0), trt.float16)
    conv32 = cast(conv, trt.float32)
    act = cast(op(conv32, sigmoid(conv32), ew.PROD), trt.float16)
    headmap = const(np.repeat(np.arange(kheads, dtype=np.int32), heads // kheads))

    def part(first, end):
        cols = const(np.arange(first, end, dtype=np.int32))
        return n.add_gather(act, cols, 1).get_output(0)

    def normalized(x):
        x = reshape(cast(x, trt.float32), (0, kheads, width))
        ss = n.add_reduce(op(x, x, ew.PROD), trt.ReduceOperation.SUM, 1 << 2, True).get_output(0)
        den = n.add_unary(
            op(ss, const(np.full((1, 1, 1), 1e-6, np.float32)), ew.SUM), un.SQRT
        ).get_output(0)
        return n.add_gather(op(x, den, ew.DIV), headmap, 1).get_output(0)

    q = op(normalized(part(0, kd)), const(np.full((1, 1, 1), width**-0.5, np.float32)), ew.PROD)
    k = normalized(part(kd, 2 * kd))
    v = reshape(cast(part(2 * kd, channels), trt.float32), (0, heads, width))
    beta = cast(sigmoid(linear(hidden, "in_proj_b", heads, 2560)), trt.float32)
    a = cast(linear(hidden, "in_proj_a", heads, 2560), trt.float32)
    a = op(a, const(weight("dt_bias", (heads,), fp32=True).reshape(1, heads)), ew.SUM)
    # Stable softplus avoids overflowing exp(a) for positive large a.
    abs_a = n.add_unary(a, un.ABS).get_output(0)
    exp = n.add_unary(n.add_unary(abs_a, un.NEG).get_output(0), un.EXP).get_output(0)
    one = const(np.ones((1, 1), np.float32))
    positive = n.add_activation(a, trt.ActivationType.RELU).get_output(0)
    sp = op(positive, n.add_unary(op(one, exp, ew.SUM), un.LOG).get_output(0), ew.SUM)
    multiplier = -np.exp(weight("A_log", (heads,), fp32=True))
    decay = n.add_unary(op(sp, const(multiplier.reshape(1, heads)), ew.PROD), un.EXP).get_output(0)

    # Chunk width is the plan's optimization-profile maximum. Neutral rows
    # have unit decay and zero beta/Q/K/V: they neither emit features nor
    # change recurrent state. The caller can therefore split any prompt
    # across chunks, including a one-token decode, without padding the cache.
    def pad(x, shape, value=0.0):
        fill = const(np.full((max_chunk_tokens, *shape), value, np.float32))
        joined = n.add_concatenation([x, fill])
        joined.axis = 0
        dims = (max_chunk_tokens, *shape)
        return n.add_slice(
            joined.get_output(0), (0,) * len(dims), dims, (1,) * len(dims)
        ).get_output(0)

    q, k, v = (pad(t, (heads, width)) for t in (q, k, v))
    beta, decay = pad(beta, (heads,)), pad(decay, (heads,), 1.0)
    outputs = []
    new_state = state
    for row in range(max_chunk_tokens):
        index = const(np.array(row, np.int32))

        def at(x):
            return n.add_gather(x, index, 0).get_output(0)

        faded = op(new_state, reshape(at(decay), (heads, 1, 1)), ew.PROD)
        kt = reshape(at(k), (heads, 1, width))
        prev = n.add_matrix_multiply(kt, mat.NONE, faded, mat.NONE).get_output(0)
        delta = op(
            op(at(v), reshape(prev, (heads, width)), ew.SUB), reshape(at(beta), (heads, 1)), ew.PROD
        )
        new_state = op(
            faded,
            op(reshape(kt, (heads, width, 1)), reshape(delta, (heads, 1, width)), ew.PROD),
            ew.SUM,
        )
        qt = reshape(at(q), (heads, 1, width))
        read = n.add_matrix_multiply(qt, mat.NONE, new_state, mat.NONE).get_output(0)
        outputs.append(reshape(read, (1, heads, width)))
    stacked = n.add_concatenation(outputs)
    stacked.axis = 0
    extent = n.add_concatenation([tokens, const(np.array([heads, width], coords))])
    extent.axis = 0
    valid = n.add_slice(
        stacked.get_output(0), (0, 0, 0), (max_chunk_tokens, heads, width), (1, 1, 1)
    )
    valid.set_input(2, extent.get_output(0))

    # Match HF's activation rounding and sigmoid (not SiLU) output gate.
    y = cast(cast(valid.get_output(0), trt.float16), trt.float32)
    var = n.add_reduce(op(y, y, ew.PROD), trt.ReduceOperation.AVG, 1 << 2, True).get_output(0)
    den = n.add_unary(
        op(var, const(np.full((1, 1, 1), eps, np.float32)), ew.SUM), un.SQRT
    ).get_output(0)
    gamma = const(weight("norm.weight", (width,)).reshape(1, 1, width))
    normed = op(cast(op(y, den, ew.DIV), trt.float16), gamma, ew.PROD)
    z = reshape(cast(linear(hidden, "in_proj_z", vd, 2560), trt.float32), (0, heads, width))
    gated = cast(op(cast(normed, trt.float32), sigmoid(z), ew.PROD), trt.float16)
    output = linear(reshape(gated, (0, vd)), "out_proj", 2560, vd)
    return output, (tail.get_output(0), new_state)


def delta_attention_sublayer(
    network, hyper, history, state, checkpoint, layer, epsilon, max_chunk_tokens
):
    """Run recurrent attention between the learned residual read and update."""
    prefix = f"model.language_model.layers.{layer}"

    def block(hidden):
        return delta_attention(
            network,
            hidden,
            history,
            state,
            checkpoint,
            prefix + ".linear_attn",
            epsilon,
            max_chunk_tokens,
        )

    return gated_sublayer(
        network, hyper, checkpoint, prefix + ".attn_hyper_connection", epsilon, block
    )
