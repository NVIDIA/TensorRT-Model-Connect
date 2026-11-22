# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D3's batched causal DeltaNet step with independent FP32 recurrent states."""
import numpy as np
import tensorrt as trt
import ml_dtypes
from . import graph_ops

BF16 = ml_dtypes.bfloat16


def delta_layer(g, x, weights, text):
    kh, vh = text["linear_num_key_heads"], text["linear_num_value_heads"]
    kd, vd = text["linear_key_head_dim"], text["linear_value_head_dim"]
    key_width, value_width = kh * kd, vh * vd
    conv_width = 2 * key_width + value_width
    kernel_size = text["linear_conv_kernel_dim"]
    batch = g.dim(x, 0)
    conv = g.input("conv", (-1, conv_width, kernel_size), axes={0: "batch"})
    state = g.input("state", (-1, vh, kd, vd), trt.float32, {0: "batch"})
    n = g.norm(x, weights["input_layernorm.weight"])

    def linear(value, name):
        return g.linear(value, weights[name + ".weight"])

    def unary(value, op):
        return g.net.add_unary(value, op).get_output(0)

    qkv = linear(n, "linear_attn.in_proj_qkv")
    z = linear(n, "linear_attn.in_proj_z")
    a = linear(n, "linear_attn.in_proj_a")
    b = linear(n, "linear_attn.in_proj_b")
    joined = g.net.add_concatenation([g.gather(conv, list(range(1, kernel_size)), 2),
                                      g.reshape(qkv, (batch, conv_width, 1))])
    joined.axis = 2
    conv_out = joined.get_output(0)
    kernel = g.constant(weights["linear_attn.conv1d.weight"].reshape(1, conv_width, kernel_size))
    products = g.mul(g.cast(conv_out, trt.float32), kernel)
    summed = g.net.add_reduce(products, trt.ReduceOperation.SUM, 1 << 2, False).get_output(0)
    activated = graph_ops.add_activation(g.net, g.cast(summed, trt.bfloat16), "silu", BF16)
    q, k, v = [g.reshape(g.gather(activated, list(range(lo, hi)), 1), (batch, heads, dim))
               for lo, hi, heads, dim in [(0, key_width, kh, kd),
                   (key_width, 2 * key_width, kh, kd),
                   (2 * key_width, conv_width, vh, vd)]]
    q = g.cast(graph_ops.add_l2_norm(g.net, q, 2, eps=1e-6, dtype=BF16), trt.float32)
    k = g.cast(graph_ops.add_l2_norm(g.net, k, 2, eps=1e-6, dtype=BF16), trt.float32)
    if vh != kh:
        repeated = [i for i in range(kh) for _ in range(vh // kh)]
        q, k = g.gather(q, repeated, 1), g.gather(k, repeated, 1)
    v = g.cast(v, trt.float32)

    biased = g.add(g.cast(a, trt.float32), g.constant(weights["linear_attn.dt_bias"].reshape(1, vh)))
    positive = g.binary(biased, g.constant([[0.0]]), trt.ElementWiseOperation.MAX)
    negative_abs = unary(unary(biased, trt.UnaryOperation.ABS), trt.UnaryOperation.NEG)
    softplus = g.add(positive, unary(g.add(unary(negative_abs, trt.UnaryOperation.EXP),
                                         g.constant([[1.0]])), trt.UnaryOperation.LOG))
    decay = unary(g.mul(softplus, g.constant(-np.exp(weights["linear_attn.A_log"]).reshape(1, vh))),
                  trt.UnaryOperation.EXP)
    decayed = g.mul(state, g.reshape(decay, (batch, vh, 1, 1)))
    beta = g.cast(g.net.add_activation(g.cast(b, trt.float32),
                                      trt.ActivationType.SIGMOID).get_output(0), trt.bfloat16)
    beta = g.reshape(g.cast(beta, trt.float32), (batch, vh, 1))
    key_col = g.reshape(k, (batch, vh, kd, 1))
    old = g.net.add_matrix_multiply(decayed, trt.MatrixOperation.TRANSPOSE,
                                    key_col, trt.MatrixOperation.NONE).get_output(0)
    delta = g.mul(g.binary(v, g.reshape(old, (batch, vh, vd)), trt.ElementWiseOperation.SUB), beta)
    update = g.net.add_matrix_multiply(key_col, trt.MatrixOperation.NONE,
                                      g.reshape(delta, (batch, vh, 1, vd)),
                                      trt.MatrixOperation.NONE).get_output(0)
    state_out = g.add(decayed, update)
    query = g.mul(q, g.constant([[[kd ** -.5]]]))
    core = g.net.add_matrix_multiply(state_out, trt.MatrixOperation.TRANSPOSE,
                                    g.reshape(query, (batch, vh, kd, 1)),
                                    trt.MatrixOperation.NONE).get_output(0)
    core = g.cast(g.reshape(core, (batch, vh, vd)), trt.bfloat16)
    value = g.cast(core, trt.float32)
    squares = g.mul(value, value)
    mean = g.net.add_reduce(squares, trt.ReduceOperation.AVG, 1 << 2, True).get_output(0)
    root = unary(g.add(mean, g.constant([[[1e-6]]])), trt.UnaryOperation.SQRT)
    norm = g.mul(value, unary(root, trt.UnaryOperation.RECIP))
    rounded = g.cast(norm, trt.bfloat16)
    scaled = g.mul(rounded, g.constant(weights["linear_attn.norm.weight"].reshape(1, 1, vd), BF16))
    gate = graph_ops.add_activation(g.net, g.cast(g.reshape(z, (batch, vh, vd)), trt.float32), "silu")
    gated = g.cast(g.mul(g.cast(scaled, trt.float32), gate), trt.bfloat16)
    residual = g.add(x, linear(g.reshape(gated, (batch, value_width)), "linear_attn.out_proj"))
    n = g.norm(residual, weights["post_attention_layernorm.weight"])
    gate, up = linear(n, "mlp.gate_proj"), linear(n, "mlp.up_proj")
    mixed = g.mul(graph_ops.add_activation(g.net, gate, "silu", BF16), up)
    hidden = g.add(residual, linear(mixed, "mlp.down_proj"))
    return {"hidden_out": hidden, "conv_out": conv_out, "state_out": state_out}
