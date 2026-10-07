# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Full-sequence Qwen backbone math for Clef, with no generation or KV cache."""

import numpy as np
import tensorrt as trt

from .graph import Graph


def delta_rule(g, q, k, v, decay, beta, heads, dim):
    """64-token chunked gated delta rule using native TensorRT matrix/loop ops.

    Invert the unit lower-triangular interaction by recursive 2x2 blocks.
    A polynomial in powers of A is algebraically equivalent but numerically
    unstable for nearly parallel keys: large alternating powers cancel and can
    overflow the recurrent state. Block substitution keeps each inverse local.
    Inter-chunk recurrent state and all chunk algebra use FP32, as upstream.
    """
    chunk = 64

    def l2(x):
        # The unmodified torch fallback performs these operations in BF16.
        square = g.mul(x, x)
        total = g.cast(g.reduce(g.cast(square, trt.float32)), x.dtype)
        inverse = g.cast(
            g.unary(g.sqrt(g.cast(g.add(total, 1e-6), trt.float32)), trt.UnaryOperation.RECIP),
            x.dtype,
        )
        return g.cast(g.mul(x, inverse), trt.float32)

    q, k = l2(q), l2(k)
    if hasattr(g, "trace"):
        g.trace.update(q_normalized=q, k_normalized=k, decay=decay, beta=beta)
    v, beta = g.cast(v, trt.float32), g.cast(beta, trt.float32)
    q = g.mul(q, dim**-0.5)
    sequence = g.n.add_gather(
        g.cast(g.n.add_shape(q).get_output(0), trt.int32), g.const(0, trt.int32), 0
    ).get_output(0)
    padded = g.mul(
        g.binary(g.add(sequence, chunk - 1), chunk, trt.ElementWiseOperation.FLOOR_DIV), chunk
    )

    def pad_tokens(x):
        tail = tuple(x.shape)[1:]
        sizes = g.concat([g.reshape(padded, (1,)), g.const(tail, trt.int32)], 0)
        layer = g.n.add_slice(x, (0,) * len(x.shape), (1, *tail), (1,) * len(x.shape))
        layer.mode = trt.SampleMode.FILL
        layer.set_input(2, sizes)
        layer.set_input(4, g.const(0.0))
        return layer.get_output(0)

    # Upstream pads only the recurrent operation, after projections, convolution
    # and Q/K normalization. Dense layers and full attention see the real length.
    q, k, v, beta, decay = map(pad_tokens, (q, k, v, beta, decay))

    def chunks(x):
        return g.reshape(g.permute(x, (1, 0, 2)), (heads, -1, chunk, dim))

    q, k, v = map(chunks, (q, k, v))
    beta = g.reshape(g.permute(beta, (1, 0)), (heads, -1, chunk, 1))
    decay = g.reshape(g.permute(decay, (1, 0)), (heads, -1, chunk))
    cumulative = g.n.add_cumulative(
        decay, g.const(2, trt.int32), trt.CumulativeOperation.SUM, False, False
    ).get_output(0)
    cumulative = g.reshape(cumulative, (heads, -1, chunk, 1))
    # Clamp before exp: upper-triangular large positive differences are unused
    # and must not create inf * 0 NaNs.
    difference = g.sub(cumulative, g.permute(cumulative, (0, 1, 3, 2)))
    difference = g.binary(difference, 0.0, trt.ElementWiseOperation.MIN)
    lower = g.const(np.tril(np.ones((1, 1, chunk, chunk), np.float32)))
    strict = g.const(np.tril(np.ones((1, 1, chunk, chunk), np.float32), -1))
    mask = g.mul(g.exp(difference), lower)
    kb, vb = g.mul(k, beta), g.mul(v, beta)
    a = g.mul(g.mul(g.mm(kb, k, True), mask), g.mul(strict, -1))
    # If L = [[L11, 0], [-A21, L22]], then
    # L^-1 = [[L11^-1, 0], [L22^-1 A21 L11^-1, L22^-1]].
    # Pack all diagonal blocks into the batch axes so each level takes two
    # batched matrix products, rather than 63 serial token-row updates.
    inverse = g.reshape(g.add(g.mul(g.slice(a, 3, 0, 1), 0), 1), (heads, -1, chunk, 1, 1))
    flattened = g.reshape(a, (heads, -1, chunk * chunk))
    for size in (2, 4, 8, 16, 32, 64):
        half, groups = size // 2, chunk // size
        even = g.const(np.arange(0, groups * 2, 2), trt.int32)
        odd = g.const(np.arange(1, groups * 2, 2), trt.int32)
        left = g.n.add_gather(inverse, even, 2).get_output(0)
        right = g.n.add_gather(inverse, odd, 2).get_output(0)
        indices = np.array(
            [
                (group * size + half + row) * chunk + group * size + col
                for group in range(groups)
                for row in range(half)
                for col in range(half)
            ]
        )
        cross = g.n.add_gather(flattened, g.const(indices, trt.int32), 2).get_output(0)
        cross = g.reshape(cross, (heads, -1, groups, half, half))
        lower_left = g.mm(g.mm(right, cross), left)
        upper = g.concat([left, g.mul(cross, 0)], 4)
        lower = g.concat([lower_left, right], 4)
        inverse = g.concat([upper, lower], 3)
    inverse = g.reshape(inverse, (heads, -1, chunk, chunk))
    if hasattr(g, "trace"):
        g.trace.update(inverse=inverse)
    values = g.mm(inverse, vb)
    k_decay = g.mm(inverse, g.mul(kb, g.exp(cumulative)))
    within = g.mul(g.mm(q, k, True), mask)
    q_decay = g.mul(q, g.exp(cumulative))
    end = g.slice(cumulative, 2, chunk - 1, 1)
    k_tail = g.mul(k, g.exp(g.sub(end, cumulative)))
    # Shape tensor is the number of chunks, including harmless causal padding.
    count = g.n.add_gather(
        g.cast(g.n.add_shape(q).get_output(0), trt.int32), g.const(1, trt.int32), 0
    ).get_output(0)
    loop = g.n.add_loop()
    loop.add_trip_limit(count, trt.TripLimit.COUNT)
    recurrent = loop.add_recurrence(g.const(np.zeros((heads, dim, dim), np.float32)))
    state = recurrent.get_output(0)

    def at(x):
        return loop.add_iterator(x, 1, False).get_output(0)

    corrected = g.sub(at(values), g.mm(at(k_decay), state))
    output = g.add(g.mm(at(q_decay), state), g.mm(at(within), corrected))
    next_state = g.add(
        g.mul(state, g.exp(at(end))), g.mm(g.permute(at(k_tail), (0, 2, 1)), corrected)
    )
    recurrent.set_input(1, next_state)
    collected = loop.add_loop_output(output, trt.LoopOutput.CONCATENATE, 1)
    collected.set_input(1, count)
    result = g.reshape(collected.get_output(0), (heads, -1, dim))
    result = g.cast(g.permute(result, (1, 0, 2)), trt.bfloat16)
    trimmed = g.n.add_slice(result, (0, 0, 0), (1, heads, dim), (1, 1, 1))
    trimmed.set_input(2, g.concat([g.reshape(sequence, (1,)), g.const([heads, dim], trt.int32)], 0))
    return trimmed.get_output(0)


def deltanet(g, x, p, config):
    heads, key_heads = config["linear_num_value_heads"], config["linear_num_key_heads"]
    dim, vdim = config["linear_key_head_dim"], config["linear_value_head_dim"]
    if dim != vdim:
        raise ValueError("Clef requires matching key and value head dimensions")
    kdim, vwidth = key_heads * dim, heads * dim
    conv_width = 2 * kdim + vwidth
    qkv = g.linear(x, p + ".in_proj_qkv")
    # The reference's BF16 Conv1d accumulates in FP32 and rounds before SiLU.
    # Native BF16 depthwise convolution differed in 163981/2621440 captured
    # values; FP32 convolution followed by this explicit BF16 boundary matched
    # all of them, including the downstream SiLU values.
    conv_input = g.reshape(g.permute(g.cast(qkv, trt.float32), (1, 0)), (1, conv_width, -1, 1))
    weight = g.weights[p + ".conv1d.weight"]
    kernel = config["linear_conv_kernel_dim"]
    convolution = g.n.add_convolution_nd(
        conv_input, conv_width, (kernel, 1), trt.Weights(), trt.Weights()
    )
    convolution.set_input(1, g.const(weight.reshape(conv_width, 1, kernel, 1), trt.float32))
    convolution.num_groups = conv_width
    convolution.pre_padding = (kernel - 1, 0)
    convolution.post_padding = (0, 0)
    convolved = g.cast(convolution.get_output(0), trt.bfloat16)
    mixed = g.silu(g.permute(g.reshape(convolved, (conv_width, -1)), (1, 0)))
    if hasattr(g, "trace"):
        g.trace.update(qkv=qkv, mixed=mixed)
    q = g.reshape(g.slice(mixed, 1, 0, kdim), (-1, key_heads, dim))
    k = g.reshape(g.slice(mixed, 1, kdim, kdim), (-1, key_heads, dim))
    v = g.reshape(g.slice(mixed, 1, kdim * 2, vwidth), (-1, heads, dim))
    # Delta-rule state has one value head per recurrence; unlike full attention,
    # the update cannot use a compact shared key-head tensor.
    indices = g.const(np.repeat(np.arange(key_heads), heads // key_heads), trt.int32)
    q = g.n.add_gather(q, indices, 1).get_output(0)
    k = g.n.add_gather(k, indices, 1).get_output(0)
    beta = g.activation(g.linear(x, p + ".in_proj_b"), trt.ActivationType.SIGMOID)
    a = g.cast(g.linear(x, p + ".in_proj_a"), trt.float32)
    dt = g.const(g.weights[p + ".dt_bias"].float().reshape(1, heads))
    decay = g.mul(
        g.activation(g.add(a, dt), trt.ActivationType.SOFTPLUS),
        g.const(-g.weights[p + ".A_log"].float().exp().reshape(1, heads)),
    )
    out = delta_rule(g, q, k, v, decay, beta, heads, dim)
    if hasattr(g, "trace"):
        g.trace.update(delta=out)
    out = g.reshape(out, (-1, dim))
    f = g.cast(out, trt.float32)
    f = g.div(
        f, g.sqrt(g.add(g.reduce(g.mul(f, f), trt.ReduceOperation.AVG), config["rms_norm_eps"]))
    )
    normalized = g.mul(
        g.cast(f, trt.bfloat16),
        g.const(g.weights[p + ".norm.weight"].reshape(1, dim), trt.bfloat16),
    )
    z = g.reshape(g.linear(x, p + ".in_proj_z"), (-1, dim))
    gated = g.cast(
        g.mul(g.cast(normalized, trt.float32), g.silu(g.cast(z, trt.float32))), trt.bfloat16
    )
    if hasattr(g, "trace"):
        g.trace.update(gated=g.reshape(gated, (-1, vwidth)))
    return g.linear(g.reshape(gated, (-1, vwidth)), p + ".out_proj")


def full_attention(g, x, p, config, cos, sin):
    heads, kv_heads, dim = (
        config["num_attention_heads"],
        config["num_key_value_heads"],
        config["head_dim"],
    )
    projection = g.reshape(g.linear(x, p + ".q_proj"), (-1, heads, dim * 2))
    q = g.rms(g.slice(projection, 2, 0, dim), p + ".q_norm", config["rms_norm_eps"])
    gate = g.reshape(g.slice(projection, 2, dim, dim), (-1, heads * dim))
    k = g.rms(
        g.reshape(g.linear(x, p + ".k_proj"), (-1, kv_heads, dim)),
        p + ".k_norm",
        config["rms_norm_eps"],
    )
    v = g.permute(g.reshape(g.linear(x, p + ".v_proj"), (1, -1, kv_heads, dim)), (0, 2, 1, 3))
    rotary = int(dim * config["rope_parameters"]["partial_rotary_factor"])

    def rope(x):
        left, right = g.slice(x, 2, 0, rotary // 2), g.slice(x, 2, rotary // 2, rotary // 2)
        rot = g.concat([g.mul(right, -1), left])
        source = g.slice(x, 2, 0, rotary)
        out = g.add(g.mul(source, cos), g.mul(rot, sin))
        return g.concat([out, g.slice(x, 2, rotary, dim - rotary)])

    q = g.permute(g.reshape(rope(q), (1, -1, heads, dim)), (0, 2, 1, 3))
    k = g.permute(g.reshape(rope(k), (1, -1, kv_heads, dim)), (0, 2, 1, 3))
    layer = g.n.add_attention(g.mul(q, dim**-0.5), k, v, trt.AttentionNormalizationOp.SOFTMAX, True)
    layer.decomposable = True
    out = g.reshape(g.permute(layer.get_output(0), (0, 2, 1, 3)), (-1, heads * dim))
    # Transformers 5.10.2 uses sigmoid here, including when the release config
    # contains output_gate_type. Follow executed reference math, not that label.
    out = g.mul(out, g.activation(gate, trt.ActivationType.SIGMOID))
    return g.linear(out, p + ".o_proj")


def decoder_block(g, x, config, index, cos=None, sin=None):
    debug = hasattr(g, "trace")
    p = f"model.language_model.layers.{index}"
    n = g.rms(x, p + ".input_layernorm", config["rms_norm_eps"])
    if debug:
        g.trace["input_norm"] = n
    if config["layer_types"][index] == "linear_attention":
        mixed = deltanet(g, n, p + ".linear_attn", config)
    else:
        if cos is None or sin is None:
            raise ValueError("full attention requires rotary position inputs")
        mixed = full_attention(g, n, p + ".self_attn", config, cos, sin)
    x = g.add(x, mixed)
    if debug:
        g.trace.update(mixer=mixed, post_mixer=x)
    n = g.rms(x, p + ".post_attention_layernorm", config["rms_norm_eps"])
    if debug:
        g.trace["mlp_norm"] = n
    mlp = g.mul(g.silu(g.linear(n, p + ".mlp.gate_proj")), g.linear(n, p + ".mlp.up_proj"))
    x = g.add(x, g.linear(mlp, p + ".mlp.down_proj"))
    return x


def decoder_layer(network, weights, config, index, debug=False):
    g = Graph(network, weights)
    if debug:
        g.trace = {}
    x = network.add_input("hidden_states", trt.bfloat16, (-1, config["hidden_size"]))
    cos = sin = None
    if config["layer_types"][index] == "full_attention":
        rotary = int(config["head_dim"] * config["rope_parameters"]["partial_rotary_factor"])
        cos = network.add_input("rope_cos", trt.bfloat16, (-1, 1, rotary))
        sin = network.add_input("rope_sin", trt.bfloat16, (-1, 1, rotary))
    x = decoder_block(g, x, config, index, cos, sin)
    x.name = "output"
    network.mark_output(x)
    if debug:
        for name, tensor in g.trace.items():
            tensor = g.cast(tensor, trt.float32)
            tensor.name = name
            network.mark_output(tensor)
    return g
