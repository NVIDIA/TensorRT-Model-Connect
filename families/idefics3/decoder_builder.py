# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native causal Llama decoder with Idefics3 image-row substitution."""

import numpy as np
import tensorrt as trt


def build_decoder(weights, cfg, length, verbose=False):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    options = builder.create_builder_config()
    options.builder_optimization_level = 3
    options.clear_flag(trt.BuilderFlag.TF32)
    width = cfg["hidden_size"]
    heads = cfg["num_attention_heads"]
    kv_heads = cfg["num_key_value_heads"]
    dim = cfg["head_dim"]

    def const(value, shape=None):
        value = np.ascontiguousarray(value, dtype=np.float32)
        return net.add_constant(tuple(value.shape) if shape is None else shape, value).get_output(0)

    def op(left, right, kind):
        return net.add_elementwise(left, right, kind).get_output(0)

    def add(left, right):
        return op(left, right, trt.ElementWiseOperation.SUM)

    def mul(left, right):
        return op(left, right, trt.ElementWiseOperation.PROD)

    def linear(x, key):
        value = weights[key + ".weight"]
        return net.add_matrix_multiply(
            x, trt.MatrixOperation.NONE, const(value), trt.MatrixOperation.TRANSPOSE
        ).get_output(0)

    def norm(x, key):
        square = mul(x, x)
        mean = net.add_reduce(square, trt.ReduceOperation.AVG, 2, True).get_output(0)
        rsqrt = net.add_unary(
            add(mean, const(np.array([[cfg["rms_norm_eps"]]]))), trt.UnaryOperation.SQRT
        ).get_output(0)
        return mul(
            op(x, rsqrt, trt.ElementWiseOperation.DIV), const(weights[key + ".weight"], (1, width))
        )

    frequencies = 1 / (cfg["rope_theta"] ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    angles = np.arange(length, dtype=np.float32)[:, None] * frequencies
    angles = np.concatenate([angles, angles], axis=-1)
    cosine = const(np.cos(angles), (1, 1, length, dim))
    sine = const(np.sin(angles), (1, 1, length, dim))

    def split(x, nheads):
        layer = net.add_shuffle(x)
        layer.reshape_dims = (1, length, nheads, dim)
        layer.second_transpose = (0, 2, 1, 3)
        return layer.get_output(0)

    def rope(x, nheads):
        first = net.add_slice(
            x, (0, 0, 0, 0), (1, nheads, length, dim // 2), (1, 1, 1, 1)
        ).get_output(0)
        second = net.add_slice(
            x, (0, 0, 0, dim // 2), (1, nheads, length, dim // 2), (1, 1, 1, 1)
        ).get_output(0)
        second = net.add_unary(second, trt.UnaryOperation.NEG).get_output(0)
        concat = net.add_concatenation([second, first])
        concat.axis = 3
        return add(mul(x, cosine), mul(concat.get_output(0), sine))

    ids = net.add_input("input_ids", trt.int32, (length,))
    valid = net.add_input("attention_mask", trt.int32, (length,))
    hidden = net.add_gather(
        const(weights["model.text_model.embed_tokens.weight"]), ids, 0
    ).get_output(0)
    image_rows = net.add_input("image_rows", trt.float32, (length, width))
    image_mask = net.add_input("image_mask", trt.float32, (length, 1))
    hidden = add(
        mul(hidden, op(const(np.ones((1, 1))), image_mask, trt.ElementWiseOperation.SUB)),
        image_rows,
    )
    valid_float = net.add_cast(valid, trt.float32).get_output(0)
    mask = mul(
        op(const(np.ones(1)), valid_float, trt.ElementWiseOperation.SUB), const(np.array([-1e9]))
    )
    reshape = net.add_shuffle(mask)
    reshape.reshape_dims = (1, 1, 1, 1, length)
    mask = reshape.get_output(0)
    mask = add(
        mask,
        const(
            np.triu(np.full((length, length), -1e9, dtype=np.float32), k=1),
            (1, 1, 1, length, length),
        ),
    )

    def grouped(x, groups):
        layer = net.add_shuffle(x)
        layer.reshape_dims = (1, kv_heads, groups, length, dim)
        return layer.get_output(0)

    for index in range(cfg["num_hidden_layers"]):
        prefix = f"model.text_model.layers.{index}"
        normalized = norm(hidden, prefix + ".input_layernorm")
        query = rope(split(linear(normalized, prefix + ".self_attn.q_proj"), heads), heads)
        key = rope(split(linear(normalized, prefix + ".self_attn.k_proj"), kv_heads), kv_heads)
        value = split(linear(normalized, prefix + ".self_attn.v_proj"), kv_heads)
        query = grouped(query, heads // kv_heads)
        key = grouped(key, 1)
        value = grouped(value, 1)
        scores = net.add_matrix_multiply(
            query, trt.MatrixOperation.NONE, key, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
        scores = add(mul(scores, const(np.array([1 / np.sqrt(dim)]), (1, 1, 1, 1, 1))), mask)
        probability = net.add_softmax(scores)
        probability.axes = 16
        attended = net.add_matrix_multiply(
            probability.get_output(0), trt.MatrixOperation.NONE, value, trt.MatrixOperation.NONE
        ).get_output(0)
        reshape = net.add_shuffle(attended)
        reshape.first_transpose = (0, 3, 1, 2, 4)
        reshape.reshape_dims = (length, heads * dim)
        hidden = add(hidden, linear(reshape.get_output(0), prefix + ".self_attn.o_proj"))
        normalized = norm(hidden, prefix + ".post_attention_layernorm")
        gate = linear(normalized, prefix + ".mlp.gate_proj")
        gate = mul(gate, net.add_activation(gate, trt.ActivationType.SIGMOID).get_output(0))
        hidden = add(
            hidden,
            linear(
                mul(gate, linear(normalized, prefix + ".mlp.up_proj")), prefix + ".mlp.down_proj"
            ),
        )
    hidden = norm(hidden, "model.text_model.norm")
    last = net.add_input("last_index", trt.int32, (1,))
    selected = net.add_gather(hidden, last, 0).get_output(0)
    output = linear(selected, "lm_head")
    output.name = "logits"
    net.mark_output(output)
    plan = builder.build_serialized_network(net, options)
    if plan is None:
        raise RuntimeError("Idefics3 decoder engine build failed")
    return bytes(plan)
