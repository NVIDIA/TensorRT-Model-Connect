# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned strongly typed bidirectional text encoder topology."""

import numpy as np


def _constant(network, values, dtype=None):
    value = np.ascontiguousarray(values)
    tensor = network.add_constant(value.shape, value).get_output(0)
    return _cast(network, tensor, dtype) if dtype is not None else tensor


def _cast(network, tensor, dtype):
    return tensor if tensor.dtype == dtype else network.add_cast(tensor, dtype).get_output(0)


def _reshape(network, tensor, shape, transpose=None):
    layer = network.add_shuffle(tensor)
    layer.reshape_dims = shape
    if transpose is not None:
        layer.second_transpose = transpose
    return layer.get_output(0)


def _elementwise(network, left, right, op):
    return network.add_elementwise(left, right, op).get_output(0)


def _scalar(network, value, tensor):
    return _constant(
        network, np.full((1,) * len(tensor.shape), value, dtype=np.float32), tensor.dtype
    )


def _linear(network, tensor, weight):
    import tensorrt as trt

    constant = _constant(network, weight, tensor.dtype)
    return network.add_matrix_multiply(
        tensor, trt.MatrixOperation.NONE, constant, trt.MatrixOperation.TRANSPOSE
    ).get_output(0)


def _rms(network, tensor, gamma, eps):
    import tensorrt as trt

    work = _cast(network, tensor, trt.float32)
    square = _elementwise(network, work, work, trt.ElementWiseOperation.PROD)
    mean = network.add_reduce(
        square, trt.ReduceOperation.AVG, 1 << (len(tensor.shape) - 1), True
    ).get_output(0)
    mean = _elementwise(network, mean, _scalar(network, eps, mean), trt.ElementWiseOperation.SUM)
    inverse = _elementwise(
        network, mean, _scalar(network, -0.5, mean), trt.ElementWiseOperation.POW
    )
    result = _elementwise(network, work, inverse, trt.ElementWiseOperation.PROD)
    if gamma is not None:
        shape = (1,) * (len(tensor.shape) - 1) + (gamma.size,)
        result = _elementwise(
            network, result, _constant(network, gamma.reshape(shape)), trt.ElementWiseOperation.PROD
        )
    return _cast(network, result, tensor.dtype)


def _gelu(network, tensor):
    import tensorrt as trt

    op = trt.ElementWiseOperation
    square = _elementwise(network, tensor, tensor, op.PROD)
    cube = _elementwise(network, square, tensor, op.PROD)
    cubic = _elementwise(network, cube, _scalar(network, 0.044715, cube), op.PROD)
    total = _elementwise(network, tensor, cubic, op.SUM)
    total = _elementwise(network, total, _scalar(network, np.sqrt(2 / np.pi), total), op.PROD)
    total = network.add_activation(total, trt.ActivationType.TANH).get_output(0)
    total = _elementwise(network, total, _scalar(network, 1, total), op.SUM)
    total = _elementwise(network, total, tensor, op.PROD)
    return _elementwise(network, total, _scalar(network, 0.5, total), op.PROD)


def local_mask(network, positions, radius):
    import tensorrt as trt

    row = _reshape(network, positions, (1, -1, 1))
    column = _reshape(network, positions, (1, 1, -1))
    distance = _elementwise(network, row, column, trt.ElementWiseOperation.SUB)
    distance = network.add_unary(distance, trt.UnaryOperation.ABS).get_output(0)
    allowed = _elementwise(
        network,
        distance,
        _constant(network, np.array([[[radius + 1]]], dtype=np.int32)),
        trt.ElementWiseOperation.LESS,
    )
    zero = _constant(network, np.array([[[0]]], dtype=np.float32))
    blocked = _constant(network, np.array([[[-np.inf]]], dtype=np.float32))
    mask = network.add_select(allowed, zero, blocked).get_output(0)
    return mask


def _rope(network, tensor, positions, dim, theta, capacity):
    import tensorrt as trt

    frequency = theta ** (-np.arange(0, dim, 2, dtype=np.float32) / dim)
    angles = np.arange(capacity, dtype=np.float32)[:, None] * frequency
    angles = np.concatenate((angles, angles), axis=-1)
    cosine = network.add_gather(
        _constant(network, np.cos(angles), tensor.dtype), positions, 0
    ).get_output(0)
    sine = network.add_gather(
        _constant(network, np.sin(angles), tensor.dtype), positions, 0
    ).get_output(0)
    cosine = _reshape(network, cosine, (1, -1, dim))
    sine = _reshape(network, sine, (1, -1, dim))
    index = np.concatenate((np.arange(dim // 2, dim), np.arange(dim // 2))).astype(np.int32)
    rotated = network.add_gather(tensor, _constant(network, index), 2).get_output(0)
    signs = (
        np.concatenate((-np.ones(dim // 2), np.ones(dim // 2)))
        .astype(np.float32)
        .reshape(1, 1, dim)
    )
    rotated = _elementwise(
        network, rotated, _constant(network, signs, tensor.dtype), trt.ElementWiseOperation.PROD
    )
    left = _elementwise(network, tensor, cosine, trt.ElementWiseOperation.PROD)
    right = _elementwise(network, rotated, sine, trt.ElementWiseOperation.PROD)
    return _elementwise(network, left, right, trt.ElementWiseOperation.SUM)


def _attention(network, tensor, positions, config, weights, index, capacity):
    import tensorrt as trt

    prefix = f"layers.{index}.self_attn."
    dim, kv_heads, theta, radius = config.layer(index)
    heads = config.num_attention_heads
    qkv = []
    for name, count in (("q", heads), ("k", kv_heads), ("v", kv_heads)):
        projected = _linear(network, tensor, weights[prefix + name + "_proj.weight"])
        projected = _reshape(network, projected, (-1, count, dim), (1, 0, 2))
        gamma = weights[prefix + name + "_norm.weight"] if name != "v" else None
        projected = _rms(network, projected, gamma, config.rms_norm_eps)
        if name != "v":
            projected = _rope(network, projected, positions, dim, theta, capacity)
        if count != heads:
            groups = _constant(network, (np.arange(heads) // (heads // count)).astype(np.int32))
            projected = network.add_gather(projected, groups, 0).get_output(0)
        qkv.append(projected)
    query, key, value = qkv
    scores = network.add_matrix_multiply(
        query, trt.MatrixOperation.NONE, key, trt.MatrixOperation.TRANSPOSE
    ).get_output(0)
    scores = _cast(network, scores, trt.float32)
    if radius is not None:
        scores = _elementwise(
            network, scores, local_mask(network, positions, radius), trt.ElementWiseOperation.SUM
        )
    softmax = network.add_softmax(scores)
    softmax.axes = 4
    probabilities = _cast(network, softmax.get_output(0), tensor.dtype)
    context = network.add_matrix_multiply(
        probabilities, trt.MatrixOperation.NONE, value, trt.MatrixOperation.NONE
    ).get_output(0)
    context = _reshape(network, context, (0, 0, dim), (1, 0, 2))
    context = _reshape(network, context, (-1, heads * dim))
    return _linear(network, context, weights[prefix + "o_proj.weight"])


def encode(network, token_ids, positions, config, weights, dtype, capacity):
    import tensorrt as trt

    op = trt.ElementWiseOperation
    table = _constant(network, weights["embed_tokens.weight"], dtype)
    hidden = network.add_gather(table, token_ids, 0).get_output(0)
    hidden = _elementwise(
        network, hidden, _scalar(network, np.sqrt(config.hidden_size), hidden), op.PROD
    )
    ple = _linear(network, hidden, weights["ple.per_layer_model_projection.weight"])
    ple = _elementwise(network, ple, _scalar(network, config.hidden_size**-0.5, ple), op.PROD)
    ple = _reshape(network, ple, (-1, config.num_hidden_layers, config.hidden_size_per_layer_input))
    ple = _rms(network, ple, weights["ple.per_layer_projection_norm.weight"], config.rms_norm_eps)
    for index in range(config.num_hidden_layers):
        prefix = f"layers.{index}."

        def norm(tensor, name):
            return _rms(network, tensor, weights[prefix + name + ".weight"], config.rms_norm_eps)

        attended = _attention(
            network, norm(hidden, "input_layernorm"), positions, config, weights, index, capacity
        )
        hidden = _elementwise(network, hidden, norm(attended, "post_attention_layernorm"), op.SUM)
        normalized = norm(hidden, "pre_feedforward_layernorm")
        gate = _gelu(
            network, _linear(network, normalized, weights[prefix + "mlp.gate_proj.weight"])
        )
        up = _linear(network, normalized, weights[prefix + "mlp.up_proj.weight"])
        feedforward = _linear(
            network,
            _elementwise(network, gate, up, op.PROD),
            weights[prefix + "mlp.down_proj.weight"],
        )
        hidden = _elementwise(
            network, hidden, norm(feedforward, "post_feedforward_layernorm"), op.SUM
        )
        signal = network.add_gather(
            ple, _constant(network, np.array([index], dtype=np.int32)), 1
        ).get_output(0)
        signal = _reshape(network, signal, (-1, config.hidden_size_per_layer_input))
        gate = _gelu(
            network,
            _linear(network, hidden, weights[prefix + "ple_block.per_layer_input_gate.weight"]),
        )
        mixed = _linear(
            network,
            _elementwise(network, gate, signal, op.PROD),
            weights[prefix + "ple_block.per_layer_projection.weight"],
        )
        hidden = _elementwise(
            network, hidden, norm(mixed, "ple_block.post_per_layer_input_norm"), op.SUM
        )
        scalar = _constant(network, weights[prefix + "layer_scalar"].reshape(1, 1), hidden.dtype)
        hidden = _elementwise(network, hidden, scalar, op.PROD)
    hidden = _rms(network, hidden, weights["norm.weight"], config.rms_norm_eps)
    return _cast(
        network, _linear(network, hidden, weights["embedding_projection.weight"]), trt.float32
    )
