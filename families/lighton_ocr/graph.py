# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strongly typed TensorRT operators and the LightOn Qwen3 decoder graph."""

from __future__ import annotations

import numpy as np
import tensorrt as trt


class Graph:
    """Family-owned mathematical operators and constant-buffer lifetimes."""

    def __init__(self, network, weights):
        self.net, self.weights = network, weights
        self.buffers, self.constants = [], {}

    def const(self, values):
        shape = np.asarray(values).shape
        values = np.ascontiguousarray(values)
        self.buffers.append(values)
        return self.net.add_constant(shape, trt.Weights(values)).get_output(0)

    def weight(self, name):
        if name not in self.constants:
            self.constants[name] = self.const(np.asarray(self.weights[name], dtype=np.float32))
        return self.constants[name]

    def op(self, left, right, kind):
        return self.net.add_elementwise(left, right, kind).get_output(0)

    def scalar(self, value, factor):
        return self.op(value, self.const(np.full((1,) * len(value.shape), factor, np.float32)),
                       trt.ElementWiseOperation.PROD)

    def add(self, left, right):
        return self.op(left, right, trt.ElementWiseOperation.SUM)

    def reshape(self, value, shape):
        layer = self.net.add_shuffle(value)
        layer.reshape_dims = shape
        return layer.get_output(0)

    def transpose(self, value, order):
        layer = self.net.add_shuffle(value)
        layer.second_transpose = order
        return layer.get_output(0)

    def gather(self, value, indices, axis=0):
        if isinstance(indices, (int, np.ndarray)):
            indices = self.const(np.asarray(indices, dtype=np.int32))
        return self.net.add_gather(value, indices, axis).get_output(0)

    def dimension(self, value, axis):
        shape = self.net.add_shape(value).get_output(0)
        return self.gather(shape, np.array([axis], np.int32))

    def shape_constant(self, values, dtype):
        return self.const(np.asarray(values, dtype=np.int64 if dtype == trt.int64 else np.int32))

    def shape_vector(self, values):
        dtype = next(value.dtype for value in values if hasattr(value, "dtype") and not isinstance(value, np.ndarray))
        tensors = [value if hasattr(value, "dtype") and not isinstance(value, np.ndarray)
                   else self.shape_constant([value], dtype) for value in values]
        layer = self.net.add_concatenation(tensors)
        layer.axis = 0
        return layer.get_output(0)

    def reshape_dynamic(self, value, dimensions):
        layer = self.net.add_shuffle(value)
        layer.zero_is_placeholder = False
        layer.set_input(1, self.shape_vector(dimensions))
        return layer.get_output(0)

    def unsqueeze_front(self, value):
        return self.reshape_dynamic(value, [1] + [self.dimension(value, axis) for axis in range(len(value.shape))])

    def linear(self, value, name):
        weight = self.weight(name + ".weight")
        for _ in range(len(value.shape) - 2):
            weight = self.unsqueeze_front(weight)
        return self.net.add_matrix_multiply(value, trt.MatrixOperation.NONE,
                                             weight, trt.MatrixOperation.TRANSPOSE).get_output(0)

    def norm(self, value, name, epsilon):
        rank = len(value.shape)
        square = self.op(value, value, trt.ElementWiseOperation.PROD)
        mean = self.net.add_reduce(square, trt.ReduceOperation.AVG, 1 << (rank - 1), True).get_output(0)
        shifted = self.add(mean, self.const(np.full((1,) * rank, epsilon, np.float32)))
        root = self.net.add_unary(shifted, trt.UnaryOperation.SQRT).get_output(0)
        inverse = self.net.add_unary(root, trt.UnaryOperation.RECIP).get_output(0)
        normalized = self.op(value, inverse, trt.ElementWiseOperation.PROD)
        gamma = self.reshape(self.weight(name + ".weight"), (1,) * (rank - 1) + (-1,))
        return self.op(normalized, gamma, trt.ElementWiseOperation.PROD)

    def silu(self, value):
        gate = self.net.add_activation(value, trt.ActivationType.SIGMOID).get_output(0)
        return self.op(value, gate, trt.ElementWiseOperation.PROD)

    def gelu(self, value):
        scaled = self.scalar(value, 2.0 ** -0.5)
        erf = self.net.add_unary(scaled, trt.UnaryOperation.ERF).get_output(0)
        offset = self.add(erf, self.const(np.ones((1,) * len(value.shape), np.float32)))
        return self.op(self.scalar(value, 0.5), offset, trt.ElementWiseOperation.PROD)

    def mlp(self, value, prefix):
        gate = self.silu(self.linear(value, prefix + ".gate_proj"))
        up = self.linear(value, prefix + ".up_proj")
        return self.linear(self.op(gate, up, trt.ElementWiseOperation.PROD), prefix + ".down_proj")

    def rope(self, value, positions, tables):
        width = value.shape[-1]
        cosine = self.reshape(self.gather(tables[0], positions), (0, 1, width))
        sine = self.reshape(self.gather(tables[1], positions), (0, 1, width))
        first = self.gather(value, np.arange(width // 2, dtype=np.int32), 2)
        second = self.gather(value, np.arange(width // 2, width, dtype=np.int32), 2)
        join = self.net.add_concatenation([self.scalar(second, -1), first])
        join.axis = 2
        return self.add(self.op(value, cosine, trt.ElementWiseOperation.PROD),
                        self.op(join.get_output(0), sine, trt.ElementWiseOperation.PROD))

    def attention(self, query, key, value, scale, mask=None):
        scores = self.net.add_matrix_multiply(query, trt.MatrixOperation.NONE,
                                              key, trt.MatrixOperation.TRANSPOSE).get_output(0)
        scores = self.scalar(scores, scale)
        if mask is not None:
            scores = self.add(scores, self.unsqueeze_front(mask))
        softmax = self.net.add_softmax(scores)
        softmax.axes = 1 << 2
        return self.net.add_matrix_multiply(softmax.get_output(0), trt.MatrixOperation.NONE,
                                             value, trt.MatrixOperation.NONE).get_output(0)

    def mark(self, value, name):
        value.name = name
        self.net.mark_output(value)


def create_network(verbose=False):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    settings = builder.create_builder_config()
    settings.builder_optimization_level = 3
    settings.clear_flag(trt.BuilderFlag.TF32)
    settings.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 32 << 30)
    return logger, builder, network, settings


def serialize(builder, network, settings, profile, buffers):
    """Keep caller-owned constants alive until TensorRT serializes the graph."""
    settings.add_optimization_profile(profile)
    result = builder.build_serialized_network(network, settings)
    if result is None:
        raise RuntimeError("lighton_ocr TensorRT engine build failed")
    return bytes(result)


def decoder_rope_tables(config, capacity):
    width = config["head_dim"]
    base = config.get("rope_parameters", {}).get("rope_theta", config.get("rope_theta", 1000000))
    inverse = 1.0 / (float(base) ** (np.arange(0, width, 2, dtype=np.float32) / width))
    angles = np.arange(capacity, dtype=np.float32)[:, None] * inverse[None]
    angles = np.concatenate([angles, angles], axis=1)
    return np.cos(angles), np.sin(angles)


def _decoder_layer(graph, hidden, positions, mask, past_key, past_value, config, index, tables):
    prefix = f"layers.{index}"
    epsilon = config["rms_norm_eps"]
    heads, kv_heads, width = config["num_attention_heads"], config["num_key_value_heads"], config["head_dim"]
    normalized = graph.norm(hidden, prefix + ".input_layernorm", epsilon)
    query = graph.reshape(graph.linear(normalized, prefix + ".self_attn.q_proj"), (0, heads, width))
    key = graph.reshape(graph.linear(normalized, prefix + ".self_attn.k_proj"), (0, kv_heads, width))
    value = graph.reshape(graph.linear(normalized, prefix + ".self_attn.v_proj"), (0, kv_heads, width))
    query = graph.norm(query, prefix + ".self_attn.q_norm", epsilon)
    key = graph.norm(key, prefix + ".self_attn.k_norm", epsilon)
    query = graph.transpose(graph.rope(query, positions, tables), (1, 0, 2))
    key = graph.transpose(graph.rope(key, positions, tables), (1, 0, 2))
    value = graph.transpose(value, (1, 0, 2))
    joined = []
    for past, new in ((past_key, key), (past_value, value)):
        layer = graph.net.add_concatenation([graph.gather(past, index), new])
        layer.axis = 1
        joined.append(layer.get_output(0))
    key, value = joined
    repeat = np.repeat(np.arange(kv_heads, dtype=np.int32), heads // kv_heads)
    expanded_key, expanded_value = graph.gather(key, repeat), graph.gather(value, repeat)
    attended = graph.attention(query, expanded_key, expanded_value, width ** -0.5, mask)
    attended = graph.reshape(graph.transpose(attended, (1, 0, 2)), (-1, heads * width))
    hidden = graph.add(hidden, graph.linear(attended, prefix + ".self_attn.o_proj"))
    normalized = graph.norm(hidden, prefix + ".post_attention_layernorm", epsilon)
    return graph.add(hidden, graph.mlp(normalized, prefix + ".mlp")), key, value


def build_decoder_engine(config, weights, capacity=4096, *, verbose=False):
    """All decoder layers with compact, explicit request-owned K/V tensors."""
    text = config.text_config
    logger, builder, network, settings = create_network(verbose)
    graph = Graph(network, weights)
    hidden = network.add_input("inputs_embeds", trt.float32, (-1, text["hidden_size"]))
    positions = network.add_input("position_ids", trt.int32, (-1,))
    mask = network.add_input("attention_mask", trt.float32, (-1, -1))
    cache_shape = (text["num_hidden_layers"], text["num_key_value_heads"], -1, text["head_dim"])
    past_key = network.add_input("past_key", trt.float32, cache_shape)
    past_value = network.add_input("past_value", trt.float32, cache_shape)
    tables = tuple(graph.const(value) for value in decoder_rope_tables(text, capacity))
    keys, values = [], []
    for index in range(text["num_hidden_layers"]):
        hidden, key, value = _decoder_layer(graph, hidden, positions, mask, past_key, past_value, text, index, tables)
        keys.append(graph.unsqueeze_front(key))
        values.append(graph.unsqueeze_front(value))
    hidden = graph.norm(hidden, "norm", text["rms_norm_eps"])
    sequence = graph.dimension(hidden, 0)
    last = graph.op(sequence, graph.shape_constant([1], sequence.dtype), trt.ElementWiseOperation.SUB)
    logits = graph.linear(graph.gather(hidden, last), "lm_head")
    graph.mark(logits, "logits")
    for tensors, name in ((keys, "present_key"), (values, "present_value")):
        layer = network.add_concatenation(tensors)
        layer.axis = 0
        graph.mark(layer.get_output(0), name)
    profile = builder.create_optimization_profile()
    optimum = min(256, capacity)
    profile.set_shape("inputs_embeds", (1, text["hidden_size"]), (optimum, text["hidden_size"]),
                      (capacity, text["hidden_size"]))
    profile.set_shape("position_ids", (1,), (optimum,), (capacity,))
    profile.set_shape("attention_mask", (1, 1), (optimum, optimum + 1), (capacity, 2 * capacity - 1))
    for name in ("past_key", "past_value"):
        first = (text["num_hidden_layers"], text["num_key_value_heads"])
        profile.set_shape(name, first + (0, text["head_dim"]), first + (1, text["head_dim"]),
                          first + (capacity - 1, text["head_dim"]))
    return serialize(builder, network, settings, profile, graph.buffers)
