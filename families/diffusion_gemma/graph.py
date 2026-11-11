# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DiffusionGemma's tied encoder/denoiser graphs, using TensorRT operators."""

from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np
import tensorrt as trt


class Graph:
    """One strongly typed network and the weight buffers backing its constants."""

    def __init__(self, network, config, weights, precision):
        self.net, self.config, self.weights = network, config, weights
        self.dtype = {"bf16": trt.bfloat16, "fp16": trt.float16, "fp32": trt.float32}[precision]
        self.buffers = []
        self.constants = {}

    def cast(self, value, dtype):
        return value if value.dtype == dtype else self.net.add_cast(value, dtype).get_output(0)

    def const(self, values, dtype=None):
        values = np.ascontiguousarray(values)
        self.buffers.append(values)
        if values.dtype.name == "bfloat16":
            weights = trt.Weights(trt.bfloat16, int(values.ctypes.data), values.size)
        else:
            weights = trt.Weights(values)
        result = self.net.add_constant(values.shape, weights).get_output(0)
        return self.cast(result, dtype) if dtype is not None else result

    def weight(self, name, dtype=None):
        dtype = self.dtype if dtype is None else dtype
        key = (name, str(dtype))
        if key not in self.constants:
            self.constants[key] = self.const(self.weights[name], dtype)
        return self.constants[key]

    def reshape(self, value, shape):
        layer = self.net.add_shuffle(value)
        layer.reshape_dims = shape
        return layer.get_output(0)

    def transpose(self, value, order):
        layer = self.net.add_shuffle(value)
        layer.second_transpose = order
        return layer.get_output(0)

    def unsqueeze_front(self, value):
        shape = self.net.add_shape(value).get_output(0)
        one = self.const(np.ones(1, np.int64 if shape.dtype == trt.int64 else np.int32))
        dimensions = self.net.add_concatenation([one, shape])
        dimensions.axis = 0
        layer = self.net.add_shuffle(value)
        layer.set_input(1, dimensions.get_output(0))
        return layer.get_output(0)

    def op(self, left, right, operation):
        return self.net.add_elementwise(left, right, operation).get_output(0)

    def scalar(self, value, scalar):
        factor = self.const(np.full((1,) * len(value.shape), scalar, np.float32), value.dtype)
        return self.op(value, factor, trt.ElementWiseOperation.PROD)

    def python_scalar(self, value, scalar):
        # PyTorch's wrapped Python scalar is evaluated in the activation's
        # opmath dtype. Rounding it to BF16 first changes the router scores.
        dtype = value.dtype
        return self.cast(self.scalar(self.cast(value, trt.float32), scalar), dtype)

    def linear(self, value, name):
        weight = self.weight(name + ".weight", value.dtype)
        for _ in range(len(value.shape) - 2):
            weight = self.unsqueeze_front(weight)
        return self.net.add_matrix_multiply(value, trt.MatrixOperation.NONE,
                                             weight, trt.MatrixOperation.TRANSPOSE).get_output(0)

    def norm(self, value, name=None):
        # The upstream module explicitly computes the reduction and learned
        # scale in FP32, uses pow(-.5), then returns the activation dtype.
        original = value.dtype
        value = self.cast(value, trt.float32)
        rank = len(value.shape)
        square = self.op(value, value, trt.ElementWiseOperation.PROD)
        mean = self.net.add_reduce(square, trt.ReduceOperation.AVG,
                                    1 << (rank - 1), True).get_output(0)
        epsilon = self.const(np.full((1,) * rank, self.config["rms_norm_eps"], np.float32))
        power = self.const(np.full((1,) * rank, -0.5, np.float32))
        inverse = self.op(self.op(mean, epsilon, trt.ElementWiseOperation.SUM), power,
                           trt.ElementWiseOperation.POW)
        result = self.op(value, inverse, trt.ElementWiseOperation.PROD)
        if name is not None:
            scale = self.reshape(self.weight(name + ".weight", trt.float32),
                                 (1,) * (rank - 1) + (-1,))
            result = self.op(result, scale, trt.ElementWiseOperation.PROD)
        return self.cast(result, original)

    def gelu(self, value):
        original = value.dtype
        value = self.cast(value, trt.float32)
        square = self.op(value, value, trt.ElementWiseOperation.PROD)
        cube = self.op(square, value, trt.ElementWiseOperation.PROD)
        inner = self.op(value, self.scalar(cube, 0.044715), trt.ElementWiseOperation.SUM)
        inner = self.scalar(inner, math.sqrt(2.0 / math.pi))
        tanh = self.net.add_activation(inner, trt.ActivationType.TANH).get_output(0)
        one = self.const(np.ones((1,) * len(value.shape), np.float32))
        result = self.op(self.scalar(value, 0.5),
                          self.op(one, tanh, trt.ElementWiseOperation.SUM),
                          trt.ElementWiseOperation.PROD)
        return self.cast(result, original)

    def mlp(self, value, prefix):
        gate = self.gelu(self.linear(value, prefix + ".gate_proj"))
        up = self.linear(value, prefix + ".up_proj")
        return self.linear(self.op(gate, up, trt.ElementWiseOperation.PROD),
                           prefix + ".down_proj")

    def softmax(self, value, axis):
        layer = self.net.add_softmax(value)
        layer.axes = 1 << axis
        return layer.get_output(0)


def rope_tables(config: Mapping, layer_index: int, capacity: int):
    """Full-width rotate-half tables, including proportional RoPE zero angles."""
    kind = config["layer_types"][layer_index]
    width = config["global_head_dim"] if kind == "full_attention" else config["head_dim"]
    parameters = config["rope_parameters"][kind]
    fraction = float(parameters.get("partial_rotary_factor", 1.0))
    count = int(width * fraction) // 2
    inv = np.zeros(width // 2, np.float32)
    # Proportional RoPE retains the full head width in the frequency exponent;
    # the unrotated frequencies are represented by zeros, not cropped pairs.
    denominator = width if parameters["rope_type"] == "proportional" else int(width * fraction)
    inv[:count] = 1.0 / (float(parameters["rope_theta"]) **
                         (np.arange(count, dtype=np.float32) * 2 / denominator))
    positions = np.arange(capacity, dtype=np.float32)[:, None] * inv[None, :]
    return np.concatenate([np.cos(positions)] * 2, axis=-1), np.concatenate([np.sin(positions)] * 2, axis=-1)


def _rope(graph, value, position_ids, tables):
    n = graph.net
    cosine = n.add_gather(graph.const(tables[0], graph.dtype), position_ids, 0).get_output(0)
    sine = n.add_gather(graph.const(tables[1], graph.dtype), position_ids, 0).get_output(0)
    width = value.shape[-1]
    first = n.add_gather(value, graph.const(np.arange(width // 2, dtype=np.int32)), 2).get_output(0)
    second = n.add_gather(value, graph.const(np.arange(width // 2, width, dtype=np.int32)), 2).get_output(0)
    halves = n.add_concatenation([graph.scalar(second, -1.0), first])
    halves.axis = 2
    cosine = graph.reshape(cosine, (0, 1, width))
    sine = graph.reshape(sine, (0, 1, width))
    # Eager BF16 RoPE rounds both products before their sum. Explicit casts
    # preserve those boundaries when TensorRT contracts multiply/add chains.
    left = graph.op(graph.cast(value, trt.float32), graph.cast(cosine, trt.float32),
                    trt.ElementWiseOperation.PROD)
    right = graph.op(graph.cast(halves.get_output(0), trt.float32),
                     graph.cast(sine, trt.float32), trt.ElementWiseOperation.PROD)
    left, right = graph.cast(left, graph.dtype), graph.cast(right, graph.dtype)
    return graph.cast(graph.op(graph.cast(left, trt.float32), graph.cast(right, trt.float32),
                               trt.ElementWiseOperation.SUM), graph.dtype)


def _qkv(graph, hidden, positions, index, capacity):
    config = graph.config
    full = config["layer_types"][index] == "full_attention"
    width = config["global_head_dim"] if full else config["head_dim"]
    kv_heads = config["num_global_key_value_heads"] if full else config["num_key_value_heads"]
    heads = config["num_attention_heads"]
    shape = (0, heads, width)
    query = graph.norm(graph.reshape(graph.linear(hidden, "self_attn.q_proj"), shape), "self_attn.q_norm")
    key = graph.reshape(graph.linear(hidden, "self_attn.k_proj"), (0, kv_heads, width))
    value = key if full else graph.reshape(graph.linear(hidden, "self_attn.v_proj"), (0, kv_heads, width))
    key = graph.norm(key, "self_attn.k_norm")
    value = graph.norm(value)
    tables = rope_tables(config, index, capacity)
    query = graph.transpose(_rope(graph, query, positions, tables), (1, 0, 2))
    key = graph.transpose(_rope(graph, key, positions, tables), (1, 0, 2))
    value = graph.transpose(value, (1, 0, 2))
    return query, key, value


def _attention(graph, query, key, value, mask, past_key, past_value, index):
    n, config = graph.net, graph.config
    full = config["layer_types"][index] == "full_attention"
    width = config["global_head_dim"] if full else config["head_dim"]
    kv_heads = config["num_global_key_value_heads"] if full else config["num_key_value_heads"]
    heads = config["num_attention_heads"]
    key_join = n.add_concatenation([past_key, key])
    key_join.axis = 1
    value_join = n.add_concatenation([past_value, value])
    value_join.axis = 1
    key, value = key_join.get_output(0), value_join.get_output(0)
    # Cache tensors stay compact. Only attention expands grouped K/V heads.
    repeat = graph.const(np.repeat(np.arange(kv_heads, dtype=np.int32), heads // kv_heads))
    expanded_key = n.add_gather(key, repeat, 0).get_output(0)
    expanded_value = n.add_gather(value, repeat, 0).get_output(0)
    scores = n.add_matrix_multiply(query, trt.MatrixOperation.NONE,
                                    expanded_key, trt.MatrixOperation.TRANSPOSE).get_output(0)
    # DiffusionGemma uses attention scale 1.0, rather than 1/sqrt(head_dim).
    scores = graph.op(graph.cast(scores, trt.float32), graph.unsqueeze_front(mask),
                      trt.ElementWiseOperation.SUM)
    probabilities = graph.cast(graph.softmax(scores, 2), graph.dtype)
    attended = n.add_matrix_multiply(probabilities, trt.MatrixOperation.NONE,
                                      expanded_value, trt.MatrixOperation.NONE).get_output(0)
    attended = graph.reshape(graph.transpose(attended, (1, 0, 2)), (-1, heads * width))
    return graph.linear(attended, "self_attn.o_proj"), key, value


def _experts(graph, residual):
    n, config = graph.net, graph.config
    hidden, selected = config["hidden_size"], config["top_k_experts"]
    width = config["moe_intermediate_size"]
    router = graph.norm(residual)
    router = graph.op(router, graph.reshape(graph.weight("router.scale"), (1, hidden)),
                      trt.ElementWiseOperation.PROD)
    router = graph.python_scalar(router, hidden ** -0.5)
    scores = graph.cast(graph.linear(router, "router.proj"), trt.float32)
    probabilities = graph.softmax(scores, 1)
    top = n.add_topk(probabilities, trt.TopKOperation.MAX, selected, 1 << 1)
    coefficients, indices = top.get_output(0), top.get_output(1)
    total = n.add_reduce(coefficients, trt.ReduceOperation.SUM, 1 << 1, True).get_output(0)
    coefficients = graph.op(coefficients, total, trt.ElementWiseOperation.DIV)
    scales = n.add_gather(graph.weight("router.per_expert_scale", trt.float32), indices, 0).get_output(0)
    coefficients = graph.op(coefficients, scales, trt.ElementWiseOperation.PROD)
    # The pinned official model defaults to grouped experts: routing weights
    # multiply BF16 expert outputs in FP32, then their sum is cast to BF16 once.
    # Keep the router's original top-k order, as restored by that implementation.
    normalized = graph.norm(residual, "pre_feedforward_layernorm_2")
    column = graph.reshape(normalized, (0, hidden, 1))
    # The checkpoint stores concatenated gate/up rows. These are independent
    # linear maps over the same expert input, with identical model semantics.
    gate_up = graph.weights["experts.gate_up_proj"]
    gate_weight = graph.const(gate_up[:, :width, :], graph.dtype)
    up_weight = graph.const(gate_up[:, width:, :], graph.dtype)
    down = graph.weight("experts.down_proj")
    loop = n.add_loop()
    trips = graph.reshape(graph.const(np.array([selected], np.int32)), ())
    loop.add_trip_limit(trips, trt.TripLimit.COUNT)
    # Each iterator emits one selected expert across the tokens. Accumulate
    # its FP32 weighted output without intermediate BF16 rounding; TensorRT
    # owns loop lowering and the actual buffer allocations.
    expert = loop.add_iterator(indices, 1, False).get_output(0)
    coefficient = loop.add_iterator(coefficients, 1, False).get_output(0)
    coefficient = graph.reshape(coefficient, (-1, 1))
    matrix = n.add_gather(gate_weight, expert, 0).get_output(0)
    gate = n.add_matrix_multiply(matrix, trt.MatrixOperation.NONE,
                                  column, trt.MatrixOperation.NONE).get_output(0)
    gate = graph.reshape(gate, (0, width))
    matrix = n.add_gather(up_weight, expert, 0).get_output(0)
    up = n.add_matrix_multiply(matrix, trt.MatrixOperation.NONE,
                                column, trt.MatrixOperation.NONE).get_output(0)
    up = graph.reshape(up, (0, width))
    product = graph.op(graph.gelu(gate), up, trt.ElementWiseOperation.PROD)
    matrix = n.add_gather(down, expert, 0).get_output(0)
    output = n.add_matrix_multiply(matrix, trt.MatrixOperation.NONE,
                                   graph.reshape(product, (0, width, 1)),
                                   trt.MatrixOperation.NONE).get_output(0)
    output = graph.reshape(output, (0, hidden))
    weighted = graph.op(graph.cast(output, trt.float32), coefficient,
                         trt.ElementWiseOperation.PROD)
    accumulation = loop.add_recurrence(graph.scalar(graph.cast(normalized, trt.float32), 0.0))
    updated = graph.op(accumulation.get_output(0), weighted, trt.ElementWiseOperation.SUM)
    accumulation.set_input(1, updated)
    # TensorRT LAST_VALUE returns X_n for the recurrence sequence
    # X_0=zero, X_(i+1)=X_i+expert_i: all selected experts are included.
    result = loop.add_loop_output(accumulation.get_output(0),
                                  trt.LoopOutput.LAST_VALUE, 0).get_output(0)
    return graph.norm(graph.cast(result, graph.dtype), "post_feedforward_layernorm_2")


def _create(precision, verbose):
    if precision not in {"bf16", "fp16", "fp32"}:
        raise ValueError("diffusion_gemma graph precision must be bf16, fp16, or fp32")
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    if precision == "fp32":
        # An FP32 diagnostic must preserve its declared mantissa precision.
        config.clear_flag(trt.BuilderFlag.TF32)
    # The exact checkpoint's 1024-token global layer requires about 98 GB
    # of execution memory. A 64-GiB workspace ceiling rejected its tactic.
    # This is an upper limit; TensorRT chooses the actual scratch allocation.
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 128 << 30)
    return logger, builder, network, config


def _serialize(builder, network, config, profile, buffers):
    config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("diffusion_gemma TensorRT engine build failed")
    return bytes(plan)


def _mark(graph, value, name):
    value.name = name
    graph.net.mark_output(value)


def build_qkv_engine(text_config, weights, layer_index, *, precision="bf16", max_sequence_length=1024, verbose=False):
    """Materialize normalized, rotated Q/K/V at the activation dtype boundary."""
    logger, builder, network, config = _create(precision, verbose)
    graph = Graph(network, text_config, weights, precision)
    hidden = text_config["hidden_size"]
    value = network.add_input("hidden_states", graph.dtype, (-1, hidden))
    positions = network.add_input("position_ids", trt.int32, (-1,))
    query, key, cache_value = _qkv(graph, graph.norm(value, "input_layernorm"), positions,
                                  layer_index, max_sequence_length)
    # These used bindings preserve eager's post-RoPE BF16 query rounding before
    # the following attention matmul, including when its producer is fused.
    _mark(graph, query, "query")
    _mark(graph, key, "key")
    _mark(graph, cache_value, "value")
    profile = builder.create_optimization_profile()
    optimum = min(256, max_sequence_length)
    profile.set_shape("hidden_states", (1, hidden), (optimum, hidden), (max_sequence_length, hidden))
    profile.set_shape("position_ids", (1,), (optimum,), (max_sequence_length,))
    return _serialize(builder, network, config, profile, graph.buffers)


def build_layer_engine(text_config, weights, layer_index, *, precision="bf16", max_sequence_length=1024, verbose=False):
    """Consume Q/K/V for attention and feedforward; caller owns cache commits."""
    logger, builder, network, config = _create(precision, verbose)
    graph = Graph(network, text_config, weights, precision)
    hidden = text_config["hidden_size"]
    full = text_config["layer_types"][layer_index] == "full_attention"
    heads = text_config["num_global_key_value_heads"] if full else text_config["num_key_value_heads"]
    width = text_config["global_head_dim"] if full else text_config["head_dim"]
    value = network.add_input("hidden_states", graph.dtype, (-1, hidden))
    query = network.add_input("query", graph.dtype, (text_config["num_attention_heads"], -1, width))
    key = network.add_input("key", graph.dtype, (heads, -1, width))
    cache_value = network.add_input("value", graph.dtype, (heads, -1, width))
    mask = network.add_input("attention_mask", trt.float32, (-1, -1))
    past_key = network.add_input("past_key", graph.dtype, (heads, -1, width))
    past_value = network.add_input("past_value", graph.dtype, (heads, -1, width))
    attention, key, cache_value = _attention(graph, query, key, cache_value, mask,
                                             past_key, past_value, layer_index)
    residual = graph.op(value, graph.norm(attention, "post_attention_layernorm"), trt.ElementWiseOperation.SUM)
    shared = graph.mlp(graph.norm(residual, "pre_feedforward_layernorm"), "mlp")
    shared = graph.norm(shared, "post_feedforward_layernorm_1")
    routed = _experts(graph, residual)
    combined = graph.op(shared, routed, trt.ElementWiseOperation.SUM)
    combined = graph.norm(combined, "post_feedforward_layernorm")
    output = graph.op(residual, combined, trt.ElementWiseOperation.SUM)
    output = graph.op(output, graph.reshape(graph.weight("layer_scalar"), (1, 1)), trt.ElementWiseOperation.PROD)
    _mark(graph, output, "hidden_states_out")
    _mark(graph, key, "present_key")
    _mark(graph, cache_value, "present_value")
    profile = builder.create_optimization_profile()
    optimum = min(256, max_sequence_length)
    profile.set_shape("hidden_states", (1, hidden), (optimum, hidden), (max_sequence_length, hidden))
    profile.set_shape("query", (text_config["num_attention_heads"], 1, width),
                      (text_config["num_attention_heads"], optimum, width),
                      (text_config["num_attention_heads"], max_sequence_length, width))
    for name in ("key", "value"):
        profile.set_shape(name, (heads, 1, width), (heads, optimum, width), (heads, max_sequence_length, width))
    profile.set_shape("attention_mask", (1, 1), (optimum, optimum + 1), (max_sequence_length, 2 * max_sequence_length))
    for name in ("past_key", "past_value"):
        profile.set_shape(name, (heads, 0, width), (heads, 1, width), (heads, max_sequence_length, width))
    return _serialize(builder, network, config, profile, graph.buffers)


def build_embedding_engine(text_config, weights, *, precision="bf16", max_sequence_length=1024, verbose=False):
    logger, builder, network, config = _create(precision, verbose)
    graph = Graph(network, text_config, weights, precision)
    hidden = text_config["hidden_size"]
    ids = network.add_input("input_ids", trt.int32, (-1,))
    signal = network.add_input("self_conditioning", graph.dtype, (-1, hidden))
    embedding = network.add_gather(graph.weight("embed_tokens.weight"), ids, 0).get_output(0)
    embedding = graph.scalar(embedding, hidden ** 0.5)
    projected = graph.mlp(graph.norm(signal, "self_conditioning.pre_norm"), "self_conditioning")
    conditioned = graph.norm(graph.op(embedding, projected, trt.ElementWiseOperation.SUM))
    _mark(graph, embedding, "encoder_embeddings")
    _mark(graph, conditioned, "decoder_embeddings")
    profile = builder.create_optimization_profile()
    optimum = min(256, max_sequence_length)
    profile.set_shape("input_ids", (1,), (optimum,), (max_sequence_length,))
    profile.set_shape("self_conditioning", (1, hidden), (optimum, hidden), (max_sequence_length, hidden))
    return _serialize(builder, network, config, profile, graph.buffers)


def build_output_engine(text_config, weights, *, precision="bf16", max_sequence_length=1024, verbose=False):
    logger, builder, network, config = _create(precision, verbose)
    graph = Graph(network, text_config, weights, precision)
    hidden = text_config["hidden_size"]
    value = network.add_input("hidden_states", graph.dtype, (-1, hidden))
    temperature = network.add_input("temperature", trt.float32, (1, 1))
    normalized = graph.norm(value, "norm")
    embedding = graph.weight("embed_tokens.weight")
    logits = network.add_matrix_multiply(normalized, trt.MatrixOperation.NONE,
                                         embedding, trt.MatrixOperation.TRANSPOSE).get_output(0)
    logits = graph.cast(logits, trt.float32)
    bound = text_config["final_logit_softcapping"]
    logits = graph.scalar(logits, 1.0 / bound)
    logits = network.add_activation(logits, trt.ActivationType.TANH).get_output(0)
    logits = graph.scalar(logits, bound)
    processed = graph.op(logits, temperature, trt.ElementWiseOperation.DIV)
    # generate() saves processed logits in the embedding dtype before the
    # next decoder converts them back to FP32 for softmax.
    probabilities = graph.softmax(graph.cast(graph.cast(processed, graph.dtype), trt.float32), 1)
    probabilities = graph.cast(probabilities, graph.dtype)
    signal = network.add_matrix_multiply(probabilities, trt.MatrixOperation.NONE,
                                         embedding, trt.MatrixOperation.NONE).get_output(0)
    signal = graph.scalar(signal, hidden ** 0.5)
    _mark(graph, logits, "raw_logits")
    _mark(graph, signal, "self_conditioning")
    profile = builder.create_optimization_profile()
    optimum = min(256, max_sequence_length)
    profile.set_shape("hidden_states", (1, hidden), (optimum, hidden), (max_sequence_length, hidden))
    return _serialize(builder, network, config, profile, graph.buffers)
