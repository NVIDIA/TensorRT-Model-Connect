# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse-Glimmer native target graph; TensorRT owns all GPU execution lowering."""
from __future__ import annotations

import math
import os

import numpy as np
import tensorrt as trt

from .config import TextConfig
from .quantization import constant, linear, weight_tensor

# Prefill chunk bounds of the dual-profile target engine.
PREFILL_OPT_TOKENS = 64
PREFILL_MAX_TOKENS = 128


class Graph:
    """A step of S query rows, including explicit compact GQA caches and draft features.

    Row r of a step is the token at ``position_id[r]``; its K/V are written to
    that cache slot before attention, and ``attention_mask[r]`` selects the
    cache slots it may read. Logits are produced for the last row only.
    """

    def __init__(self, network, checkpoint, config: TextConfig, capacity: int):
        self.network = network
        self.checkpoint = checkpoint
        self.config = config
        self.capacity = capacity
        self.keep_alive = []

    def const(self, value, dtype=np.float32):
        return constant(self.network, np.asarray(value, dtype=dtype), self.keep_alive)

    def cast(self, x, dtype):
        return x if x.dtype == dtype else self.network.add_cast(x, dtype).get_output(0)

    def reshape(self, x, shape, transpose=None, *, first_transpose=None):
        layer = self.network.add_shuffle(x)
        if first_transpose is not None:
            layer.first_transpose = first_transpose
        layer.reshape_dims = shape
        if transpose is not None:
            layer.second_transpose = transpose
        return layer.get_output(0)

    def ew(self, a, b, op):
        return self.network.add_elementwise(a, b, op).get_output(0)

    def scale(self, x, value):
        scale = self.cast(self.const(np.full((1,) * len(x.shape), value)), x.dtype)
        return self.ew(x, scale, trt.ElementWiseOperation.PROD)

    def norm(self, x, eps, centered_weight=None):
        dtype = x.dtype
        value = self.cast(x, trt.float32)
        square = self.ew(value, value, trt.ElementWiseOperation.PROD)
        variance = self.network.add_reduce(square, trt.ReduceOperation.AVG,
                                          1 << (len(value.shape) - 1), True).get_output(0)
        variance = self.ew(variance, self.const(np.full((1,) * len(value.shape), eps)),
                           trt.ElementWiseOperation.SUM)
        root = self.network.add_unary(variance, trt.UnaryOperation.SQRT).get_output(0)
        inverse = self.network.add_unary(root, trt.UnaryOperation.RECIP).get_output(0)
        result = self.ew(value, inverse, trt.ElementWiseOperation.PROD)
        if centered_weight is not None:
            gain = self.const((1.0 + centered_weight).reshape((1,) * (len(x.shape) - 1) + (-1,)))
            result = self.ew(result, gain, trt.ElementWiseOperation.PROD)
        return self.cast(result, dtype)

    def project(self, x, name, output_width):
        p = self.checkpoint.projection(name, output_width, int(x.shape[-1]))
        return linear(self.network, x, p, self.keep_alive)

    def rotary(self, x, position, heads, theta):
        d = self.config.head_dim
        value = self.reshape(x, (-1, heads, d))
        table = np.arange(self.capacity, dtype=np.float32)[:, None] * (
            1.0 / theta ** (np.arange(0, d, 2, dtype=np.float32) / d)
        )[None, :]
        cos = self.network.add_gather(self.const(np.concatenate((np.cos(table), np.cos(table)), -1)),
                                      position, 0).get_output(0)
        sin = self.network.add_gather(self.const(np.concatenate((np.sin(table), np.sin(table)), -1)),
                                      position, 0).get_output(0)
        cos = self.cast(self.reshape(cos, (-1, 1, d)), x.dtype)
        sin = self.cast(self.reshape(sin, (-1, 1, d)), x.dtype)
        # rotate_half([a, b]) = [-b, a]: swap the halves, then negate the first.
        swap = np.concatenate((np.arange(d // 2, d), np.arange(d // 2))).astype(np.int32)
        swapped = self.network.add_gather(value, self.const(swap, np.int32), 2).get_output(0)
        sign = np.concatenate((-np.ones(d // 2), np.ones(d // 2))).reshape(1, 1, d)
        rotated = self.ew(swapped, self.cast(self.const(sign), x.dtype), trt.ElementWiseOperation.PROD)
        result = self.ew(self.ew(value, cos, trt.ElementWiseOperation.PROD),
                         self.ew(rotated, sin, trt.ElementWiseOperation.PROD),
                         trt.ElementWiseOperation.SUM)
        return self.reshape(result, (-1, heads * d))

    def attention(self, x, position, mask, layer):
        c = self.config
        prefix = f"layers.{layer}.self_attn"
        q = self.project(x, prefix + ".q_proj", c.num_attention_heads * c.head_dim)
        k = self.project(x, prefix + ".k_proj", c.num_key_value_heads * c.head_dim)
        v = self.project(x, prefix + ".v_proj", c.num_key_value_heads * c.head_dim)
        q = self.reshape(self.norm(self.reshape(q, (-1, c.num_attention_heads, c.head_dim)),
                                   c.rms_norm_eps), (-1, c.num_attention_heads * c.head_dim))
        k = self.reshape(self.norm(self.reshape(k, (-1, c.num_key_value_heads, c.head_dim)),
                                   c.rms_norm_eps), (-1, c.num_key_value_heads * c.head_dim))
        if c.layer_types[layer] == "sliding_attention":
            q = self.rotary(q, position, c.num_attention_heads, c.layer_rope_theta[layer])
            k = self.rotary(k, position, c.num_key_value_heads, c.layer_rope_theta[layer])
            # The public runtime provides causal masking; sliding history is a
            # separate mathematical constraint, independent of GPU architecture.
            indices = self.const(np.arange(self.capacity).reshape(1, -1), np.int32)
            distance = self.ew(self.reshape(position, (-1, 1)), indices, trt.ElementWiseOperation.SUB)
            in_window = self.ew(distance, self.const([[c.sliding_window]], np.int32),
                                trt.ElementWiseOperation.LESS)
            mask = self.network.add_select(in_window, mask,
                                            self.const(np.full((1, self.capacity), -np.inf))).get_output(0)
        caches = []
        kv_width = c.num_key_value_heads * c.head_dim
        for stem, current in (("k", k), ("v", v)):
            cache = self.network.add_input(f"cache_{stem}_{layer}", x.dtype, (self.capacity, kv_width))
            updated = self.network.add_scatter(cache, self.reshape(position, (-1, 1)), current,
                                               trt.ScatterMode.ND).get_output(0)
            current.name = f"present_{stem}_{layer}"
            self.network.mark_output(current)
            heads = self.reshape(updated, (self.capacity, c.num_key_value_heads, c.head_dim), (1, 0, 2))
            gather = self.const(np.arange(c.num_attention_heads) //
                                (c.num_attention_heads // c.num_key_value_heads), np.int32)
            caches.append(self.network.add_gather(heads, gather, 0).get_output(0))
        q = self.cast(self.reshape(q, (-1, c.num_attention_heads, c.head_dim), (1, 0, 2)), trt.float32)
        score = self.network.add_matrix_multiply(q, trt.MatrixOperation.NONE,
                                                 self.cast(caches[0], trt.float32),
                                                 trt.MatrixOperation.TRANSPOSE).get_output(0)
        score = self.scale(score, c.qk_scale_factor / math.sqrt(c.head_dim))
        score = self.ew(score, self.reshape(mask, (1, -1, self.capacity)), trt.ElementWiseOperation.SUM)
        probability = self.network.add_softmax(score)
        probability.axes = 1 << 2
        context = self.network.add_matrix_multiply(probability.get_output(0), trt.MatrixOperation.NONE,
                                                   self.cast(caches[1], trt.float32),
                                                   trt.MatrixOperation.NONE).get_output(0)
        context = self.cast(self.reshape(context, (-1, c.num_attention_heads * c.head_dim),
                                         first_transpose=(1, 0, 2)), x.dtype)
        gate = self.project(x, prefix + ".gate_proj", c.num_attention_heads * c.head_dim)
        gate = self.network.add_activation(gate, trt.ActivationType.SIGMOID).get_output(0)
        return self.project(self.ew(context, gate, trt.ElementWiseOperation.PROD),
                            prefix + ".o_proj", c.hidden_size)

    def last_row(self, x):
        """Slice ``[rows, width]`` to its last row."""
        one_row = self.const(np.array([1, int(x.shape[1])]), np.int64)
        shape = self.network.add_shape(x).get_output(0)
        start = self.ew(shape, one_row, trt.ElementWiseOperation.SUB)
        layer = self.network.add_slice(x, (0, 0), (1, 1), (1, 1))
        layer.set_input(1, start)
        layer.set_input(2, one_row)
        return layer.get_output(0)

    def populate(self, target_layer_ids=()):
        c = self.config
        if len(set(target_layer_ids)) != len(target_layer_ids) or any(i < 0 or i >= c.num_hidden_layers for i in target_layer_ids):
            raise ValueError("Muse draft feature layers must be unique target layer indices")
        token = self.network.add_input("token_id", trt.int32, (-1,))
        position = self.network.add_input("position_id", trt.int32, (-1,))
        mask = self.network.add_input("attention_mask", trt.float32, (-1, self.capacity))
        embedding = self.checkpoint.projection("embed_tokens", c.vocab_size, c.hidden_size)
        embedding = weight_tensor(self.network, embedding, self.keep_alive, trt.float16)
        x = self.network.add_gather(embedding, token, 0).get_output(0)
        x = self.norm(x, c.rms_norm_eps)
        hidden = {}
        for i in range(c.num_hidden_layers):
            prefix = f"layers.{i}"
            normalized = self.norm(x, c.rms_norm_eps,
                                   self.checkpoint.vector(prefix + ".input_layernorm.weight", c.hidden_size))
            attention = self.attention(normalized, position, mask, i)
            attention = self.norm(attention, c.post_norm_eps,
                                  self.checkpoint.vector(prefix + ".post_attention_layernorm.weight", c.hidden_size))
            x = self.ew(x, attention, trt.ElementWiseOperation.SUM)
            normalized = self.norm(x, c.rms_norm_eps,
                                   self.checkpoint.vector(prefix + ".pre_feedforward_layernorm.weight", c.hidden_size))
            gate = self.project(normalized, prefix + ".mlp.gate_proj", c.intermediate_size)
            up = self.project(normalized, prefix + ".mlp.up_proj", c.intermediate_size)
            sigmoid = self.network.add_activation(gate, trt.ActivationType.SIGMOID).get_output(0)
            activated = self.ew(self.ew(gate, sigmoid, trt.ElementWiseOperation.PROD), up,
                                trt.ElementWiseOperation.PROD)
            down = self.project(activated, prefix + ".mlp.down_proj", c.hidden_size)
            down = self.norm(down, c.post_norm_eps,
                             self.checkpoint.vector(prefix + ".post_feedforward_layernorm.weight", c.hidden_size))
            x = self.ew(x, down, trt.ElementWiseOperation.SUM)
            if i in target_layer_ids:
                hidden[i] = x
        if target_layer_ids:
            features = self.network.add_concatenation([hidden[i] for i in target_layer_ids])
            features.axis = 1
            output = features.get_output(0)
            output.name = "target_features"
            self.network.mark_output(output)
        x = self.last_row(x)
        x = self.norm(x, c.rms_norm_eps, self.checkpoint.vector("norm.weight", c.hidden_size))
        logits = self.cast(self.project(x, "lm_head", c.vocab_size), trt.float32)
        logits = self.scale(logits, c.output_multiplier / c.final_logit_softcapping)
        logits = self.network.add_activation(logits, trt.ActivationType.TANH).get_output(0)
        logits = self.scale(logits, c.final_logit_softcapping)
        logits.name = "logits"
        self.network.mark_output(logits)


def build_target(config, checkpoint, capacity, *, target_layer_ids=(), verbose=False):
    if type(capacity) is not int or capacity < 1 or capacity > config.max_position_embeddings:
        raise ValueError("Muse native capacity must fit the checkpoint context")
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    builder.max_threads = min(8, os.cpu_count() or 1)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    graph = Graph(network, checkpoint, config, capacity)
    graph.populate(target_layer_ids)
    settings = builder.create_builder_config()
    settings.builder_optimization_level = 1
    # Profile 0 prefills 1..max_rows prompt rows; profile 1 decodes one row.
    max_rows = min(PREFILL_MAX_TOKENS, capacity)
    for low, opt, high in ((1, min(PREFILL_OPT_TOKENS, max_rows), max_rows), (1, 1, 1)):
        profile = builder.create_optimization_profile()
        for name in ("token_id", "position_id"):
            profile.set_shape(name, (low,), (opt,), (high,))
        profile.set_shape("attention_mask", (low, capacity), (opt, capacity), (high, capacity))
        settings.add_optimization_profile(profile)
    if "NVFP4" in checkpoint.formats.values():
        settings.max_aux_streams = 0
    engine = builder.build_serialized_network(network, settings)
    if engine is None:
        raise RuntimeError("TensorRT failed to build the Muse native target; see builder diagnostics")
    return bytes(engine)
