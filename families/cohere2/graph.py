# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cohere2 parallel LayerNorm decoder with GPT-J RoPE on local-attention layers.

The bounded full-prefix engine recomputes attention for each generated token.
It has no external KV cache. All floating-point operations and weights are FP32.
"""

import math

import numpy as np
import tensorrt as trt

from .config import Config


class Graph:
    def __init__(self, network):
        self.n = network
        self.arrays = []
        self.constants = {}

    def constant(self, value):
        value = np.ascontiguousarray(value)
        key = (value.ctypes.data, value.shape, value.dtype.str)
        if key not in self.constants:
            self.arrays.append(value)
            self.constants[key] = self.n.add_constant(value.shape, value).get_output(0)
        return self.constants[key]

    def scalar(self, value, rank):
        return self.constant(np.full((1,) * rank, value, np.float32))

    def op(self, a, b, op):
        return self.n.add_elementwise(a, b, op).get_output(0)

    def add(self, a, b):
        return self.op(a, b, trt.ElementWiseOperation.SUM)

    def mul(self, a, b):
        return self.op(a, b, trt.ElementWiseOperation.PROD)

    def reshape(self, x, shape):
        layer = self.n.add_shuffle(x)
        layer.reshape_dims = shape
        return layer.get_output(0)

    def matmul(self, x, w):
        return self.n.add_matrix_multiply(
            x, trt.MatrixOperation.NONE, w, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)

    def linear(self, x, weight):
        return self.matmul(x, self.constant(weight))

    def norm(self, x, weight, eps):
        mean = self.n.add_reduce(x, trt.ReduceOperation.AVG, 2, True).get_output(0)
        centered = self.op(x, mean, trt.ElementWiseOperation.SUB)
        var = self.n.add_reduce(
            self.mul(centered, centered), trt.ReduceOperation.AVG, 2, True
        ).get_output(0)
        root = self.n.add_unary(
            self.add(var, self.scalar(eps, 2)), trt.UnaryOperation.SQRT
        ).get_output(0)
        normalized = self.op(centered, root, trt.ElementWiseOperation.DIV)
        return self.mul(normalized, self.constant(weight.reshape(1, -1)))

    def heads(self, x, heads, size):
        layer = self.n.add_shuffle(x)
        layer.reshape_dims = (1, -1, heads, size)
        layer.second_transpose = (0, 2, 1, 3)
        return layer.get_output(0)

    def rotary(self, x, cos, sin, length, heads, dim):
        pairs = self.reshape(x, (1, heads, length, dim // 2, 2))
        swapped = self.n.add_gather(pairs, self.constant(np.array([1, 0], np.int32)), 4).get_output(
            0
        )
        rotated = self.mul(
            swapped, self.constant(np.array([-1, 1], np.float32).reshape(1, 1, 1, 1, 2))
        )
        rotated = self.reshape(rotated, (1, heads, length, dim))
        return self.add(self.mul(x, cos), self.mul(rotated, sin))


def build_engine(config: Config, weights: dict, length: int, verbose=False) -> bytes:
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    g = Graph(network)
    tokens = network.add_input("token_ids", trt.int32, (length,))
    last = network.add_input("last_position", trt.int32, (1,))
    embedding = g.constant(weights["model.embed_tokens.weight"])
    x = network.add_gather(embedding, tokens, 0).get_output(0)
    dim, heads, kv = config.head_dim, config.num_attention_heads, config.num_key_value_heads
    inv_freq = 1.0 / (
        np.float32(config.rope_theta) ** (np.arange(0, dim, 2, dtype=np.float32) / dim)
    )
    angles = np.arange(length, dtype=np.float32)[:, None] * inv_freq[None, :]
    angles = np.repeat(angles, 2, axis=-1)
    cos = g.constant(np.cos(angles).reshape(1, 1, length, dim))
    sin = g.constant(np.sin(angles).reshape(1, 1, length, dim))
    rows, cols = np.arange(length)[:, None], np.arange(length)[None, :]
    head_map = g.constant(np.repeat(np.arange(kv, dtype=np.int32), heads // kv))
    for i, layer_type in enumerate(config.layer_types):
        base = f"model.layers.{i}"
        norm = g.norm(x, weights[f"{base}.input_layernorm.weight"], config.layer_norm_eps)
        q = g.heads(g.linear(norm, weights[f"{base}.self_attn.q_proj.weight"]), heads, dim)
        k = g.heads(g.linear(norm, weights[f"{base}.self_attn.k_proj.weight"]), kv, dim)
        v = g.heads(g.linear(norm, weights[f"{base}.self_attn.v_proj.weight"]), kv, dim)
        if layer_type == "sliding_attention":
            q = g.rotary(q, cos, sin, length, heads, dim)
            k = g.rotary(k, cos, sin, length, kv, dim)
        k = network.add_gather(k, head_map, 1).get_output(0)
        v = network.add_gather(v, head_map, 1).get_output(0)
        scores = g.mul(g.matmul(q, k), g.scalar(1 / math.sqrt(dim), 4))
        allowed = cols <= rows
        if layer_type == "sliding_attention":
            allowed = allowed & (cols > rows - config.sliding_window)
        mask = g.constant(
            np.where(allowed, 0, np.finfo(np.float32).min)
            .astype(np.float32)
            .reshape(1, 1, length, length)
        )
        softmax = network.add_softmax(g.add(scores, mask))
        softmax.axes = 8
        context = network.add_matrix_multiply(
            softmax.get_output(0), trt.MatrixOperation.NONE, v, trt.MatrixOperation.NONE
        ).get_output(0)
        merged = network.add_shuffle(context)
        merged.first_transpose = (0, 2, 1, 3)
        merged.reshape_dims = (length, config.hidden_size)
        attn = g.linear(merged.get_output(0), weights[f"{base}.self_attn.o_proj.weight"])
        gate = g.linear(norm, weights[f"{base}.mlp.gate_proj.weight"])
        silu = g.mul(gate, network.add_activation(gate, trt.ActivationType.SIGMOID).get_output(0))
        mlp = g.linear(
            g.mul(silu, g.linear(norm, weights[f"{base}.mlp.up_proj.weight"])),
            weights[f"{base}.mlp.down_proj.weight"],
        )
        x = g.add(g.add(x, attn), mlp)
    x = g.norm(x, weights["model.norm.weight"], config.layer_norm_eps)
    x = network.add_gather(x, last, 0).get_output(0)
    logits = g.mul(g.linear(x, weights["lm_head.weight"]), g.scalar(config.logit_scale, 2))
    logits.name = "logits"
    network.mark_output(logits)
    settings = builder.create_builder_config()
    settings.builder_optimization_level = 3
    settings.clear_flag(trt.BuilderFlag.TF32)
    plan = builder.build_serialized_network(network, settings)
    if plan is None:
        raise RuntimeError("cohere2 TensorRT engine build failed")
    return bytes(plan)
