# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Flint's mathematical graph. TensorRT owns GPU lowering and execution."""

import numpy as np
import tensorrt as trt


class Graph:
    def __init__(self, network, weights):
        self.n = network
        self.weights = weights
        self.storage = []

    def cast(self, x, dtype):
        return x if x.dtype == dtype else self.n.add_cast(x, dtype).get_output(0)

    def constant(self, data, dtype=trt.float32):
        data = np.ascontiguousarray(data, dtype=np.float32)
        self.storage.append(data)
        return self.cast(self.n.add_constant(data.shape, trt.Weights(data)).get_output(0), dtype)

    def array(self, name):
        _, _, _, entry = self.weights.describe(name)
        with self.weights.view(name) as raw:
            if entry["dtype"] == "BF16":
                result = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(
                    np.float32
                )
            else:
                result = np.frombuffer(raw, dtype=np.float32).copy()
        return result.reshape(entry["shape"])

    def weight(self, name, dtype=trt.bfloat16, shape=None):
        value = self.array(name)
        if shape is not None:
            value = value.reshape(shape)
        return self.constant(value, dtype)

    def reshape(self, x, shape):
        layer = self.n.add_shuffle(x)
        layer.reshape_dims = shape
        return layer.get_output(0)

    def op(self, a, b, op):
        return self.n.add_elementwise(a, b, op).get_output(0)

    def add(self, a, b):
        return self.op(a, b, trt.ElementWiseOperation.SUM)

    def mul(self, a, b):
        return self.op(a, b, trt.ElementWiseOperation.PROD)

    def mm(self, a, b):
        return self.n.add_matrix_multiply(
            a, trt.MatrixOperation.NONE, b, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)

    def linear(self, x, name):
        return self.mm(x, self.weight(name, x.dtype))

    def norm(self, x, name, eps):
        f = self.cast(x, trt.float32)
        rank = len(x.shape)
        square = self.mul(f, f)
        mean = self.n.add_reduce(square, trt.ReduceOperation.AVG, 1 << (rank - 1), True).get_output(
            0
        )
        mean = self.add(mean, self.constant(np.full((1,) * rank, eps)))
        root = self.n.add_unary(mean, trt.UnaryOperation.SQRT).get_output(0)
        inv = self.n.add_unary(root, trt.UnaryOperation.RECIP).get_output(0)
        scale = self.weight(name, trt.float32, (1,) * (rank - 1) + (x.shape[-1],))
        return self.cast(self.mul(self.mul(f, inv), scale), x.dtype)

    def swiglu(self, x, prefix):
        gate = self.cast(self.linear(x, prefix + ".gate_proj.weight"), trt.float32)
        up = self.cast(self.linear(x, prefix + ".up_proj.weight"), trt.float32)
        silu = self.mul(gate, self.n.add_activation(gate, trt.ActivationType.SIGMOID).get_output(0))
        hidden = self.cast(self.mul(silu, up), x.dtype)
        return self.linear(hidden, prefix + ".down_proj.weight")

    def rope(self, x, position, theta, dim):
        shape = (1,) * (len(x.shape) - 1) + (dim,)
        inv = theta ** (-np.arange(0, dim, 2, dtype=np.float32) / dim)
        frequencies = np.concatenate((inv, inv)).reshape(shape)
        angle = self.mul(
            self.reshape(self.cast(position, trt.float32), (1,) * len(shape)),
            self.constant(frequencies),
        )
        cos = self.n.add_unary(angle, trt.UnaryOperation.COS).get_output(0)
        sin = self.n.add_unary(angle, trt.UnaryOperation.SIN).get_output(0)
        # NeoX RoPE: rotate the two contiguous half-heads.
        rotation = np.concatenate((np.arange(dim // 2, dim), np.arange(dim // 2))).astype(np.int32)
        self.storage.append(rotation)
        indices = self.n.add_constant((dim,), trt.Weights(rotation)).get_output(0)
        rotated = self.n.add_gather(x, indices, len(shape) - 1).get_output(0)
        signs = np.concatenate((-np.ones(dim // 2), np.ones(dim // 2))).reshape(shape)
        result = self.add(
            self.mul(self.cast(x, trt.float32), cos),
            self.mul(self.mul(self.cast(rotated, trt.float32), self.constant(signs)), sin),
        )
        return self.cast(result, x.dtype)

    def attention(self, x, prefix, config, layer):
        c = config
        h, kv, d = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        q = self.reshape(self.linear(x, prefix + ".q_proj.weight"), (h, 1, d))
        k = self.reshape(self.linear(x, prefix + ".k_proj.weight"), (kv, 1, d))
        v = self.reshape(self.linear(x, prefix + ".v_proj.weight"), (kv, 1, d))
        q = self.norm(q, prefix + ".q_norm.weight", c.rms_norm_eps)
        k = self.norm(k, prefix + ".k_norm.weight", c.rms_norm_eps)
        position = self.n.add_input("position", trt.int32, (1,))
        if layer not in c.no_rope_layer_indices:
            q = self.rope(q, position, c.rope_theta, d)
            k = self.rope(k, position, c.rope_theta, d)
        # Each cache includes a permanently masked initial dummy slot. This
        # keeps the first invocation within non-empty TensorRT shape profiles.
        past_k = self.n.add_input("past_k", trt.bfloat16, (kv, -1, d))
        past_v = self.n.add_input("past_v", trt.bfloat16, (kv, -1, d))
        joined = []
        for name, past, current in (("present_k", past_k, k), ("present_v", past_v, v)):
            concat = self.n.add_concatenation([past, current])
            concat.axis = 1
            out = concat.get_output(0)
            out.name = name
            self.n.mark_output(out)
            joined.append(out)
        k, v = joined
        # GQA: gather KV heads in query-head order without expanding the cache.
        ids = np.repeat(np.arange(kv, dtype=np.int32), h // kv)
        self.storage.append(ids)
        ids = self.n.add_constant((h,), trt.Weights(ids)).get_output(0)
        k = self.n.add_gather(k, ids, 0).get_output(0)
        v = self.n.add_gather(v, ids, 0).get_output(0)
        scores = self.mm(self.cast(q, trt.float32), self.cast(k, trt.float32))
        scores = self.mul(scores, self.constant(np.full((1, 1, 1), d**-0.5)))
        mask = self.n.add_input("mask", trt.float32, (1, 1, -1))
        scores = self.add(scores, mask)
        sinks = self.weight(prefix + ".softmax_offset", trt.float32, (h, 1, 1))
        logits = self.n.add_concatenation([sinks, scores])
        logits.axis = 2
        probs = self.n.add_softmax(logits.get_output(0))
        probs.axes = 1 << 2
        # Prepend a zero-value sink so its probability has no value contribution.
        zeros = self.constant(np.zeros((h, 1, d)), trt.bfloat16)
        values = self.n.add_concatenation([zeros, v])
        values.axis = 1
        context = self.n.add_matrix_multiply(
            self.cast(probs.get_output(0), trt.bfloat16),
            trt.MatrixOperation.NONE,
            values.get_output(0),
            trt.MatrixOperation.NONE,
        ).get_output(0)
        return self.linear(self.reshape(context, (1, h * d)), prefix + ".o_proj.weight")

    def moe(self, x, prefix, c):
        scores = self.cast(self.linear(x, prefix + ".gate.weight"), trt.float32)
        scores = self.n.add_activation(scores, trt.ActivationType.SIGMOID).get_output(0)
        bias_name = prefix + ".gate.expert_bias"
        choices = (
            self.add(scores, self.weight(bias_name, trt.float32, (1, c.num_experts)))
            if bias_name in self.weights.files
            else scores
        )
        top = self.n.add_topk(choices, trt.TopKOperation.MAX, c.num_experts_per_tok, 1 << 1)
        ids = self.reshape(top.get_output(1), (c.num_experts_per_tok,))
        selected = self.n.add_gather(scores, ids, 1).get_output(0)
        total = self.n.add_reduce(selected, trt.ReduceOperation.SUM, 1 << 1, True).get_output(0)
        total = self.add(total, self.constant(np.full((1, 1), 1e-20)))
        probabilities = self.op(selected, total, trt.ElementWiseOperation.DIV)
        packed = []
        for projection in ("gate_proj", "up_proj", "down_proj"):
            matrices = np.stack(
                [
                    self.array(f"{prefix}.experts.{e}.{projection}.weight")
                    for e in range(c.num_experts)
                ]
            )
            weights = self.constant(matrices, trt.bfloat16)
            packed.append(self.n.add_gather(weights, ids, 0).get_output(0))
        expanded = self.reshape(x, (1, 1, c.hidden_size))
        gate = self.cast(self.mm(expanded, packed[0]), trt.float32)
        up = self.cast(self.mm(expanded, packed[1]), trt.float32)
        silu = self.mul(gate, self.n.add_activation(gate, trt.ActivationType.SIGMOID).get_output(0))
        activation = self.cast(self.mul(silu, up), trt.bfloat16)
        outputs = self.cast(self.mm(activation, packed[2]), trt.float32)
        weighted = self.mul(outputs, self.reshape(probabilities, (c.num_experts_per_tok, 1, 1)))
        result = self.n.add_reduce(weighted, trt.ReduceOperation.SUM, 1 << 0, False).get_output(0)
        shared = self.swiglu(x, prefix + ".shared_expert")
        return self.add(self.cast(result, trt.bfloat16), shared)


def layer_network(network, weights, c, index):
    g = Graph(network, weights)
    x = network.add_input("hidden", trt.bfloat16, (1, c.hidden_size))
    prefix = f"model.layers.{index}"
    norm = g.norm(x, prefix + ".input_layernorm.weight", c.rms_norm_eps)
    residual = g.add(x, g.attention(norm, prefix + ".self_attn", c, index))
    norm = g.norm(residual, prefix + ".post_attention_layernorm.weight", c.rms_norm_eps)
    mlp = (
        g.swiglu(norm, prefix + ".mlp")
        if index in c.mlp_only_layers
        else g.moe(norm, prefix + ".mlp", c)
    )
    out = g.add(residual, mlp)
    out.name = "hidden_out"
    network.mark_output(out)
    return g


def head_network(network, weights, c):
    g = Graph(network, weights)
    hidden = network.add_input("hidden", trt.bfloat16, (1, c.hidden_size))
    norm = g.norm(hidden, "model.norm.weight", c.rms_norm_eps)
    # The upstream serving contract explicitly enables an FP32 LM head.
    out = g.linear(g.cast(norm, trt.float32), "lm_head.weight")
    out.name = "logits"
    network.mark_output(out)
    return g
