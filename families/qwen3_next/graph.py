# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Next equations expressed as strongly typed TensorRT graphs."""

import numpy as np
import tensorrt as trt


class Graph:
    def __init__(self, network, weights):
        self.n = network
        self.weights = weights
        self.storage = []

    def cast(self, value, dtype):
        return value if value.dtype == dtype else self.n.add_cast(value, dtype).get_output(0)

    def constant(self, values, dtype=trt.float32):
        values = np.ascontiguousarray(values, dtype=np.float32)
        self.storage.append(values)
        result = self.n.add_constant(values.shape, trt.Weights(values)).get_output(0)
        return self.cast(result, dtype)

    def integers(self, values):
        values = np.ascontiguousarray(values, dtype=np.int32)
        self.storage.append(values)
        return self.n.add_constant(values.shape, trt.Weights(values)).get_output(0)

    def raw_array(self, name):
        _, _, _, entry = self.weights.describe(name)
        with self.weights.view(name) as view:
            dtype = np.uint16 if entry["dtype"] == "BF16" else np.float32
            result = np.frombuffer(view, dtype=dtype).copy()
        return result.reshape(entry["shape"]), entry["dtype"]

    def bf16_constant(self, array):
        array = np.ascontiguousarray(array, dtype=np.uint16)
        self.storage.append(array)
        weights = trt.Weights(trt.bfloat16, array.ctypes.data, array.size)
        return self.n.add_constant(array.shape, weights).get_output(0)

    def weight(self, name, dtype=trt.bfloat16, shape=None):
        array, storage_dtype = self.raw_array(name)
        if shape is not None:
            array = array.reshape(shape)
        result = self.bf16_constant(array) if storage_dtype == "BF16" else self.constant(array)
        return self.cast(result, dtype)

    def reshape(self, value, shape):
        layer = self.n.add_shuffle(value)
        layer.reshape_dims = shape
        return layer.get_output(0)

    def slice(self, value, start, size):
        return self.n.add_slice(value, start, size, (1,) * len(start)).get_output(0)

    def concat(self, values, axis):
        result = self.n.add_concatenation(values)
        result.axis = axis
        return result.get_output(0)

    def binary(self, a, b, operation):
        return self.n.add_elementwise(a, b, operation).get_output(0)

    def add(self, a, b):
        return self.binary(a, b, trt.ElementWiseOperation.SUM)

    def sub(self, a, b):
        return self.binary(a, b, trt.ElementWiseOperation.SUB)

    def mul(self, a, b):
        return self.binary(a, b, trt.ElementWiseOperation.PROD)

    def divide(self, a, b):
        return self.binary(a, b, trt.ElementWiseOperation.DIV)

    def reduce(self, value, operation, axis, keep=True):
        return self.n.add_reduce(value, operation, 1 << axis, keep).get_output(0)

    def unary(self, value, operation):
        return self.n.add_unary(value, operation).get_output(0)

    def sigmoid(self, value):
        # PyTorch's BF16 activation computes in FP32 and rounds once.
        result = self.n.add_activation(
            self.cast(value, trt.float32), trt.ActivationType.SIGMOID
        ).get_output(0)
        return self.cast(result, value.dtype)

    def silu(self, value):
        f = self.cast(value, trt.float32)
        return self.cast(self.mul(f, self.sigmoid(f)), value.dtype)

    def mm(self, a, b):
        return self.n.add_matrix_multiply(
            a, trt.MatrixOperation.NONE, b, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)

    def linear(self, value, name):
        # Accumulate dense projections in FP32, then round once to BF16.
        # BF16 GEMV rounding differences perturb the recurrent mixer and can
        # change which experts are selected at tied router cutoffs.
        kernel = self.weight(name, trt.float32)
        width = value.shape[-1]
        products = self.mul(
            self.reshape(self.cast(value, trt.float32), (1, 1, width)),
            self.reshape(kernel, (1, kernel.shape[0], width)),
        )
        return self.cast(self.reduce(products, trt.ReduceOperation.SUM, 2, False), value.dtype)

    def inverse_root(self, value):
        return self.unary(self.unary(value, trt.UnaryOperation.SQRT), trt.UnaryOperation.RECIP)

    def normalize(self, value, eps):
        f = self.cast(value, trt.float32)
        shape = (1,) * len(value.shape)
        variance = self.reduce(self.mul(f, f), trt.ReduceOperation.AVG, len(shape) - 1)
        return self.mul(
            f, self.inverse_root(self.add(variance, self.constant(np.full(shape, eps))))
        )

    def norm(self, value, name, eps):
        shape = (1,) * (len(value.shape) - 1) + (value.shape[-1],)
        gamma = self.add(
            self.weight(name, trt.float32, shape), self.constant(np.ones((1,) * len(shape)))
        )
        return self.cast(self.mul(self.normalize(value, eps), gamma), value.dtype)

    def swiglu(self, value, prefix):
        gate = self.silu(self.linear(value, prefix + ".gate_proj.weight"))
        up = self.linear(value, prefix + ".up_proj.weight")
        return self.linear(self.mul(gate, up), prefix + ".down_proj.weight")

    def l2norm(self, value):
        # Match the official torch fallback's BF16 product/reduction boundary.
        square = self.mul(value, value)
        total = self.reduce(square, trt.ReduceOperation.SUM, len(value.shape) - 1)
        epsilon = self.constant(np.full((1,) * len(value.shape), 1e-6), value.dtype)
        denominator = self.add(total, epsilon)
        inverse = self.cast(self.inverse_root(self.cast(denominator, trt.float32)), value.dtype)
        return self.mul(value, inverse)

    def partial_rope(self, value, position, config):
        dimension = int(config.head_dim * config.partial_rotary_factor)
        h = value.shape[0]
        rotated = self.slice(value, (0, 0, 0), (h, 1, dimension))
        passed = self.slice(value, (0, 0, dimension), (h, 1, config.head_dim - dimension))
        inv = config.rope_theta ** (-np.arange(0, dimension, 2, dtype=np.float32) / dimension)
        frequency = self.constant(np.concatenate((inv, inv)).reshape(1, 1, dimension))
        angle = self.mul(self.reshape(self.cast(position, trt.float32), (1, 1, 1)), frequency)
        cosine = self.cast(self.unary(angle, trt.UnaryOperation.COS), value.dtype)
        sine = self.cast(self.unary(angle, trt.UnaryOperation.SIN), value.dtype)
        ids = self.integers(
            np.concatenate((np.arange(dimension // 2, dimension), np.arange(dimension // 2)))
        )
        swapped = self.n.add_gather(rotated, ids, 2).get_output(0)
        signs = self.constant(
            np.concatenate((-np.ones(dimension // 2), np.ones(dimension // 2))).reshape(
                1, 1, dimension
            ),
            value.dtype,
        )
        output = self.add(self.mul(rotated, cosine), self.mul(self.mul(swapped, signs), sine))
        return self.concat([output, passed], 2)

    def attention(self, value, prefix, c):
        h, kv, d = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        mixed = self.reshape(self.linear(value, prefix + ".q_proj.weight"), (h, 1, d * 2))
        query = self.slice(mixed, (0, 0, 0), (h, 1, d))
        gate = self.reshape(self.slice(mixed, (0, 0, d), (h, 1, d)), (1, h * d))
        key = self.reshape(self.linear(value, prefix + ".k_proj.weight"), (kv, 1, d))
        val = self.reshape(self.linear(value, prefix + ".v_proj.weight"), (kv, 1, d))
        query = self.norm(query, prefix + ".q_norm.weight", c.rms_norm_eps)
        key = self.norm(key, prefix + ".k_norm.weight", c.rms_norm_eps)
        position = self.n.add_input("position", trt.int32, (1,))
        query, key = self.partial_rope(query, position, c), self.partial_rope(key, position, c)
        for name, current in (("k", key), ("v", val)):
            past = self.n.add_input("past_" + name, trt.bfloat16, (kv, -1, d))
            joined = self.concat([past, current], 1)
            joined.name = "present_" + name
            self.n.mark_output(joined)
            if name == "k":
                key = joined
            else:
                val = joined
        ids = self.integers(np.repeat(np.arange(kv), h // kv))
        key = self.n.add_gather(key, ids, 0).get_output(0)
        val = self.n.add_gather(val, ids, 0).get_output(0)
        scores = self.mm(query, key)
        scores = self.mul(scores, self.constant(np.full((1, 1, 1), d**-0.5), scores.dtype))
        mask = self.n.add_input("mask", trt.float32, (1, 1, -1))
        scores = self.add(scores, self.cast(mask, scores.dtype))
        probabilities = self.n.add_softmax(self.cast(scores, trt.float32))
        probabilities.axes = 1 << 2
        context = self.n.add_matrix_multiply(
            self.cast(probabilities.get_output(0), value.dtype),
            trt.MatrixOperation.NONE,
            val,
            trt.MatrixOperation.NONE,
        ).get_output(0)
        context = self.mul(self.reshape(context, (1, h * d)), self.sigmoid(gate))
        return self.linear(context, prefix + ".o_proj.weight")

    def delta(self, value, prefix, c):
        kh, vh = c.linear_num_key_heads, c.linear_num_value_heads
        kd, vd = c.linear_key_head_dim, c.linear_value_head_dim
        ratio = vh // kh
        mixed = self.reshape(
            self.linear(value, prefix + ".in_proj_qkvz.weight"), (kh, 2 * kd + 2 * ratio * vd)
        )
        query = self.slice(mixed, (0, 0), (kh, kd))
        key = self.slice(mixed, (0, kd), (kh, kd))
        val = self.slice(mixed, (0, kd * 2), (kh, ratio * vd))
        z = self.reshape(self.slice(mixed, (0, kd * 2 + ratio * vd), (kh, ratio * vd)), (vh, vd))
        mixed_ba = self.reshape(self.linear(value, prefix + ".in_proj_ba.weight"), (kh, ratio * 2))
        b = self.reshape(self.slice(mixed_ba, (0, 0), (kh, ratio)), (vh,))
        a = self.reshape(self.slice(mixed_ba, (0, ratio), (kh, ratio)), (vh,))
        channels = c.conv_dim
        current = self.reshape(
            self.concat(
                [
                    self.reshape(query, (kh * kd,)),
                    self.reshape(key, (kh * kd,)),
                    self.reshape(val, (vh * vd,)),
                ],
                0,
            ),
            (channels, 1),
        )
        history = self.n.add_input(
            "conv_state", trt.bfloat16, (channels, c.linear_conv_kernel_dim - 1)
        )
        window = self.concat([history, current], 1)
        present = self.slice(window, (0, 1), (channels, c.linear_conv_kernel_dim - 1))
        present.name = "present_conv"
        self.n.mark_output(present)
        coefficients = self.weight(
            prefix + ".conv1d.weight", trt.float32, (channels, c.linear_conv_kernel_dim)
        )
        convolved = self.reduce(
            self.mul(self.cast(window, trt.float32), coefficients),
            trt.ReduceOperation.SUM,
            1,
            False,
        )
        convolved = self.silu(self.cast(convolved, trt.bfloat16))
        query = self.reshape(self.slice(convolved, (0,), (kh * kd,)), (kh, kd))
        key = self.reshape(self.slice(convolved, (kh * kd,), (kh * kd,)), (kh, kd))
        val = self.reshape(self.slice(convolved, (kh * kd * 2,), (vh * vd,)), (vh, vd))
        ids = self.integers(np.repeat(np.arange(kh), ratio))
        query = self.n.add_gather(query, ids, 0).get_output(0)
        key = self.n.add_gather(key, ids, 0).get_output(0)
        query = self.cast(self.l2norm(query), trt.float32)
        key = self.cast(self.l2norm(key), trt.float32)
        query = self.mul(query, self.constant(np.full((1, 1), kd**-0.5)))
        beta = self.cast(self.sigmoid(b), trt.float32)
        timestep = self.add(
            self.cast(a, trt.float32), self.weight(prefix + ".dt_bias", trt.float32)
        )
        softplus = self.n.add_activation(timestep, trt.ActivationType.SOFTPLUS)
        softplus.alpha = 1.0
        softplus.beta = 1.0
        rate = self.unary(self.weight(prefix + ".A_log", trt.float32), trt.UnaryOperation.EXP)
        decay = self.unary(
            self.unary(self.mul(rate, softplus.get_output(0)), trt.UnaryOperation.NEG),
            trt.UnaryOperation.EXP,
        )
        state = self.n.add_input("recurrent_state", trt.float32, (vh, kd, vd))
        decayed = self.mul(state, self.reshape(decay, (vh, 1, 1)))
        memory = self.reduce(
            self.mul(decayed, self.reshape(key, (vh, kd, 1))), trt.ReduceOperation.SUM, 1, False
        )
        innovation = self.mul(
            self.sub(self.cast(val, trt.float32), memory), self.reshape(beta, (vh, 1))
        )
        state = self.add(
            decayed, self.mul(self.reshape(key, (vh, kd, 1)), self.reshape(innovation, (vh, 1, vd)))
        )
        state.name = "present_recurrent"
        self.n.mark_output(state)
        output = self.cast(
            self.reduce(
                self.mul(state, self.reshape(query, (vh, kd, 1))), trt.ReduceOperation.SUM, 1, False
            ),
            trt.bfloat16,
        )
        # The gated norm deliberately rounds normalization and affine scaling
        # to BF16 before the FP32 SiLU gate, as in the official torch fallback.
        normalized = self.cast(self.normalize(output, c.rms_norm_eps), trt.bfloat16)
        normalized = self.mul(
            normalized, self.weight(prefix + ".norm.weight", trt.bfloat16, (1, vd))
        )
        normalized = self.cast(
            self.mul(self.cast(normalized, trt.float32), self.silu(self.cast(z, trt.float32))),
            trt.bfloat16,
        )
        return self.linear(self.reshape(normalized, (1, vh * vd)), prefix + ".out_proj.weight")

    def routing(self, scores, c):
        # Export the unmodified probabilities so the family runtime can resolve
        # cutoff ties with the reference selection algorithm before state commits.
        scores.name = "router_probabilities"
        self.n.mark_output(scores)
        top = self.n.add_topk(scores, trt.TopKOperation.MAX, c.num_experts_per_tok, 1 << 1)
        forced = self.n.add_input("forced_experts", trt.int32, (c.num_experts_per_tok,))
        control = self.n.add_input("force_routing", trt.int32, (1,))
        use_forced = self.binary(control, self.integers([0]), trt.ElementWiseOperation.GREATER)
        selected = self.n.add_select(
            self.reshape(use_forced, (1, 1)),
            self.reshape(forced, (1, c.num_experts_per_tok)),
            top.get_output(1),
        ).get_output(0)
        selected.name = "selected_experts"
        self.n.mark_output(selected)
        probabilities = self.n.add_gather(
            self.reshape(scores, (c.num_experts,)),
            self.reshape(selected, (c.num_experts_per_tok,)),
            0,
        ).get_output(0)
        probabilities = self.reshape(probabilities, (1, c.num_experts_per_tok))
        probabilities = self.divide(
            probabilities, self.reduce(probabilities, trt.ReduceOperation.SUM, 1)
        )
        return selected, probabilities

    def moe(self, value, prefix, c):
        scores = self.n.add_softmax(
            self.cast(self.linear(value, prefix + ".gate.weight"), trt.float32)
        )
        scores.axes = 1 << 1
        selected, probabilities = self.routing(scores.get_output(0), c)
        # The full upstream model uses grouped expert execution. Preserve the
        # router slot order through FP32 weighting and reduction.
        ids = self.reshape(selected, (c.num_experts_per_tok,))
        projections = []
        for projection in ("gate_proj", "up_proj", "down_proj"):
            values = []
            for expert in range(c.num_experts):
                array, dtype = self.raw_array(f"{prefix}.experts.{expert}.{projection}.weight")
                if dtype != "BF16":
                    raise ValueError("Qwen3-Next routed experts must use BF16 checkpoint storage")
                values.append(array)
            packed = self.bf16_constant(np.stack(values))
            projections.append(self.n.add_gather(packed, ids, 0).get_output(0))
        expanded = self.reshape(value, (1, 1, c.hidden_size))
        gate = self.silu(self.mm(expanded, projections[0]))
        up = self.mm(expanded, projections[1])
        outputs = self.mm(self.mul(gate, up), projections[2])
        weighted = self.mul(
            self.cast(outputs, trt.float32),
            self.reshape(probabilities, (c.num_experts_per_tok, 1, 1)),
        )
        total = self.cast(self.reduce(weighted, trt.ReduceOperation.SUM, 0, False), trt.bfloat16)
        shared = self.swiglu(value, prefix + ".shared_expert")
        shared = self.mul(
            shared, self.sigmoid(self.linear(value, prefix + ".shared_expert_gate.weight"))
        )
        return self.add(total, shared)


def layer_network(network, weights, config, index):
    g = Graph(network, weights)
    value = network.add_input("hidden", trt.bfloat16, (1, config.hidden_size))
    prefix = f"model.layers.{index}"
    normalized = g.norm(value, prefix + ".input_layernorm.weight", config.rms_norm_eps)
    mixer = (
        g.attention(normalized, prefix + ".self_attn", config)
        if config.layer_types[index] == "full_attention"
        else g.delta(normalized, prefix + ".linear_attn", config)
    )
    residual = g.add(value, mixer)
    normalized = g.norm(residual, prefix + ".post_attention_layernorm.weight", config.rms_norm_eps)
    output = g.add(residual, g.moe(normalized, prefix + ".mlp", config))
    output.name = "hidden_out"
    network.mark_output(output)
    return g


def head_network(network, weights, config):
    g = Graph(network, weights)
    value = network.add_input("hidden", trt.bfloat16, (1, config.hidden_size))
    normalized = g.norm(value, "model.norm.weight", config.rms_norm_eps)
    output = g.cast(g.linear(normalized, "lm_head.weight"), trt.float32)
    output.name = "logits"
    network.mark_output(output)
    return g
