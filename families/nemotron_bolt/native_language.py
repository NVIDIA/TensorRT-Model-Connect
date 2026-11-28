# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-token Bolt decoder plans. Repeated application implements both prefill and decode."""

import numpy as np

from .puzzle_moe import puzzle_moe, q8_linear


class Graph:
    def __init__(self, network, weights):
        import tensorrt as trt

        self.n, self.w, self.t = network, weights, trt
        self.live = []

    def const(self, a):
        a = np.array(a, copy=True, order="C")
        self.live.append(a)
        return self.n.add_constant(a.shape, self.t.Weights(a)).get_output(0)

    def cast(self, x, dtype):
        return x if x.dtype == dtype else self.n.add_cast(x, dtype).get_output(0)

    def f(self, x):
        return self.cast(x, self.t.float32)

    def bf(self, x):
        return self.cast(x, self.t.bfloat16)

    def ew(self, a, b, op):
        return self.n.add_elementwise(a, b, getattr(self.t.ElementWiseOperation, op)).get_output(0)

    def sh(self, a, *shape):
        x = self.n.add_shuffle(a)
        x.reshape_dims = shape
        return x.get_output(0)

    def transpose(self, a, order):
        x = self.n.add_shuffle(a)
        x.first_transpose = order
        return x.get_output(0)

    def col(self, a, offset, size):
        return self.n.add_slice(a, (0, offset), (1, size), (1, 1)).get_output(0)

    def scalar(self, k, rank):
        return self.const(np.full((1,) * rank, k, np.float32))

    def reduce(self, x, op, axis, keep=True):
        return self.n.add_reduce(
            x, getattr(self.t.ReduceOperation, op), 1 << axis, keep
        ).get_output(0)

    def silu(self, x):
        return self.ew(
            x, self.n.add_activation(x, self.t.ActivationType.SIGMOID).get_output(0), "PROD"
        )

    def norm(self, x, scale, eps, groups=1):
        a = self.sh(self.f(x), groups, scale.size // groups)
        variance = self.reduce(self.ew(a, a, "PROD"), "AVG", 1)
        root = self.n.add_unary(
            self.ew(variance, self.scalar(eps, 2), "SUM"), self.t.UnaryOperation.SQRT
        ).get_output(0)
        z = self.sh(self.ew(a, root, "DIV"), 1, scale.size)
        return self.bf(self.ew(z, self.const(scale.reshape(1, -1)), "PROD"))

    def w4a16(self, x, key):
        """E2M1 weights, E4M3 scales and FP32 global scale; do not quantize activations."""
        raw = np.ascontiguousarray(self.w.tensor(key + ".weight"))
        scale = self.w.tensor(key + ".weight_scale")
        factor = self.w.f32(key + ".weight_scale_2").reshape(())
        n, k2 = raw.shape
        if (
            raw.dtype != np.uint8
            or scale.dtype.name != "float8_e4m3fn"
            or scale.shape != (n, k2 // 8)
            or x.shape[-1] != 2 * k2
            or not np.isfinite(factor)
            or factor <= 0
        ):
            raise ValueError("Malformed Bolt W4A16 projection " + key)
        self.live.extend([raw, scale])
        t = self.t
        weight = self.n.add_constant(
            (n, 2 * k2), t.Weights(t.DataType.FP4, raw.ctypes.data, 2 * raw.size)
        ).get_output(0)
        if x.dtype not in (t.bfloat16, t.float16):
            raise ValueError("Bolt W4A16 requires a 16-bit activation")
        # These E4M3 block scales are immutable checkpoint metadata, not activations.
        # Precompute their global scale in FP32, then round once to the activation dtype.
        # This keeps the E2M1 weights packed without requiring an FP8-capable GPU.
        real_scales = self.cast(self.const(scale.astype(np.float32) * factor), x.dtype)
        dq = self.n.add_dequantize(weight, real_scales, x.dtype)
        dq.axis = 1
        return self.n.add_matrix_multiply(
            x, t.MatrixOperation.NONE, dq.get_output(0), t.MatrixOperation.TRANSPOSE
        ).get_output(0)

    def output(self, x, name):
        x.name = name
        self.n.mark_output(x)

    def mamba(self, x, k, c):
        n, t = self.n, self.t
        h, p, g, s = c["mamba_num_heads"], c["mamba_head_dim"], c["n_groups"], c["ssm_state_size"]
        inner, span, D = h * p, g * s, c["conv_kernel"]
        ch = inner + 2 * span
        u = self.w4a16(x, k + "in_proj")
        gate = self.col(u, 0, inner)
        v = self.col(u, inner, ch)
        dt = self.f(self.col(u, inner + ch, h))
        old = n.add_input("conv", t.bfloat16, (ch, D))
        rest = n.add_slice(old, (0, 1), (ch, D - 1), (1, 1)).get_output(0)
        cat = n.add_concatenation([rest, self.sh(v, ch, 1)])
        cat.axis = 1
        new = cat.get_output(0)
        self.output(new, "next_conv")
        filt = self.const(self.w.f32(k + "conv1d.weight").reshape(ch, D))
        vc = self.reduce(self.ew(self.f(new), filt, "PROD"), "SUM", 1, False)
        vc = self.ew(vc, self.const(self.w.f32(k + "conv1d.bias")), "SUM")
        activated = self.sh(self.f(self.bf(self.silu(vc))), 1, ch)
        xh = self.sh(self.col(activated, 0, inner), g, h // g, p, 1)
        B = self.sh(self.col(activated, inner, span), g, 1, 1, s)
        C = self.sh(self.col(activated, inner + span, span), g, 1, 1, s)
        dt = self.ew(dt, self.const(self.w.f32(k + "dt_bias").reshape(1, h)), "SUM")
        soft = n.add_activation(dt, t.ActivationType.SOFTPLUS)
        soft.alpha = 1.0
        soft.beta = 1.0
        dt = self.sh(soft.get_output(0), g, h // g, 1, 1)
        a = self.const(-np.exp(self.w.f32(k + "A_log")).reshape(g, h // g, 1, 1))
        decay = n.add_unary(self.ew(dt, a, "PROD"), t.UnaryOperation.EXP).get_output(0)
        state = n.add_input("ssm", t.float32, (g, h // g, p, s))
        drive = self.ew(self.ew(dt, B, "PROD"), xh, "PROD")
        ns = self.ew(self.ew(state, decay, "PROD"), drive, "SUM")
        self.output(ns, "next_ssm")
        y = self.reduce(self.ew(ns, C, "PROD"), "SUM", 3)
        y = self.ew(
            y, self.ew(xh, self.const(self.w.f32(k + "D").reshape(g, h // g, 1, 1)), "PROD"), "SUM"
        )
        gated = self.ew(self.sh(self.f(self.bf(y)), 1, inner), self.silu(self.f(gate)), "PROD")
        y = self.norm(gated, self.w.f32(k + "norm.weight"), c["layer_norm_epsilon"], g)
        return self.w4a16(y, k + "out_proj")

    def attention(self, x, k, c, capacity):
        n, t = self.n, self.t
        h, kv, d = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        if c.get("use_qk_norm") or c.get("attention_bias") or c.get("sliding_window") or h % kv:
            raise ValueError("Unexpected Bolt attention contract")
        q, key, value = (
            q8_linear(n, x, self.w.tensor(k + role + "_proj.weight")) for role in ["q", "k", "v"]
        )
        q = self.sh(self.f(q), kv, h // kv, d)
        mask = n.add_input("attention_mask", t.float32, (1, 1, capacity + 1))
        cache = []
        for z, name in [(key, "k"), (value, "v")]:
            current = self.sh(z, 1, kv, d)
            self.output(current, "append_" + name)
            old = n.add_input(name, t.bfloat16, (capacity, kv, d))
            cat = n.add_concatenation([old, current])
            cat.axis = 0
            cache.append(self.f(self.transpose(cat.get_output(0), (1, 0, 2))))
        logits = n.add_matrix_multiply(
            q, t.MatrixOperation.NONE, cache[0], t.MatrixOperation.TRANSPOSE
        ).get_output(0)
        logits = self.ew(self.ew(logits, self.scalar(d**-0.5, 3), "PROD"), mask, "SUM")
        soft = n.add_softmax(logits)
        soft.axes = 4
        probs = self.f(self.bf(soft.get_output(0)))
        y = n.add_matrix_multiply(
            probs, t.MatrixOperation.NONE, cache[1], t.MatrixOperation.NONE
        ).get_output(0)
        return q8_linear(n, self.bf(self.sh(y, 1, h * d)), self.w.tensor(k + "o_proj.weight"))


def build_plan(weights, config, index, capacity, *, logger):
    import tensorrt as trt

    builder = trt.Builder(logger)
    n = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    g = Graph(n, weights)
    x = n.add_input("hidden", trt.bfloat16, (1, config["hidden_size"]))
    if index == "head":
        norm = g.norm(
            x, weights.f32("language_model.backbone.norm_f.weight"), config["rms_norm_eps"]
        )
        logits = q8_linear(
            n, norm, weights.tensor("language_model.lm_head.weight"), output_dtype=trt.float32
        )
        g.output(g.f(logits), "logits")
    else:
        b = config["block_configs"][index]
        prefix = f"language_model.backbone.layers.{index}."
        hidden = g.norm(x, weights.f32(prefix + "norm.weight"), config["rms_norm_eps"])
        k = prefix + "mixer."
        kind = b["block_type"]
        if kind == "mamba":
            y = g.mamba(hidden, k, config)
        elif kind == "moe":
            y, _, _ = puzzle_moe(
                n,
                hidden,
                weights.experts(k, b["n_routed_experts"]),
                selected=b["num_experts_per_tok"],
                normalise=config["norm_topk_prob"],
                routed_scale=config["routed_scaling_factor"],
            )
        elif kind == "attention":
            y = g.attention(hidden, k, config, capacity)
        else:
            raise ValueError(f"Unsupported Bolt layer {kind}")
        g.output(g.ew(x, y, "SUM"), "next_hidden")
    config_builder = builder.create_builder_config()
    config_builder.builder_optimization_level = 3
    plan = builder.build_serialized_network(n, config_builder)
    if plan is None:
        raise RuntimeError(f"TensorRT could not build Bolt language plan {index}")
    return plan
