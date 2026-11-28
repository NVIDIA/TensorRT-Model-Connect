# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bolt LiLiCorr: accepted-token KV projection and noncausal eight-lane proposal blocks."""

import numpy as np

from .native_language import Graph


class ProposalGraph(Graph):
    def dense(self, x, prefix):
        op = self.t.MatrixOperation
        w = self.const(self.w.tensor(prefix + ".weight").astype(np.float16))
        return self.n.add_matrix_multiply(x, op.NONE, w, op.TRANSPOSE).get_output(0)

    def slice(self, x, start, size):
        return self.n.add_slice(x, start, size, (1,) * len(size)).get_output(0)

    def cat(self, tensors, axis):
        op = self.n.add_concatenation(tensors)
        op.axis = axis
        return op.get_output(0)

    def rms(self, x, prefix, epsilon):
        values = self.f(x)
        square = self.ew(values, values, "PROD")
        var = self.reduce(square, "AVG", len(x.shape) - 1)
        root = self.n.add_unary(
            self.ew(var, self.scalar(epsilon, len(x.shape)), "SUM"), self.t.UnaryOperation.SQRT
        ).get_output(0)
        normalized = self.cast(self.ew(values, root, "DIV"), x.dtype)
        weight = self.w.tensor(prefix + ".weight").astype(np.float16)
        shape = (1,) * (len(x.shape) - 1) + (weight.size,)
        return self.ew(normalized, self.cast(self.const(weight.reshape(shape)), x.dtype), "PROD")

    def rotate(self, x, trig):
        rows, heads, dim = x.shape
        half = dim // 2
        c = self.sh(self.slice(trig, (0, 0), (rows, half)), rows, 1, half)
        s = self.sh(self.slice(trig, (0, half), (rows, half)), rows, 1, half)
        a = self.f(self.slice(x, (0, 0, 0), (rows, heads, half)))
        b = self.f(self.slice(x, (0, 0, half), (rows, heads, half)))
        left = self.ew(self.ew(a, c, "PROD"), self.ew(b, s, "PROD"), "SUB")
        right = self.ew(self.ew(b, c, "PROD"), self.ew(a, s, "PROD"), "SUM")
        return self.cast(self.cat([left, right], 2), x.dtype)

    def conv(self, x, coeff, base, group, residual=None):
        rows, h = x.shape
        groups, kernel = h // group, base.shape[0]
        result = None
        for tap in range(kernel):
            delta = self.slice(coeff, (0, tap * groups), (rows, groups))
            gain = self.ew(
                self.sh(self.f(delta), rows, groups, 1),
                self.const(base[tap].reshape(1, groups, group)),
                "SUM",
            )
            shifted = (
                x
                if tap == 0
                else self.cat(
                    [
                        self.const(np.zeros((tap, h), np.float16)),
                        self.slice(x, (0, 0), (rows - tap, h)),
                    ],
                    0,
                )
            )
            term = self.ew(gain, self.sh(self.f(shifted), rows, groups, group), "PROD")
            result = term if result is None else self.ew(result, term, "SUM")
        out = self.sh(result, rows, h)
        if residual is not None:
            return self.ew(
                self.ew(out, self.ew(residual, self.scalar(1 / 1024, 2), "PROD"), "SUM"),
                self.scalar(1024, 2),
                "PROD",
            )
        clip = self.n.add_activation(out, self.t.ActivationType.CLIP)
        clip.alpha, clip.beta = -65504.0, 65504.0
        return self.cast(clip.get_output(0), self.t.float16)

    def prepare(self, x, name, group):
        base = self.w.tensor(name + ".base_kernel").astype(np.float16).astype(np.float32)
        rows, h = x.shape
        side = base.shape[1] * (h // group)
        coeff = self.dense(x, name + ".kernel_projection")
        pre = self.slice(coeff, (0, 0), (rows, side))
        post = self.slice(coeff, (0, side), (rows, side))
        return self.conv(x, pre, base[0], group), post, base[1]


def _network(weights, logger):
    import tensorrt as trt

    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    return builder, net, ProposalGraph(net, weights)


def _finish(builder, net):
    config_builder = builder.create_builder_config()
    config_builder.builder_optimization_level = 3
    result = builder.build_serialized_network(net, config_builder)
    if result is None:
        raise RuntimeError("Cannot build native Bolt LiLiCorr proposal network")
    return result


def build_cache(weights, cfg, *, logger):
    """Project one accepted feature into all five layers' permanent K/V. Never cache proposals."""
    import tensorrt as t

    b, n, g = _network(weights, logger)
    h, kv, dim = cfg["hidden_size"], cfg["num_key_value_heads"], cfg["head_dim"]
    x = n.add_input("accepted_hidden", t.float16, (1, h))
    trig = n.add_input("rope", t.float32, (1, dim))
    for i in range(cfg["num_hidden_layers"]):
        prefix = f"layers.{i}.self_attn"
        k = g.sh(g.dense(x, prefix + ".k_proj"), 1, kv, dim)
        g.output(g.rotate(g.rms(k, prefix + ".k_norm", cfg["rms_norm_eps"]), trig), f"key_{i}")
        g.output(g.sh(g.dense(x, prefix + ".v_proj"), 1, kv, dim), f"value_{i}")
    return _finish(b, n)


def build_proposal(weights, cfg, index, *, logger):
    """Compute one real checkpoint layer. The final one returns only the seven proposal lanes."""
    import tensorrt as t

    b, n, g = _network(weights, logger)
    rows, h, heads = cfg["block_size"], cfg["hidden_size"], cfg["num_attention_heads"]
    kv, dim = cfg["num_key_value_heads"], cfg["head_dim"]
    dc = cfg["dflash_config"]
    if (
        rows != 8
        or heads % kv
        or not dc["use_swa"]
        or dc["causal"]
        or not dc["attention_sink_bias"]
        or not 0 <= index < cfg["num_hidden_layers"]
    ):
        raise ValueError("Unsupported Bolt LiLiCorr proposal geometry")
    window, group, eps = dc["swa_window_size"], dc["conv_group_size"], cfg["rms_norm_eps"]
    total, repeats = window + rows, heads // kv
    hidden = n.add_input("hidden", t.float32, (rows, h))
    keys = n.add_input("cached_keys", t.float16, (window, kv, dim))
    values = n.add_input("cached_values", t.float16, (window, kv, dim))
    trig = n.add_input("rope", t.float32, (rows, dim))
    mask = n.add_input("attention_bias", t.float32, (rows, total))
    prefix = f"layers.{index}"
    attn = prefix + ".self_attn"
    norm = g.cast(g.rms(hidden, prefix + ".input_layernorm", eps), t.float16)
    x, post, base = g.prepare(norm, prefix + ".attention_conv", group)

    def projected(part, count):
        raw = g.sh(g.dense(x, attn + "." + part + "_proj"), rows, count, dim)
        return g.rotate(g.rms(raw, attn + "." + part + "_norm", eps), trig)

    q = g.sh(g.f(g.transpose(projected("q", heads), (1, 0, 2))), kv, repeats, rows, dim)
    k = g.cat([keys, projected("k", kv)], 0)
    v = g.cat([values, g.sh(g.dense(x, attn + ".v_proj"), rows, kv, dim)], 0)
    k = g.sh(g.f(g.transpose(k, (1, 0, 2))), kv, 1, total, dim)
    v = g.sh(g.f(g.transpose(v, (1, 0, 2))), kv, 1, total, dim)
    op = t.MatrixOperation
    logits = n.add_matrix_multiply(q, op.NONE, k, op.TRANSPOSE).get_output(0)
    logits = g.ew(
        g.ew(logits, g.scalar(1 / np.sqrt(dim), 4), "PROD"), g.sh(mask, 1, 1, rows, total), "SUM"
    )
    # The learned sink contributes to the softmax denominator but has no value vector.
    sinks = weights.f32(attn + ".attention_sink_bias").reshape(kv, repeats, 1, 1)
    scores = g.cat([logits, g.const(np.repeat(sinks, rows, axis=2))], 3)
    prob = n.add_softmax(scores)
    prob.axes = 1 << 3
    p = g.slice(prob.get_output(0), (0, 0, 0, 0), (kv, repeats, rows, total))
    out = n.add_matrix_multiply(g.f(g.cast(p, t.float16)), op.NONE, v, op.NONE).get_output(0)
    out = g.sh(g.transpose(g.sh(out, heads, rows, dim), (1, 0, 2)), rows, h)
    scale = g.const(np.array([[1 / 1024]], np.float16))
    att = g.dense(g.ew(g.cast(out, t.float16), scale, "PROD"), attn + ".o_proj")
    hidden = g.conv(att, post, base, group, hidden)
    norm = g.cast(g.rms(hidden, prefix + ".post_attention_layernorm", eps), t.float16)
    x, post, base = g.prepare(norm, prefix + ".mlp_conv", group)
    mlp = prefix + ".mlp"
    gate = g.cast(g.silu(g.f(g.w4a16(x, mlp + ".gate_proj"))), t.float16)
    up = g.w4a16(g.ew(x, scale, "PROD"), mlp + ".up_proj")
    down = g.w4a16(g.ew(gate, up, "PROD"), mlp + ".down_proj")
    hidden = g.conv(down, post, base, group, hidden)
    if index == cfg["num_hidden_layers"] - 1:
        prediction = g.cast(g.rms(hidden, "norm", eps), t.float16)
        g.output(g.slice(prediction, (1, 0), (rows - 1, h)), "predictions")
    else:
        g.output(hidden, "next_hidden")
    return _finish(b, n)
