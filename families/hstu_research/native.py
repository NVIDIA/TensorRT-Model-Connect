# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strongly typed TensorRT lowering of the public HSTU reference."""

import numpy as np
import tensorrt as trt


class Network:
    def __init__(self, network, config, weights, precision):
        self.n, self.c, self.w = network, config, weights
        self.low = trt.bfloat16 if precision == "bf16" else trt.float32
        self.precision = precision
        self.keep = []

    def const(self, value, dtype=None):
        a = np.ascontiguousarray(np.asarray(value, dtype=dtype))
        self.keep.append(a)
        return self.n.add_constant(a.shape, a).get_output(0)

    def cast(self, x, dtype):
        return x if x.dtype == dtype else self.n.add_cast(x, dtype).get_output(0)

    def reshape(self, x, shape):
        layer = self.n.add_shuffle(x)
        layer.reshape_dims = tuple(shape)
        return layer.get_output(0)

    def transpose(self, x, axes):
        layer = self.n.add_shuffle(x)
        layer.second_transpose = tuple(axes)
        return layer.get_output(0)

    def op(self, a, b, kind):
        rank = max(len(a.shape), len(b.shape))
        if len(a.shape) < rank:
            a = self.reshape(a, (1,) * (rank - len(a.shape)) + tuple(a.shape))
        if len(b.shape) < rank:
            b = self.reshape(b, (1,) * (rank - len(b.shape)) + tuple(b.shape))
        return self.n.add_elementwise(a, b, getattr(trt.ElementWiseOperation, kind)).get_output(0)

    def add(self, a, b):
        return self.op(a, b, "SUM")

    def mul(self, a, b):
        return self.op(a, b, "PROD")

    def less(self, a, b):
        return self.op(a, b, "LESS")

    def equal(self, a, b):
        return self.op(a, b, "EQUAL")

    def select(self, p, a, b):
        rank = max(len(p.shape), len(a.shape), len(b.shape))
        values = [
            self.reshape(v, (1,) * (rank - len(v.shape)) + tuple(v.shape))
            if len(v.shape) < rank
            else v
            for v in (p, a, b)
        ]
        return self.n.add_select(*values).get_output(0)

    def reduce(self, x, kind, axis, keep=True):
        return self.n.add_reduce(x, getattr(trt.ReduceOperation, kind), 1 << axis, keep).get_output(
            0
        )

    def silu(self, x):
        return self.mul(x, self.n.add_activation(x, trt.ActivationType.SIGMOID).get_output(0))

    def norm(self, x):
        # Sensitive reductions and reciprocal sqrt are explicitly FP32.
        x = self.cast(x, trt.float32)
        mean = self.reduce(x, "AVG", len(x.shape) - 1)
        centered = self.op(x, mean, "SUB")
        var = self.reduce(self.mul(centered, centered), "AVG", len(x.shape) - 1)
        denom = self.n.add_unary(
            self.add(var, self.const([self.c.architecture.epsilon], np.float32)),
            trt.UnaryOperation.SQRT,
        ).get_output(0)
        return self.op(centered, denom, "DIV")

    def mm(self, a, b, low=False, transpose=False):
        dtype = self.low if low else trt.float32
        a, b = self.cast(a, dtype), self.cast(b, dtype)
        layer = self.n.add_matrix_multiply(
            a,
            trt.MatrixOperation.NONE,
            b,
            trt.MatrixOperation.TRANSPOSE if transpose else trt.MatrixOperation.NONE,
        )
        return self.cast(layer.get_output(0), trt.float32)

    def linear(self, x, name, low=False):
        weight = self.const(self.w[name])
        weight = self.reshape(weight, (1,) * (len(x.shape) - 2) + tuple(weight.shape))
        return self.mm(x, weight, low)

    def gather(self, table, indices):
        return self.n.add_gather(self.const(self.w[table]), indices, 0).get_output(0)

    def concat(self, values, axis):
        layer = self.n.add_concatenation(values)
        layer.axis = axis
        return layer.get_output(0)

    def batch_gather(self, value, indices, axis):
        layer = self.n.add_gather(value, indices, axis)
        layer.num_elementwise_dims = 1
        return layer.get_output(0)

    def slice_last(self, x, start, count):
        return self.n.add_gather(
            x, self.const(np.arange(start, start + count), np.int32), 2
        ).get_output(0)

    def project(self, x, layer):
        a = self.c.architecture
        width, qwidth = a.num_heads * a.value_dim, a.num_heads * a.attention_dim
        reduced = self.c.bf16_projections == "all" and layer not in self.c.fp32_block_indices
        projected = self.silu(self.linear(self.norm(x), f"layers.{layer}.uvqk", low=reduced))
        u, v = self.slice_last(projected, 0, width), self.slice_last(projected, width, width)
        q = self.slice_last(projected, 2 * width, qwidth)
        key = self.slice_last(projected, 2 * width + qwidth, qwidth)

        def heads(value, dim):
            return self.transpose(
                self.reshape(value, (-1, x.shape[1], a.num_heads, dim)), (0, 2, 1, 3)
            )

        return u, heads(v, a.value_dim), heads(q, a.attention_dim), heads(key, a.attention_dim)

    def attention(self, x, u, q, key, value, bias, mask, layer):
        a = self.c.architecture
        reduced = layer not in self.c.fp32_block_indices
        logits = self.add(self.mm(q, key, low=reduced, transpose=True), bias)
        epi = self.low if reduced and self.c.bf16_attention_epilogue else trt.float32
        weights = self.mul(
            self.silu(self.cast(logits, epi)),
            self.cast(self.const([1 / a.max_sequence_length], np.float32), epi),
        )
        attended = self.mm(self.mul(weights, self.cast(mask, epi)), value, low=reduced)
        attended = self.reshape(
            self.transpose(attended, (0, 2, 1, 3)), (-1, x.shape[1], a.num_heads * a.value_dim)
        )
        return self.add(
            x,
            self.add(
                self.linear(
                    self.mul(u, self.norm(attended)),
                    f"layers.{layer}.output_weight",
                    low=reduced and self.c.bf16_projections in ("output", "all"),
                ),
                self.const(self.w[f"layers.{layer}.output_bias"]),
            ),
        )

    def unit(self, value):
        squared = self.reduce(self.mul(value, value), "SUM", len(value.shape) - 1)
        length = self.n.add_unary(squared, trt.UnaryOperation.SQRT).get_output(0)
        length = self.op(length, self.const([self.c.architecture.l2_epsilon], np.float32), "MAX")
        return self.op(value, length, "DIV")

    def buckets(self, delta):
        value = self.n.add_unary(delta, trt.UnaryOperation.ABS).get_output(0)
        value = self.cast(self.op(value, self.const([1], np.int64), "MAX"), trt.float32)
        value = self.n.add_unary(value, trt.UnaryOperation.LOG).get_output(0)
        value = self.op(value, self.const([0.301], np.float32), "DIV")
        return self.cast(
            self.op(value, self.const([self.c.architecture.time_buckets], np.float32), "MIN"),
            trt.int32,
        )

    def prepare_history(self, inp):
        a, s = self.c.architecture, self.c.serving
        t, d, h = s.history_length, a.hidden_size, a.num_heads
        positions = self.const(np.arange(t, dtype=np.int64)[None, :])
        ids, lengths = inp["history_ids"], inp["history_lengths"]
        last = self.op(lengths, self.const([1], np.int64), "SUB")
        valid = self.cast(self.less(positions, last), trt.float32)
        raw = self.gather("item_embeddings", ids)
        x = self.add(
            self.mul(raw, self.const([np.sqrt(d)], np.float32)),
            self.gather("position_embeddings", positions),
        )
        nonzero = self.cast(self.less(self.const([0], np.int64), ids), trt.float32)
        x = self.mul(x, self.reshape(nonzero, (-1, t, 1)))
        query_x = self.batch_gather(x, last, 1)
        mask = self.mul(
            self.const(np.tril(np.ones((1, 1, t, t), np.float32))),
            self.reshape(valid, (-1, 1, 1, t)),
        )
        timestamps = inp["history_timestamps"]
        before = self.n.add_gather(timestamps, self.const(np.arange(t), np.int32), 1).get_output(0)
        after = self.n.add_gather(
            timestamps, self.const(np.arange(1, t + 1), np.int32), 1
        ).get_output(0)
        delta = self.op(self.reshape(after, (-1, t, 1)), self.reshape(before, (-1, 1, t)), "SUB")
        buckets = self.buckets(delta)
        self_bucket = self.buckets(
            self.op(self.batch_gather(after, last, 1), self.batch_gather(before, last, 1), "SUB")
        )
        relpos = a.max_sequence_length - 1 + np.arange(t)[None, :] - np.arange(t)[:, None]
        keys, values, biases = [], [], []
        for layer in range(a.num_layers):
            prefix = f"layers.{layer}."
            bias = self.add(
                self.gather(prefix + "time_bias", buckets),
                self.gather(prefix + "position_bias", self.const(relpos, np.int32)),
            )
            query_bias = self.batch_gather(bias, last, 1)
            diagonal = self.add(
                self.gather(prefix + "time_bias", self_bucket),
                self.const(
                    [self.w[prefix + "position_bias"][a.max_sequence_length - 1]], np.float32
                ),
            )
            qb = self.concat([query_bias, self.reshape(diagonal, (-1, 1, 1))], 2)
            biases.append(self.reshape(qb, (-1, 1, 1, t + 1)))
            u, v, q, key = self.project(x, layer)
            keep = self.reshape(valid, (-1, 1, t, 1))
            keys.append(self.reshape(self.mul(key, keep), (-1, 1, h, t, a.attention_dim)))
            values.append(self.reshape(self.mul(v, keep), (-1, 1, h, t, a.value_dim)))
            if layer + 1 < a.num_layers:
                x = self.attention(x, u, q, key, v, self.reshape(bias, (-1, 1, t, t)), mask, layer)
        return [
            self.concat(keys, 1),
            self.concat(values, 1),
            self.cast(valid, trt.int32),
            query_x,
            self.concat(biases, 1),
        ]

    def score_candidates(self, inp):
        a, s = self.c.architecture, self.c.serving
        t, h = s.history_length, a.num_heads
        mask = self.reshape(self.cast(inp["history_mask"], trt.float32), (-1, 1, 1, t))
        ones = self.add(
            self.mul(self.reduce(mask, "SUM", 3), self.const([0], np.float32)),
            self.const([1], np.float32),
        )
        mask = self.concat([mask, ones], 3)
        x = inp["query_x"]
        for layer in range(a.num_layers):
            index = self.const([layer], np.int32)
            key = self.n.add_gather(inp["history_k_b"], index, 1).get_output(0)
            val = self.n.add_gather(inp["history_v_b"], index, 1).get_output(0)
            bias = self.n.add_gather(inp["query_bias"], index, 1).get_output(0)
            key = self.reshape(key, (-1, h, t, a.attention_dim))
            val = self.reshape(val, (-1, h, t, a.value_dim))
            bias = self.reshape(bias, (-1, 1, 1, t + 1))
            u, self_val, q, self_key = self.project(x, layer)
            x = self.attention(
                x,
                u,
                q,
                self.concat([key, self_key], 2),
                self.concat([val, self_val], 2),
                bias,
                mask,
                layer,
            )
        score = self.reduce(self.mul(self.unit(x), inp["candidate_emb"]), "SUM", 2, False)
        valid = self.equal(inp["valid_mask"], self.const([1], np.int32))
        scores = self.select(valid, score, self.const([0], np.float32))
        masked = self.select(valid, score, self.const([-1e30], np.float32))
        best = self.reduce(masked, "MAX", 1)
        index = self.const(np.arange(s.candidates, dtype=np.int64)[None, :])
        selected = self.op(valid, self.equal(masked, best), "AND")
        picked = self.reduce(
            self.select(selected, index, self.const([s.candidates], np.int64)), "MIN", 1
        )
        picked = self.select(
            self.less(picked, self.const([s.candidates], np.int64)),
            picked,
            self.const([-1], np.int64),
        )
        return [scores, picked]

    def define(self, stage):
        batch = max(self.c.serving.min_batch_size, self.c.serving.opt_batch_size or 1)
        schema = next(
            s for s in self.c.runtime(batch, self.precision)["stages"] if s["name"] == stage
        )
        dtype = {"int64": trt.int64, "int32": trt.int32, "float32": trt.float32}
        inp = {
            s["name"]: self.n.add_input(s["name"], dtype[s["dtype"]], s["shape"])
            for s in schema["inputs"]
        }
        if stage == "history_prepare":
            outputs = self.prepare_history(inp)
        elif stage == "request_prepare":
            emb = self.unit(self.gather("item_embeddings", inp["candidate_ids"]))
            positions = self.const(np.arange(self.c.serving.candidates, dtype=np.int64)[None, :])
            outputs = [emb, self.cast(self.less(positions, inp["num_candidates"]), trt.int32)]
        else:
            outputs = self.score_candidates(inp)
        for tensor, spec in zip(outputs, schema["outputs"], strict=True):
            tensor = self.n.add_identity(tensor).get_output(0)
            tensor.name = spec["name"]
            self.n.mark_output(tensor)
        return self
