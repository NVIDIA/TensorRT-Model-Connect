# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Strongly typed TensorRT lowering of the public HSTU reference."""

import numpy as np
import tensorrt as trt
from .checkpoint import BLOCK, HEAD, POS, POST, PRE


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

    def norm(self, x, weight, bias, eps=1e-5):
        x = self.cast(x, trt.float32)
        centered = self.op(x, self.reduce(x, "AVG", len(x.shape) - 1), "SUB")
        variance = self.reduce(self.mul(centered, centered), "AVG", len(x.shape) - 1)
        denom = self.n.add_unary(
            self.add(variance, self.const([eps], np.float32)), trt.UnaryOperation.SQRT
        ).get_output(0)
        return self.add(
            self.mul(self.op(centered, denom, "DIV"), self.const(weight)), self.const(bias)
        )

    def ln(self, x, prefix):
        return self.norm(x, self.w[prefix + "weight"], self.w[prefix + "bias"])

    def dense(self, x, prefix):
        weight = self.const(self.w[prefix + "weight"])
        weight = self.reshape(weight, (1,) * (len(x.shape) - 2) + tuple(weight.shape))
        return self.add(self.mm(x, weight, transpose=True), self.const(self.w[prefix + "bias"]))

    def swish_ln(self, x, prefix):
        return self.mul(
            x, self.n.add_activation(self.ln(x, prefix), trt.ActivationType.SIGMOID).get_output(0)
        )

    def mlp(self, x, prefix):
        return self.ln(
            self.dense(self.swish_ln(self.dense(x, prefix + "0."), prefix + "1."), prefix + "2."),
            prefix + "3.",
        )

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

    def slice_last(self, x, start, count):
        return self.n.add_gather(
            x, self.const(np.arange(start, start + count), np.int32), 2
        ).get_output(0)

    def reduced(self, layer):
        return self.precision == "bf16" and layer not in self.c.fp32_block_indices

    def project(self, x, layer):
        p = BLOCK + str(layer) + "."
        normalized = self.norm(
            x, self.w[p + "_input_norm_weight"], self.w[p + "_input_norm_bias"], 1e-6
        )
        projection = self.add(
            self.linear(normalized, p + "_uvqk_weight", low=self.reduced(layer)),
            self.const(self.w[p + "_uvqk_beta"]),
        )
        u, v, q, key = [self.slice_last(projection, i * 512, 512) for i in range(4)]

        def heads(value):
            return self.transpose(self.reshape(value, (-1, x.shape[1], 4, 128)), (0, 2, 1, 3))

        return self.silu(u), heads(v), heads(q), heads(key)

    def weights(self, logits, mask, layer):
        n = self.c.serving.history_length + self.c.serving.candidates + 1
        dtype = self.low if self.reduced(layer) else trt.float32
        logits = self.mul(
            self.cast(logits, dtype), self.cast(self.const([128**-0.5], np.float32), dtype)
        )
        weights = self.mul(self.silu(logits), self.cast(self.const([1 / n], np.float32), dtype))
        return self.mul(weights, self.cast(mask, dtype))

    def finish(self, x, u, attended, layer):
        p = BLOCK + str(layer) + "."
        attended = self.reshape(self.transpose(attended, (0, 2, 1, 3)), (-1, x.shape[1], 512))
        normalized = self.norm(
            attended, self.w[p + "_output_norm_weight"], self.w[p + "_output_norm_bias"], 1e-6
        )
        output = self.concat([u, attended, self.mul(u, normalized)], 2)
        return self.add(x, self.linear(output, p + "_output_weight", low=self.reduced(layer)))

    def preprocess(self, item, category, actions):
        return self.add(
            self.add(
                self.mlp(item, PRE + "_content_embedding_mlp."),
                self.mlp(category, PRE + "_additional_embedding_mlp."),
            ),
            self.mlp(actions, PRE + "_action_embedding_mlp."),
        )

    def position(self, x, indices, timestamps, query_time):
        delta = self.cast(self.op(query_time, timestamps, "SUB"), trt.float32)
        delta = self.op(
            self.op(delta, self.const([1e-6], np.float32), "MAX"),
            self.const([60], np.float32),
            "DIV",
        )
        bucket = self.n.add_unary(delta, trt.UnaryOperation.SQRT).get_output(0)
        bucket = self.cast(self.op(bucket, self.const([2048], np.float32), "MIN"), trt.int32)
        return self.add(
            self.mul(x, self.const([np.sqrt(512)], np.float32)),
            self.add(
                self.gather(POS + "_position_embeddings_weight", indices),
                self.gather(POS + "_timestamp_embeddings_weight", bucket),
            ),
        )

    def prepare_history(self, inp):
        t = self.c.serving.history_length
        lengths = inp["history_lengths"]
        bits = self.const(
            self.w[PRE + "_action_encoder._combined_action_weights"]
            .astype(np.int32)
            .reshape(1, 1, 5)
        )
        action = self.reshape(inp["history_actions"], (-1, t, 1))
        quotient = self.op(action, bits, "FLOOR_DIV")
        exploded = self.op(
            quotient,
            self.mul(
                self.op(quotient, self.const([2], np.int32), "FLOOR_DIV"), self.const([2], np.int32)
            ),
            "SUB",
        )
        exploded = self.cast(self.less(self.const([0], np.int32), exploded), trt.float32)
        actions = self.reshape(
            self.mul(
                self.reshape(exploded, (-1, t, 5, 1)),
                self.const(
                    self.w[PRE + "_action_encoder._action_embedding_table"].reshape(1, 1, 5, 8)
                ),
            ),
            (-1, t, 40),
        )
        x = self.preprocess(
            inp["history_item_embeddings"], inp["history_category_embeddings"], actions
        )
        enough = self.cast(
            self.less(self.const([19], np.int32), self.add(lengths, inp["num_candidates"])),
            trt.float32,
        )
        contextual = self.mul(inp["user_embeddings"], self.reshape(enough, (-1, 1, 1)))
        context = self.add(
            self.mm(contextual, self.const(self.w[PRE + "_batched_contextual_linear_weights"])),
            self.const(self.w[PRE + "_batched_contextual_linear_bias"]),
        )
        x = self.concat([context, x], 1)
        zero = self.mul(lengths, self.const([0], np.int32))
        positions = self.op(
            self.add(lengths, self.const([1], np.int32)),
            self.const(np.arange(t)[None], np.int32),
            "SUB",
        )
        positions = self.op(
            self.op(positions, self.const([1], np.int32), "MAX"),
            self.const([8191], np.int32),
            "MIN",
        )
        positions = self.concat([zero, positions], 1)
        timestamps = self.concat([self.cast(zero, trt.int64), inp["history_timestamps"]], 1)
        x = self.position(x, positions, timestamps, inp["query_timestamp"])
        valid = self.cast(
            self.less(
                self.const(np.arange(t + 1)[None], np.int32),
                self.add(lengths, self.const([1], np.int32)),
            ),
            trt.float32,
        )
        mask = self.mul(
            self.const(np.tril(np.ones((1, 1, t + 1, t + 1), np.float32))),
            self.reshape(valid, (-1, 1, 1, t + 1)),
        )
        keys, values = [], []
        keep = self.reshape(valid, (-1, 1, t + 1, 1))
        for layer in range(5):
            u, v, q, key = self.project(x, layer)
            keys.append(self.reshape(self.mul(key, keep), (-1, 1, 4, t + 1, 128)))
            values.append(self.reshape(self.mul(v, keep), (-1, 1, 4, t + 1, 128)))
            if layer < 4:
                weights = self.weights(
                    self.mm(q, key, low=self.reduced(layer), transpose=True), mask, layer
                )
                x = self.finish(x, u, self.mm(weights, v, low=self.reduced(layer)), layer)
        candidate_mask = self.cast(
            self.less(
                self.const(np.arange(self.c.serving.candidates)[None], np.int32),
                inp["num_candidates"],
            ),
            trt.int32,
        )
        return [
            self.concat(keys, 1),
            self.concat(values, 1),
            self.cast(valid, trt.int32),
            candidate_mask,
            inp["query_timestamp"],
        ]

    def prepare_request(self, inp):
        item, category = inp["candidate_item_embeddings"], inp["candidate_category_embeddings"]
        zeros = self.mul(self.slice_last(item, 0, 40), self.const([0], np.float32))
        actions = self.add(
            zeros, self.const(self.w[PRE + "_action_encoder._target_action_embedding_table"])
        )
        x = self.preprocess(item, category, actions)
        emb = self.mlp(self.concat([item, category], 2), "_item_embedding_mlp.")
        return [x, emb]

    def score_candidates(self, inp):
        t, c = self.c.serving.history_length, self.c.serving.candidates
        x = self.add(
            self.mul(inp["candidate_x"], self.const([np.sqrt(512)], np.float32)),
            self.const(
                (
                    self.w[POS + "_position_embeddings_weight"][1]
                    + self.w[POS + "_timestamp_embeddings_weight"][0]
                ).reshape(1, 1, 512)
            ),
        )
        mask = self.reshape(self.cast(inp["history_mask"], trt.float32), (-1, 1, 1, t + 1))
        for layer in range(5):
            index = self.const([layer], np.int32)
            key = self.reshape(
                self.n.add_gather(inp["history_k_b"], index, 1).get_output(0), (-1, 4, t + 1, 128)
            )
            val = self.reshape(
                self.n.add_gather(inp["history_v_b"], index, 1).get_output(0), (-1, 4, t + 1, 128)
            )
            u, self_val, q, self_key = self.project(x, layer)
            weights = self.weights(
                self.mm(q, key, low=self.reduced(layer), transpose=True), mask, layer
            )
            # Targets see the immutable prefix and themselves, never other targets.
            self_logit = self.reduce(self.mul(q, self_key), "SUM", 3)
            self_weight = self.weights(self_logit, self.const([1], np.float32), layer)
            attended = self.add(
                self.mm(weights, val, low=self.reduced(layer)),
                self.mul(self.cast(self_weight, trt.float32), self_val),
            )
            x = self.finish(x, u, attended, layer)
        timestamp = self.cast(self.reshape(inp["query_timestamp_b"], (-1, 1, 1)), trt.float32)
        units = self.n.add_unary(
            self.op(timestamp, self.const(self.w[POST + "_period_units"].reshape(1, 1, 2)), "DIV"),
            trt.UnaryOperation.FLOOR,
        ).get_output(0)
        periods = self.const(self.w[POST + "_units_per_period"].reshape(1, 1, 2))
        remainder = self.op(
            units,
            self.mul(
                self.n.add_unary(
                    self.op(units, periods, "DIV"), trt.UnaryOperation.FLOOR
                ).get_output(0),
                periods,
            ),
            "SUB",
        )
        phase = self.mul(self.op(remainder, periods, "DIV"), self.const([2 * 3.14], np.float32))
        cos = self.n.add_unary(phase, trt.UnaryOperation.COS).get_output(0)
        sin = self.n.add_unary(phase, trt.UnaryOperation.SIN).get_output(0)
        features = self.reshape(
            self.concat([self.reshape(cos, (-1, 1, 2, 1)), self.reshape(sin, (-1, 1, 2, 1))], 3),
            (-1, 1, 4),
        )
        features = self.add(
            self.mul(self.slice_last(x, 0, 4), self.const([0], np.float32)), features
        )
        user = self.ln(
            self.dense(self.concat([x, features], 2), POST + "_time_feature_combiner."),
            POST + "_layer_norm.",
        )
        head = self.swish_ln(
            self.dense(self.mul(user, inp["candidate_emb"]), HEAD + "0."), HEAD + "1."
        )
        score = self.reshape(
            self.n.add_activation(
                self.dense(head, HEAD + "2."), trt.ActivationType.SIGMOID
            ).get_output(0),
            (-1, c),
        )
        return [
            self.select(
                self.equal(inp["candidate_mask"], self.const([1], np.int32)),
                score,
                self.const([0], np.float32),
            )
        ]

    def define(self, stage):
        batch = max(self.c.serving.min_batch_size, self.c.serving.opt_batch_size)
        schema = next(
            s for s in self.c.runtime(batch, self.precision)["stages"] if s["name"] == stage
        )
        dtype = {"int64": trt.int64, "int32": trt.int32, "float32": trt.float32}
        inp = {
            s["name"]: self.n.add_input(s["name"], dtype[s["dtype"]], s["shape"])
            for s in schema["inputs"]
        }
        functions = {
            "history_prepare": self.prepare_history,
            "request_prepare": self.prepare_request,
            "step": self.score_candidates,
        }
        for tensor, spec in zip(functions[stage](inp), schema["outputs"], strict=True):
            tensor = self.n.add_identity(tensor).get_output(0)
            tensor.name = spec["name"]
            self.n.mark_output(tensor)
        return self
