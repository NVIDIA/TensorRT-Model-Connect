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
        self.fast_bf16 = precision == "bf16" and config.bf16_policy == "fast"
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
            self.add(var, self.const([self.c.epsilon], np.float32)), trt.UnaryOperation.SQRT
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

    def project(self, raw, positions, tokens):
        c = self.c
        d, h = c.hidden_size, c.num_heads
        x = self.add(
            self.mul(raw, self.const([np.sqrt(d)], np.float32)),
            self.gather("position_embeddings", positions),
        )
        uvqk = self.silu(self.linear(self.norm(x), "uvqk"))
        parts = [self.slice_last(uvqk, i * d, d) for i in range(4)]

        def heads(a):
            return self.transpose(self.reshape(a, (-1, tokens, h, d // h)), (0, 2, 1, 3))

        return x, parts[0], heads(parts[1]), heads(parts[2]), heads(parts[3])

    def slice_last(self, x, start, size):
        layer = self.n.add_slice(x, (0, 0, start), (1, int(x.shape[1]), size), (1, 1, 1))
        shape = self.n.add_shape(x).get_output(0)
        batch = self.n.add_slice(shape, (0,), (1,), (1,)).get_output(0)
        dimensions = self.concat([batch, self.const([int(x.shape[1]), size], np.int64)], 0)
        layer.set_input(2, dimensions)
        return layer.get_output(0)

    def define(self, stage):
        c = self.c
        d, t, r, k = c.hidden_size, c.history_length, c.decode_steps, c.candidates
        dtype = {"float32": trt.float32, "int32": trt.int32, "int64": trt.int64}
        specs = c.runtime(1, "fp32")["stages"]
        schema = next(s for s in specs if s["name"] == stage)
        inp = {
            s["name"]: self.n.add_input(s["name"], dtype[s["dtype"]], s["shape"])
            for s in schema["inputs"]
        }
        if stage == "history_prepare":
            positions = self.const(np.arange(t, dtype=np.int64)[None, :])
            _, _, v, _, key = self.project(
                self.gather("item_embeddings", inp["history_ids"]), positions, t
            )
            mask = self.cast(self.less(positions, inp["history_lengths"]), trt.int32)
            outputs = [key, v, mask]
        elif stage == "request_prepare":
            raw = self.gather("item_embeddings", inp["candidate_ids"])
            valid = self.cast(
                self.less(self.const(np.arange(k, dtype=np.int64)[None, :]), inp["num_candidates"]),
                trt.int32,
            )
            outputs = [raw, valid, inp["candidate_ids"]]
        else:
            hist_mask = self.cast(inp["history_mask"], trt.float32)
            lengths = self.cast(self.reduce(inp["history_mask"], "SUM", 1), trt.int64)
            step = self.reshape(inp["step_index"], (1, 1))
            x, u, cv, q, ck = self.project(inp["candidate_emb"], self.add(lengths, step), k)
            _, _, fv, _, fk = self.project(
                inp["frozen_raw"],
                self.add(lengths, self.const(np.arange(r, dtype=np.int64)[None, :])),
                r,
            )
            keys = self.concat([inp["history_k_b"], fk, ck], 2)
            values = self.concat([inp["history_v_b"], fv, cv], 2)
            history = self.reshape(hist_mask, (-1, 1, 1, t))
            previous = self.reshape(
                self.cast(
                    self.less(self.const(np.arange(r, dtype=np.int64)[None, :]), step), trt.float32
                ),
                (1, 1, 1, r),
            )
            # Broadcast masks explicitly using the query/candidate axes.
            query_ones = self.const(np.ones((1, 1, k, 1), np.float32))
            history = self.mul(history, query_ones)
            previous = self.mul(previous, query_ones)
            # Include a dynamic batch axis so concatenation has equal leading dims.
            batch_ones = self.add(
                self.mul(
                    self.reshape(self.cast(lengths, trt.float32), (-1, 1, 1, 1)),
                    self.const([0], np.float32),
                ),
                self.const([1], np.float32),
            )
            previous = self.mul(previous, batch_ones)
            target = self.mul(self.const(np.eye(k, dtype=np.float32)[None, None, :, :]), batch_ones)
            mask = self.concat([history, previous, target], 3)

            def attention_scores(low):
                logits = self.mm(q, keys, low=low, transpose=True)
                # Keep the complete SiLU/scale/mask epilogue in one explicit
                # dtype. TensorRT owns whether this graph becomes fused HSTU.
                epi = self.low if low and c.bf16_attention_epilogue else trt.float32
                logits = self.cast(logits, epi)
                scale = self.cast(self.const([1 / (t + r + k)], np.float32), epi)
                attention = self.mul(self.silu(logits), scale)
                attended = self.mm(self.mul(attention, self.cast(mask, epi)), values, low=low)
                attended = self.reshape(self.transpose(attended, (0, 2, 1, 3)), (-1, k, d))
                encoded = self.add(
                    self.add(
                        self.linear(
                            self.mul(u, self.norm(attended)), "output_weight", low=self.fast_bf16
                        ),
                        self.const(self.w["output_bias"]),
                    ),
                    x,
                )
                return self.reshape(
                    self.add(
                        self.linear(self.norm(encoded), "head_weight"),
                        self.const(self.w["head_bias"]),
                    ),
                    (-1, k),
                )

            raw_scores = attention_scores(low=True)
            # Strict mode retains an FP32 selector for greedy slate stability.
            # Fast mode uses one BF16 attention/output-projection path for both
            # scores and selections, accepting bounded ranking differences.
            decision_scores = (
                attention_scores(low=False)
                if self.precision == "bf16" and not self.fast_bf16
                else raw_scores
            )
            # Equal item IDs denote the same mathematical query. Canonicalize the
            # score to its first occurrence before stable argmax: GEMM row tiling
            # may otherwise perturb exact duplicates by an FP32 ULP.
            ids = inp["candidate_ids_context"]
            same = self.equal(self.reshape(ids, (-1, k, 1)), self.reshape(ids, (-1, 1, k)))
            canonical = self.reduce(
                self.select(
                    same,
                    self.const(np.arange(k, dtype=np.int64)[None, None, :]),
                    self.const([k], np.int64),
                ),
                "MIN",
                2,
                False,
            )
            gather = self.n.add_gather(raw_scores, canonical, 1)
            gather.num_elementwise_dims = 1
            raw_scores = gather.get_output(0)
            if self.precision == "bf16" and not self.fast_bf16:
                decision = self.n.add_gather(decision_scores, canonical, 1)
                decision.num_elementwise_dims = 1
                decision_scores = decision.get_output(0)
            else:
                decision_scores = raw_scores
            eligible = self.op(
                self.equal(inp["placed_mask"], self.const([0], np.int32)),
                self.equal(inp["valid_mask"], self.const([1], np.int32)),
                "AND",
            )
            masked = self.select(eligible, decision_scores, self.const([-1e30], np.float32))
            maximum = self.reduce(masked, "MAX", 1)
            indices = self.const(np.arange(k, dtype=np.int64)[None, :])
            matches = self.op(eligible, self.equal(masked, maximum), "AND")
            best = self.reduce(self.select(matches, indices, self.const([k], np.int64)), "MIN", 1)
            exists = self.less(best, self.const([k], np.int64))
            picked = self.select(exists, best, self.const([-1], np.int64))
            chosen = self.equal(indices, picked)
            scores = self.select(chosen, raw_scores, inp["scores"])
            placed = self.select(chosen, self.const([1], np.int32), inp["placed_mask"])
            chosen_emb = self.reduce(
                self.mul(
                    inp["candidate_emb"], self.reshape(self.cast(chosen, trt.float32), (-1, k, 1))
                ),
                "SUM",
                1,
            )
            slot = self.equal(self.const(np.arange(r, dtype=np.int64)[None, :]), step)
            slot = self.op(slot, exists, "AND")
            frozen = self.select(self.reshape(slot, (-1, r, 1)), chosen_emb, inp["frozen_raw"])
            count = self.add(inp["placed_count"], self.reshape(self.cast(exists, trt.int64), (-1,)))
            outputs = [scores, placed, frozen, count, self.reshape(picked, (-1,))]
        for value, spec in zip(outputs, schema["outputs"], strict=True):
            value = self.n.add_identity(value).get_output(0)
            value.name = spec["name"]
            self.n.mark_output(value)
        return self
