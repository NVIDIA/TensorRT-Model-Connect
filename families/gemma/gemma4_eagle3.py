# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gemma4 single-layer EAGLE3 math, using TensorRT operators only."""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from . import graph_ops
from .checkpoint_mapper import _load_tensor
from .gemma4_text import DenseTextGraph, attention_geometry, rope_tables


class EagleDraftGraph(DenseTextGraph):
    """Recompute committed draft context and unroll the trained recurrent head.

    Target feature i pairs with token i+1, but uses RoPE position i. A
    single-layer draft's K/V depend directly on those per-token inputs, so
    recomputation needs no speculative mutable cache and no rejected history.
    """

    def __init__(self, network, raw, readers, precision, capacity, *, branch_tree=False):
        if (raw.get("architectures") != ["Gemma4Eagle3Model"]
                or raw.get("num_hidden_layers") != 1 or raw.get("draft_num_hidden_layers") != 1
                or raw.get("ttt_length") != 7 or raw.get("layer_types") != ["full_attention"]
                or raw.get("enable_moe_block") or raw.get("hidden_size_per_layer_input", 0)
                or raw.get("attention_bias") or raw.get("num_kv_shared_layers", 0)):
            raise NotImplementedError("Native Gemma EAGLE3 requires the dense single-layer TTT7 head")
        self.net, self.raw, self.readers = network, raw, readers
        self.branch_tree = branch_tree
        self.capacity, self.buffers, self.weights = capacity, [], {}
        self.dtype = {"fp16": trt.float16, "bf16": trt.bfloat16, "fp32": trt.float32}[precision]
        self.storage = np.float16 if precision == "fp16" else np.float32
        self.target_layers = list(raw["target_layer_ids"])
        if not self.target_layers or len(set(self.target_layers)) != len(self.target_layers):
            raise ValueError("Gemma EAGLE3 requires distinct target feature layers")
        if not 1 < capacity <= int(raw["max_position_embeddings"]):
            raise ValueError("Gemma EAGLE3 capacity exceeds the checkpoint")

    def weight(self, name, *, dtype=None):
        key = name, dtype
        if key not in self.weights:
            value = _load_tensor(self.readers, name)
            self.weights[key] = self.const(
                value.astype(self.storage if dtype is None else np.float32),
                self.dtype if dtype is None else dtype)
        return self.weights[key]

    def linear(self, value, name):
        weight = self.cast(self.weight(name + ".weight"), value.dtype)
        return self.net.add_matrix_multiply(
            value, trt.MatrixOperation.NONE, weight, trt.MatrixOperation.TRANSPOSE).get_output(0)

    def build(self):
        n, raw = self.net, self.raw
        hidden, heads = int(raw["hidden_size"]), int(raw["num_attention_heads"])
        dim, kv_heads = attention_geometry(raw, "full_attention")
        tokens = n.add_input("draft_tokens", trt.int32, (-1,))
        positions = n.add_input("draft_positions", trt.int32, (-1,))
        features = n.add_input("target_features", trt.float32,
                               (-1, len(self.target_layers) * hidden))
        past = self.cast(n.add_shape(tokens).get_output(0), trt.int32)
        last = self.op(past, self.const(np.array([1], np.int32)), trt.ElementWiseOperation.SUB)

        def joined(values, axis=0):
            layer = n.add_concatenation(values)
            layer.axis = axis
            return layer.get_output(0)

        prefix, attn = "layers.0", "layers.0.self_attn"
        embedding = self.weight("embed_tokens.weight")
        fused = self.linear(features, "fc")  # FP32 through normalization and residual.
        embed = self.scale(n.add_gather(embedding, tokens, 0).get_output(0), hidden**0.5)
        source = joined([self.norm(embed, prefix + ".input_layernorm"),
                         self.cast(self.norm(fused, prefix + ".hidden_norm"), self.dtype)], 1)
        state = n.add_gather(fused, last, 0).get_output(0)
        query_source = n.add_gather(source, last, 0).get_output(0)
        cos, sin = [self.const(t.astype(self.storage), self.dtype)
                    for t in rope_tables(raw, "full_attention", self.capacity)]

        def key_value(value, pos):
            key_raw = self.linear(value, attn + ".k_proj")
            value_raw = (key_raw if raw.get("attention_k_eq_v")
                         else self.linear(value, attn + ".v_proj"))
            key = self.head_norm(key_raw, kv_heads, dim, attn + ".k_norm")
            key = graph_ops.add_apply_rope_native(
                n, key, kv_heads, dim, cos, sin, pos, dim, sequence_length=None)
            return key, self.head_norm(value_raw, kv_heads, dim)

        key, value = key_value(source, positions)
        zero_mask = self.const(np.zeros((1, 1, 1, 1), np.float32))
        token = n.add_gather(tokens, last, 0).get_output(0)
        rows, states, proposals = [], [], []
        if self.branch_tree:
            # Score one particular ancestry. Only its causal prefix up to
            # branch_depth is used; siblings never share a speculative state.
            branch_tokens = n.add_input("branch_tokens", trt.int32, (5,))
            depth = n.add_input("branch_depth", trt.int32, (1,))

        def project(state):
            logits = self.linear(self.norm(state, "norm"), "lm_head")
            if raw.get("final_logit_softcapping"):
                logits = graph_ops.add_tanh_softcap(n, logits, float(raw["final_logit_softcapping"]),
                                                   scalar_shape=(1, 1))
            return logits

        for row in range(6 if self.branch_tree else raw["ttt_length"]):
            # Tail rows past capacity are discarded, but all gathers remain valid.
            pos = self.op(last, self.const(np.array([row], np.int32)), trt.ElementWiseOperation.SUM)
            pos = self.op(pos, self.const(np.array([self.capacity - 1], np.int32)),
                          trt.ElementWiseOperation.MIN)
            if row:
                if self.branch_tree:
                    index = self.const(np.array([row - 1], np.int32))
                    token = n.add_gather(branch_tokens, index, 0).get_output(0)
                embed = self.scale(n.add_gather(embedding, token, 0).get_output(0), hidden**0.5)
                query_source = joined([self.norm(embed, prefix + ".input_layernorm"),
                    self.cast(self.norm(state, prefix + ".hidden_norm"), self.dtype)], 1)
                next_key, next_value = key_value(query_source, pos)
                key, value = joined([key, next_key]), joined([value, next_value])
            query = self.head_norm(self.linear(query_source, attn + ".q_proj"),
                                   heads, dim, attn + ".q_norm")
            query = graph_ops.add_apply_rope_native(
                n, query, heads, dim, cos, sin, pos, dim, sequence_length=None)
            attended = self.attention(query, key, value, heads, kv_heads, dim, zero_mask)
            attended = self.norm(self.linear(attended, attn + ".o_proj"),
                                 prefix + ".post_attention_layernorm")
            state = self.op(state, self.cast(attended, trt.float32), trt.ElementWiseOperation.SUM)
            normalized = self.norm(state, prefix + ".pre_feedforward_layernorm")
            gate, up = (self.linear(normalized, prefix + ".mlp." + name)
                        for name in ("gate_proj", "up_proj"))
            product = self.op(self.activation(gate), up, trt.ElementWiseOperation.PROD)
            product = self.norm(self.linear(product, prefix + ".mlp.down_proj"),
                                prefix + ".post_feedforward_layernorm")
            state = self.op(state, product, trt.ElementWiseOperation.SUM)
            state = self.op(state,
                self.reshape(self.weight(prefix + ".layer_scalar", dtype=trt.float32), (1, 1)),
                trt.ElementWiseOperation.PROD)
            states.append(state)
            if self.branch_tree:
                continue
            logits = project(state)
            token = self.reshape(n.add_topk(logits, trt.TopKOperation.MAX, 1, 2).get_output(1), (1,))
            rows.append(logits)
            proposals.append(token)

        if self.branch_tree:
            # One vocabulary projection per visited parent. Normalize over
            # the whole vocabulary, not just the ten retained children.
            logits = self.cast(project(n.add_gather(joined(states), depth, 0).get_output(0)),
                               trt.float32)
            best = n.add_topk(logits, trt.TopKOperation.MAX, 10, 2)
            maximum = n.add_reduce(logits, trt.ReduceOperation.MAX, 2, True).get_output(0)
            centered = self.op(logits, maximum, trt.ElementWiseOperation.SUB)
            exp = n.add_unary(centered, trt.UnaryOperation.EXP).get_output(0)
            total = n.add_reduce(exp, trt.ReduceOperation.SUM, 2, True).get_output(0)
            log_total = n.add_unary(total, trt.UnaryOperation.LOG).get_output(0)
            scores = self.op(self.op(best.get_output(0), maximum, trt.ElementWiseOperation.SUB),
                             log_total, trt.ElementWiseOperation.SUB)
            for name, tensor in (("branch_ids", best.get_output(1)), ("branch_log_probs", scores)):
                tensor.name = name
                n.mark_output(tensor)
            return

        # Public module buffers are provisioned at the maximum input shape.
        # A shrinking output would underallocate at shorter contexts. Keep
        # the trained block bound static; the runtime consumes only the prefix
        # that fits the remaining target capacity.
        for name, values in (("proposal_ids", proposals), ("draft_logits", rows),
                             ("draft_hidden", states)):
            tensor = joined(values)
            tensor.name = name
            n.mark_output(tensor)
