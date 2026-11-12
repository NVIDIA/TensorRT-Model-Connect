# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gemma4 dense block-7 DFlash/DSpark math, without an Edge or ONNX dependency."""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from . import graph_ops
from .checkpoint_mapper import _load_tensor
from .gemma4_text import DenseTextGraph, attention_geometry, rope_tables


class BlockDraftGraph(DenseTextGraph):
    """Recompute draft context from committed target features.

    Draft proposals attend bidirectionally within their own block and to all
    committed target features. Recomputing context avoids a second mutable KV
    history; only the target verifier decides which proposal prefix is committed.
    This is a mathematical representation, not a GPU-specific kernel policy.
    """

    def __init__(self, network, raw, readers, precision, capacity):
        if (raw.get("architectures") != ["Gemma4DSparkModel"]
                or raw.get("enable_moe_block") or raw.get("hidden_size_per_layer_input", 0)
                or raw.get("attention_bias") or raw.get("num_kv_shared_layers", 0)
                or raw.get("block_size") != 7
                or raw.get("layer_types") != ["full_attention"] * raw["num_hidden_layers"]):
            raise NotImplementedError(
                "Native Gemma block draft requires the dense block7 DFlash/DSpark topology")
        self.markov_rank = int(raw.get("markov_rank", 0))
        if self.markov_rank:
            if (self.markov_rank < 0 or raw.get("markov_head_type") != "vanilla"
                    or not raw.get("enable_confidence_head")
                    or not raw.get("confidence_head_with_markov")):
                raise NotImplementedError(
                    "Native Gemma DSpark requires vanilla Markov and Markov-conditioned confidence")
        elif raw.get("enable_confidence_head") or raw.get("confidence_head_with_markov"):
            raise NotImplementedError("Native Gemma DFlash does not map independent confidence heads")
        self.net, self.raw, self.readers = network, raw, readers
        self.capacity, self.buffers = capacity, []
        self.dtype = {"fp16": trt.float16, "bf16": trt.bfloat16, "fp32": trt.float32}[precision]
        self.storage = np.float16 if precision == "fp16" else np.float32
        self.target_layers = list(raw["target_layer_ids"])
        if not self.target_layers or len(set(self.target_layers)) != len(self.target_layers):
            raise ValueError("Gemma block draft requires distinct target feature layers")
        if not 1 < capacity <= int(raw["max_position_embeddings"]):
            raise ValueError("Gemma block draft capacity exceeds the checkpoint")

    def weight(self, name, *, dtype=None):
        value = _load_tensor(self.readers, name)
        return self.const(value.astype(self.storage if dtype is None else np.float32),
                          self.dtype if dtype is None else dtype)

    def linear(self, value, name):
        weight = self.cast(self.weight(name + ".weight"), value.dtype)
        return self.net.add_matrix_multiply(
            value, trt.MatrixOperation.NONE, weight, trt.MatrixOperation.TRANSPOSE).get_output(0)

    def dspark_heads(self, tokens, state, logits):
        """Select sequential greedy proposals with the original Markov correction.

        The fixed checkpoint block is unrolled in the TensorRT graph. Short
        capacity-boundary blocks reuse a valid row for the unused tail, then
        slice it away; no out-of-range gather or host-side weight math is needed.
        Confidence is a raw logit, not a probability. The default scheduler is
        off, matching the paired Edge adapter.
        """
        n = self.net
        count = self.cast(n.add_shape(tokens).get_output(0), trt.int32)
        last = self.op(count, self.const(np.array([1], np.int32)), trt.ElementWiseOperation.SUB)
        previous = n.add_gather(tokens, self.const(np.array([0], np.int32)), 0).get_output(0)
        embedding = self.weight("markov_head.markov_w1.weight")
        projection = self.weight("markov_head.markov_w2.weight")
        confidence_weight = self.weight("confidence_head.proj.weight")
        confidence_bias = self.reshape(self.weight("confidence_head.proj.bias"), (1, 1))
        corrected, proposals, confidences = [], [], []
        for row in range(int(self.raw["block_size"])):
            index = self.op(self.const(np.array([row], np.int32)), last,
                            trt.ElementWiseOperation.MIN)
            latent = n.add_gather(embedding, previous, 0).get_output(0)
            bias = n.add_matrix_multiply(
                latent, trt.MatrixOperation.NONE, projection,
                trt.MatrixOperation.TRANSPOSE).get_output(0)
            base = n.add_gather(logits, index, 0).get_output(0)
            step = self.op(base, self.cast(bias, trt.float32), trt.ElementWiseOperation.SUM)
            corrected.append(step)
            features = n.add_concatenation([
                n.add_gather(state, index, 0).get_output(0), latent])
            features.axis = 1
            confidence = n.add_matrix_multiply(
                features.get_output(0), trt.MatrixOperation.NONE, confidence_weight,
                trt.MatrixOperation.TRANSPOSE).get_output(0)
            confidences.append(self.cast(
                self.op(confidence, confidence_bias, trt.ElementWiseOperation.SUM), trt.float32))
            previous = self.reshape(n.add_topk(step, trt.TopKOperation.MAX, 1, 2).get_output(1),
                                    (1,))
            proposals.append(previous)

        def prefix(values, width=None):
            joined = n.add_concatenation(values)
            joined.axis = 0
            if width is None:
                value = self.reshape(joined.get_output(0), (-1,))
                sliced = n.add_slice(value, (0,), (self.raw["block_size"],), (1,))
                sliced.set_input(2, count)
            else:
                size = n.add_concatenation([count, self.const(np.array([width], np.int32))])
                size.axis = 0
                sliced = n.add_slice(joined.get_output(0), (0, 0),
                                     (self.raw["block_size"], width), (1, 1))
                sliced.set_input(2, size.get_output(0))
            return sliced.get_output(0)

        for name, value in (("proposal_ids", prefix(proposals)),
                            ("proposal_confidence", prefix(confidences))):
            value.name = name
            n.mark_output(value)
        return prefix(corrected, self.raw["vocab_size"])

    def build(self):
        n, raw = self.net, self.raw
        hidden, heads = int(raw["hidden_size"]), int(raw["num_attention_heads"])
        dim, kv_heads = attention_geometry(raw, "full_attention")
        tokens = n.add_input("draft_tokens", trt.int32, (-1,))
        positions = n.add_input("draft_positions", trt.int32, (-1,))
        context_positions = n.add_input("context_positions", trt.int32, (-1,))
        context = n.add_input("target_features", trt.float32,
                              (-1, len(self.target_layers) * hidden))

        # The visible fusion result stays FP32 through RMS normalization.
        # Casting an overflowing FP16 projection back to FP32 would be too late.
        context = self.cast(self.norm(self.linear(context, "fc"), "hidden_norm"), self.dtype)
        state = self.scale(n.add_gather(self.weight("embed_tokens.weight"), tokens, 0).get_output(0),
                           hidden**0.5)
        joined_positions = n.add_concatenation([context_positions, positions])
        joined_positions.axis = 0
        all_positions = joined_positions.get_output(0)
        cos, sin = [self.const(t.astype(self.storage), self.dtype)
                    for t in rope_tables(raw, "full_attention", self.capacity)]
        zero_mask = self.const(np.zeros((1, 1, 1, 1), np.float32))

        for index in range(int(raw["num_hidden_layers"])):
            prefix = f"layers.{index}"
            attn = prefix + ".self_attn"
            normalized = self.norm(state, prefix + ".input_layernorm")
            query = self.head_norm(self.linear(normalized, attn + ".q_proj"),
                                   heads, dim, attn + ".q_norm")
            joined = n.add_concatenation([context, normalized])
            joined.axis = 0
            source = joined.get_output(0)
            key_raw = self.linear(source, attn + ".k_proj")
            value_raw = (key_raw if raw.get("attention_k_eq_v")
                         else self.linear(source, attn + ".v_proj"))
            key = self.head_norm(key_raw, kv_heads, dim, attn + ".k_norm")
            value = self.head_norm(value_raw, kv_heads, dim)
            query = graph_ops.add_apply_rope_native(
                n, query, heads, dim, cos, sin, positions, dim, sequence_length=None)
            key = graph_ops.add_apply_rope_native(
                n, key, kv_heads, dim, cos, sin, all_positions, dim, sequence_length=None)
            attended = self.attention(query, key, value, heads, kv_heads, dim, zero_mask)
            attended = self.norm(self.linear(attended, attn + ".o_proj"),
                                 prefix + ".post_attention_layernorm")
            state = self.op(state, attended, trt.ElementWiseOperation.SUM)
            normalized = self.norm(state, prefix + ".pre_feedforward_layernorm")
            gate = self.cast(self.linear(normalized, prefix + ".mlp.gate_proj"), trt.float32)
            up = self.cast(self.linear(normalized, prefix + ".mlp.up_proj"), trt.float32)
            product = self.op(self.activation(gate), up, trt.ElementWiseOperation.PROD)
            product = self.norm(self.linear(product, prefix + ".mlp.down_proj"),
                                prefix + ".post_feedforward_layernorm")
            state = self.op(state, self.cast(product, self.dtype), trt.ElementWiseOperation.SUM)
            state = self.op(state, self.reshape(self.weight(prefix + ".layer_scalar"), (1, 1)),
                            trt.ElementWiseOperation.PROD)

        state = self.norm(state, "norm")
        logits = self.cast(self.linear(state, "lm_head"), trt.float32)
        if raw.get("final_logit_softcapping"):
            logits = graph_ops.add_tanh_softcap(n, logits, float(raw["final_logit_softcapping"]),
                                               scalar_shape=(1, 1))
        if self.markov_rank:
            logits = self.dspark_heads(tokens, state, logits)
        logits.name = "draft_logits"
        n.mark_output(logits)
        features = self.cast(state, trt.float32)
        features.name = "draft_hidden"
        n.mark_output(features)
