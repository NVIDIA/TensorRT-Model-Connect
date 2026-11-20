# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gemma4 Q-only assistant math; the target owns all persistent K/V state."""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from . import graph_ops
from .checkpoint_mapper import _has_tensor, _load_tensor
from .gemma4_text import DenseTextGraph, attention_geometry, cache_geometry, rope_tables


def shared_target_layers(raw):
    """Last owning target layer for each assistant attention type."""
    owners, _ = cache_geometry(raw)
    return {kind: owners[index] for index, kind in enumerate(raw["layer_types"])}


class AssistantGraph(DenseTextGraph):
    """One recurrent proposal, without pretending the assistant is a causal LM."""

    def __init__(self, network, config, readers, target, target_readers, precision, capacity):
        self.net, self.raw, self.readers = network, config["text_config"], readers
        self.config, self.target = config, target
        self.capacity, self.prefix, self.buffers = capacity, "model", []
        self.dtype = {"fp16": trt.float16, "bf16": trt.bfloat16, "fp32": trt.float32}[precision]
        self.storage = np.float16 if precision == "fp16" else np.float32
        self.target_readers = target_readers
        self.target_prefix = next((prefix for prefix in
                                   ("model.language_model", "language_model.model", "model")
                                   if _has_tensor(target_readers, prefix + ".embed_tokens.weight")), None)
        if self.target_prefix is None:
            raise ValueError("Gemma4 MTP target embedding is missing")
        if (self.raw["num_kv_shared_layers"] != self.raw["num_hidden_layers"]
                or self.raw.get("hidden_size_per_layer_input", 0)
                or self.raw.get("enable_moe_block") or self.raw.get("attention_bias")):
            raise NotImplementedError("Gemma4 native MTP requires the dense Q-only assistant topology")
        if config.get("use_ordered_embeddings"):
            centroids = config["num_centroids"]
            top_k = config["centroid_intermediate_top_k"]
            if (type(centroids) is not int or type(top_k) is not int or
                    not 0 < top_k <= centroids or self.raw["vocab_size"] % centroids):
                raise ValueError("Gemma4 MTP ordered vocabulary has invalid centroid geometry")
            ordering = _load_tensor(readers, "masked_embedding.token_ordering")
            if (ordering.shape != (self.raw["vocab_size"],) or
                    not np.array_equal(np.sort(ordering), np.arange(self.raw["vocab_size"]))):
                raise ValueError("Gemma4 MTP token ordering must be a vocabulary permutation")
        if (config["backbone_hidden_size"] != target["hidden_size"]
                or self.raw["vocab_size"] != target["vocab_size"]
                or not 0 < capacity <= min(target["max_position_embeddings"],
                                          self.raw["max_position_embeddings"])):
            raise ValueError("Gemma4 MTP target and assistant dimensions or capacity disagree")
        self.target_layers = shared_target_layers(target)
        for kind in self.raw["layer_types"]:
            if kind not in self.target_layers or attention_geometry(self.raw, kind) != attention_geometry(target, kind):
                raise ValueError("Gemma4 MTP shared target K/V geometry does not match the assistant")

    def weight(self, name, *, dtype=None):
        if (name in {"pre_projection.weight", "post_projection.weight", "lm_head.weight"}
                or name.startswith("masked_embedding.")):
            key = name
            if key == "lm_head.weight" and not _has_tensor(self.readers, key):
                key = "model.embed_tokens.weight"
        else:
            key = self.prefix + "." + name
        values = _load_tensor(self.readers, key)
        return self.const(values.astype(self.storage if dtype is None else np.float32),
                          self.dtype if dtype is None else dtype)

    def build(self):
        n, raw = self.net, self.raw
        backbone = int(self.config["backbone_hidden_size"])
        tokens = n.add_input("token_id", trt.int32, (1,))
        previous = n.add_input("target_hidden", trt.float32, (1, backbone))
        positions = n.add_input("position_id", trt.int32, (1,))
        length = n.add_input("shared_length", trt.int32, (1,))
        rows = self.const(np.arange(self.capacity, dtype=np.int32).reshape(1, -1))
        length = self.reshape(length, (1, 1))
        valid = self.op(rows, length, trt.ElementWiseOperation.LESS)
        # The assistant flips a bidirectional window anchored at its last KV row.
        # Unlike target causal attention, the original includes that row plus
        # sliding_window preceding rows (verified beyond the window boundary).
        lower = self.op(length, self.const(np.array([[raw["sliding_window"] + 1]], np.int32)),
                        trt.ElementWiseOperation.SUB)
        local = self.op(self.op(rows, lower, trt.ElementWiseOperation.GREATER),
                        self.op(rows, lower, trt.ElementWiseOperation.EQUAL),
                        trt.ElementWiseOperation.OR)
        local = self.op(local, valid, trt.ElementWiseOperation.AND)
        masks, caches = {}, {}
        for kind in set(raw["layer_types"]):
            mask = n.add_select(local if kind == "sliding_attention" else valid,
                                self.const(np.zeros((1, 1), np.float32)),
                                self.const(np.full((1, 1), -np.inf, np.float32))).get_output(0)
            masks[kind] = self.reshape(mask, (1, 1, 1, self.capacity))
            dim, heads = attention_geometry(raw, kind)
            caches[kind] = [n.add_input(f"shared_{label}_{kind}", self.dtype,
                                       (self.capacity, dim * heads)) for label in ("k", "v")]
        values = _load_tensor(self.target_readers, self.target_prefix + ".embed_tokens.weight")
        target_embedding = self.const(values.astype(self.storage), self.dtype)
        embedding = self.scale(n.add_gather(target_embedding, tokens, 0).get_output(0), backbone**0.5)
        fused = n.add_concatenation([embedding, self.cast(previous, self.dtype)])
        fused.axis = 1
        state = self.linear(fused.get_output(0), "pre_projection")
        tables = {kind: tuple(self.const(value.astype(self.storage), self.dtype)
                              for value in rope_tables(raw, kind, self.capacity))
                  for kind in set(raw["layer_types"])}
        heads = int(raw["num_attention_heads"])
        for index, kind in enumerate(raw["layer_types"]):
            prefix = f"layers.{index}"
            dim, kv_heads = attention_geometry(raw, kind)
            normalized = self.norm(state, prefix + ".input_layernorm")
            query = self.head_norm(self.linear(normalized, prefix + ".self_attn.q_proj"),
                                   heads, dim, prefix + ".self_attn.q_norm")
            cos, sin = tables[kind]
            query = graph_ops.add_apply_rope_native(n, query, heads, dim, cos, sin, positions,
                                                    dim, sequence_length=1)
            key, value = caches[kind]
            attention = self.attention(query, key, value, heads, kv_heads, dim, masks[kind])
            attention = self.norm(self.linear(attention, prefix + ".self_attn.o_proj"),
                                  prefix + ".post_attention_layernorm")
            state = self.op(state, attention, trt.ElementWiseOperation.SUM)
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
        projected = self.cast(self.linear(state, "post_projection"), trt.float32)
        projected.name = "assistant_hidden"
        n.mark_output(projected)
        logits = self.ordered_logits(state) if self.config.get("use_ordered_embeddings") else self.linear(state, "lm_head")
        logits = self.cast(logits, trt.float32)
        logits.name = "logits"
        n.mark_output(logits)

    def ordered_logits(self, state):
        """Select centroid vocabulary blocks and restore canonical token order."""
        n, config = self.net, self.config
        vocabulary = self.raw["vocab_size"]
        centroid_logits = self.linear(state, "masked_embedding.centroids")
        selected = n.add_topk(centroid_logits, trt.TopKOperation.MAX,
                              config["centroid_intermediate_top_k"], 1 << 1).get_output(1)
        ordering = _load_tensor(self.readers, "masked_embedding.token_ordering")
        ordering = self.const(ordering.astype(np.int32).reshape(config["num_centroids"], -1))
        canonical = n.add_gather(ordering, selected, 0).get_output(0)
        canonical = self.reshape(canonical, (-1,))
        embeddings = n.add_gather(self.weight("lm_head.weight"), canonical, 0).get_output(0)
        selected_logits = n.add_matrix_multiply(
            state, trt.MatrixOperation.NONE, embeddings, trt.MatrixOperation.TRANSPOSE).get_output(0)
        # Match the original head's selected_min - 1 fill, not Edge's fixed
        # FP16 export sentinel. No unselected token may become the proposal.
        minimum = n.add_reduce(self.cast(selected_logits, trt.float32),
                               trt.ReduceOperation.MIN, 1 << 1, True).get_output(0)
        fill = self.op(minimum, self.const(np.ones((1, 1), np.float32)),
                       trt.ElementWiseOperation.SUB)
        base = self.op(self.const(np.zeros((1, vocabulary), self.storage), self.dtype),
                       self.cast(fill, self.dtype), trt.ElementWiseOperation.SUM)
        scatter = n.add_scatter(base, self.reshape(canonical, (1, -1)), selected_logits,
                                trt.ScatterMode.ELEMENT)
        scatter.axis = 1
        return scatter.get_output(0)
