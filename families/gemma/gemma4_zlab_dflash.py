# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gemma-target z-lab DFlash math; this does not add a Qwen target adapter."""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from . import graph_ops
from .checkpoint_mapper import _has_tensor, _load_tensor
from .gemma4_text import DenseTextGraph


def draft_rope_theta(raw):
    """Read equivalent original and current unscaled rotary config spellings."""
    parameters = raw.get("rope_parameters")
    if parameters is None:
        if raw.get("rope_scaling") is not None:
            raise NotImplementedError("Native Gemma z-lab draft requires unscaled rotary positions")
        parameters = {"rope_type": "default", "rope_theta": raw.get("rope_theta")}
    if not isinstance(parameters, dict) or parameters.get("rope_type") != "default":
        raise NotImplementedError("Native Gemma z-lab draft requires default rotary positions")
    theta = parameters.get("rope_theta")
    if type(theta) not in (int, float) or not np.isfinite(theta) or theta <= 0:
        raise ValueError("Gemma z-lab draft rotary theta must be finite and positive")
    return float(theta)


class ZlabDraftGraph(DenseTextGraph):
    """Original block16 draft, including its pending-anchor output row."""

    def __init__(self, network, raw, readers, precision, capacity, target_readers):
        if (raw.get("architectures") != ["DFlashDraftModel"] or raw.get("block_size") != 16
                or raw.get("model_type") != "qwen3" or raw.get("hidden_act") != "silu"
                or raw.get("attention_bias") or raw.get("conv_kernel_size", 0)
                or len(raw.get("layer_types", [])) != raw["num_hidden_layers"]
                or set(raw["layer_types"]) - {"sliding_attention", "full_attention"}):
            raise NotImplementedError("Native Gemma z-lab draft requires its original block16 topology")
        self.rope_theta = draft_rope_theta(raw)
        self.net, self.raw, self.readers = network, raw, readers
        self.capacity, self.buffers = capacity, []
        self.dtype = {"fp16": trt.float16, "bf16": trt.bfloat16, "fp32": trt.float32}[precision]
        self.storage = np.float16 if precision == "fp16" else np.float32
        self.target_layers = list(raw["dflash_config"]["target_layer_ids"])
        if not self.target_layers or len(set(self.target_layers)) != len(self.target_layers):
            raise ValueError("Gemma z-lab draft requires distinct target feature layers")
        if not 1 < capacity <= int(raw["max_position_embeddings"]):
            raise ValueError("Gemma z-lab draft capacity exceeds the checkpoint")
        self.target_readers = target_readers
        self.target_embedding = next((p + ".embed_tokens.weight" for p in
            ("model.language_model", "language_model.model", "model")
            if _has_tensor(target_readers, p + ".embed_tokens.weight")), None)
        if self.target_embedding is None:
            raise ValueError("Gemma z-lab draft requires its target embedding")

    def weight(self, name, *, dtype=None):
        value = _load_tensor(self.readers, name)
        return self.const(value.astype(self.storage if dtype is None else np.float32),
                          self.dtype if dtype is None else dtype)

    def linear(self, value, name):
        weight = self.cast(self.weight(name + ".weight"), value.dtype)
        return self.net.add_matrix_multiply(
            value, trt.MatrixOperation.NONE, weight, trt.MatrixOperation.TRANSPOSE).get_output(0)

    def build(self):
        n, raw = self.net, self.raw
        hidden, heads = int(raw["hidden_size"]), int(raw["num_attention_heads"])
        dim, kv_heads = int(raw["head_dim"]), int(raw["num_key_value_heads"])
        tokens = n.add_input("draft_tokens", trt.int32, (-1,))
        positions = n.add_input("draft_positions", trt.int32, (-1,))
        context_positions = n.add_input("context_positions", trt.int32, (-1,))
        context = n.add_input("target_features", trt.float32,
                              (-1, len(self.target_layers) * hidden))
        context = self.cast(self.norm(self.linear(context, "fc"), "hidden_norm"), self.dtype)
        table = _load_tensor(self.target_readers, self.target_embedding)
        if table.shape != (raw["vocab_size"], hidden):
            raise ValueError("Gemma target embedding and z-lab draft geometry differ")
        embedding = self.const(table.astype(self.storage), self.dtype)
        state = n.add_gather(embedding, tokens, 0).get_output(0)
        # The original draft deliberately bypasses Gemma's scaled embedding.
        metadata = raw["dflash_config"]
        state = self.scale(state, float(metadata.get("input_embedding_scale",
                                                     raw.get("input_embedding_scale", 1.0))))
        joined_positions = n.add_concatenation([context_positions, positions])
        joined_positions.axis = 0
        all_positions = joined_positions.get_output(0)
        inv = 1.0 / (self.rope_theta **
                      (np.arange(0, dim, 2, dtype=np.float32) / dim))
        angles = np.arange(self.capacity, dtype=np.float32)[:, None] * inv[None, :]
        cos, sin = [self.const(value.astype(self.storage), self.dtype)
                    for value in (np.cos(angles), np.sin(angles))]
        query_pos, key_pos = self.reshape(positions, (-1, 1)), self.reshape(all_positions, (1, -1))
        before = self.op(key_pos, self.op(query_pos, self.const(np.array([[1]], np.int32)),
                                         trt.ElementWiseOperation.SUM), trt.ElementWiseOperation.LESS)
        lower = self.op(query_pos, self.const(np.array([[raw["sliding_window"]]], np.int32)),
                        trt.ElementWiseOperation.SUB)
        visible = self.op(before, self.op(key_pos, lower, trt.ElementWiseOperation.GREATER),
                          trt.ElementWiseOperation.AND)
        mask = n.add_select(visible, self.const(np.zeros((1, 1), np.float32)),
                            self.const(np.full((1, 1), -np.inf, np.float32))).get_output(0)
        shape = n.add_concatenation([self.const(np.array([1, 1], np.int32)),
                                    self.cast(n.add_shape(mask).get_output(0), trt.int32)])
        shape.axis = 0
        expanded = n.add_shuffle(mask)
        expanded.set_input(1, shape.get_output(0))
        sliding_mask = expanded.get_output(0)
        full_mask = self.const(np.zeros((1, 1, 1, 1), np.float32))

        for index, kind in enumerate(raw["layer_types"]):
            prefix = f"layers.{index}"
            attn = prefix + ".self_attn"
            normalized = self.norm(state, prefix + ".input_layernorm")
            joined = n.add_concatenation([context, normalized])
            joined.axis = 0
            query = self.head_norm(self.linear(normalized, attn + ".q_proj"),
                                   heads, dim, attn + ".q_norm")
            key = self.head_norm(self.linear(joined.get_output(0), attn + ".k_proj"),
                                 kv_heads, dim, attn + ".k_norm")
            value = self.linear(joined.get_output(0), attn + ".v_proj")  # No value RMSNorm.
            query = graph_ops.add_apply_rope_native(
                n, query, heads, dim, cos, sin, positions, dim, sequence_length=None)
            key = graph_ops.add_apply_rope_native(
                n, key, kv_heads, dim, cos, sin, all_positions, dim, sequence_length=None)
            attended = self.attention(self.scale(self.cast(query, trt.float32), dim**-0.5),
                key, value, heads, kv_heads, dim,
                sliding_mask if kind == "sliding_attention" else full_mask)
            state = self.op(state, self.linear(attended, attn + ".o_proj"), trt.ElementWiseOperation.SUM)
            normalized = self.norm(state, prefix + ".post_attention_layernorm")
            gate, up = (self.cast(self.linear(normalized, prefix + ".mlp." + name), trt.float32)
                        for name in ("gate_proj", "up_proj"))
            gate = graph_ops.add_activation(n, gate, "silu", dtype=np.float32)
            product = self.op(gate, up, trt.ElementWiseOperation.PROD)
            down = self.cast(self.linear(product, prefix + ".mlp.down_proj"), self.dtype)
            state = self.op(state, down, trt.ElementWiseOperation.SUM)
        state = self.norm(state, "norm")
        logits = n.add_matrix_multiply(state, trt.MatrixOperation.NONE, embedding,
                                       trt.MatrixOperation.TRANSPOSE).get_output(0)
        logits = self.scale(self.cast(logits, trt.float32),
                           float(metadata.get("output_multiplier", raw.get("output_multiplier", 1.0))))
        cap = metadata.get("final_logit_softcapping", raw.get("final_logit_softcapping"))
        if cap is not None:
            logits = graph_ops.add_tanh_softcap(n, logits, float(cap), scalar_shape=(1, 1))
        for name, tensor in (("draft_logits", logits), ("draft_hidden", self.cast(state, trt.float32))):
            tensor.name = name
            n.mark_output(tensor)
