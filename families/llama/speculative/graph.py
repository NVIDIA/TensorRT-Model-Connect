# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graphs for the linear-state speculative ABI.

Uses family-owned graph primitives and TensorRT APIs for target feature
extraction, EAGLE3 draft conditioning, and explicit cache updates.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from .. import graph_ops as ops
from ..native_kv_attention_builder import NativeKvMasks, add_explicit_masked_grouped_query_attention
from .contract import EngineContract


class Graph:
    def __init__(self, contract: EngineContract, config, *, verbose=False):
        self.contract = contract
        self.config = config
        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        self.builder = trt.Builder(self.logger)
        self.network = self.builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        )
        self.build_config = self.builder.create_builder_config()
        self.build_config.builder_optimization_level = 1
        self.build_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 16 << 30)
        self.profiles = [self.builder.create_optimization_profile()
                         for _ in range(len(contract.execution_profiles) or 1)]
        self.tokens = self.input("token_id", trt.int32, (-1,))
        self.positions = self.input("position_id", trt.int32, (-1,))
        self.selected = self.input("logits_indices", trt.int32, (-1,))
        self.write_start = self.input("cache_write_indices", trt.int32, (1,))
        self.length = self.input("key_value_lengths", trt.int32, (1,))
        # INT32 visibility values match the backend Tensor ABI.
        mask = self.input("attention_mask", trt.int32, (-1, contract.capacity))
        mask = self.network.add_cast(mask, trt.bool).get_output(0)
        reshape = self.network.add_shuffle(mask)
        reshape.reshape_dims = (1, 1, -1, contract.capacity)
        slots = self.constant(np.arange(contract.capacity, dtype=np.int32).reshape(1, -1))
        length = self.network.add_shuffle(self.length)
        length.reshape_dims = (1, 1)
        active = self.network.add_elementwise(
            slots, length.get_output(0), trt.ElementWiseOperation.LESS
        ).get_output(0)
        active4 = self.network.add_shuffle(active)
        active4.reshape_dims = (1, 1, 1, contract.capacity)
        visible = self.network.add_elementwise(
            reshape.get_output(0), active4.get_output(0), trt.ElementWiseOperation.AND
        ).get_output(0)
        self.masks = NativeKvMasks(visible, active4.get_output(0))
        self.eps = self.constant(np.array([[config.rms_norm_eps]], dtype=np.float32))
        inv_freq = ops.make_native_active_rope_inv_freq(
            contract.head_dim, config.rope_theta, 1.0,
            rope_scaling=config.raw.get("rope_parameters") or config.raw.get("rope_scaling"),
        )
        self.cos, self.sin = ops.add_active_rope_cache(
            self.network, self.positions, inv_freq, trt.float16
        )

    def input(self, name, dtype, shape):
        value = self.network.add_input(name, dtype, shape)
        if -1 in shape:
            for profile, bounds in zip(self.profiles, self.contract.profile_shapes(name, shape)):
                profile.set_shape(name, *bounds)
        return value

    def constant(self, value):
        value = np.ascontiguousarray(value)
        return self.network.add_constant(value.shape, value).get_output(0)

    def output(self, value, name):
        value.name = name
        self.network.mark_output(value)

    def add(self, left, right):
        return self.network.add_elementwise(left, right, trt.ElementWiseOperation.SUM).get_output(0)

    def linear(self, value, weight):
        return ops.add_matmul_rhs_constant(
            self.network, value, weight.shape[0], weight.shape[1], weight, dtype=np.float16
        )

    def norm(self, value, weight):
        return ops.add_rms_norm(
            self.network, value, self.contract.hidden_size, weight, self.eps, dtype=np.float16
        )

    def attention(self, value, weights, prefix, layer):
        c, cfg = self.contract, self.config
        q = self.linear(value, weights[f"{prefix}.w_q"])
        k = self.linear(value, weights[f"{prefix}.w_k"])
        v = self.linear(value, weights[f"{prefix}.w_v"])
        q = ops.add_apply_rope_native(
            self.network, q, cfg.num_attention_heads, c.head_dim,
            self.cos, self.sin, None, c.head_dim, sequence_length=None,
        )
        k = ops.add_apply_rope_native(
            self.network, k, c.kv_heads, c.head_dim,
            self.cos, self.sin, None, c.head_dim, sequence_length=None,
        )
        shape = (1, c.kv_heads, c.capacity, c.head_dim)
        cache_k = self.input(f"cache_k_{layer}", trt.float16, shape)
        cache_v = self.input(f"cache_v_{layer}", trt.float16, shape)
        # This is the lowering seam: all state effects remain explicit graph
        # inputs/outputs, independent of the proposal and acceptance policy.
        k4 = ops.reshape_rows_to_heads_4d(self.network, k, c.kv_heads, c.head_dim, None)
        v4 = ops.reshape_rows_to_heads_4d(self.network, v, c.kv_heads, c.head_dim, None)
        q4 = ops.reshape_rows_to_heads_4d(self.network, q, cfg.num_attention_heads, c.head_dim, None)
        updated_k = self.network.add_kv_cache_update(
            cache_k, k4, self.write_start, trt.KVCacheMode.LINEAR
        ).get_output(0)
        updated_v = self.network.add_kv_cache_update(
            cache_v, v4, self.write_start, trt.KVCacheMode.LINEAR
        ).get_output(0)
        context = add_explicit_masked_grouped_query_attention(
            self.network, q4, updated_k, updated_v, self.masks,
            num_heads=cfg.num_attention_heads, num_kv_heads=c.kv_heads,
            head_dim=c.head_dim, tag=f"{prefix}.spec_attention",
        )
        self.output(updated_k, f"present_k_{layer}")
        self.output(updated_v, f"present_v_{layer}")
        rows = ops.reshape_heads_4d_to_rows(
            self.network, context, cfg.num_attention_heads * c.head_dim, None
        )
        return self.linear(rows, weights[f"{prefix}.w_o"])

    def mlp(self, value, weights, prefix):
        gate = self.linear(value, weights[f"{prefix}.w_gate"])
        up = self.linear(value, weights[f"{prefix}.w_up"])
        sigmoid = self.network.add_activation(gate, trt.ActivationType.SIGMOID).get_output(0)
        silu = self.network.add_elementwise(gate, sigmoid, trt.ElementWiseOperation.PROD).get_output(0)
        product = self.network.add_elementwise(silu, up, trt.ElementWiseOperation.PROD).get_output(0)
        return self.linear(product, weights[f"{prefix}.w_down"])

    def logits(self, hidden, weights):
        rows = self.network.add_gather(hidden, self.selected, 0).get_output(0)
        normed = self.norm(rows, weights["final_norm"])
        logits = self.linear(normed, weights["w_out"])
        self.output(self.network.add_cast(logits, trt.float32).get_output(0), "logits")

    def finish(self):
        for profile in self.profiles:
            self.build_config.add_optimization_profile(profile)
        result = self.builder.build_serialized_network(self.network, self.build_config)
        if result is None:
            raise RuntimeError(f"failed to compile speculative {self.contract.role} engine")
        return bytes(result)


def build_target(config, weights, contract, *, feature_indices, verbose=False):
    graph = Graph(contract, config, verbose=verbose)
    embedding = graph.constant(np.asarray(weights["embedding"], dtype=np.float16))
    hidden = graph.network.add_gather(embedding, graph.tokens, 0).get_output(0)
    features = {}
    # Feature taps index the inputs to layers (embedding is layer 0's input).
    for layer in range(config.num_hidden_layers):
        if layer in feature_indices:
            features[layer] = hidden
        prefix = f"layer.{layer}"
        normalized = graph.norm(hidden, weights[f"{prefix}.input_norm"])
        hidden = graph.add(hidden, graph.attention(normalized, weights, prefix, layer))
        normalized = graph.norm(hidden, weights[f"{prefix}.post_attn_norm"])
        hidden = graph.add(hidden, graph.mlp(normalized, weights, prefix))
    concat = graph.network.add_concatenation([features[layer] for layer in feature_indices])
    concat.axis = 1
    graph.output(concat.get_output(0), "features")
    graph.logits(hidden, weights)
    return graph.finish()


def build_draft(config, weights, contract, *, verbose=False):
    graph = Graph(contract, config, verbose=verbose)
    features = graph.input("target_features", trt.float16, (-1, contract.feature_width))
    recurrent = graph.input("draft_features", trt.float16, (-1, contract.hidden_size))
    hidden = graph.add(graph.linear(features, weights["fc"]), recurrent)
    embedding = graph.constant(np.asarray(weights["embedding"], dtype=np.float16))
    embedded = graph.network.add_gather(embedding, graph.tokens, 0).get_output(0)
    for layer in range(config.num_hidden_layers):
        prefix = f"layer.{layer}"
        hidden_norm = graph.norm(hidden, weights[f"{prefix}.hidden_norm"])
        embed_norm = graph.norm(embedded, weights[f"{prefix}.input_norm"])
        concat = graph.network.add_concatenation([embed_norm, hidden_norm])
        concat.axis = 1
        hidden = graph.add(hidden, graph.attention(concat.get_output(0), weights, prefix, layer))
        normalized = graph.norm(hidden, weights[f"{prefix}.post_attn_norm"])
        hidden = graph.add(hidden, graph.mlp(normalized, weights, prefix))
    graph.output(hidden, "features")
    graph.logits(hidden, weights)
    return graph.finish()
