# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Uncached original NVIDIA HSTU attention with ordinary padded public inputs.

All token-wise math stays in the shared flat block. No KV cache tensor or update
exists in this graph. The native attention adapter ignores its one-element dummy.
"""

from __future__ import annotations

import numpy as np

from .paged_graph import PagedGraph


class DenseGraph(PagedGraph):
    def updated_pages(self, projection, index, metadata):
        # Keep the shared three-input host adapter boundary. The dense C ABI has
        # no page arguments, and never constructs a page/TMA view of this value.
        if not hasattr(self, "_unused_page"):
            self._unused_page = self.constant(
                np.zeros((1,), np.float32), "attention.unused_page", self.dtype
            )
        return self._unused_page

    def _restore_rows(self, value, batch_sequence_shape, name):
        width = self.constant(np.asarray([value.shape[-1]], np.int64), f"{name}.width")
        shape = self.net.add_concatenation([batch_sequence_shape, width])
        shape.axis = 0
        layer = self.net.add_shuffle(value)
        layer.set_input(1, self.out(shape, f"{name}.shape"))
        return self.out(layer, name)

    def outputs(self):
        c = self.config
        if c["enable_history_cache"]:
            raise ValueError("Dense HSTU graph requires history caching disabled")
        ids = self.net.add_input("token_ids", self.trt.int32, (-1, -1))
        ids.set_dimension_name(0, "batch")
        ids.set_dimension_name(1, "sequence")
        public_shape = self.out(self.net.add_shape(ids), "dense.batch_sequence_shape")
        metadata = self.net.add_input("attention_metadata", self.trt.int32, (5, -1, 8))
        merged = np.concatenate(
            [self.weights[f"embeddings.{table['name']}.weight"] for table in c["embedding_tables"]]
        )
        table = self.constant(merged, "embedding.weight", self.dtype)
        x = self.out(self.net.add_gather(table, ids, 0), "embedding.lookup")
        x = self.reshape(x, (-1, c["hidden_size"]), "dense.input_rows")
        if c["position_buckets"]:
            x = self.cast(x, self.trt.float32, "position.input.fp32")
            x = self.binary(x, self.scalar(c["hidden_size"] ** 0.5, x, "position.scale"),
                            "PROD", "position.scaled_embeddings")
            positions = self.net.add_input("position_ids", self.trt.int32, (-1, -1))
            positions.set_dimension_name(0, "batch")
            positions.set_dimension_name(1, "sequence")
            weight = self.constant(self.weights["position.weight"], "position.weight", self.dtype)
            positional = self.out(self.net.add_gather(weight, positions, 0), "position.lookup")
            positional = self.reshape(positional, (-1, c["hidden_size"]), "dense.position_rows")
            positional = self.cast(positional, self.trt.float32, "position.fp32")
            x = self.cast(self.binary(x, positional, "SUM", "position.add"),
                          self.dtype, "position.output")
        for index in range(c["num_layers"]):
            x = self.block(x, None, None, index, {"attention_metadata": metadata})
        embeddings = self.l2(x, "output.l2")
        logits = embeddings
        for index in range(len(c["prediction_head"])):
            logits = self.linear(logits, f"head.{index}", c["prediction_bias"])
            if index + 1 < len(c["prediction_head"]):
                activation = (self.trt.ActivationType.RELU if c["prediction_activation"] == "relu"
                              else self.trt.ActivationType.GELU_ERF)
                logits = self.out(self.net.add_activation(logits, activation),
                                  f"head.{index}.activation")
        for name, value in (("embeddings", embeddings), ("logits", logits)):
            value = self._restore_rows(value, public_shape, f"dense.{name}.public_shape")
            output = self.cast(value, self.trt.float32, f"{name}.float_output")
            output.name = name
            self.net.mark_output(output)


def build_dense_engine(config, weights, max_batch_size, library, namespace, verbose=False):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    registry = builder.get_plugin_registry()
    registry.parent_search_enabled = False
    handle = registry.load_library(str(library))
    if handle is None:
        raise RuntimeError("Could not load HSTU dense attention library into builder")
    graph = net = creator = linear_creator = value = None
    try:
        creator = registry.get_creator("HstuDenseAttention", "1", namespace)
        if creator is None:
            raise RuntimeError("HSTU dense attention creator is missing")
        if config["hidden_size"] == 256 and config["add_uvqk_bias"]:
            linear_creator = registry.get_creator("HstuUvqk", "1", namespace)
            if linear_creator is None:
                raise RuntimeError("HSTU native projection creator is missing")
        net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        graph = DenseGraph(trt, net, config, weights, "bf16", creator=creator,
                           linear_creator=linear_creator)
        graph.outputs()
        opt_b, max_s = min(max_batch_size, 4), config["max_sequence_length"]
        opt_s = min(max_s, 128)
        profile = builder.create_optimization_profile()
        for index in range(net.num_inputs):
            value = net.get_input(index)
            if value.name == "attention_metadata":
                shapes = (5, 2, 8), (5, opt_b + 1, 8), (5, max_batch_size + 1, 8)
            else:
                shapes = (1, 1), (opt_b, opt_s), (max_batch_size, max_s)
            profile.set_shape(value.name, *shapes)
            if tuple(tuple(shape) for shape in profile.get_shape(value.name)) != shapes:
                raise RuntimeError(f"TensorRT rejected dense HSTU profile for {value.name}")
        options = builder.create_builder_config()
        options.builder_optimization_level = 3
        options.clear_flag(trt.BuilderFlag.TF32)
        options.add_optimization_profile(profile)
        plan = builder.build_serialized_network(net, options)
        if plan is None:
            raise RuntimeError("TensorRT failed to build the uncached HSTU engine")
        return bytes(plan)
    finally:
        value = graph = net = creator = linear_creator = None
        registry.deregister_library(handle)
