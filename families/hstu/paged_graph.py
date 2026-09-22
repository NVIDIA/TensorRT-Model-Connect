# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native TensorRT page updates and the original NVIDIA HSTU attention body.

Requests use compact query rows. Historical K/V lives in owned pages; candidate
K/V is read directly from the current projection and is never put in those pages.
The runtime owns and validates every offset and the page-tail initialization.
"""

from __future__ import annotations

import numpy as np

from .model import _Graph


class PagedGraph(_Graph):
    def __init__(self, *args, creator, linear_creator=None, projection_creator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.creator = creator
        self.linear_creator = linear_creator
        self.projection_creator = projection_creator
        self.plugins = []
        width = self.config["num_heads"] * self.config["head_dim"]
        for index in range(self.config["num_layers"]):
            prefix = f"blocks.{index}.uvqk"
            # The base constructor canonicalizes head-major UVQK exactly once.
            # UQKV makes both current attention and native K/V writes views.
            self.weights[f"{prefix}.weight"] = np.ascontiguousarray(
                self.weights[f"{prefix}.weight"]
                .reshape(4, width, self.config["hidden_size"])[[0, 2, 3, 1]]
                .reshape(4 * width, self.config["hidden_size"])
            )
            if self.config["add_uvqk_bias"]:
                self.weights[f"{prefix}.bias"] = np.ascontiguousarray(
                    self.weights[f"{prefix}.bias"].reshape(4, width)[[0, 2, 3, 1]]
                    .reshape(4 * width)
                )

    def linear(self, value, prefix, bias=True):
        # A model-owned stock cuBLASLt specialization. All token counts and
        # batches use the same pointer-aware selection policy and BF16/F32 math.
        if self.linear_creator is None or not prefix.endswith(".uvqk") or not bias:
            return super().linear(value, prefix, bias)
        weight = self.constant(np.ascontiguousarray(self.weights[f"{prefix}.weight"].T),
                               f"{prefix}.weight", value.dtype)
        vector = self.constant(self.weights[f"{prefix}.bias"], f"{prefix}.bias", value.dtype)
        plugin = self.linear_creator.create_plugin(
            prefix, self.trt.PluginFieldCollection([]), self.trt.TensorRTPhase.BUILD
        )
        if plugin is None:
            raise RuntimeError("Could not create HSTU cuBLASLt projection adapter")
        self.plugins.append(plugin)
        return self.out(self.net.add_plugin_v3([value, weight, vector], [], plugin), prefix)

    def materialize_projection(self, projection, prefix):
        # Dense passes no creator and keeps the same tensor/graph. Paged uses a
        # shared opaque boundary before U, attention and native-KV consumers.
        if self.projection_creator is None:
            return projection
        name = f"{prefix}.projection_barrier"
        plugin = self.projection_creator.create_plugin(
            name, self.trt.PluginFieldCollection([]), self.trt.TensorRTPhase.BUILD
        )
        if plugin is None:
            raise RuntimeError("Could not create HSTU projection barrier")
        self.plugins.append(plugin)
        return self.out(self.net.add_plugin_v3([projection], [], plugin), name)

    def updated_pages(self, projection, index, metadata):
        width = self.config["num_heads"] * self.config["head_dim"]
        prefix = f"pages.{index}"
        rows = metadata["page_update_rows"]
        zero = self.constant(np.asarray([0], np.int32), f"{prefix}.zero_index")
        sentinel = self.constant(np.asarray([-1], np.int32), f"{prefix}.sentinel")
        safe_rows = self.binary(rows, zero, "MAX", f"{prefix}.safe_rows")
        selected = self.out(
            self.net.add_gather(projection, safe_rows, 0), f"{prefix}.selected_projection"
        )
        kv = self.last_axis_slice(selected, 2 * width, 2 * width, f"{prefix}.kv")
        update = self.reshape(kv, (-1, 2, width), f"{prefix}.packed_kv")
        valid = self.reshape(
            self.binary(rows, sentinel, "GREATER", f"{prefix}.real_rows"),
            (0, 1, 1), f"{prefix}.real_row_mask",
        )
        # Select, not multiplication: unknown page tails may contain NaN/Inf.
        update = self.out(
            self.net.add_select(valid, update, self.scalar(0.0, update, f"{prefix}.zero")),
            f"{prefix}.sanitized_updates",
        )
        pages = self.net.add_input(f"cache_{index}_pages", self.dtype, (-1, 2, 128, width))
        pages.set_dimension_name(0, "cache_pages")
        layer = self.net.add_kv_cache_update(
            pages, update, metadata["page_write_indices"], self.trt.KVCacheMode.LINEAR
        )
        if layer is None:
            raise RuntimeError("TensorRT rejected the HSTU native page update")
        layer.update_form = self.trt.AttentionIOForm.PACKED_NHD
        layer.update_lengths = metadata["page_update_lengths"]
        present = self.out(layer, f"{prefix}.update")
        present.name = f"present_{index}_pages"
        self.net.mark_output(present)
        return present

    def block(self, value, mask, scale, index, cache=None):
        c, prefix = self.config, f"blocks.{index}"
        width = c["num_heads"] * c["head_dim"]
        x = self.norm(value, f"{prefix}.input_norm", c["learnable_input_layernorm"])
        projection = self.silu(
            self.linear(x, f"{prefix}.uvqk", c["add_uvqk_bias"]),
            f"{prefix}.silu", model_dtype=True,
        )
        projection = self.materialize_projection(projection, prefix)
        u = self.last_axis_slice(projection, 0, width, f"{prefix}.u")
        pages = self.updated_pages(projection, index, cache)
        plugin = self.creator.create_plugin(
            f"{prefix}.attention", self.trt.PluginFieldCollection([]), self.trt.TensorRTPhase.BUILD
        )
        if plugin is None:
            raise RuntimeError("Could not create HSTU attention adapter")
        self.plugins.append(plugin)
        attention = self.out(
            self.net.add_plugin_v3([projection, pages, cache["attention_metadata"]], [], plugin),
            f"{prefix}.attention",
        )
        attention = self.reshape(attention, (-1, width), f"{prefix}.attention_rows")
        attention = self.norm(
            attention, f"{prefix}.output_norm", c["learnable_output_layernorm"], preserve_fp32=True
        )
        gated = self.binary(
            self.cast(u, self.trt.float32, f"{prefix}.u.fp32"), attention, "PROD", f"{prefix}.gate"
        )
        projected = self.linear(self.cast(gated, value.dtype, f"{prefix}.gate.output"),
                                f"{prefix}.proj", False)
        return self.binary(projected, value, "SUM", f"{prefix}.residual") if c["residual"] else projected

    def outputs(self):
        c = self.config
        ids = self.net.add_input("token_ids", self.trt.int32, (-1,))
        cache = {name: self.net.add_input(name, self.trt.int32, (-1,)) for name in
                 ("page_write_indices", "page_update_rows", "page_update_lengths")}
        cache["page_write_indices"].set_dimension_name(0, "cache_pages")
        cache["attention_metadata"] = self.net.add_input(
            "attention_metadata", self.trt.int32, (5, -1, 8)
        )
        merged = np.concatenate(
            [self.weights[f"embeddings.{table['name']}.weight"] for table in c["embedding_tables"]]
        )
        table = self.constant(merged, "embedding.weight", self.dtype)
        x = self.out(self.net.add_gather(table, ids, 0), "embedding.lookup")
        if c["position_buckets"]:
            # Same FP32 position-only path as the ordinary HSTU builder.
            x = self.cast(x, self.trt.float32, "position.input.fp32")
            x = self.binary(x, self.scalar(c["hidden_size"] ** 0.5, x, "position.scale"),
                            "PROD", "position.scaled_embeddings")
            positions = self.net.add_input("position_ids", self.trt.int32, (-1,))
            weight = self.constant(self.weights["position.weight"], "position.weight", self.dtype)
            positional = self.out(self.net.add_gather(weight, positions, 0), "position.lookup")
            positional = self.cast(positional, self.trt.float32, "position.fp32")
            x = self.cast(self.binary(x, positional, "SUM", "position.add"),
                          self.dtype, "position.output")
        for index in range(c["num_layers"]):
            x = self.block(x, None, None, index, cache)
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
            output = self.cast(value, self.trt.float32, f"{name}.float_output")
            output.name = name
            self.net.mark_output(output)


def build_paged_engine(config, weights, max_batch_size, library, namespace, verbose=False):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    registry = builder.get_plugin_registry()
    registry.parent_search_enabled = False
    handle = registry.load_library(str(library))
    if handle is None:
        raise RuntimeError("Could not load HSTU native attention library into builder")
    graph = net = creator = linear_creator = projection_creator = value = None
    try:
        creator = registry.get_creator("HstuPagedAttention", "1", namespace)
        if creator is None:
            raise RuntimeError("HSTU native attention creator is missing")
        if config["hidden_size"] == 256 and config["add_uvqk_bias"]:
            linear_creator = registry.get_creator("HstuUvqk", "1", namespace)
            if linear_creator is None:
                raise RuntimeError("HSTU native projection creator is missing")
        projection_creator = registry.get_creator("HstuProjectionBarrier", "1", namespace)
        if projection_creator is None:
            raise RuntimeError("HSTU native projection barrier creator is missing")
        net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        graph = PagedGraph(trt, net, config, weights, "bf16", creator=creator,
                           linear_creator=linear_creator, projection_creator=projection_creator)
        graph.outputs()
        opt_b, max_s = min(max_batch_size, 4), config["max_sequence_length"]
        opt_s = min(max_s, 128)
        max_pages = max_batch_size * ((max_s + 127) // 128)
        opt_pages = min(max_pages, opt_b * 4)
        width = config["num_heads"] * config["head_dim"]
        profile = builder.create_optimization_profile()
        for index in range(net.num_inputs):
            value = net.get_input(index)
            if value.name.endswith("_pages"):
                shapes = (1, 2, 128, width), (opt_pages, 2, 128, width), (max_pages, 2, 128, width)
            elif value.name == "attention_metadata":
                shapes = (5, 2, 8), (5, opt_b + 1, 8), (5, max_batch_size + 1, 8)
            elif value.name == "page_write_indices":
                shapes = (1,), (opt_pages,), (max_pages,)
            elif value.name == "page_update_lengths":
                shapes = (2,), (opt_pages + 1,), (max_pages + 1,)
            elif value.name == "page_update_rows":
                shapes = (0,), (opt_b * opt_s,), (max_pages * 128,)
            else:
                shapes = (1,), (opt_b * opt_s,), (max_batch_size * max_s,)
            profile.set_shape(value.name, *shapes)
            if tuple(tuple(shape) for shape in profile.get_shape(value.name)) != shapes:
                raise RuntimeError(f"TensorRT rejected HSTU profile for {value.name}")
        options = builder.create_builder_config()
        options.builder_optimization_level = 3
        options.clear_flag(trt.BuilderFlag.TF32)
        options.set_preview_feature(trt.PreviewFeature.ALIASED_PLUGIN_IO_10_03, True)
        if not options.get_preview_feature(trt.PreviewFeature.ALIASED_PLUGIN_IO_10_03):
            raise RuntimeError("TensorRT did not enable HSTU projection I/O aliasing")
        options.add_optimization_profile(profile)
        plan = builder.build_serialized_network(net, options)
        if plan is None:
            raise RuntimeError("TensorRT failed to build the HSTU paged engine")
        return bytes(plan)
    finally:
        # Destroy every wrapper using the library before unloading it.
        value = graph = net = creator = linear_creator = projection_creator = None
        registry.deregister_library(handle)
