# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BLIP-2 ViT, query-only Q-Former and learned-position OPT computation graphs."""

import numpy as np
import tensorrt as trt

from . import graph as g


class Graph:
    def __init__(self, weights, verbose):
        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        self.builder = trt.Builder(self.logger)
        self.net = self.builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        )
        self.options = self.builder.create_builder_config()
        self.options.builder_optimization_level = 3
        self.options.clear_flag(trt.BuilderFlag.TF32)
        self.weights = weights

    def const(self, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        return self.net.add_constant(x.shape, x).get_output(0)

    def reshape(self, x, shape):
        layer = self.net.add_shuffle(x)
        layer.reshape_dims = shape
        return layer.get_output(0)

    def add(self, a, b):
        return self.net.add_elementwise(a, b, trt.ElementWiseOperation.SUM).get_output(0)

    def mul(self, a, b):
        return self.net.add_elementwise(a, b, trt.ElementWiseOperation.PROD).get_output(0)

    def linear(self, x, name):
        return g.add_linear(
            self.net, x, self.weights[name + ".weight"], self.weights.get(name + ".bias")
        )

    def norm(self, x, name, eps):
        return g.add_layer_norm(
            self.net, x, self.weights[name + ".weight"], self.weights[name + ".bias"], eps
        )

    def attention(self, q, k, v, heads, mask=None):
        qlen, width = q.shape
        klen = k.shape[0]
        dim = width // heads

        def split(x, length):
            layer = self.net.add_shuffle(x)
            layer.reshape_dims = (1, length, heads, dim)
            layer.second_transpose = (0, 2, 1, 3)
            return layer.get_output(0)

        scores = self.net.add_matrix_multiply(
            split(q, qlen), trt.MatrixOperation.NONE, split(k, klen), trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
        scores = self.mul(scores, self.const(np.full((1, 1, 1, 1), dim**-0.5)))
        if mask is not None:
            scores = self.add(scores, mask)
        softmax = self.net.add_softmax(scores)
        softmax.axes = 8
        context = self.net.add_matrix_multiply(
            softmax.get_output(0),
            trt.MatrixOperation.NONE,
            split(v, klen),
            trt.MatrixOperation.NONE,
        ).get_output(0)
        layer = self.net.add_shuffle(context)
        layer.first_transpose = (0, 2, 1, 3)
        layer.reshape_dims = (qlen, width)
        return layer.get_output(0)

    def finish(self):
        plan = self.builder.build_serialized_network(self.net, self.options)
        if plan is None:
            raise RuntimeError("BLIP-2 engine build failed")
        return bytes(plan)


def build_vision(weights, config, verbose=False):
    graph = Graph(weights, verbose)
    net = graph.net
    cfg = config["vision_config"]
    width = cfg["hidden_size"]
    image = cfg["image_size"]
    patch = cfg["patch_size"]
    patches = (image // patch) ** 2
    length = patches + 1
    pixels = net.add_input("pixel_values", trt.float32, (1, 3, image, image))
    name = "vision_model.embeddings.patch_embedding"
    conv = net.add_convolution_nd(
        pixels, width, (patch, patch), weights[name + ".weight"], weights[name + ".bias"]
    )
    conv.stride_nd = (patch, patch)
    layer = net.add_shuffle(conv.get_output(0))
    layer.reshape_dims = (width, patches)
    layer.second_transpose = (1, 0)
    cls = graph.const(weights["vision_model.embeddings.class_embedding"].reshape(1, width))
    concat = net.add_concatenation([cls, layer.get_output(0)])
    concat.axis = 0
    hidden = graph.add(
        concat.get_output(0),
        graph.const(weights["vision_model.embeddings.position_embedding"].reshape(length, width)),
    )
    for index in range(cfg["num_hidden_layers"]):
        prefix = f"vision_model.encoder.layers.{index}"
        normalized = graph.norm(hidden, prefix + ".layer_norm1", cfg["layer_norm_eps"])
        qkv = graph.linear(normalized, prefix + ".self_attn.qkv")
        q, k, v = [
            net.add_slice(qkv, (0, j * width), (length, width), (1, 1)).get_output(0)
            for j in range(3)
        ]
        attended = graph.attention(q, k, v, cfg["num_attention_heads"])
        hidden = graph.add(hidden, graph.linear(attended, prefix + ".self_attn.projection"))
        inner = graph.linear(
            graph.norm(hidden, prefix + ".layer_norm2", cfg["layer_norm_eps"]), prefix + ".mlp.fc1"
        )
        inner = net.add_activation(inner, trt.ActivationType.GELU_ERF).get_output(0)
        hidden = graph.add(hidden, graph.linear(inner, prefix + ".mlp.fc2"))
    memory = graph.norm(hidden, "vision_model.post_layernorm", cfg["layer_norm_eps"])
    cfg = config["qformer_config"]
    hidden = graph.norm(
        graph.const(
            weights["query_tokens"].reshape(config["num_query_tokens"], cfg["hidden_size"])
        ),
        "qformer.layernorm",
        cfg["layer_norm_eps"],
    )
    for index in range(cfg["num_hidden_layers"]):
        prefix = f"qformer.encoder.layer.{index}"
        for kind in ("attention", "crossattention"):
            if kind == "crossattention" and index % cfg["cross_attention_frequency"]:
                continue
            name = prefix + "." + kind
            source = memory if kind == "crossattention" else hidden
            q = graph.linear(hidden, name + ".attention.query")
            k = graph.linear(source, name + ".attention.key")
            v = graph.linear(source, name + ".attention.value")
            attended = graph.attention(q, k, v, cfg["num_attention_heads"])
            hidden = graph.norm(
                graph.add(hidden, graph.linear(attended, name + ".output.dense")),
                name + ".output.LayerNorm",
                cfg["layer_norm_eps"],
            )
        inner = graph.linear(hidden, prefix + ".intermediate_query.dense")
        inner = net.add_activation(inner, trt.ActivationType.GELU_ERF).get_output(0)
        hidden = graph.norm(
            graph.add(hidden, graph.linear(inner, prefix + ".output_query.dense")),
            prefix + ".output_query.LayerNorm",
            cfg["layer_norm_eps"],
        )
    hidden = graph.linear(hidden, "language_projection")
    hidden.name = "image_features"
    net.mark_output(hidden)
    return graph.finish()


def build_decoder(weights, config, length, verbose=False):
    graph = Graph(weights, verbose)
    net = graph.net
    cfg = config["text_config"]
    width = cfg["hidden_size"]
    prefix = "language_model.model.decoder"
    ids = net.add_input("input_ids", trt.int32, (length,))
    valid = net.add_input("attention_mask", trt.int32, (length,))
    image = net.add_input("image_rows", trt.float32, (length, width))
    image_mask = net.add_input("image_mask", trt.float32, (length, 1))
    last = net.add_input("last_index", trt.int32, (1,))
    hidden = net.add_gather(
        graph.const(weights[prefix + ".embed_tokens.weight"]), ids, 0
    ).get_output(0)
    keep = net.add_elementwise(
        graph.const(np.ones((length, 1))), image_mask, trt.ElementWiseOperation.SUB
    ).get_output(0)
    hidden = graph.add(graph.mul(hidden, keep), image)
    # All real tokens form a contiguous prefix. OPT positions start at learned row 2.
    hidden = graph.add(
        hidden, graph.const(weights[prefix + ".embed_positions.weight"][2 : length + 2])
    )
    mask = graph.reshape(net.add_cast(valid, trt.float32).get_output(0), (1, 1, 1, length))
    mask = net.add_elementwise(
        graph.const(np.ones((1, 1, 1, 1))), mask, trt.ElementWiseOperation.SUB
    ).get_output(0)
    mask = graph.mul(mask, graph.const(np.full((1, 1, 1, 1), -1e9)))
    mask = graph.add(
        mask, graph.const(np.triu(np.full((1, 1, length, length), -1e9, np.float32), k=1))
    )
    for index in range(cfg["num_hidden_layers"]):
        name = prefix + f".layers.{index}"
        normalized = graph.norm(hidden, name + ".self_attn_layer_norm", 1e-5)
        q, k, v = [
            graph.linear(normalized, name + ".self_attn." + x)
            for x in ("q_proj", "k_proj", "v_proj")
        ]
        attended = graph.attention(q, k, v, cfg["num_attention_heads"], mask)
        hidden = graph.add(hidden, graph.linear(attended, name + ".self_attn.out_proj"))
        inner = graph.linear(graph.norm(hidden, name + ".final_layer_norm", 1e-5), name + ".fc1")
        inner = net.add_activation(inner, trt.ActivationType.RELU).get_output(0)
        hidden = graph.add(hidden, graph.linear(inner, name + ".fc2"))
    hidden = graph.norm(hidden, prefix + ".final_layer_norm", 1e-5)
    selected = net.add_gather(hidden, last, 0).get_output(0)
    logits = g.add_linear(net, selected, weights[prefix + ".embed_tokens.weight"])
    logits.name = "logits"
    net.mark_output(logits)
    return graph.finish()
