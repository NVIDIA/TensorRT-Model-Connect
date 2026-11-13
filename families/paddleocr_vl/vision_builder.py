# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PaddleOCR vision encoder with learned positions, axial RoPE and 2x2 merging."""

import numpy as np
import tensorrt as trt

from . import graph as g


def build_vision(weights, cfg, height, width, verbose=False):
    import torch
    from transformers.models.paddleocr_vl.modeling_paddleocr_vl import (
        PaddleOCRVisionRotaryEmbedding,
    )
    from transformers.vision_utils import (
        get_vision_interpolation_indices_and_weights,
        get_vision_position_ids,
    )

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    options = builder.create_builder_config()
    options.builder_optimization_level = 3
    options.clear_flag(trt.BuilderFlag.TF32)
    hidden_size = cfg.hidden_size
    heads = cfg.num_attention_heads
    dim = hidden_size // heads
    patch = cfg.patch_size
    gh, gw = height // patch, width // patch
    count = gh * gw
    prefix = "visual.vision_model"
    grid = torch.tensor([[1, gh, gw]])
    indices, factors = get_vision_interpolation_indices_and_weights(
        grid,
        num_grid_per_side=cfg.image_size // patch,
        mode="bilinear",
        align_corners=True,
        spatial_merge_size=1,
        kwargs={},
    )
    positions = (
        (
            torch.from_numpy(weights[prefix + ".embeddings.position_embedding.weight"])[indices]
            * factors[:, :, None]
        )
        .sum(1)
        .numpy()
    )
    rotary = PaddleOCRVisionRotaryEmbedding(cfg)
    with torch.no_grad():
        cosine, sine = rotary(
            torch.empty(count, hidden_size), get_vision_position_ids(grid, 1, kwargs={})
        )

    def const(value):
        value = np.ascontiguousarray(value, dtype=np.float32)
        return net.add_constant(value.shape, value).get_output(0)

    def linear(x, name):
        return g.add_linear(net, x, weights[name + ".weight"], weights[name + ".bias"])

    def norm(x, name, eps):
        return g.add_layer_norm(net, x, weights[name + ".weight"], weights[name + ".bias"], eps)

    def split(x):
        layer = net.add_shuffle(x)
        layer.reshape_dims = (1, count, heads, dim)
        layer.second_transpose = (0, 2, 1, 3)
        return layer.get_output(0)

    cos = const(cosine.numpy().reshape(1, 1, count, dim))
    sin = const(sine.numpy().reshape(1, 1, count, dim))

    def rope(x):
        first = net.add_slice(
            x, (0, 0, 0, 0), (1, heads, count, dim // 2), (1, 1, 1, 1)
        ).get_output(0)
        second = net.add_slice(
            x, (0, 0, 0, dim // 2), (1, heads, count, dim // 2), (1, 1, 1, 1)
        ).get_output(0)
        second = net.add_unary(second, trt.UnaryOperation.NEG).get_output(0)
        joined = net.add_concatenation([second, first])
        joined.axis = 3
        return g.add_sum(
            net,
            net.add_elementwise(x, cos, trt.ElementWiseOperation.PROD).get_output(0),
            net.add_elementwise(
                joined.get_output(0), sin, trt.ElementWiseOperation.PROD
            ).get_output(0),
        )

    pixels = net.add_input("pixel_values", trt.float32, (1, 3, height, width))
    embedded = g.add_conv2d(
        net,
        pixels,
        weights[prefix + ".embeddings.patch_embedding.weight"],
        weights[prefix + ".embeddings.patch_embedding.bias"],
        stride=(patch, patch),
    )
    flat = net.add_shuffle(embedded)
    flat.reshape_dims = (hidden_size, count)
    flat.second_transpose = (1, 0)
    hidden = g.add_sum(net, flat.get_output(0), const(positions))
    for index in range(cfg.num_hidden_layers):
        layer = f"{prefix}.encoder.layers.{index}"
        normal = norm(hidden, layer + ".layer_norm1", cfg.layer_norm_eps)
        query, key, value = [
            split(linear(normal, layer + ".self_attn." + name))
            for name in ("q_proj", "k_proj", "v_proj")
        ]
        scores = net.add_matrix_multiply(
            rope(query), trt.MatrixOperation.NONE, rope(key), trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
        scores = net.add_elementwise(
            scores, const(np.full((1, 1, 1, 1), dim**-0.5)), trt.ElementWiseOperation.PROD
        ).get_output(0)
        softmax = net.add_softmax(scores)
        softmax.axes = 8
        context = net.add_matrix_multiply(
            softmax.get_output(0), trt.MatrixOperation.NONE, value, trt.MatrixOperation.NONE
        ).get_output(0)
        flat = net.add_shuffle(context)
        flat.first_transpose = (0, 2, 1, 3)
        flat.reshape_dims = (count, hidden_size)
        hidden = g.add_sum(net, hidden, linear(flat.get_output(0), layer + ".self_attn.out_proj"))
        inner = linear(norm(hidden, layer + ".layer_norm2", cfg.layer_norm_eps), layer + ".mlp.fc1")
        hidden = g.add_sum(net, hidden, linear(g.add_gelu_tanh(net, inner), layer + ".mlp.fc2"))
    hidden = norm(hidden, prefix + ".post_layernorm", cfg.layer_norm_eps)
    hidden = norm(hidden, "mlp_AR.pre_norm", 1e-5)
    grouped = net.add_shuffle(hidden)
    grouped.reshape_dims = (gh // 2, 2, gw // 2, 2, hidden_size)
    grouped.second_transpose = (0, 2, 1, 3, 4)
    merged = net.add_shuffle(grouped.get_output(0))
    merged.reshape_dims = (count // 4, hidden_size * 4)
    hidden = net.add_activation(
        linear(merged.get_output(0), "mlp_AR.linear_1"), trt.ActivationType.GELU_ERF
    ).get_output(0)
    hidden = linear(hidden, "mlp_AR.linear_2")
    hidden.name = "image_features"
    net.mark_output(hidden)
    plan = builder.build_serialized_network(net, options)
    if plan is None:
        raise RuntimeError("PaddleOCR vision engine build failed")
    return bytes(plan)
