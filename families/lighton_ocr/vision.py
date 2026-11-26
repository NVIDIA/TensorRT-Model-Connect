# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pixtral encoder and LightOn's channel-major spatial patch projector."""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from .graph import Graph, create_network, serialize


def vision_rope_tables(config):
    """Axial H/W frequencies in the checkpoint's rotate-half arrangement."""
    width = config["head_dim"]
    side = config["image_size"] // config["patch_size"]
    base = config.get("rope_parameters", {}).get("rope_theta", config.get("rope_theta", 10000))
    inverse = 1.0 / (float(base) ** (np.arange(0, width, 2, dtype=np.float32) / width))
    positions = np.arange(side, dtype=np.float32)
    height = positions[:, None] * inverse[None, ::2]
    horizontal = positions[:, None] * inverse[None, 1::2]
    height = np.broadcast_to(height[:, None], (side, side, width // 4))
    horizontal = np.broadcast_to(horizontal[None], (side, side, width // 4))
    angles = np.concatenate([height, horizontal], axis=-1).reshape(-1, width // 2)
    angles = np.concatenate([angles, angles], axis=-1)
    return np.cos(angles), np.sin(angles)


def _vision_layer(graph, hidden, positions, config, index, tables):
    prefix = f"transformer.layers.{index}"
    heads, width = config["num_attention_heads"], config["head_dim"]
    normalized = graph.norm(hidden, prefix + ".attention_norm", 1e-5)
    query = graph.reshape(graph.linear(normalized, prefix + ".attention.q_proj"), (0, heads, width))
    key = graph.reshape(graph.linear(normalized, prefix + ".attention.k_proj"), (0, heads, width))
    value = graph.reshape(graph.linear(normalized, prefix + ".attention.v_proj"), (0, heads, width))
    query = graph.transpose(graph.rope(query, positions, tables), (1, 0, 2))
    key = graph.transpose(graph.rope(key, positions, tables), (1, 0, 2))
    value = graph.transpose(value, (1, 0, 2))
    attended = graph.attention(query, key, value, width ** -0.5)
    attended = graph.reshape(graph.transpose(attended, (1, 0, 2)), (-1, heads * width))
    hidden = graph.add(hidden, graph.linear(attended, prefix + ".attention.o_proj"))
    normalized = graph.norm(hidden, prefix + ".ffn_norm", 1e-5)
    return graph.add(hidden, graph.mlp(normalized, prefix + ".feed_forward"))


def _merge_patches(graph, features, patch_grid, config):
    """Match unfold(kernel=merge,stride=merge), including odd-grid cropping."""
    merge, hidden = config.spatial_merge_size, config.vision_config["hidden_size"]
    height, width = graph.dimension(patch_grid, 2), graph.dimension(patch_grid, 3)
    divisor = graph.shape_constant([merge], height.dtype)
    rows = graph.op(height, divisor, trt.ElementWiseOperation.FLOOR_DIV)
    columns = graph.op(width, divisor, trt.ElementWiseOperation.FLOOR_DIV)
    grid = graph.reshape_dynamic(features, [height, width, hidden])
    cropped_height = graph.op(rows, divisor, trt.ElementWiseOperation.PROD)
    cropped_width = graph.op(columns, divisor, trt.ElementWiseOperation.PROD)
    crop = graph.net.add_slice(grid, (0, 0, 0), (1, 1, hidden), (1, 1, 1))
    crop.set_input(2, graph.shape_vector([cropped_height, cropped_width, hidden]))
    grid = graph.reshape_dynamic(crop.get_output(0), [rows, merge, columns, merge, hidden])
    # PyTorch unfold flattens channel first, then the two spatial kernel axes.
    grid = graph.transpose(grid, (0, 2, 4, 1, 3))
    return graph.reshape(grid, (-1, hidden * merge * merge))


def build_vision_engine(config, weights, *, verbose=False):
    """One image's complete vision encoder and multimodal projector."""
    vision = config.vision_config
    logger, builder, network, settings = create_network(verbose)
    graph = Graph(network, weights)
    image = network.add_input("pixel_values", trt.float32, (1, 3, -1, -1))
    positions = network.add_input("vision_position_ids", trt.int32, (-1,))
    patch = vision["patch_size"]
    kernel = np.ascontiguousarray(weights["patch_conv.weight"], dtype=np.float32)
    graph.buffers.append(kernel)
    convolution = network.add_convolution_nd(image, vision["hidden_size"], (patch, patch), kernel,
                                              trt.Weights())
    convolution.stride_nd = (patch, patch)
    patch_grid = convolution.get_output(0)
    hidden = graph.reshape(graph.transpose(patch_grid, (0, 2, 3, 1)), (-1, vision["hidden_size"]))
    hidden = graph.norm(hidden, "ln_pre", 1e-5)
    tables = tuple(graph.const(value) for value in vision_rope_tables(vision))
    for index in range(vision["num_hidden_layers"]):
        hidden = _vision_layer(graph, hidden, positions, vision, index, tables)
    hidden = graph.norm(hidden, "vision_projection.norm", config.text_config["rms_norm_eps"])
    hidden = _merge_patches(graph, hidden, patch_grid, config)
    hidden = graph.linear(hidden, "vision_projection.patch_merger.merging_layer")
    hidden = graph.gelu(graph.linear(hidden, "vision_projection.linear_1"))
    hidden = graph.linear(hidden, "vision_projection.linear_2")
    graph.mark(hidden, "image_features")
    minimum = patch * config.spatial_merge_size
    maximum = vision["image_size"]
    optimal_height, optimal_width = min(448, maximum), min(672, maximum)
    optimal_height = max(minimum, optimal_height // minimum * minimum)
    optimal_width = max(minimum, optimal_width // minimum * minimum)
    profile = builder.create_optimization_profile()
    profile.set_shape("pixel_values", (1, 3, minimum, minimum), (1, 3, optimal_height, optimal_width),
                      (1, 3, maximum, maximum))
    profile.set_shape("vision_position_ids", (config.spatial_merge_size ** 2,),
                      ((optimal_height // patch) * (optimal_width // patch),),
                      ((maximum // patch) ** 2,))
    return serialize(builder, network, settings, profile, graph.buffers)
