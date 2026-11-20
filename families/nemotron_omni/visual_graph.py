# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native RADIO image/video encoders; TensorRT owns all GPU execution.

One trunk serves image patches and true multi-frame tubelet patches. Separate
small patch graphs avoid duplicating the complete transformer for each modality.
The owning C++ runtime provides position interpolation and pixel-shuffle indices.
"""

from __future__ import annotations

import tensorrt as trt


from .component_graph import _Graph


def _input(graph, name, width, maximum):
    tensor = graph.network.add_input(name, graph.dtype, (-1, width))
    tensor.set_dimension_name(0, "patch_count")
    graph.profile.set_shape(name, (4, width), (min(1024, maximum), width), (maximum, width))
    return tensor


def build_patch(config, weights, max_patches, *, video=False, precision="fp32", verbose=False):
    """Build actual checkpoint image or temporal-video patch projection."""
    graph = _Graph(precision, verbose)
    frames = int(config.vision.get("video_temporal_patch_size", 2)) if video else 1
    if frames <= 0 or (video and not config.vision.get("separate_video_embedder", False)):
        raise ValueError("Omni video requires its checkpoint-owned temporal patch embedder")
    width = frames * 3 * config.raw["patch_size"] ** 2
    prefix = "vision_model.radio_model.model.patch_generator."
    prefix += "video_embedder" if video else "embedder"
    if weights[prefix + ".weight"].shape != (config.raw["vit_hidden_size"], width):
        raise ValueError("Omni patch embedder dimensions differ from the checkpoint configuration")
    pixels = _input(graph, "patch_pixels", width, max_patches)
    output = graph.linear(pixels, weights, prefix)
    graph.output(output, "patch_embeddings")
    return graph.finish()


def build_visual(config, weights, max_patches, *, precision="fp32", verbose=False):
    """Build complete RADIO registers, transformer and language projector."""
    if type(max_patches) is not int or max_patches < 4 or max_patches % 4:
        raise ValueError("Omni visual capacity must contain complete 2x2 patch groups")
    graph = _Graph(precision, verbose)
    net = graph.network
    hidden = config.raw["vit_hidden_size"]
    heads = int(config.vision.get("num_attention_heads", hidden // 80))
    if heads <= 0 or hidden % heads:
        raise ValueError("Omni visual heads must divide its hidden width")
    dim = hidden // heads
    root = "vision_model.radio_model.model"
    patches = _input(graph, "patch_embeddings", hidden, max_patches)
    positions = _input(graph, "position_embeddings", hidden, max_patches)
    shuffle = net.add_input("shuffle_indices", trt.int32, (-1, 4))
    graph.profile.set_shape(
        "shuffle_indices", (1, 4), (min(256, max_patches // 4), 4), (max_patches // 4, 4)
    )
    registers = weights[root + ".patch_generator.cls_token.token"]
    if registers.ndim != 2 or registers.shape[1] != hidden or registers.shape[0] < 1:
        raise ValueError("Omni RADIO register tokens have invalid dimensions")
    joined = net.add_concatenation([graph.constant(registers), graph.add(patches, positions)])
    joined.axis = 0
    state = joined.get_output(0)
    layers = int(config.vision.get("num_hidden_layers", 32))
    for index in range(layers):
        prefix = f"{root}.blocks.{index}"
        normalized = graph.norm(state, weights, prefix + ".norm1", 1e-6, centered=True)
        qkv = graph.linear(normalized, weights, prefix + ".attn.qkv", bias=True)
        qkv = graph.reshape(qkv, (-1, 3, heads, dim), (1, 2, 0, 3))
        parts = []
        for part in range(3):
            tensor = net.add_gather(qkv, graph.constant([part], trt.int32), 0).get_output(0)
            parts.append(graph.reshape(tensor, (heads, -1, dim)))
        attended = graph.attention(*parts, dim)
        reorder = net.add_shuffle(attended)
        reorder.first_transpose = (1, 0, 2)
        reorder.reshape_dims = (-1, hidden)
        state = graph.add(
            state, graph.linear(reorder.get_output(0), weights, prefix + ".attn.proj", bias=True)
        )
        normalized = graph.norm(state, weights, prefix + ".norm2", 1e-6, centered=True)
        feed_forward = graph.gelu(graph.linear(normalized, weights, prefix + ".mlp.fc1", bias=True))
        state = graph.add(
            state, graph.linear(feed_forward, weights, prefix + ".mlp.fc2", bias=True)
        )
    # Offset the actual patch indices, excluding all position-free registers.
    indices = graph.add(shuffle, graph.constant([[registers.shape[0]]], trt.int32))
    gathered = net.add_gather(state, indices, 0).get_output(0)
    merged = graph.reshape(gathered, (-1, hidden * 4))
    merged = graph.norm(merged, weights, "mlp1.0", 1e-5)
    projected = graph.linear(merged, weights, "mlp1.1")
    projected = net.add_activation(projected, trt.ActivationType.RELU).get_output(0)
    projected = graph.linear(graph.mul(projected, projected), weights, "mlp1.3")
    graph.output(projected, "image_features", trt.float32)
    return graph.finish()
