# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLIPVisionModelWithProjection: preprocessed pixels -> projected image embedding.

transformers' CLIP vision tower: a bias-free patch convolution, a class token
and learned position embeddings, ``pre_layrnorm``, pre-norm encoder layers with
exact-erf GELU MLPs, ``post_layernorm`` on the class token and a bias-free
``visual_projection``.
"""

from __future__ import annotations

import math

import numpy as np
import tensorrt as trt

from . import graph as g

_PREFIX = "vision_model"


def _attention(network, tensor, weights, prefix: str, heads: int, work):
    """CLIPAttention; the 1/sqrt(head_dim) scale multiplies q_proj's FP32 result."""
    width = int(np.asarray(weights[f"{prefix}.q_proj.weight"]).shape[0])
    query = g.linear(network, tensor, weights[f"{prefix}.q_proj.weight"],
                     weights[f"{prefix}.q_proj.bias"], work,
                     output_scale=1.0 / math.sqrt(width // heads))
    key = g.linear(network, tensor, weights[f"{prefix}.k_proj.weight"],
                   weights[f"{prefix}.k_proj.bias"], work)
    value = g.linear(network, tensor, weights[f"{prefix}.v_proj.weight"],
                     weights[f"{prefix}.v_proj.bias"], work)
    # One 257-token image: the attention itself runs in FP32 at negligible cost.
    query, key, value = (g.cast(network, tensor, trt.float32) for tensor in (query, key, value))
    attended = g.attention(network, g.split_heads(network, query, heads),
                           g.split_heads(network, key, heads), g.split_heads(network, value, heads))
    attended = g.cast(network, attended, work)
    return g.linear(network, g.merge_heads(network, attended),
                    weights[f"{prefix}.out_proj.weight"], weights[f"{prefix}.out_proj.bias"], work)


def build_image_encoder(network, weights, cfg: dict, work) -> None:
    """Mark ``image_embeds`` [1, projection_dim] for ``pixel_values`` [1, 3, S, S]."""
    size = cfg["image_size"]
    patch = cfg["patch_size"]
    hidden = cfg["hidden_size"]
    heads = cfg["num_attention_heads"]
    eps = cfg["layer_norm_eps"]
    if size % patch:
        raise ValueError("CLIP image size must be a multiple of the patch size")
    grid = size // patch

    pixels = network.add_input("pixel_values", trt.float32, (1, 3, size, size))
    x = g.cast(network, pixels, work)
    patches = g.conv(network, x, weights[f"{_PREFIX}.embeddings.patch_embedding.weight"], None,
                     work, stride=(patch, patch))
    tokens = g.transpose(network, g.reshape(network, patches, (1, hidden, grid * grid)), (0, 2, 1))
    dtype = g.numpy_dtype(work)
    class_token = g.constant(
        network, np.asarray(weights[f"{_PREFIX}.embeddings.class_embedding"]).reshape(1, 1, hidden),
        dtype)
    x = g.concat(network, [class_token, tokens], axis=1)
    positions = np.asarray(weights[f"{_PREFIX}.embeddings.position_embedding.weight"])
    if positions.shape != (grid * grid + 1, hidden):
        raise ValueError("CLIP position table does not match the image grid")
    x = g.add(network, x, g.constant(network, positions.reshape(1, grid * grid + 1, hidden), dtype))
    x = g.layer_norm(network, x, weights[f"{_PREFIX}.pre_layrnorm.weight"],
                     weights[f"{_PREFIX}.pre_layrnorm.bias"], eps, work)

    for index in range(cfg["num_hidden_layers"]):
        prefix = f"{_PREFIX}.encoder.layers.{index}"
        normed = g.layer_norm(network, x, weights[f"{prefix}.layer_norm1.weight"],
                              weights[f"{prefix}.layer_norm1.bias"], eps, work)
        x = g.add(network, x, _attention(network, normed, weights, f"{prefix}.self_attn", heads, work))
        normed = g.layer_norm(network, x, weights[f"{prefix}.layer_norm2.weight"],
                              weights[f"{prefix}.layer_norm2.bias"], eps, work)
        inner = g.gelu(network, g.linear(network, normed, weights[f"{prefix}.mlp.fc1.weight"],
                                         weights[f"{prefix}.mlp.fc1.bias"], work))
        x = g.add(network, x, g.linear(network, inner, weights[f"{prefix}.mlp.fc2.weight"],
                                       weights[f"{prefix}.mlp.fc2.bias"], work))

    pooled = g.reshape(network, g.slice_axis(network, x, 1, 0, 1), (1, hidden))
    pooled = g.layer_norm(network, pooled, weights[f"{_PREFIX}.post_layernorm.weight"],
                          weights[f"{_PREFIX}.post_layernorm.bias"], eps, work)
    embeds = g.linear(network, pooled, weights["visual_projection.weight"], None, work)
    embeds = g.cast(network, embeds, trt.float32)
    embeds.name = "image_embeds"
    network.mark_output(embeds)
