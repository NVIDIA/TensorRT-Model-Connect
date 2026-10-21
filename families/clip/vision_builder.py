# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLIP vision tower: ViT patches to one projected embedding.

Two details here are stated in the checkpoint rather than in the paper, and
each produces a same-shaped wrong answer if guessed:

* the class token is **prepended**, so the position table has one more row than
  the patch grid and row 0 belongs to the class token;
* there are **two** layer norms outside the encoder stack. ``pre_layrnorm``
  (the checkpoint's spelling, not a typo on our side) runs before the first
  layer, and ``post_layernorm`` runs on the class row only, after the last.
  Measured: the pooled vector is exactly ``post_layernorm(last_hidden[:, 0])``.
"""

from __future__ import annotations

import numpy as np

from . import encoder_builder, graph as g

PREFIX = "vision_model"


def build_vision_tower(network, pixel_values, weights, cfg, projection, *, dtype=np.float32):
    """``[1, 3, H, W]`` pixels to an unnormalised ``[1, projection_dim]`` embedding."""
    width = cfg["hidden_size"]
    patch = cfg["patch_size"]

    # Patch embedding is a bias-free stride-patch convolution.
    patches = g.add_conv2d(network, pixel_values,
                           weights[f"{PREFIX}.embeddings.patch_embedding.weight"],
                           stride=(patch, patch), dtype=dtype)
    flat = network.add_shuffle(patches)
    grid = cfg["image_size"] // patch
    flat.reshape_dims = (width, grid * grid)
    flat.second_transpose = (1, 0)
    tokens = flat.get_output(0)

    class_token = g.add_constant(
        network, (1, width), weights[f"{PREFIX}.embeddings.class_embedding"], dtype=dtype)
    concat = network.add_concatenation([class_token, tokens])
    concat.axis = 0
    hidden = concat.get_output(0)

    positions = g.add_constant(network, (cfg["num_positions"], width),
                               weights[f"{PREFIX}.embeddings.position_embedding.weight"],
                               dtype=dtype)
    hidden = g.add_sum(network, hidden, positions)
    hidden = g.add_layer_norm(network, hidden, weights[f"{PREFIX}.pre_layrnorm.weight"],
                              weights[f"{PREFIX}.pre_layrnorm.bias"], cfg["eps"])

    hidden = encoder_builder.build_layers(network, hidden, weights, PREFIX, cfg, dtype=dtype)

    class_row = network.add_slice(hidden, start=(0, 0), shape=(1, width), stride=(1, 1))
    pooled = g.add_layer_norm(network, class_row.get_output(0),
                              weights[f"{PREFIX}.post_layernorm.weight"],
                              weights[f"{PREFIX}.post_layernorm.bias"], cfg["eps"])
    return g.add_linear(network, pooled, projection, dtype=dtype)
