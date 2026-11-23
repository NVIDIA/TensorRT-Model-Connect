# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Idefics3 vision transformer and spatial pixel-shuffle connector."""

from __future__ import annotations

import numpy as np

from . import encoder_builder, graph as g

PREFIX = "model.vision_model"


def build_vision_tower(network, pixel_values, weights, cfg, *, dtype=np.float32, quant=None):
    """One square image view to its connector rows ``[image_sequence, text_width]``."""
    width = cfg["hidden_size"]
    patch = cfg["patch_size"]
    grid = cfg["grid"]

    patches = g.add_conv2d(
        network,
        pixel_values,
        weights[f"{PREFIX}.embeddings.patch_embedding.weight"],
        weights[f"{PREFIX}.embeddings.patch_embedding.bias"],
        stride=(patch, patch),
        dtype=dtype,
    )
    flat = network.add_shuffle(patches)
    flat.reshape_dims = (width, grid * grid)
    flat.second_transpose = (1, 0)
    hidden = flat.get_output(0)

    positions = g.add_constant(
        network,
        (cfg["num_positions"], width),
        weights[f"{PREFIX}.embeddings.position_embedding.weight"],
        dtype=dtype,
    )
    hidden = g.add_sum(network, hidden, positions)

    hidden = encoder_builder.build_layers(
        network, hidden, weights, PREFIX, cfg, dtype=dtype, quant=quant
    )
    hidden = g.add_layer_norm(
        network,
        hidden,
        weights[f"{PREFIX}.post_layernorm.weight"],
        weights[f"{PREFIX}.post_layernorm.bias"],
        cfg["eps"],
    )
    factor = cfg["scale_factor"]
    merged = network.add_shuffle(hidden)
    merged.reshape_dims = (1, grid, grid // factor, width * factor)
    merged.second_transpose = (0, 2, 1, 3)
    packed = network.add_shuffle(merged.get_output(0))
    packed.reshape_dims = (1, grid // factor, grid // factor, width * factor * factor)
    packed.second_transpose = (0, 2, 1, 3)
    rows = network.add_shuffle(packed.get_output(0))
    rows.reshape_dims = (grid * grid // (factor * factor), width * factor * factor)
    return g.add_linear(
        network,
        rows.get_output(0),
        weights["model.connector.modality_projection.proj.weight"],
        None,
        dtype=dtype,
    )
