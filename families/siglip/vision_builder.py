# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SigLIP vision tower: ViT patches to one pooled embedding.

Three details separate this from the CLIP tower it otherwise resembles, and
each one produces a same-shaped wrong answer if carried over:

* there is **no class token**. The position table has exactly one row per
  patch, and nothing is prepended, so a tower that reserves row 0 for a class
  token reads the whole table off by one patch.
* the patch convolution **has a bias**. CLIP's does not.
* pooling is a learned attention head, not a row of the sequence. A single
  learned ``probe`` row attends over the whole grid, and the result is run
  through ``residual + mlp(layernorm(residual))`` where the residual is the
  **attention output** rather than the attention input. Taking row 0 of the
  grid instead, or normalising the input rather than the output, both give a
  vector of the right shape pointing somewhere else.
"""

from __future__ import annotations

import numpy as np

from . import encoder_builder, graph as g

PREFIX = "vision_model"


def _pooling_head(network, hidden, weights, cfg, *, dtype=np.float32):
    """torch ``MultiheadAttention`` with one learned query row, then an MLP."""
    width = cfg["hidden_size"]
    head = f"{PREFIX}.head"

    # torch packs this head's projections into one [3 * width, width] tensor,
    # in q, k, v order. The encoder layers above ship them separately; this one
    # does not, so it is split here rather than looked up under q/k/v names.
    packed_weight = np.asarray(weights[f"{head}.attention.in_proj_weight"])
    packed_bias = np.asarray(weights[f"{head}.attention.in_proj_bias"])
    q_w, k_w, v_w = np.split(packed_weight, 3, axis=0)
    q_b, k_b, v_b = np.split(packed_bias, 3, axis=0)

    probe = g.add_constant(network, (1, width),
                           np.asarray(weights[f"{head}.probe"]).reshape(1, width), dtype=dtype)

    query = g.add_linear(network, probe, q_w, q_b, dtype=dtype)
    key = g.add_linear(network, hidden, k_w, k_b, dtype=dtype)
    value = g.add_linear(network, hidden, v_w, v_b, dtype=dtype)
    attended = g.add_attention(network, query, key, value,
                               num_heads=cfg["num_heads"], head_dim=cfg["head_dim"])
    attended = g.add_linear(network, attended,
                            weights[f"{head}.attention.out_proj.weight"],
                            weights[f"{head}.attention.out_proj.bias"], dtype=dtype)

    # residual = attention output; the norm feeds the MLP only.
    normed = g.add_layer_norm(network, attended, weights[f"{head}.layernorm.weight"],
                              weights[f"{head}.layernorm.bias"], cfg["eps"])
    inner = g.add_linear(network, normed, weights[f"{head}.mlp.fc1.weight"],
                         weights[f"{head}.mlp.fc1.bias"], dtype=dtype)
    inner = g.add_gelu_tanh(network, inner, dtype=dtype)
    inner = g.add_linear(network, inner, weights[f"{head}.mlp.fc2.weight"],
                         weights[f"{head}.mlp.fc2.bias"], dtype=dtype)
    return g.add_sum(network, attended, inner)


def build_vision_tower(network, pixel_values, weights, cfg, *, dtype=np.float32):
    """``[1, 3, H, W]`` pixels to an unnormalised ``[1, width]`` embedding."""
    width = cfg["hidden_size"]
    patch = cfg["patch_size"]
    grid = cfg["grid"]

    patches = g.add_conv2d(network, pixel_values,
                           weights[f"{PREFIX}.embeddings.patch_embedding.weight"],
                           weights[f"{PREFIX}.embeddings.patch_embedding.bias"],
                           stride=(patch, patch), dtype=dtype)
    flat = network.add_shuffle(patches)
    flat.reshape_dims = (width, grid * grid)
    flat.second_transpose = (1, 0)
    hidden = flat.get_output(0)

    positions = g.add_constant(network, (cfg["num_positions"], width),
                               weights[f"{PREFIX}.embeddings.position_embedding.weight"],
                               dtype=dtype)
    hidden = g.add_sum(network, hidden, positions)

    hidden = encoder_builder.build_layers(network, hidden, weights, PREFIX, cfg, dtype=dtype)
    hidden = g.add_layer_norm(network, hidden, weights[f"{PREFIX}.post_layernorm.weight"],
                              weights[f"{PREFIX}.post_layernorm.bias"], cfg["eps"])
    return _pooling_head(network, hidden, weights, cfg, dtype=dtype)
