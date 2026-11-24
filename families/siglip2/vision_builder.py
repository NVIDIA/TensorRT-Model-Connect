# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SigLIP 2 NaFlex vision tower: one image's patches to one pooled embedding.

The engine sees exactly the request's patches, ``[patches, channels x patch x patch]``
in (row, column, channel) order, with the patch count dynamic up to the processor's
budget. Three details separate it from the fixed-grid SigLIP tower:

* patches are embedded by a **Linear** layer over already-flattened pixels, not by a
  convolution over the image;
* the position table is a square grid **resampled to the image's own grid** with an
  antialiased bilinear filter. The filter taps depend only on the two grid shapes, so
  the runtime computes them on the host as one ``[patches, table rows]`` matrix and
  the engine applies it to the table it carries as a constant. The table never leaves
  the engine; only geometry crosses the boundary;
* pooling is a learned attention head: one ``probe`` row attends over every patch and
  the result goes through ``residual + mlp(layernorm(residual))``, where the residual
  is the attention **output**.

The reference pads every image to the patch budget and masks the padding out of both
the encoder and the pooling head. A masked key contributes exactly zero after the
softmax, so running only the real patches computes the same function without a mask.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from . import encoder_builder, graph as g

PREFIX = "vision_model"


def _pooling_head(network, hidden, weights, cfg, *, dtype=np.float32):
    """torch ``MultiheadAttention`` with one learned query row, then an MLP."""
    width = cfg["hidden_size"]
    head = f"{PREFIX}.head"

    # torch packs this head's projections into one [3 * width, width] tensor, in
    # q, k, v order; the encoder layers ship them separately.
    q_w, k_w, v_w = np.split(np.asarray(weights[f"{head}.attention.in_proj_weight"]), 3, axis=0)
    q_b, k_b, v_b = np.split(np.asarray(weights[f"{head}.attention.in_proj_bias"]), 3, axis=0)

    probe = g.add_constant(network, (1, width),
                           np.asarray(weights[f"{head}.probe"]).reshape(1, width), dtype=dtype)
    query = g.add_linear(network, probe, q_w, q_b, dtype=dtype)
    key = g.add_linear(network, hidden, k_w, k_b, dtype=dtype)
    value = g.add_linear(network, hidden, v_w, v_b, dtype=dtype)
    attended = g.add_attention(network, query, key, value, num_heads=cfg["num_heads"],
                               head_dim=cfg["head_dim"])
    attended = g.add_linear(network, attended, weights[f"{head}.attention.out_proj.weight"],
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


def build_vision_tower(network, patches, position_weights, weights, cfg, *, dtype=np.float32):
    """``[patches, features]`` pixels and ``[patches, table rows]`` taps to ``[1, width]``."""
    width = cfg["hidden_size"]

    hidden = g.add_linear(network, patches,
                          weights[f"{PREFIX}.embeddings.patch_embedding.weight"],
                          weights[f"{PREFIX}.embeddings.patch_embedding.bias"], dtype=dtype)

    table = g.add_constant(network, (cfg["num_positions"], width),
                           weights[f"{PREFIX}.embeddings.position_embedding.weight"], dtype=dtype)
    positions = network.add_matrix_multiply(position_weights, trt.MatrixOperation.NONE,
                                            table, trt.MatrixOperation.NONE).get_output(0)
    hidden = g.add_sum(network, hidden, positions)

    hidden = encoder_builder.build_layers(network, hidden, weights, PREFIX, cfg, dtype=dtype)
    hidden = g.add_layer_norm(network, hidden, weights[f"{PREFIX}.post_layernorm.weight"],
                              weights[f"{PREFIX}.post_layernorm.bias"], cfg["eps"])
    return _pooling_head(network, hidden, weights, cfg, dtype=dtype)
