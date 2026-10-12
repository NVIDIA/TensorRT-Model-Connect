# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deformable DETR encoder: six post-norm deformable self-attention layers.

The layer order is post-norm, read from ``DeformableDetrEncoderLayer.forward``:

    h = self_attn_layer_norm(h + attention(h))
    h = final_layer_norm(h + mlp(h))

Encoder reference points are the normalised centres of each token's own
feature map - ``linspace(0.5, size - 0.5, size) / size`` - and, because nothing
is padded here, the same point is reused for all four levels.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from . import graph as g
from .deformable_attention import build_deformable_attention

_NORM_EPS = 1e-5


def reference_points(shapes, dtype=np.float32) -> np.ndarray:
    """``(1, tokens, levels, 2)`` normalised centres, in (x, y) order."""
    pieces = []
    for height, width in shapes:
        ref_y = (np.arange(height, dtype=np.float64) + 0.5) / height
        ref_x = (np.arange(width, dtype=np.float64) + 0.5) / width
        grid_y, grid_x = np.meshgrid(ref_y, ref_x, indexing="ij")
        pieces.append(np.stack((grid_x.reshape(-1), grid_y.reshape(-1)), axis=-1))
    flat = np.concatenate(pieces, axis=0)
    # Every level shares the same point, which is what valid_ratios of 1 gives.
    return np.tile(flat[None, :, None, :], (1, 1, len(shapes), 1)).astype(dtype)


def build_encoder(network, hidden, position, weights, shapes, config,
                  prefix="model.encoder", dtype=np.float32):
    """Run the six encoder layers over the flattened pyramid."""
    reference = g.add_constant(
        network, (1, int(hidden.shape[1]), len(shapes), 2),
        reference_points(shapes, dtype), dtype,
    )
    for layer in range(config["encoder_layers"]):
        base = f"{prefix}.layers.{layer}"
        posed = network.add_elementwise(
            hidden, position, trt.ElementWiseOperation.SUM).get_output(0)
        attended = build_deformable_attention(
            network, hidden, posed, reference, weights, f"{base}.self_attn",
            shapes, config["encoder_attention_heads"], config["encoder_n_points"], dtype,
        )
        hidden = g.add_sum(network, hidden, attended)
        hidden = g.add_layer_norm(
            network, hidden,
            weights[f"{base}.self_attn_layer_norm.weight"],
            weights[f"{base}.self_attn_layer_norm.bias"], _NORM_EPS, dtype,
        )

        inner = g.add_linear(network, hidden, weights[f"{base}.mlp.fc1.weight"],
                             weights[f"{base}.mlp.fc1.bias"], dtype)
        inner = g.add_relu(network, inner)
        inner = g.add_linear(network, inner, weights[f"{base}.mlp.fc2.weight"],
                             weights[f"{base}.mlp.fc2.bias"], dtype)
        hidden = g.add_sum(network, hidden, inner)
        hidden = g.add_layer_norm(
            network, hidden,
            weights[f"{base}.final_layer_norm.weight"],
            weights[f"{base}.final_layer_norm.bias"], _NORM_EPS, dtype,
        )
    return hidden
