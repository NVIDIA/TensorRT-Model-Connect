# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grounding DINO encoder: six layers that update vision and text together.

Each layer runs three blocks, and the order is the one thing here that cannot
be read off ``named_children()`` - the modules are declared
``text_enhancer_layer, fusion_layer, deformable_layer`` but
``GroundingDinoEncoderLayer.forward`` runs them as:

    fusion  ->  text_enhancer  ->  deformable

Taking the declaration order would be shape-identical and wrong.

The deformable block itself is the same post-norm layer
``families/deformable_detr`` builds, with centre-only reference points: the
encoder's references are the two-dimensional level centres, and only the
decoder uses the four-dimensional box branch. The module is copied rather than
imported because families are self-contained.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from . import graph as g
from .deformable_attention import build_deformable_attention
from .fusion_builder import build_fusion, build_text_enhancer

_LAYER_NORM_EPS = 1e-5


def reference_points(shapes, dtype=np.float32) -> np.ndarray:
    """``(1, tokens, levels, 2)`` normalised level centres, in (x, y) order."""
    pieces = []
    for height, width in shapes:
        ref_y = (np.arange(height, dtype=np.float64) + 0.5) / height
        ref_x = (np.arange(width, dtype=np.float64) + 0.5) / width
        grid_y, grid_x = np.meshgrid(ref_y, ref_x, indexing="ij")
        pieces.append(np.stack((grid_x.reshape(-1), grid_y.reshape(-1)), axis=-1))
    flat = np.concatenate(pieces, axis=0)
    return np.tile(flat[None, :, None, :], (1, 1, len(shapes), 1)).astype(dtype)


def _deformable_block(network, hidden, posed, reference, weights, prefix, shapes,
                      heads, points, dtype):
    """One post-norm deformable self-attention layer over the vision tokens."""
    attended = build_deformable_attention(
        network, hidden, posed, reference, weights, f"{prefix}.self_attn",
        shapes, heads, points, dtype)
    hidden = g.add_layer_norm(
        network, g.add_sum(network, hidden, attended),
        weights[f"{prefix}.self_attn_layer_norm.weight"],
        weights[f"{prefix}.self_attn_layer_norm.bias"], _LAYER_NORM_EPS, dtype=dtype)

    inner = g.add_relu(network, g.add_linear(
        network, hidden, weights[f"{prefix}.fc1.weight"],
        weights[f"{prefix}.fc1.bias"], dtype=dtype))
    inner = g.add_linear(network, inner, weights[f"{prefix}.fc2.weight"],
                         weights[f"{prefix}.fc2.bias"], dtype=dtype)
    return g.add_layer_norm(
        network, g.add_sum(network, hidden, inner),
        weights[f"{prefix}.final_layer_norm.weight"],
        weights[f"{prefix}.final_layer_norm.bias"], _LAYER_NORM_EPS, dtype=dtype)


def build_encoder(network, vision, text, position, weights, shapes, config,
                  prefix="model.encoder", dtype=np.float32,
                  text_mask=None, text_key_mask=None, text_position=None, debug=None):
    """Run the six fused encoder layers.

    ``position`` is the vision position + level embedding constant;
    ``text_position`` is the enhancer's sine embedding over the per-phrase
    position ids; ``text_mask`` is the additive phrase block mask used by the
    enhancer, and ``text_key_mask`` the additive ``(1, 1, length)`` term that
    hides padded prompt positions from the fusion's vision queries. Without
    the latter the vision tokens attend to whatever the padding produced.
    """
    reference = g.add_constant(
        network, (1, int(vision.shape[1]), len(shapes), 2),
        reference_points(shapes, dtype), dtype)

    for layer in range(config["encoder_layers"]):
        base = f"{prefix}.layers.{layer}"
        # 1. fusion, 2. text enhancer, 3. deformable - forward order, not
        # declaration order.
        vision, text = build_fusion(
            network, vision, text, weights, f"{base}.fusion_layer",
            config["fusion_attention_heads"], dtype, text_mask=text_key_mask)
        text = build_text_enhancer(
            network, text, weights, f"{base}.text_enhancer_layer",
            config["text_enhancer_heads"], dtype,
            mask=text_mask, position=text_position)
        posed = network.add_elementwise(
            vision, position, trt.ElementWiseOperation.SUM).get_output(0)
        vision = _deformable_block(
            network, vision, posed, reference, weights, f"{base}.deformable_layer",
            shapes, config["encoder_attention_heads"], config["encoder_n_points"], dtype)
        if debug is not None:
            debug[f"vision_{layer}"] = vision
            debug[f"text_{layer}"] = text
    return vision, text
