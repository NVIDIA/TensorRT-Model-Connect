# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The OWLv2 detection heads that read the per-patch features.

Three things here are easy to get subtly wrong, so each is spelled out:

* The box head uses the **exact** GELU, while both towers use QuickGELU. Two
  activations in one model is unusual enough to look like a mistake; it is not.
* The boxes carry a fixed per-patch bias in logit space, so each patch predicts
  a box near its own grid cell and about one patch wide. Dropping it leaves
  every box centred on the image.
* The class head normalises with the epsilon added to the norm, not under the
  root, and applies a per-patch shift and scale after the dot product.

The objectness head is not built. It exists in the checkpoint, but the
text-prompted post-processing this family serves never reads it: scores come
from the class logits alone.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from . import graph as g


def box_bias(grid: int) -> np.ndarray:
    """The ``[grid * grid, 4]`` additive bias, in logit space, as the reference computes it."""
    coordinates = np.arange(1, grid + 1, dtype=np.float32) / float(grid)
    # indexing="xy": x varies along the row, y down the column.
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    centres = np.clip(np.stack((xx, yy), axis=-1).reshape(-1, 2), 0.0, 1.0)
    centre_bias = np.log(centres + 1e-4) - np.log1p(-centres + 1e-4)
    # The box starts out one patch wide and one patch tall.
    sizes = np.full_like(centre_bias, 1.0 / float(grid))
    size_bias = np.log(sizes + 1e-4) - np.log1p(-sizes + 1e-4)
    return np.concatenate([centre_bias, size_bias], axis=-1).astype(np.float32)


def _mlp(network, feats, weights, prefix, *, dtype):
    hidden = g.add_linear(network, feats, weights[f"{prefix}.dense0.weight"],
                          weights[f"{prefix}.dense0.bias"], dtype=dtype)
    hidden = network.add_activation(hidden, trt.ActivationType.GELU_ERF).get_output(0)
    hidden = g.add_linear(network, hidden, weights[f"{prefix}.dense1.weight"],
                          weights[f"{prefix}.dense1.bias"], dtype=dtype)
    hidden = network.add_activation(hidden, trt.ActivationType.GELU_ERF).get_output(0)
    return g.add_linear(network, hidden, weights[f"{prefix}.dense2.weight"],
                        weights[f"{prefix}.dense2.bias"], dtype=dtype)


def build_boxes(network, feats, weights, *, grid, dtype=np.float32):
    """Per-patch boxes as normalised ``cxcywh`` in ``[0, 1]``, shape ``[patches, 4]``."""
    raw = _mlp(network, feats, weights, "box_head", dtype=dtype)
    biased = g.add_sum(network, raw, g.add_constant(network, (grid * grid, 4), box_bias(grid),
                                                    dtype=dtype))
    return network.add_activation(biased, trt.ActivationType.SIGMOID).get_output(0)


def build_class_terms(network, feats, weights, *, dtype=np.float32):
    """The three per-patch tensors the runtime needs to score any set of queries.

    Returns normalised class embeddings ``[patches, query_dim]``, a shift
    ``[patches, 1]`` and a strictly positive scale ``[patches, 1]``. Keeping
    them apart is what lets the prompt stay out of the engine.
    """
    embeds = g.add_linear(network, feats, weights["class_head.dense0.weight"],
                          weights["class_head.dense0.bias"], dtype=dtype)
    embeds = g.add_l2_normalise_eps(network, embeds)
    shift = g.add_linear(network, feats, weights["class_head.logit_shift.weight"],
                         weights["class_head.logit_shift.bias"], dtype=dtype)
    scale = g.add_linear(network, feats, weights["class_head.logit_scale.weight"],
                         weights["class_head.logit_scale.bias"], dtype=dtype)
    return embeds, shift, g.add_elu_plus_one(network, scale, dtype=dtype)
