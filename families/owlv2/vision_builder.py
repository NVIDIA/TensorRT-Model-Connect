# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The OWLv2 image tower, up to the per-patch features the three heads read.

This is a CLIP vision tower with one extra step at the end. CLIP pools the
class token and throws the patch rows away; OWLv2 keeps every patch row and
*multiplies* it by the class token, then normalises. Measured: the reference
applies ``post_layernorm`` to the whole sequence here, not only to the pooled
row the way CLIP does.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from . import encoder_builder
from . import graph as g

PREFIX = "owlv2.vision_model"


def build_features(network, pixels, weights, cfg, *, dtype=np.float32):
    """Pixels ``[1, 3, size, size]`` to per-patch features ``[num_patches, width]``."""
    width = cfg["hidden_size"]
    count = cfg["num_patches"]

    patches = g.add_conv2d(network, pixels,
                           weights[f"{PREFIX}.embeddings.patch_embedding.weight"],
                           stride=(cfg["patch_size"], cfg["patch_size"]), dtype=dtype)
    rows = network.add_shuffle(patches)
    rows.reshape_dims = (width, count)
    rows.second_transpose = (1, 0)

    class_token = g.add_constant(network, (1, width),
                                 weights[f"{PREFIX}.embeddings.class_embedding"], dtype=dtype)
    sequence = network.add_concatenation([class_token, rows.get_output(0)])
    sequence.axis = 0
    hidden = g.add_sum(network, sequence.get_output(0),
                       g.add_constant(network, (cfg["num_positions"], width),
                                      weights[f"{PREFIX}.embeddings.position_embedding.weight"],
                                      dtype=dtype))
    hidden = g.add_layer_norm(network, hidden, weights[f"{PREFIX}.pre_layernorm.weight"],
                              weights[f"{PREFIX}.pre_layernorm.bias"], cfg["eps"])
    hidden = encoder_builder.build_layers(network, hidden, weights, PREFIX, cfg,
                                          mask=None, dtype=dtype)
    # The whole sequence, not just the pooled row.
    hidden = g.add_layer_norm(network, hidden, weights[f"{PREFIX}.post_layernorm.weight"],
                              weights[f"{PREFIX}.post_layernorm.bias"], cfg["eps"])

    class_row = network.add_slice(hidden, (0, 0), (1, width), (1, 1)).get_output(0)
    patch_rows = network.add_slice(hidden, (1, 0), (count, width), (1, 1)).get_output(0)
    # [1, width] against [count, width]: the class token multiplies every patch.
    merged = network.add_elementwise(patch_rows, class_row,
                                     trt.ElementWiseOperation.PROD).get_output(0)
    return merged


def merge_norm(network, merged, weights, cfg, *, dtype=np.float32):
    """The top-level ``layer_norm`` that follows the class-token merge."""
    return g.add_layer_norm(network, merged, weights["layer_norm.weight"],
                            weights["layer_norm.bias"], cfg["eps"])
