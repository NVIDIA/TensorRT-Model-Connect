# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vision input projections: the Swin pyramid mapped onto the model width.

Three 1x1 convolutions over the backbone's stride 8, 16 and 32 outputs
(192, 384 and 768 channels), plus a fourth level that is a 3x3 stride-2
convolution over the **raw last backbone feature**, not over the projected
level 2 - its weight has 768 input channels, which only the raw map carries.

Every projection is followed by GroupNorm(32, 256), written as an explicit
reduction because TensorRT's INormalizationLayer rejected the axis/scale
combinations tried here.
"""

from __future__ import annotations

import numpy as np

from . import graph as g

_GROUPS = 32
_NORM_EPS = 1e-5
PREFIX = "model.input_proj_vision"


def _project(network, x, weights, index, stride, padding, dtype):
    conv = g.add_conv2d(
        network, x,
        weights[f"{PREFIX}.{index}.0.weight"], weights[f"{PREFIX}.{index}.0.bias"],
        stride=stride, padding=padding, dtype=dtype)
    return g.add_group_norm(
        network, conv,
        weights[f"{PREFIX}.{index}.1.weight"], weights[f"{PREFIX}.{index}.1.bias"],
        _GROUPS, _NORM_EPS, dtype)


def build_projections(network, features, weights, dtype=np.float32):
    """Return the four 256-channel levels the encoder flattens."""
    if len(features) != 3:
        raise ValueError("grounding_dino expects three Swin stages")
    levels = [
        _project(network, feature, weights, index, (1, 1), (0, 0), dtype)
        for index, feature in enumerate(features)
    ]
    # The extra level strides the raw stride-32 map, not the projected one.
    levels.append(_project(network, features[-1], weights, 3, (2, 2), (1, 1), dtype))
    return levels
