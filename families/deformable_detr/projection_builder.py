# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input projections: the backbone pyramid mapped onto the model width.

Three of the four levels are 1x1 convolutions over the backbone's stride 8, 16
and 32 outputs. The fourth is the one worth stating plainly, because it is easy
to wire the obvious way and be wrong:

    level 3 = conv3x3 stride 2 over the **raw C5 feature**, not over the
    already-projected level 2.

Read from the checkpoint rather than inferred - ``input_proj.3.0`` has 2048
input channels, which only the raw C5 carries; the projected level 2 has 256.
That channel count is what makes this safe: wiring level 3 off level 2 does not
merely score badly, it cannot be built at all. Checked, not assumed - the
negative control for this rule fails at network construction rather than at the
comparison.

Every projection is followed by GroupNorm(32, 256), eps 1e-5.
"""

from __future__ import annotations

import numpy as np

from . import graph as g

_GROUPS = 32
_NORM_EPS = 1e-5


def _project(network, x, weights, index, stride, padding, dtype):
    conv = g.add_conv2d(
        network, x,
        weights[f"model.input_proj.{index}.0.weight"],
        weights[f"model.input_proj.{index}.0.bias"],
        stride=stride, padding=padding, dtype=dtype,
    )
    return g.add_group_norm(
        network, conv,
        weights[f"model.input_proj.{index}.1.weight"],
        weights[f"model.input_proj.{index}.1.bias"],
        _GROUPS, _NORM_EPS, dtype,
    )


def build_projections(network, features, weights, dtype=np.float32):
    """Return the four 256-channel feature levels the encoder flattens."""
    if len(features) != 3:
        raise ValueError("deformable_detr expects three backbone features")
    levels = [
        _project(network, feature, weights, index, (1, 1), (0, 0), dtype)
        for index, feature in enumerate(features)
    ]
    # The extra level comes off the raw C5, striding it down by two.
    levels.append(_project(network, features[-1], weights, 3, (2, 2), (1, 1), dtype))
    return levels
