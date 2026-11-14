# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the ResNet-D backbone RT-DETR v2 uses.

The "vd" suffix is not cosmetic and both of its differences from a stock
ResNet were read off the checkpoint rather than inferred:

* the stem is **three** 3x3 convolutions (3->32, 32->32, 32->64), only the
  first of which strides, followed by a 3x3 max pool. A stock ResNet uses one
  7x7 stride-2 convolution instead.
* a downsampling shortcut is **average pool 2x2 then a stride-1 1x1
  convolution**, not a stride-2 1x1 convolution. Both give the same output
  shape, so a wrong choice here is invisible until the numbers are compared.

Only the last three stages leave the backbone, at strides 8, 16 and 32.
"""

from __future__ import annotations

import numpy as np

from . import graph as g

_BN_EPS = 1e-5


def _conv_bn(network, x, weights, prefix, stride, padding, dtype):
    """One convolution with its BatchNorm folded in."""
    folded, bias = g.fold_batch_norm(
        weights[f"{prefix}.convolution.weight"],
        weights[f"{prefix}.normalization.weight"],
        weights[f"{prefix}.normalization.bias"],
        weights[f"{prefix}.normalization.running_mean"],
        weights[f"{prefix}.normalization.running_var"],
        _BN_EPS,
    )
    return g.add_conv2d(network, x, folded, bias, stride=stride, padding=padding, dtype=dtype)


def _basic_block(network, x, weights, prefix, stride, dtype):
    """BasicBlock: 3x3 stride-s, 3x3 stride-1, plus the shortcut."""
    residual = x
    if f"{prefix}.shortcut.1.convolution.weight" in weights:
        # Downsampling shortcut: average pool first, then a stride-1 projection.
        pooled = g.add_avg_pool(network, residual, (2, 2), (2, 2))
        residual = _conv_bn(network, pooled, weights, f"{prefix}.shortcut.1", (1, 1), (0, 0), dtype)
    elif f"{prefix}.shortcut.convolution.weight" in weights:
        # Same-resolution projection, used once to widen the first stage.
        residual = _conv_bn(network, residual, weights, f"{prefix}.shortcut", (1, 1), (0, 0), dtype)

    h = _conv_bn(network, x, weights, f"{prefix}.layer.0", stride, (1, 1), dtype)
    h = g.add_relu(network, h)
    h = _conv_bn(network, h, weights, f"{prefix}.layer.1", (1, 1), (1, 1), dtype)
    return g.add_relu(network, g.add_sum(network, h, residual))


def build_backbone(network, pixel_values, weights, cfg, dtype=np.float32):
    """Return the three feature maps the encoder consumes, strides 8/16/32."""
    root = "model.backbone.model"
    x = pixel_values
    for index in range(3):
        x = _conv_bn(network, x, weights, f"{root}.embedder.embedder.{index}",
                     (2, 2) if index == 0 else (1, 1), (1, 1), dtype)
        x = g.add_relu(network, x)
    x = g.add_max_pool(network, x, (3, 3), (2, 2), (1, 1))

    outputs = []
    depths = cfg["depths"]
    for stage in range(len(depths)):
        for block in range(depths[stage]):
            # Only the first block of a stage strides, and never in stage 0.
            stride = (2, 2) if (block == 0 and stage > 0) else (1, 1)
            x = _basic_block(network, x, weights,
                             f"{root}.encoder.stages.{stage}.layers.{block}", stride, dtype)
        if stage > 0:
            outputs.append(x)
    if len(outputs) != 3:
        raise ValueError(f"RT-DETR v2 backbone produced {len(outputs)} levels, expected 3")
    return outputs
