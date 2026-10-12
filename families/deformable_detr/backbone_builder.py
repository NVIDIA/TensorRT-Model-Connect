# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ResNet-50 backbone for Deformable DETR.

This is the torchvision ResNet-50, which is a different animal from the PResNet
that RT-DETR v2 carries, and the differences are the kind that stay invisible
until the numbers are compared:

* the stem is a single 7x7 stride-2 convolution, not three stacked 3x3s;
* a downsampling shortcut is a plain stride-2 1x1 convolution, with no average
  pool in front of it;
* the stride inside a bottleneck sits on the 3x3 ``conv2``, not on the leading
  1x1. This is the "V1.5" layout, read from the torchvision source rather than
  assumed - putting the stride on conv1 gives the same output shape and so
  fails silently.

Every BatchNorm is frozen at inference, so each one folds into the convolution
ahead of it. The epsilon is 1e-5, read from
``DeformableDetrFrozenBatchNorm2d.forward``.

Only the last three stages leave the backbone, at strides 8, 16 and 32. The
stem and ``layer1`` (stride 4) stay inside: treating layer1 as a level is the
mistake that turns a 100x100 leading feature map into a 200x200 one.
"""

from __future__ import annotations

import numpy as np

from . import graph as g

_BN_EPS = 1e-5

# planes, blocks, stride. Channels out of each stage are planes * 4.
_STAGES = (
    ("layer1", 64, 3, 1),
    ("layer2", 128, 4, 2),
    ("layer3", 256, 6, 2),
    ("layer4", 512, 3, 2),
)

# The stages that leave the backbone, at strides 8, 16 and 32.
_OUTPUT_STAGES = ("layer2", "layer3", "layer4")


def _conv_bn(network, x, weights, conv_key, bn_prefix, stride, padding, dtype):
    """One convolution with its frozen BatchNorm folded in."""
    folded, bias = g.fold_batch_norm(
        weights[conv_key],
        weights[f"{bn_prefix}.weight"],
        weights[f"{bn_prefix}.bias"],
        weights[f"{bn_prefix}.running_mean"],
        weights[f"{bn_prefix}.running_var"],
        _BN_EPS,
    )
    return g.add_conv2d(network, x, folded, bias, stride=stride, padding=padding, dtype=dtype)


def _bottleneck(network, x, weights, prefix, planes, stride, downsample, dtype):
    """conv1 1x1 -> conv2 3x3 (carries the stride) -> conv3 1x1, plus shortcut."""
    out = _conv_bn(network, x, weights, f"{prefix}.conv1.weight", f"{prefix}.bn1",
                   (1, 1), (0, 0), dtype)
    out = g.add_relu(network, out)
    out = _conv_bn(network, out, weights, f"{prefix}.conv2.weight", f"{prefix}.bn2",
                   (stride, stride), (1, 1), dtype)
    out = g.add_relu(network, out)
    out = _conv_bn(network, out, weights, f"{prefix}.conv3.weight", f"{prefix}.bn3",
                   (1, 1), (0, 0), dtype)

    identity = x
    if downsample:
        # A plain strided 1x1. No average pool - that is the PResNet shortcut.
        identity = _conv_bn(network, x, weights, f"{prefix}.downsample.0.weight",
                            f"{prefix}.downsample.1", (stride, stride), (0, 0), dtype)
    return g.add_relu(network, g.add_sum(network, out, identity))


# The builder vocabulary. families/deformable_detr/checkpoint.py maps the
# file's own names onto these.
BACKBONE_PREFIX = "model.backbone.model"


def build_backbone(network, pixels, weights, dtype=np.float32, prefix=BACKBONE_PREFIX):
    """Return the three feature maps the neck consumes, at strides 8, 16 and 32."""
    x = _conv_bn(network, pixels, weights, f"{prefix}.conv1.weight", f"{prefix}.bn1",
                 (2, 2), (3, 3), dtype)
    x = g.add_relu(network, x)
    x = g.add_max_pool(network, x, (3, 3), (2, 2), (1, 1))

    features = []
    in_channels = 64
    for name, planes, blocks, stride in _STAGES:
        out_channels = planes * 4
        for index in range(blocks):
            # Only the first block of a stage strides, and only it reshapes the
            # channel count, so only it carries a shortcut convolution.
            first = index == 0
            x = _bottleneck(
                network, x, weights, f"{prefix}.{name}.{index}", planes,
                stride if first else 1,
                downsample=first and (stride != 1 or in_channels != out_channels),
                dtype=dtype,
            )
        in_channels = out_channels
        if name in _OUTPUT_STAGES:
            features.append(x)
    return features
