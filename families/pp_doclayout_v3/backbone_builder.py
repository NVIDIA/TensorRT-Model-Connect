# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HGNetV2 backbone as native TensorRT convolutions and concatenations."""

from . import graph as g


def conv(network, x, weights, prefix, *, stride=1, groups=1, activation="relu"):
    weight, bias = g.fold_batch_norm(
        weights[f"{prefix}.convolution.weight"],
        weights[f"{prefix}.normalization.weight"],
        weights[f"{prefix}.normalization.bias"],
        weights[f"{prefix}.normalization.running_mean"],
        weights[f"{prefix}.normalization.running_var"],
        1e-5,
    )
    layer = network.add_convolution_nd(x, weight.shape[0], weight.shape[2:], weight, bias)
    layer.stride_nd = (stride, stride)
    layer.padding_nd = tuple((size - 1) // 2 for size in weight.shape[2:])
    layer.num_groups = groups
    out = layer.get_output(0)
    return (
        g.add_relu(network, out)
        if activation == "relu"
        else (g.add_silu(network, out) if activation == "silu" else out)
    )


def build_backbone(network, pixels, weights):
    root = "model.backbone.model"
    stem = f"{root}.embedder"
    x = conv(network, pixels, weights, f"{stem}.stem1", stride=2)
    # Both even-kernel stem convolutions have explicit right/bottom zero padding.
    x = network.add_padding_nd(x, (0, 0), (1, 1)).get_output(0)
    branch = conv(network, x, weights, f"{stem}.stem2a")
    branch = network.add_padding_nd(branch, (0, 0), (1, 1)).get_output(0)
    branch = conv(network, branch, weights, f"{stem}.stem2b")
    pooled = g.add_max_pool(network, x, (2, 2), (1, 1), (0, 0))
    x = conv(network, g.concat(network, [pooled, branch], 1), weights, f"{stem}.stem3", stride=2)
    x = conv(network, x, weights, f"{stem}.stem4")
    outputs = []
    for stage, blocks in enumerate((1, 1, 3, 1)):
        prefix = f"{root}.encoder.stages.{stage}"
        if stage:
            x = conv(
                network,
                x,
                weights,
                f"{prefix}.downsample",
                stride=2,
                groups=int(x.shape[1]),
                activation=None,
            )
        for block in range(blocks):
            block_prefix = f"{prefix}.blocks.{block}"
            residual = x
            features = [x]
            for index in range(6):
                p = f"{block_prefix}.layers.{index}"
                if stage >= 2:
                    x = conv(network, x, weights, f"{p}.conv1", activation=None)
                    x = conv(network, x, weights, f"{p}.conv2", groups=int(x.shape[1]))
                else:
                    x = conv(network, x, weights, p)
                features.append(x)
            x = conv(
                network, g.concat(network, features, 1), weights, f"{block_prefix}.aggregation.0"
            )
            x = conv(network, x, weights, f"{block_prefix}.aggregation.1")
            if block:
                x = g.add_sum(network, x, residual)
        outputs.append(x)
    return outputs
