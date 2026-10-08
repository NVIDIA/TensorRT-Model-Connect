# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build BiRefNet's ASPPDeformable block.

The deformable convolution is the operator this repository had never built.
TensorRT has no DCNv2 layer, but it does not need one: for each of the K*K
kernel taps, sample the input where that tap's learned offset points, scale by
the learned modulation, convolve with that tap's 1x1 slice of the kernel, and
sum. Measured against ``torchvision.ops.deform_conv2d`` at cosine 0.99999994.

Two details were read off the reference and neither is visible in the shapes:

* the modulation is ``2 * sigmoid(...)``, not ``sigmoid(...)``. Dropping the
  factor of two halves every sampled contribution.
* ``regular_conv`` carries no bias; the only bias in the branch is the one
  folded out of the BatchNorm that follows it.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from . import graph as g

_BN_EPS = 1e-5


def _deformable_conv(network, x, weights, prefix, dtype):
    """Modulated deformable convolution, built from K*K grid samples."""
    offset_w = np.asarray(weights[f"{prefix}.offset_conv.weight"])
    kernel = int(offset_w.shape[2])
    pad = kernel // 2
    shape = tuple(int(v) for v in x.shape)
    height, width = shape[2], shape[3]

    offsets = g.add_conv2d(network, x, offset_w, weights[f"{prefix}.offset_conv.bias"],
                           padding=(pad, pad), dtype=dtype)
    raw_mod = g.add_conv2d(network, x, weights[f"{prefix}.modulator_conv.weight"],
                           weights[f"{prefix}.modulator_conv.bias"],
                           padding=(pad, pad), dtype=dtype)
    sigmoid = network.add_activation(raw_mod, trt.ActivationType.SIGMOID).get_output(0)
    modulator = network.add_elementwise(
        sigmoid, g.add_constant(network, (1, 1, 1, 1), np.full((1,), 2.0), dtype=dtype),
        trt.ElementWiseOperation.PROD).get_output(0)

    rows, cols = np.meshgrid(np.arange(height, dtype=np.float32),
                             np.arange(width, dtype=np.float32), indexing="ij")
    regular = np.asarray(weights[f"{prefix}.regular_conv.weight"], dtype=np.float32)
    out_channels = int(regular.shape[0])

    accumulated = None
    for tap in range(kernel * kernel):
        ky, kx = divmod(tap, kernel)
        base = np.stack([cols + (kx - pad), rows + (ky - pad)], axis=-1)[None]
        base_const = g.add_constant(network, (1, height, width, 2), base, dtype=dtype)
        # torchvision lays the offsets out as (dy, dx) per tap.
        off_y = network.add_slice(offsets, (0, 2 * tap, 0, 0), (1, 1, height, width),
                                  (1, 1, 1, 1)).get_output(0)
        off_x = network.add_slice(offsets, (0, 2 * tap + 1, 0, 0), (1, 1, height, width),
                                  (1, 1, 1, 1)).get_output(0)
        pair = network.add_concatenation([off_x, off_y])
        pair.axis = 1
        moved = network.add_shuffle(pair.get_output(0))
        moved.first_transpose = (0, 2, 3, 1)
        grid_px = network.add_elementwise(base_const, moved.get_output(0),
                                          trt.ElementWiseOperation.SUM).get_output(0)
        scale = g.add_constant(network, (1, 1, 1, 2),
                               np.array([2.0 / width, 2.0 / height]), dtype=dtype)
        shift = g.add_constant(network, (1, 1, 1, 2),
                               np.array([1.0 / width - 1.0, 1.0 / height - 1.0]), dtype=dtype)
        grid = network.add_elementwise(
            network.add_elementwise(grid_px, scale,
                                    trt.ElementWiseOperation.PROD).get_output(0),
            shift, trt.ElementWiseOperation.SUM).get_output(0)
        sampler = network.add_grid_sample(x, grid)
        sampler.interpolation_mode = trt.InterpolationMode.LINEAR
        sampler.sample_mode = trt.SampleMode.FILL
        sampler.align_corners = False
        sampled = sampler.get_output(0)

        gate = network.add_slice(modulator, (0, tap, 0, 0), (1, 1, height, width),
                                 (1, 1, 1, 1)).get_output(0)
        gated = network.add_elementwise(sampled, gate,
                                        trt.ElementWiseOperation.PROD).get_output(0)
        tap_weight = np.ascontiguousarray(
            regular[:, :, ky, kx].reshape(out_channels, -1, 1, 1))
        contribution = g.add_conv2d(network, gated, tap_weight, None, dtype=dtype)
        accumulated = contribution if accumulated is None else network.add_elementwise(
            accumulated, contribution, trt.ElementWiseOperation.SUM).get_output(0)
    return accumulated


def _deform_branch(network, x, weights, prefix, dtype):
    """Deformable convolution, BatchNorm folded onto it, then ReLU."""
    out = _deformable_conv(network, x, weights, f"{prefix}.atrous_conv", dtype)
    gamma = np.asarray(weights[f"{prefix}.bn.weight"], dtype=np.float32)
    beta = np.asarray(weights[f"{prefix}.bn.bias"], dtype=np.float32)
    mean = np.asarray(weights[f"{prefix}.bn.running_mean"], dtype=np.float32)
    variance = np.asarray(weights[f"{prefix}.bn.running_var"], dtype=np.float32)
    scale = gamma / np.sqrt(variance + _BN_EPS)
    bias = beta - mean * scale
    channels = scale.shape[0]
    out = network.add_elementwise(
        out, g.add_constant(network, (1, channels, 1, 1), scale.reshape(1, -1, 1, 1),
                            dtype=dtype), trt.ElementWiseOperation.PROD).get_output(0)
    out = network.add_elementwise(
        out, g.add_constant(network, (1, channels, 1, 1), bias.reshape(1, -1, 1, 1),
                            dtype=dtype), trt.ElementWiseOperation.SUM).get_output(0)
    return g.add_relu(network, out)


def build_aspp_deformable(network, x, weights, prefix, dtype=np.float32):
    """aspp1, three deformable branches, a pooled branch, then the projection."""
    shape = tuple(int(v) for v in x.shape)
    height, width = shape[2], shape[3]

    branches = [_deform_branch(network, x, weights, f"{prefix}.aspp1", dtype)]
    index = 0
    while f"{prefix}.aspp_deforms.{index}.atrous_conv.regular_conv.weight" in weights:
        branches.append(_deform_branch(network, x, weights,
                                       f"{prefix}.aspp_deforms.{index}", dtype))
        index += 1

    # Global average pool, 1x1 projection, BatchNorm, ReLU, then broadcast back.
    pooled = network.add_reduce(x, trt.ReduceOperation.AVG, (1 << 2) | (1 << 3), True)
    pooled_out = g.add_conv2d(network, pooled.get_output(0),
                              weights[f"{prefix}.global_avg_pool.1.weight"], None, dtype=dtype)
    folded, bias = g.fold_batch_norm(
        np.ones((int(np.asarray(weights[f"{prefix}.global_avg_pool.2.weight"]).shape[0]),
                 1, 1, 1), dtype=np.float32),
        weights[f"{prefix}.global_avg_pool.2.weight"],
        weights[f"{prefix}.global_avg_pool.2.bias"],
        weights[f"{prefix}.global_avg_pool.2.running_mean"],
        weights[f"{prefix}.global_avg_pool.2.running_var"], _BN_EPS)
    channels = folded.shape[0]
    pooled_out = network.add_elementwise(
        pooled_out, g.add_constant(network, (1, channels, 1, 1),
                                   folded.reshape(1, -1, 1, 1), dtype=dtype),
        trt.ElementWiseOperation.PROD).get_output(0)
    pooled_out = network.add_elementwise(
        pooled_out, g.add_constant(network, (1, channels, 1, 1), bias.reshape(1, -1, 1, 1),
                                   dtype=dtype), trt.ElementWiseOperation.SUM).get_output(0)
    pooled_out = g.add_relu(network, pooled_out)
    # A 1x1 map resized to the full size is a plain broadcast; a stride of zero
    # expresses that exactly, without relying on resize behaviour at size one.
    spread = network.add_slice(pooled_out, (0, 0, 0, 0), (1, channels, height, width),
                               (1, 1, 0, 0)).get_output(0)
    branches.append(spread)

    merged = g.concat(network, branches, axis=1)
    out = g.add_conv2d(network, merged, weights[f"{prefix}.conv1.weight"], None, dtype=dtype)
    gamma = np.asarray(weights[f"{prefix}.bn1.weight"], dtype=np.float32)
    beta = np.asarray(weights[f"{prefix}.bn1.bias"], dtype=np.float32)
    mean = np.asarray(weights[f"{prefix}.bn1.running_mean"], dtype=np.float32)
    variance = np.asarray(weights[f"{prefix}.bn1.running_var"], dtype=np.float32)
    scale = gamma / np.sqrt(variance + _BN_EPS)
    final_bias = beta - mean * scale
    out = network.add_elementwise(
        out, g.add_constant(network, (1, scale.shape[0], 1, 1), scale.reshape(1, -1, 1, 1),
                            dtype=dtype), trt.ElementWiseOperation.PROD).get_output(0)
    out = network.add_elementwise(
        out, g.add_constant(network, (1, scale.shape[0], 1, 1),
                            final_bias.reshape(1, -1, 1, 1), dtype=dtype),
        trt.ElementWiseOperation.SUM).get_output(0)
    return g.add_relu(network, out)
