# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Family-owned FP32 BigVGAN with alias-free SnakeBeta activations."""

import numpy as np
import tensorrt as trt


def _conv(g, hidden, name, *, stride=1, dilation=1, transpose=False, padding=None):
    if name + ".weight" in g.weights.index:
        weight = g.weights.read(name + ".weight")
    else:
        vector = g.weights.read(name + ".weight_v")
        gain = g.weights.read(name + ".weight_g")
        norm = np.sqrt(
            np.sum(
                vector * vector, axis=tuple(range(1, vector.ndim)), keepdims=True, dtype=np.float32
            )
        )
        if np.any(norm == 0):
            raise ValueError("Pika audio weight normalization has a zero norm")
        weight = vector * (gain / norm)
    weight = np.ascontiguousarray(weight.reshape(*weight.shape[:2], 1, weight.shape[2]))
    bias = (
        g.weights.read(name + ".bias")
        if name + ".bias" in g.weights.index
        else trt.Weights(trt.float32)
    )
    operation = g.net.add_deconvolution_nd if transpose else g.net.add_convolution_nd
    layer = operation(
        hidden,
        weight.shape[1] if transpose else weight.shape[0],
        (1, weight.shape[-1]),
        weight,
        bias,
    )
    layer.stride_nd = (1, stride)
    if not transpose:
        layer.dilation_nd = (1, dilation)
    layer.padding_nd = (
        0,
        (weight.shape[-1] * dilation - dilation) // 2 if padding is None else padding,
    )
    return layer.get_output(0)


def _pad(g, hidden, left, right):
    length = hidden.shape[-1]
    indices = np.clip(np.arange(-left, length + right), 0, length - 1)
    return g.gather(hidden, g.const(indices, trt.int32), 3)


def _filter(g, hidden, name, *, transpose=False):
    channels = hidden.shape[1]
    weights = g.weights.read(name).reshape(1, 1, 1, -1)
    weights = np.repeat(weights, channels, axis=0)
    operation = g.net.add_deconvolution_nd if transpose else g.net.add_convolution_nd
    layer = operation(
        hidden,
        channels,
        (1, weights.shape[-1]),
        np.ascontiguousarray(weights),
        trt.Weights(trt.float32),
    )
    layer.num_groups = channels
    layer.stride_nd = (1, 2)
    return layer.get_output(0)


def _activation(g, hidden, name):
    # Source ratio=2, kernel=12: replicate five samples on both sides,
    # transposed-filter, then remove the 15/15-sample convolution margins.
    up = g.mul(_filter(g, _pad(g, hidden, 5, 5), name + ".upsample.filter", transpose=True), 2)
    length = up.shape[-1]
    up = g.gather(up, g.const(np.arange(15, length - 15), trt.int32), 3)
    alpha = g.const(g.weights.read(name + ".act.alpha").reshape(1, -1, 1, 1), trt.float32)
    beta = g.const(g.weights.read(name + ".act.beta").reshape(1, -1, 1, 1), trt.float32)
    alpha = g.net.add_unary(alpha, trt.UnaryOperation.EXP).get_output(0)
    beta = g.net.add_unary(beta, trt.UnaryOperation.EXP).get_output(0)
    sine = g.net.add_unary(g.mul(alpha, up), trt.UnaryOperation.SIN).get_output(0)
    inverse = g.net.add_unary(g.add(beta, 1e-9), trt.UnaryOperation.RECIP).get_output(0)
    activated = g.add(up, g.mul(inverse, g.mul(sine, sine)))
    return _filter(g, _pad(g, activated, 5, 6), name + ".downsample.lowpass.filter")


def input_graph(g, frames):
    latents = g.net.add_input("latents", trt.float32, (2, 32, 1, frames))
    hidden = _conv(g, latents, "dec_in_proj", padding=0)
    g.output(_conv(g, hidden, "decoder.conv_pre", padding=3), "result")


def stage_graph(g, cfg, stage, frames):
    channels = cfg["decoder_dim"] // (2**stage)
    hidden = g.net.add_input("hidden", trt.float32, (2, channels, 1, frames))
    rate, kernel = cfg["decoder_rates"][stage], cfg["decoder_kernel_sizes"][stage]
    hidden = _conv(
        g,
        hidden,
        f"decoder.ups.{stage}.0",
        stride=rate,
        transpose=True,
        padding=(kernel - rate) // 2,
    )
    result = None
    for block, dilations in enumerate(cfg["resblock_dilation_sizes"]):
        prefix = f"decoder.resblocks.{stage * len(cfg['resblock_kernel_sizes']) + block}"
        residual = hidden
        for index, dilation in enumerate(dilations):
            value = _activation(g, residual, prefix + f".activations.{2 * index}")
            value = _conv(g, value, prefix + f".convs1.{index}", dilation=dilation)
            value = _activation(g, value, prefix + f".activations.{2 * index + 1}")
            value = _conv(g, value, prefix + f".convs2.{index}")
            residual = g.add(residual, value)
        result = residual if result is None else g.add(result, residual)
    g.output(g.div(result, len(cfg["resblock_kernel_sizes"])), "result")


def head_graph(g, frames):
    hidden = g.net.add_input("hidden", trt.float32, (2, 8, 1, frames))
    hidden = _activation(g, hidden, "decoder.activation_post")
    hidden = _conv(g, hidden, "decoder.conv_post", padding=3)
    g.output(
        g.op(g.op(hidden, -1, trt.ElementWiseOperation.MAX), 1, trt.ElementWiseOperation.MIN),
        "result",
    )
