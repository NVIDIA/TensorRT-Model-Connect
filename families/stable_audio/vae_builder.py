# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``AutoencoderOobleck`` decoder as one TensorRT network: latents [1, C, T] -> audio [1, A, T * hop].

The checkpoint stores weight-normalised convolutions (``weight_g`` / ``weight_v``); the
effective kernel ``g * v / ||v||`` (norm over every axis but the first, which for a
``ConvTranspose1d`` is its input-channel axis) is folded here. Snake activations
(``x + sin(e^alpha x)^2 / (e^beta + 1e-9)``) fold their parameters into per-channel
constants. Activations run in a [1, C, 1, T] layout so every convolution is a 1 x k 2-D one.
In an FP16 build the activations are FP16 and the convolutions multiply FP16-rounded
weights as FP32 and accumulate in FP32 (see ``Graph._fp32_operands``).
"""

from __future__ import annotations

import math

import numpy as np
import tensorrt as trt

from .graph_ops import Graph

_RESIDUAL_DILATIONS = (1, 3, 9)


def fold_weight_norm(weights: dict, name: str) -> np.ndarray:
    """``torch.nn.utils.weight_norm`` with ``dim=0``: ``g * v / ||v||`` per slice of axis 0."""
    v = np.asarray(weights[f"{name}.weight_v"], dtype=np.float64)
    g = np.asarray(weights[f"{name}.weight_g"], dtype=np.float64)
    norm = np.sqrt(np.sum(v * v, axis=tuple(range(1, v.ndim)), keepdims=True))
    return (v * (g / norm)).astype(np.float32)


def snake_constants(weights: dict, name: str) -> tuple[np.ndarray, np.ndarray]:
    """``Snake1d(logscale=True)`` as (e^alpha, 1 / (e^beta + 1e-9)) in float32, per channel."""
    alpha = np.exp(np.asarray(weights[f"{name}.alpha"], dtype=np.float32).reshape(-1))
    beta = np.exp(np.asarray(weights[f"{name}.beta"], dtype=np.float32).reshape(-1))
    return alpha.astype(np.float32), (np.float32(1.0) / (beta + np.float32(1e-9))).astype(np.float32)


def _snake(g: Graph, tensor, weights: dict, name: str):
    alpha, inverse_beta = snake_constants(weights, name)
    shape = (1, alpha.size, 1, 1)
    wave = g.unary(g.mul(tensor, g.constant(alpha, tensor.dtype, shape=shape)), trt.UnaryOperation.SIN)
    return g.add(tensor, g.mul(g.mul(wave, wave), g.constant(inverse_beta, tensor.dtype, shape=shape)))


def _residual_unit(g: Graph, tensor, weights: dict, name: str, dilation: int):
    out = g.conv1d(_snake(g, tensor, weights, f"{name}.snake1"), fold_weight_norm(weights, f"{name}.conv1"),
                   weights[f"{name}.conv1.bias"], dilation=dilation, padding=3 * dilation)
    out = g.conv1d(_snake(g, out, weights, f"{name}.snake2"), fold_weight_norm(weights, f"{name}.conv2"),
                   weights[f"{name}.conv2.bias"])
    # Both convolutions keep the length, so the reference's centre crop never applies.
    return g.add(tensor, out)


def populate(network: trt.INetworkDefinition, weights: dict, cfg: dict, work: trt.DataType) -> Graph:
    """Add the decoder network; ``weights`` are the ``decoder.*`` tensors of the vae checkpoint."""
    g = Graph(network, tf32_engine=work == trt.float16)
    vae = cfg["vae"]
    frames = cfg["transformer"]["sample_size"]
    latents = network.add_input("latents", trt.float32, (1, vae["latent_channels"], frames))
    hidden = g.cast(g.reshape(latents, (1, vae["latent_channels"], 1, frames)), work)
    hidden = g.conv1d(hidden, fold_weight_norm(weights, "decoder.conv1"), weights["decoder.conv1.bias"], padding=3)
    for index, stride in enumerate(vae["upsampling_ratios"]):
        name = f"decoder.block.{index}"
        hidden = _snake(g, hidden, weights, f"{name}.snake1")
        hidden = g.conv_transpose1d(hidden, fold_weight_norm(weights, f"{name}.conv_t1"),
                                    weights[f"{name}.conv_t1.bias"], stride=stride,
                                    padding=math.ceil(stride / 2))
        for unit, dilation in enumerate(_RESIDUAL_DILATIONS, start=1):
            hidden = _residual_unit(g, hidden, weights, f"{name}.res_unit{unit}", dilation)
    hidden = _snake(g, hidden, weights, "decoder.snake1")
    hidden = g.conv1d(hidden, fold_weight_norm(weights, "decoder.conv2"), None, padding=3)
    audio = g.cast(g.reshape(hidden, (1, vae["audio_channels"], cfg["audio_samples"])), trt.float32)
    audio.name = "audio"
    network.mark_output(audio)
    return g
