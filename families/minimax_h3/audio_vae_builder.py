# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native TensorRT waveform decoder for the MiniMax-H3 audio VAE."""

from __future__ import annotations

import gc
import sys

import numpy as np
import tensorrt as trt

from . import graph_ops as op
from .config import AUDIO_VAE_DECODER_DEFAULT_WORKSPACE_BYTES


BATCH = 2  # H3 represents stereo as two independently decoded mono batch items.
LATENT_CHANNELS = 32
LATENT_FRAMES = 207
LATENT_DIM = 2048
DECODER_DIM = 1024
UPSAMPLE_RATES = (5, 5, 2, 2, 2, 2, 2)
UPSAMPLE_KERNELS = (9, 9, 4, 4, 4, 4, 4)
RESBLOCK_KERNELS = (3, 7, 11)
RESBLOCK_DILATIONS = (1, 3, 5)
HOP_LENGTH = 800
SAMPLES = LATENT_FRAMES * HOP_LENGTH
FILTER_SIZE = 12


def checkpoint_keys() -> tuple[str, ...]:
    names = [
        "dec_in_proj.weight",
        "dec_in_proj.bias",
        "decoder.conv_pre.weight_g",
        "decoder.conv_pre.weight_v",
        "decoder.conv_pre.bias",
        "decoder.activation_post.act.alpha",
        "decoder.activation_post.act.beta",
        "decoder.activation_post.upsample.filter",
        "decoder.activation_post.downsample.lowpass.filter",
        "decoder.conv_post.weight_g",
        "decoder.conv_post.weight_v",
    ]
    for stage in range(len(UPSAMPLE_RATES)):
        names.extend(
            (
                f"decoder.ups.{stage}.0.weight_g",
                f"decoder.ups.{stage}.0.weight_v",
                f"decoder.ups.{stage}.0.bias",
            )
        )
        for kernel_index in range(len(RESBLOCK_KERNELS)):
            block = stage * len(RESBLOCK_KERNELS) + kernel_index
            for activation in range(2 * len(RESBLOCK_DILATIONS)):
                prefix = f"decoder.resblocks.{block}.activations.{activation}"
                names.extend(
                    (
                        f"{prefix}.act.alpha",
                        f"{prefix}.act.beta",
                        f"{prefix}.upsample.filter",
                        f"{prefix}.downsample.lowpass.filter",
                    )
                )
            for group in ("convs1", "convs2"):
                for layer in range(len(RESBLOCK_DILATIONS)):
                    prefix = f"decoder.resblocks.{block}.{group}.{layer}"
                    names.extend(
                        (
                            f"{prefix}.weight_g",
                            f"{prefix}.weight_v",
                            f"{prefix}.bias",
                        )
                    )
    return tuple(names)


def _weight_norm(weights: dict, prefix: str) -> np.ndarray:
    value = np.asarray(weights[f"{prefix}.weight_v"], dtype=np.float32)
    scale = np.asarray(weights[f"{prefix}.weight_g"], dtype=np.float32)
    axes = tuple(range(1, value.ndim))
    norm = np.sqrt(np.sum(value * value, axis=axes, keepdims=True, dtype=np.float32))
    return np.ascontiguousarray(value * (scale / norm), dtype=np.float32)


def _conv1d(
    network,
    tensor,
    weight: np.ndarray,
    bias: np.ndarray | None,
    *,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
):
    batch, channels, length = tuple(tensor.shape)
    output_channels, grouped_channels, kernel = weight.shape
    reshape = network.add_shuffle(tensor)
    reshape.reshape_dims = (batch, channels, 1, length)
    layer = network.add_convolution_nd(
        reshape.get_output(0),
        output_channels,
        (1, kernel),
        trt.Weights(np.ascontiguousarray(weight[:, :, None, :], dtype=np.float32)),
        trt.Weights()
        if bias is None
        else trt.Weights(np.ascontiguousarray(bias, dtype=np.float32)),
    )
    layer.stride_nd = (1, stride)
    layer.padding_nd = (0, padding)
    layer.dilation_nd = (1, dilation)
    layer.num_groups = groups
    output = layer.get_output(0)
    flatten = network.add_shuffle(output)
    flatten.reshape_dims = (batch, output_channels, output.shape[3])
    return flatten.get_output(0)


def _deconv1d(
    network,
    tensor,
    weight: np.ndarray,
    bias: np.ndarray | None,
    *,
    stride: int,
    padding: int = 0,
    groups: int = 1,
):
    batch, input_channels, length = tuple(tensor.shape)
    _, output_channels_per_group, kernel = weight.shape
    output_channels = output_channels_per_group * groups
    reshape = network.add_shuffle(tensor)
    reshape.reshape_dims = (batch, input_channels, 1, length)
    layer = network.add_deconvolution_nd(
        reshape.get_output(0),
        output_channels,
        (1, kernel),
        trt.Weights(np.ascontiguousarray(weight[:, :, None, :], dtype=np.float32)),
        trt.Weights()
        if bias is None
        else trt.Weights(np.ascontiguousarray(bias, dtype=np.float32)),
    )
    layer.stride_nd = (1, stride)
    layer.padding_nd = (0, padding)
    layer.num_groups = groups
    output = layer.get_output(0)
    flatten = network.add_shuffle(output)
    flatten.reshape_dims = (batch, output_channels, output.shape[3])
    return flatten.get_output(0)


def _constant(network, value: np.ndarray):
    value = np.ascontiguousarray(value, dtype=np.float32)
    return network.add_constant(value.shape, trt.Weights(value)).get_output(0)


def _replicate_pad(network, tensor, left: int, right: int):
    batch, channels, length = tuple(tensor.shape)
    pieces = []
    if left:
        first = network.add_slice(tensor, (0, 0, 0), (batch, channels, 1), (1, 1, 1))
        pieces.extend([first.get_output(0)] * left)
    pieces.append(tensor)
    if right:
        last = network.add_slice(
            tensor, (0, 0, length - 1), (batch, channels, 1), (1, 1, 1)
        )
        pieces.extend([last.get_output(0)] * right)
    concat = network.add_concatenation(pieces)
    concat.axis = 2
    return concat.get_output(0)


def _snake_beta(network, tensor, alpha: np.ndarray, beta: np.ndarray):
    channels = tuple(tensor.shape)[1]
    alpha = np.exp(np.asarray(alpha, dtype=np.float32)).reshape(1, channels, 1)
    beta = np.exp(np.asarray(beta, dtype=np.float32)).reshape(1, channels, 1)
    scaled = network.add_elementwise(
        tensor, _constant(network, alpha), trt.ElementWiseOperation.PROD
    ).get_output(0)
    sine = network.add_unary(scaled, trt.UnaryOperation.SIN).get_output(0)
    squared = network.add_elementwise(sine, sine, trt.ElementWiseOperation.PROD).get_output(0)
    reciprocal = _constant(network, 1.0 / (beta + np.float32(1.0e-9)))
    update = network.add_elementwise(
        squared, reciprocal, trt.ElementWiseOperation.PROD
    ).get_output(0)
    return network.add_elementwise(tensor, update, trt.ElementWiseOperation.SUM).get_output(0)


def _activation1d(network, tensor, weights: dict, prefix: str):
    channels = tuple(tensor.shape)[1]
    up_filter = np.asarray(weights[f"{prefix}.upsample.filter"], dtype=np.float32)
    up_filter = np.broadcast_to(up_filter, (channels, 1, FILTER_SIZE)).copy()
    tensor = _replicate_pad(network, tensor, 5, 5)
    tensor = _deconv1d(network, tensor, up_filter, None, stride=2, groups=channels)
    ratio = _constant(network, np.asarray([[[2.0]]], dtype=np.float32))
    tensor = network.add_elementwise(tensor, ratio, trt.ElementWiseOperation.PROD).get_output(0)
    length = tuple(tensor.shape)[2]
    tensor = network.add_slice(
        tensor, (0, 0, 15), (BATCH, channels, length - 30), (1, 1, 1)
    ).get_output(0)
    tensor = _snake_beta(
        network,
        tensor,
        weights[f"{prefix}.act.alpha"],
        weights[f"{prefix}.act.beta"],
    )
    down_filter = np.asarray(
        weights[f"{prefix}.downsample.lowpass.filter"], dtype=np.float32
    )
    down_filter = np.broadcast_to(down_filter, (channels, 1, FILTER_SIZE)).copy()
    tensor = _replicate_pad(network, tensor, 5, 6)
    return _conv1d(network, tensor, down_filter, None, stride=2, groups=channels)


def _resblock(network, tensor, weights: dict, block: int, kernel: int):
    hidden = tensor
    for layer, dilation in enumerate(RESBLOCK_DILATIONS):
        first_activation = f"decoder.resblocks.{block}.activations.{2 * layer}"
        second_activation = f"decoder.resblocks.{block}.activations.{2 * layer + 1}"
        residual = _activation1d(network, hidden, weights, first_activation)
        first_conv = f"decoder.resblocks.{block}.convs1.{layer}"
        residual = _conv1d(
            network,
            residual,
            _weight_norm(weights, first_conv),
            weights[f"{first_conv}.bias"],
            padding=(kernel * dilation - dilation) // 2,
            dilation=dilation,
        )
        residual = _activation1d(network, residual, weights, second_activation)
        second_conv = f"decoder.resblocks.{block}.convs2.{layer}"
        residual = _conv1d(
            network,
            residual,
            _weight_norm(weights, second_conv),
            weights[f"{second_conv}.bias"],
            padding=(kernel - 1) // 2,
        )
        hidden = network.add_elementwise(
            hidden, residual, trt.ElementWiseOperation.SUM
        ).get_output(0)
    return hidden


def build_audio_vae_decoder_engine(
    weights: dict,
    latents_mean: tuple[float, ...] | list[float],
    latents_std: tuple[float, ...] | list[float],
    *,
    verbose: bool = False,
    consume_weights: bool = False,
    workspace_bytes: int | None = None,
) -> bytes:
    if len(latents_mean) != LATENT_CHANNELS or len(latents_std) != LATENT_CHANNELS:
        raise ValueError("MiniMax-H3 audio VAE requires 32 latent mean/std values")
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 1
    op.configure_builder(config)
    op.configure_workspace(
        config,
        workspace_bytes,
        default_bytes=AUDIO_VAE_DECODER_DEFAULT_WORKSPACE_BYTES,
    )

    latent = network.add_input(
        "normalized_audio_latents", trt.float32, (BATCH, LATENT_CHANNELS, LATENT_FRAMES)
    )
    mean = _constant(network, np.asarray(latents_mean, np.float32).reshape(1, -1, 1))
    std = _constant(network, np.asarray(latents_std, np.float32).reshape(1, -1, 1))
    hidden = network.add_elementwise(latent, std, trt.ElementWiseOperation.PROD).get_output(0)
    hidden = network.add_elementwise(hidden, mean, trt.ElementWiseOperation.SUM).get_output(0)
    hidden = _conv1d(
        network,
        hidden,
        weights["dec_in_proj.weight"],
        weights["dec_in_proj.bias"],
    )
    hidden = _conv1d(
        network,
        hidden,
        _weight_norm(weights, "decoder.conv_pre"),
        weights["decoder.conv_pre.bias"],
        padding=3,
    )

    for stage, (rate, kernel) in enumerate(zip(UPSAMPLE_RATES, UPSAMPLE_KERNELS, strict=True)):
        prefix = f"decoder.ups.{stage}.0"
        hidden = _deconv1d(
            network,
            hidden,
            _weight_norm(weights, prefix),
            weights[f"{prefix}.bias"],
            stride=rate,
            padding=(kernel - rate) // 2,
        )
        blocks = [
            _resblock(
                network,
                hidden,
                weights,
                stage * len(RESBLOCK_KERNELS) + kernel_index,
                block_kernel,
            )
            for kernel_index, block_kernel in enumerate(RESBLOCK_KERNELS)
        ]
        hidden = blocks[0]
        for block in blocks[1:]:
            hidden = network.add_elementwise(
                hidden, block, trt.ElementWiseOperation.SUM
            ).get_output(0)
        divisor = _constant(network, np.asarray([[[1.0 / len(blocks)]]], np.float32))
        hidden = network.add_elementwise(
            hidden, divisor, trt.ElementWiseOperation.PROD
        ).get_output(0)

    hidden = _activation1d(network, hidden, weights, "decoder.activation_post")
    waveform = _conv1d(
        network,
        hidden,
        _weight_norm(weights, "decoder.conv_post"),
        None,
        padding=3,
    )
    lower = _constant(network, np.asarray([[[-1.0]]], np.float32))
    upper = _constant(network, np.asarray([[[1.0]]], np.float32))
    waveform = network.add_elementwise(
        waveform, lower, trt.ElementWiseOperation.MAX
    ).get_output(0)
    waveform = network.add_elementwise(
        waveform, upper, trt.ElementWiseOperation.MIN
    ).get_output(0)
    if tuple(waveform.shape) != (BATCH, 1, SAMPLES):
        raise RuntimeError(f"MiniMax-H3 audio decoder produced shape {tuple(waveform.shape)}")
    waveform.name = "waveform"
    network.mark_output(waveform)
    print(
        f"[minimax-h3] building native audio VAE decoder: batch={BATCH}, "
        f"latent_frames={LATENT_FRAMES}, samples={SAMPLES}",
        file=sys.stderr,
    )
    try:
        plan = builder.build_serialized_network(network, config)
    finally:
        op.release_weight_buffers(network)
        if consume_weights:
            weights.clear()
    if plan is None:
        raise RuntimeError("TensorRT failed to build MiniMax-H3 audio VAE decoder")
    del network, config, builder
    gc.collect()
    return bytes(plan)
