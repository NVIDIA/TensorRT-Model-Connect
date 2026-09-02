# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native FP32 TensorRT decoder for the MiniMax-H3 audio VAE.

The released checkpoint uses a BigVGAN decoder.  Stereo is represented as two
batch items, so this plan maps denormalized ``[2, 32, frames]`` latents to
``[2, 1, frames * 800]`` mono waveforms.  Interleaving the two batch items and
applying the checkpoint's per-channel latent normalization belong to the
pipeline boundary, not to this decoder.

TensorRT-RTX convolution layers are two-dimensional.  The graph therefore
keeps a singleton height dimension and expresses every Conv1d/ConvTranspose1d
as a native 2-D convolution with a ``(1, kernel)`` kernel.  No plugin or
framework runtime is required by the serialized engine.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import math
import sys
from pathlib import Path
from typing import Mapping

import numpy as np

from tensorrt_model_connect import trt_compat

from . import graph_ops as op
from .config import (
    AUDIO_LATENT_FRAMES_MAX,
    AUDIO_LATENT_FRAMES_MIN,
    AUDIO_LATENT_FRAMES_OPT,
    AUDIO_VAE_DECODER_DEFAULT_WORKSPACE_BYTES,
)


trt = trt_compat.get_trt()


@dataclass(frozen=True)
class AudioVAEDecoderConfig:
    """Static architecture and input shape of one audio decoder plan."""

    batch_size: int = 2
    latent_channels: int = 32
    latent_frames: int = AUDIO_LATENT_FRAMES_OPT
    min_latent_frames: int = AUDIO_LATENT_FRAMES_MIN
    max_latent_frames: int = AUDIO_LATENT_FRAMES_MAX
    latent_dim: int = 2048
    decoder_dim: int = 1024
    decoder_rates: tuple[int, ...] = (5, 5, 2, 2, 2, 2, 2)
    decoder_kernel_sizes: tuple[int, ...] = (9, 9, 4, 4, 4, 4, 4)
    resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: tuple[tuple[int, ...], ...] = (
        (1, 3, 5),
        (1, 3, 5),
        (1, 3, 5),
    )
    sampling_rate: int = 32000

    @property
    def num_upsamples(self) -> int:
        return len(self.decoder_rates)

    @property
    def num_kernels(self) -> int:
        return len(self.resblock_kernel_sizes)

    @property
    def hop_length(self) -> int:
        return math.prod(self.decoder_rates)

    @property
    def output_samples(self) -> int:
        return self.latent_frames * self.hop_length

    @property
    def min_output_samples(self) -> int:
        return self.min_latent_frames * self.hop_length

    @property
    def max_output_samples(self) -> int:
        return self.max_latent_frames * self.hop_length

    def validate(self) -> None:
        integer_fields = {
            "batch_size": self.batch_size,
            "latent_channels": self.latent_channels,
            "latent_frames": self.latent_frames,
            "min_latent_frames": self.min_latent_frames,
            "max_latent_frames": self.max_latent_frames,
            "latent_dim": self.latent_dim,
            "decoder_dim": self.decoder_dim,
            "sampling_rate": self.sampling_rate,
        }
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in integer_fields.values()
        ):
            raise ValueError("MiniMax-H3 AudioVAE dimensions must be positive integers")
        if self.batch_size != 2:
            raise ValueError("MiniMax-H3 AudioVAE stereo requires batch_size=2")
        if not self.min_latent_frames <= self.latent_frames <= self.max_latent_frames:
            raise ValueError(
                "MiniMax-H3 AudioVAE latent frames must satisfy min <= opt <= max"
            )
        if len(self.decoder_rates) != len(self.decoder_kernel_sizes):
            raise ValueError("MiniMax-H3 AudioVAE decoder rates and kernels must align")
        if not self.decoder_rates or not self.resblock_kernel_sizes:
            raise ValueError("MiniMax-H3 AudioVAE decoder stages and residual kernels are required")
        if len(self.resblock_kernel_sizes) != len(self.resblock_dilation_sizes):
            raise ValueError("MiniMax-H3 AudioVAE residual kernels and dilations must align")
        if any(rate <= 0 for rate in self.decoder_rates):
            raise ValueError("MiniMax-H3 AudioVAE decoder rates must be positive")
        if any(kernel <= 0 for kernel in self.decoder_kernel_sizes):
            raise ValueError("MiniMax-H3 AudioVAE decoder kernels must be positive")
        if any(kernel <= 0 or kernel % 2 == 0 for kernel in self.resblock_kernel_sizes):
            raise ValueError("MiniMax-H3 AudioVAE residual kernels must be positive and odd")
        if any(not dilations or any(value <= 0 for value in dilations) for dilations in self.resblock_dilation_sizes):
            raise ValueError("MiniMax-H3 AudioVAE residual dilations must be positive")
        divisor = 1 << self.num_upsamples
        if self.decoder_dim % divisor:
            raise ValueError("MiniMax-H3 AudioVAE decoder_dim must halve at every stage")
        if self.decoder_dim // divisor <= 0:
            raise ValueError("MiniMax-H3 AudioVAE final decoder width must be positive")


DEFAULT_AUDIO_VAE_DECODER_CONFIG = AudioVAEDecoderConfig()


def decoder_config_from_checkpoint(
    raw: Mapping[str, object],
    *,
    latent_frames: int,
    min_latent_frames: int | None = None,
    max_latent_frames: int | None = None,
) -> AudioVAEDecoderConfig:
    """Validate and convert the public Diffusers audio-VAE config."""

    required = (
        "latent_channels",
        "latent_dim",
        "decoder_dim",
        "decoder_rates",
        "decoder_kernel_sizes",
        "resblock_kernel_sizes",
        "resblock_dilation_sizes",
        "sampling_rate",
    )
    missing = [name for name in required if name not in raw]
    if missing:
        raise ValueError(f"MiniMax-H3 AudioVAE config is missing fields: {missing}")

    try:
        profile = AudioVAEDecoderConfig(
            latent_channels=int(raw["latent_channels"]),
            latent_frames=latent_frames,
            min_latent_frames=(
                latent_frames if min_latent_frames is None else min_latent_frames
            ),
            max_latent_frames=(
                latent_frames if max_latent_frames is None else max_latent_frames
            ),
            latent_dim=int(raw["latent_dim"]),
            decoder_dim=int(raw["decoder_dim"]),
            decoder_rates=tuple(int(value) for value in raw["decoder_rates"]),
            decoder_kernel_sizes=tuple(int(value) for value in raw["decoder_kernel_sizes"]),
            resblock_kernel_sizes=tuple(int(value) for value in raw["resblock_kernel_sizes"]),
            resblock_dilation_sizes=tuple(
                tuple(int(value) for value in values)
                for values in raw["resblock_dilation_sizes"]
            ),
            sampling_rate=int(raw["sampling_rate"]),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("MiniMax-H3 AudioVAE config has invalid decoder fields") from error
    profile.validate()

    encoder_rates = raw.get("encoder_rates")
    if not isinstance(encoder_rates, (list, tuple)):
        raise ValueError("MiniMax-H3 AudioVAE config is missing encoder_rates")
    try:
        encoder_hop = math.prod(int(value) for value in encoder_rates)
    except (TypeError, ValueError) as error:
        raise ValueError("MiniMax-H3 AudioVAE encoder_rates are invalid") from error
    if encoder_hop != profile.hop_length:
        raise ValueError(
            "MiniMax-H3 AudioVAE encoder/decoder hop mismatch: "
            f"encoder={encoder_hop}, decoder={profile.hop_length}"
        )
    return profile


def checkpoint_keys(profile: AudioVAEDecoderConfig = DEFAULT_AUDIO_VAE_DECODER_CONFIG) -> tuple[str, ...]:
    """Return exactly the checkpoint tensors consumed by the decoder plan."""

    profile.validate()
    names = [
        "dec_in_proj.weight",
        "dec_in_proj.bias",
        "decoder.conv_pre.weight_g",
        "decoder.conv_pre.weight_v",
        "decoder.conv_pre.bias",
    ]
    for stage in range(profile.num_upsamples):
        prefix = f"decoder.ups.{stage}.0"
        names.extend((f"{prefix}.weight_g", f"{prefix}.weight_v", f"{prefix}.bias"))

    for block in range(profile.num_upsamples * profile.num_kernels):
        dilation_count = len(
            profile.resblock_dilation_sizes[block % profile.num_kernels]
        )
        prefix = f"decoder.resblocks.{block}"
        for activation in range(2 * dilation_count):
            activation_prefix = f"{prefix}.activations.{activation}"
            names.extend(
                (
                    f"{activation_prefix}.act.alpha",
                    f"{activation_prefix}.act.beta",
                    f"{activation_prefix}.upsample.filter",
                    f"{activation_prefix}.downsample.lowpass.filter",
                )
            )
        for group in ("convs1", "convs2"):
            for index in range(dilation_count):
                convolution_prefix = f"{prefix}.{group}.{index}"
                names.extend(
                    (
                        f"{convolution_prefix}.weight_g",
                        f"{convolution_prefix}.weight_v",
                        f"{convolution_prefix}.bias",
                    )
                )

    names.extend(
        (
            "decoder.activation_post.act.alpha",
            "decoder.activation_post.act.beta",
            "decoder.activation_post.upsample.filter",
            "decoder.activation_post.downsample.lowpass.filter",
            "decoder.conv_post.weight_g",
            "decoder.conv_post.weight_v",
        )
    )
    return tuple(names)


def _require_array(weights: Mapping[str, object], name: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        value = np.asarray(weights[name])
    except KeyError as error:
        raise ValueError(f"MiniMax-H3 AudioVAE checkpoint is missing tensor: {name}") from error
    if tuple(value.shape) != shape:
        raise ValueError(
            f"MiniMax-H3 AudioVAE tensor {name} has shape {tuple(value.shape)}, expected {shape}"
        )
    if value.dtype != np.float32:
        raise ValueError(
            f"MiniMax-H3 AudioVAE tensor {name} must remain FP32, got {value.dtype}"
        )
    return np.ascontiguousarray(value)


def fold_weight_norm(weight_g: object, weight_v: object) -> np.ndarray:
    """Fold PyTorch ``weight_norm(..., dim=0)`` without changing FP32 dtype."""

    gain = np.asarray(weight_g)
    direction = np.asarray(weight_v)
    if gain.dtype != np.float32 or direction.dtype != np.float32:
        raise ValueError("MiniMax-H3 AudioVAE weight-normalized tensors must be FP32")
    if gain.ndim != direction.ndim or gain.shape[0] != direction.shape[0]:
        raise ValueError("MiniMax-H3 AudioVAE weight-normalized tensor shapes do not align")
    if gain.shape[1:] != (1,) * (direction.ndim - 1):
        raise ValueError("MiniMax-H3 AudioVAE weight_g must normalize dimension zero")
    axes = tuple(range(1, direction.ndim))
    norm = np.sqrt(
        np.sum(direction * direction, axis=axes, keepdims=True, dtype=np.float32)
    )
    if np.any(norm == 0.0):
        raise ValueError("MiniMax-H3 AudioVAE weight_v contains a zero-norm filter")
    return np.ascontiguousarray(direction * (gain / norm), dtype=np.float32)


def _fold_checkpoint_weight(
    weights: Mapping[str, object], prefix: str, shape: tuple[int, ...]
) -> np.ndarray:
    gain_shape = (shape[0],) + (1,) * (len(shape) - 1)
    gain = _require_array(weights, f"{prefix}.weight_g", gain_shape)
    direction = _require_array(weights, f"{prefix}.weight_v", shape)
    return fold_weight_norm(gain, direction)


def _conv1d(
    network,
    hidden,
    weight: np.ndarray,
    bias: np.ndarray | None,
    *,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    name: str,
    owned_weights: list[np.ndarray],
):
    out_channels, in_channels_per_group, kernel_size = weight.shape
    kernel = np.ascontiguousarray(weight[:, :, None, :], dtype=np.float32)
    owned_weights.append(kernel)
    if bias is not None:
        bias = np.ascontiguousarray(bias, dtype=np.float32)
        owned_weights.append(bias)
    layer = network.add_convolution_nd(
        hidden,
        out_channels,
        (1, kernel_size),
        kernel,
        bias,
    )
    if layer is None:
        raise RuntimeError(f"TensorRT rejected MiniMax-H3 AudioVAE convolution {name}")
    layer.name = name
    layer.stride_nd = (1, stride)
    layer.padding_nd = (0, padding)
    layer.dilation_nd = (1, dilation)
    layer.num_groups = groups
    expected_inputs = int(tuple(hidden.shape)[1]) // groups
    if in_channels_per_group != expected_inputs:
        raise ValueError(
            f"MiniMax-H3 AudioVAE convolution {name} expects {in_channels_per_group} "
            f"channels per group, got {expected_inputs}"
        )
    return layer.get_output(0)


def _deconv1d(
    network,
    hidden,
    weight: np.ndarray,
    bias: np.ndarray | None,
    *,
    out_channels: int,
    stride: int = 1,
    padding: int = 0,
    groups: int = 1,
    name: str,
    owned_weights: list[np.ndarray],
):
    in_channels, out_channels_per_group, kernel_size = weight.shape
    kernel = np.ascontiguousarray(weight[:, :, None, :], dtype=np.float32)
    owned_weights.append(kernel)
    if bias is not None:
        bias = np.ascontiguousarray(bias, dtype=np.float32)
        owned_weights.append(bias)
    if in_channels != int(tuple(hidden.shape)[1]):
        raise ValueError(
            f"MiniMax-H3 AudioVAE deconvolution {name} expects {in_channels} input channels, "
            f"got {tuple(hidden.shape)[1]}"
        )
    if out_channels_per_group * groups != out_channels:
        raise ValueError(
            f"MiniMax-H3 AudioVAE deconvolution {name} has an invalid grouped output width"
        )
    layer = network.add_deconvolution_nd(
        hidden,
        out_channels,
        (1, kernel_size),
        kernel,
        bias,
    )
    if layer is None:
        raise RuntimeError(f"TensorRT rejected MiniMax-H3 AudioVAE deconvolution {name}")
    layer.name = name
    layer.stride_nd = (1, stride)
    layer.padding_nd = (0, padding)
    layer.num_groups = groups
    return layer.get_output(0)


def _replicate_pad(network, hidden, left: int, right: int, *, name: str):
    batch, channels, height, _width = tuple(int(value) for value in hidden.shape)
    layer = network.add_slice(
        hidden,
        (0, 0, 0, -left),
        (batch, channels, height, 1),
        (1, 1, 1, 1),
    )
    if layer is None:
        raise RuntimeError(f"TensorRT rejected MiniMax-H3 AudioVAE replicate pad {name}")
    layer.name = name
    layer.mode = trt.SampleMode.CLAMP
    shape = network.add_shape(hidden).get_output(0)
    delta = op.constant(
        network,
        np.asarray([0, 0, 0, left + right], dtype=np.int64),
        dtype=np.int64,
    )
    size = network.add_elementwise(shape, delta, trt.ElementWiseOperation.SUM).get_output(0)
    layer.set_input(2, size)
    return layer.get_output(0)


def _crop_time(network, hidden, left: int, right: int, *, name: str):
    batch, channels, height, _width = tuple(int(value) for value in hidden.shape)
    layer = network.add_slice(
        hidden,
        (0, 0, 0, left),
        (batch, channels, height, 1),
        (1, 1, 1, 1),
    )
    if layer is None:
        raise RuntimeError(f"TensorRT rejected MiniMax-H3 AudioVAE crop {name}")
    layer.name = name
    shape = network.add_shape(hidden).get_output(0)
    delta = op.constant(
        network,
        np.asarray([0, 0, 0, -(left + right)], dtype=np.int64),
        dtype=np.int64,
    )
    size = network.add_elementwise(shape, delta, trt.ElementWiseOperation.SUM).get_output(0)
    layer.set_input(2, size)
    return layer.get_output(0)


def _snake_beta(network, hidden, weights: Mapping[str, object], prefix: str):
    channels = int(tuple(hidden.shape)[1])
    alpha = _require_array(weights, f"{prefix}.alpha", (channels,)).reshape(1, channels, 1, 1)
    beta = _require_array(weights, f"{prefix}.beta", (channels,)).reshape(1, channels, 1, 1)
    alpha = network.add_unary(op.weight_constant(network, alpha), trt.UnaryOperation.EXP).get_output(0)
    beta = network.add_unary(op.weight_constant(network, beta), trt.UnaryOperation.EXP).get_output(0)
    phase = network.add_elementwise(hidden, alpha, trt.ElementWiseOperation.PROD).get_output(0)
    sine = network.add_unary(phase, trt.UnaryOperation.SIN).get_output(0)
    sine_square = network.add_elementwise(sine, sine, trt.ElementWiseOperation.PROD).get_output(0)
    epsilon = op.constant(network, np.full((1, 1, 1, 1), 1.0e-9, np.float32))
    denominator = network.add_elementwise(beta, epsilon, trt.ElementWiseOperation.SUM).get_output(0)
    periodic = network.add_elementwise(
        sine_square, denominator, trt.ElementWiseOperation.DIV
    ).get_output(0)
    return network.add_elementwise(hidden, periodic, trt.ElementWiseOperation.SUM).get_output(0)


def _alias_free_activation(
    network,
    hidden,
    weights: Mapping[str, object],
    prefix: str,
    *,
    owned_weights: list[np.ndarray],
):
    ratio = 2
    kernel_size = 12
    channels = int(tuple(hidden.shape)[1])
    pad = kernel_size // ratio - 1
    pad_left = pad * ratio + (kernel_size - ratio) // 2
    pad_right = pad * ratio + (kernel_size - ratio + 1) // 2

    padded = _replicate_pad(network, hidden, pad, pad, name=f"{prefix}.upsample.pad")
    upsample_filter = _require_array(weights, f"{prefix}.upsample.filter", (1, 1, kernel_size))
    upsample_filter = np.broadcast_to(
        upsample_filter, (channels, 1, kernel_size)
    ).copy()
    hidden = _deconv1d(
        network,
        padded,
        upsample_filter,
        None,
        out_channels=channels,
        stride=ratio,
        groups=channels,
        name=f"{prefix}.upsample",
        owned_weights=owned_weights,
    )
    scale = op.constant(network, np.full((1, 1, 1, 1), float(ratio), np.float32))
    hidden = network.add_elementwise(hidden, scale, trt.ElementWiseOperation.PROD).get_output(0)
    hidden = _crop_time(
        network,
        hidden,
        pad_left,
        pad_right,
        name=f"{prefix}.upsample.crop",
    )
    hidden = _snake_beta(network, hidden, weights, f"{prefix}.act")

    even = kernel_size % 2 == 0
    down_left = kernel_size // 2 - int(even)
    down_right = kernel_size // 2
    hidden = _replicate_pad(
        network,
        hidden,
        down_left,
        down_right,
        name=f"{prefix}.downsample.pad",
    )
    downsample_filter = _require_array(
        weights, f"{prefix}.downsample.lowpass.filter", (1, 1, kernel_size)
    )
    downsample_filter = np.broadcast_to(
        downsample_filter, (channels, 1, kernel_size)
    ).copy()
    return _conv1d(
        network,
        hidden,
        downsample_filter,
        None,
        stride=ratio,
        groups=channels,
        name=f"{prefix}.downsample",
        owned_weights=owned_weights,
    )


def _amp_block(
    network,
    hidden,
    weights: Mapping[str, object],
    prefix: str,
    *,
    kernel_size: int,
    dilations: tuple[int, ...],
    owned_weights: list[np.ndarray],
):
    channels = int(tuple(hidden.shape)[1])
    for index, dilation in enumerate(dilations):
        residual = _alias_free_activation(
            network,
            hidden,
            weights,
            f"{prefix}.activations.{2 * index}",
            owned_weights=owned_weights,
        )
        conv1_prefix = f"{prefix}.convs1.{index}"
        conv1_weight = _fold_checkpoint_weight(
            weights, conv1_prefix, (channels, channels, kernel_size)
        )
        conv1_bias = _require_array(weights, f"{conv1_prefix}.bias", (channels,))
        residual = _conv1d(
            network,
            residual,
            conv1_weight,
            conv1_bias,
            padding=(kernel_size * dilation - dilation) // 2,
            dilation=dilation,
            name=conv1_prefix,
            owned_weights=owned_weights,
        )
        residual = _alias_free_activation(
            network,
            residual,
            weights,
            f"{prefix}.activations.{2 * index + 1}",
            owned_weights=owned_weights,
        )
        conv2_prefix = f"{prefix}.convs2.{index}"
        conv2_weight = _fold_checkpoint_weight(
            weights, conv2_prefix, (channels, channels, kernel_size)
        )
        conv2_bias = _require_array(weights, f"{conv2_prefix}.bias", (channels,))
        residual = _conv1d(
            network,
            residual,
            conv2_weight,
            conv2_bias,
            padding=(kernel_size - 1) // 2,
            name=conv2_prefix,
            owned_weights=owned_weights,
        )
        hidden = network.add_elementwise(
            hidden, residual, trt.ElementWiseOperation.SUM
        ).get_output(0)
    return hidden


@op.cleanup_failed_build
def build_audio_vae_decoder_engine(
    weights: dict,
    profile: AudioVAEDecoderConfig = DEFAULT_AUDIO_VAE_DECODER_CONFIG,
    *,
    verbose: bool = False,
    consume_weights: bool = False,
    workspace_bytes: int | None = None,
    weight_streaming: bool = False,
    output_path: str | Path | None = None,
) -> bytes | dict[str, int | str]:
    """Build the released MiniMax-H3 BigVGAN decoder as a native FP32 plan."""

    profile.validate()
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    op.configure_builder(config, weight_streaming=weight_streaming)
    op.configure_workspace(
        config,
        workspace_bytes,
        default_bytes=AUDIO_VAE_DECODER_DEFAULT_WORKSPACE_BYTES,
    )
    missing = sorted(set(checkpoint_keys(profile)) - set(weights))
    if missing:
        raise ValueError(f"MiniMax-H3 AudioVAE checkpoint is missing tensors: {missing}")
    owned_weights: list[np.ndarray] = []

    latents = network.add_input(
        "audio_latents",
        trt.float32,
        (profile.batch_size, profile.latent_channels, -1),
    )
    shape_profile = builder.create_optimization_profile()
    shape_profile.set_shape(
        "audio_latents",
        (profile.batch_size, profile.latent_channels, profile.min_latent_frames),
        (profile.batch_size, profile.latent_channels, profile.latent_frames),
        (profile.batch_size, profile.latent_channels, profile.max_latent_frames),
    )
    profile_index = config.add_optimization_profile(shape_profile)
    if profile_index != 0:
        raise RuntimeError("TensorRT rejected MiniMax-H3 AudioVAE dynamic shape profile")
    expanded = network.add_shuffle(latents)
    expanded.reshape_dims = (
        profile.batch_size,
        profile.latent_channels,
        1,
        -1,
    )
    hidden = expanded.get_output(0)

    dec_in_weight = _require_array(
        weights,
        "dec_in_proj.weight",
        (profile.latent_dim, profile.latent_channels, 1),
    )
    dec_in_bias = _require_array(weights, "dec_in_proj.bias", (profile.latent_dim,))
    hidden = _conv1d(
        network,
        hidden,
        dec_in_weight,
        dec_in_bias,
        name="dec_in_proj",
        owned_weights=owned_weights,
    )

    conv_pre_weight = _fold_checkpoint_weight(
        weights,
        "decoder.conv_pre",
        (profile.decoder_dim, profile.latent_dim, 7),
    )
    conv_pre_bias = _require_array(weights, "decoder.conv_pre.bias", (profile.decoder_dim,))
    hidden = _conv1d(
        network,
        hidden,
        conv_pre_weight,
        conv_pre_bias,
        padding=3,
        name="decoder.conv_pre",
        owned_weights=owned_weights,
    )

    for stage, (rate, kernel_size) in enumerate(
        zip(profile.decoder_rates, profile.decoder_kernel_sizes)
    ):
        input_channels = profile.decoder_dim // (1 << stage)
        output_channels = profile.decoder_dim // (1 << (stage + 1))
        upsample_prefix = f"decoder.ups.{stage}.0"
        upsample_weight = _fold_checkpoint_weight(
            weights,
            upsample_prefix,
            (input_channels, output_channels, kernel_size),
        )
        upsample_bias = _require_array(weights, f"{upsample_prefix}.bias", (output_channels,))
        hidden = _deconv1d(
            network,
            hidden,
            upsample_weight,
            upsample_bias,
            out_channels=output_channels,
            stride=rate,
            padding=(kernel_size - rate) // 2,
            name=upsample_prefix,
            owned_weights=owned_weights,
        )

        block_outputs = []
        for kernel_index, (resblock_kernel, dilations) in enumerate(
            zip(profile.resblock_kernel_sizes, profile.resblock_dilation_sizes)
        ):
            block_index = stage * profile.num_kernels + kernel_index
            block_outputs.append(
                _amp_block(
                    network,
                    hidden,
                    weights,
                    f"decoder.resblocks.{block_index}",
                    kernel_size=resblock_kernel,
                    dilations=dilations,
                    owned_weights=owned_weights,
                )
            )
        combined = block_outputs[0]
        for block_output in block_outputs[1:]:
            combined = network.add_elementwise(
                combined, block_output, trt.ElementWiseOperation.SUM
            ).get_output(0)
        divisor = op.constant(
            network, np.full((1, 1, 1, 1), float(profile.num_kernels), np.float32)
        )
        hidden = network.add_elementwise(
            combined, divisor, trt.ElementWiseOperation.DIV
        ).get_output(0)

    hidden = _alias_free_activation(
        network,
        hidden,
        weights,
        "decoder.activation_post",
        owned_weights=owned_weights,
    )
    final_channels = profile.decoder_dim // (1 << profile.num_upsamples)
    conv_post_weight = _fold_checkpoint_weight(
        weights,
        "decoder.conv_post",
        (1, final_channels, 7),
    )
    hidden = _conv1d(
        network,
        hidden,
        conv_post_weight,
        None,
        padding=3,
        name="decoder.conv_post",
        owned_weights=owned_weights,
    )
    clamp = network.add_activation(hidden, trt.ActivationType.CLIP)
    if clamp is None:
        raise RuntimeError("TensorRT rejected MiniMax-H3 AudioVAE output clamp")
    clamp.alpha = -1.0
    clamp.beta = 1.0
    output = network.add_shuffle(clamp.get_output(0))
    output.reshape_dims = (profile.batch_size, 1, -1)
    decoded = output.get_output(0)
    decoded.name = "decoded_audio"
    network.mark_output(decoded)
    op.validate_native_network(network, expected_attentions=0, label="AudioVAE decoder")

    print(
        "[minimax-h3] building native FP32 AudioVAE decoder: "
        f"latents={profile.min_latent_frames}/{profile.latent_frames}/"
        f"{profile.max_latent_frames}, samples={profile.min_output_samples}/"
        f"{profile.output_samples}/{profile.max_output_samples}, "
        f"upsamples={profile.num_upsamples}",
        file=sys.stderr,
    )
    plan = None
    record = None
    try:
        if output_path is None:
            plan = builder.build_serialized_network(network, config)
        else:
            record = trt_compat.build_serialized_network_to_file(
                builder, network, config, output_path
            )
    finally:
        op.release_weight_buffers(network)
        if consume_weights:
            weights.clear()
    if output_path is None and plan is None:
        raise RuntimeError("TensorRT failed to build MiniMax-H3 AudioVAE decoder")
    del owned_weights, network, config, builder
    gc.collect()
    return record if record is not None else bytes(plan)
