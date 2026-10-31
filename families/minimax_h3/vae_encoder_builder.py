# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native FP32 H3 VisualVAE encoder for one spatial tile / temporal clip.

The input is ImageNet-normalized NCTHW, not raw RGB. The output contains
posterior moments, not sampled latents. Runtime must stitch tile moments
before constructing the posterior, and must preserve the original single
image / 17-frame clip distinction. No tiling, sampling or dtype boundary is
silently replaced by a full-image encoder or a posterior mean.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import math

import numpy as np
import tensorrt as trt

from . import graph_ops as op


@dataclass(frozen=True)
class VideoEncoderConfig:
    in_channels: int = 3
    latent_channels: int = 24
    block_out_channels: tuple[int, ...] = (128, 256, 256, 512, 512, 1024)
    layers_per_block: int = 2
    spatial_downsample_factors: tuple[int, ...] = (2, 2, 2, 2, 1, 1)
    temporal_downsample_factors: tuple[int, ...] = (1, 2, 2, 1, 1, 1)
    norm_num_groups: int = 32
    norm_eps: float = 1e-6

    @property
    def spatial_ratio(self) -> int:
        return math.prod(self.spatial_downsample_factors)

    @property
    def temporal_ratio(self) -> int:
        return math.prod(self.temporal_downsample_factors)

    def validate(self) -> None:
        levels = len(self.block_out_channels)
        if (
            not levels
            or len(self.spatial_downsample_factors) != levels
            or len(self.temporal_downsample_factors) != levels
        ):
            raise ValueError("H3 encoder channel and stride levels must agree")
        values = (
            self.in_channels,
            self.latent_channels,
            self.layers_per_block,
            self.norm_num_groups,
            *self.block_out_channels,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values
        ):
            raise ValueError("H3 encoder channel/group/layer counts must be positive integers")
        if any(channels % self.norm_num_groups for channels in self.block_out_channels):
            raise ValueError("H3 encoder channels must be divisible by the GroupNorm group count")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value not in (1, 2)
            for value in (*self.spatial_downsample_factors, *self.temporal_downsample_factors)
        ):
            raise ValueError("H3 encoder downsample factors must be one or two")
        if any(
            temporal > spatial
            for temporal, spatial in zip(
                self.temporal_downsample_factors, self.spatial_downsample_factors
            )
        ):
            raise ValueError(
                "H3 encoder supports temporal downsampling only with spatial downsampling"
            )
        if not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError("H3 encoder normalization epsilon must be finite and positive")


def checkpoint_shapes(
    config: VideoEncoderConfig = VideoEncoderConfig(),
) -> dict[str, tuple[int, ...]]:
    """All encoder and quant-convolution tensors; no decoder payload."""

    config.validate()
    shapes = {}

    def norm(prefix, channels):
        shapes[prefix + ".weight"] = (channels,)
        shapes[prefix + ".bias"] = (channels,)

    def conv(prefix, in_channels, out_channels, kernel):
        shapes[prefix + ".weight"] = (out_channels, in_channels, kernel, kernel, kernel)
        shapes[prefix + ".bias"] = (out_channels,)

    channels = config.block_out_channels[0]
    conv("encoder.conv_in", config.in_channels, channels, 3)
    for level, out_channels in enumerate(config.block_out_channels):
        for index in range(config.layers_per_block):
            prefix = f"encoder.down_blocks.{level}.resnets.{index}"
            norm(prefix + ".norm1", channels)
            conv(prefix + ".conv1", channels, out_channels, 3)
            norm(prefix + ".norm2", out_channels)
            conv(prefix + ".conv2", out_channels, out_channels, 3)
            if channels != out_channels:
                conv(prefix + ".conv_shortcut", channels, out_channels, 1)
            channels = out_channels
        if config.spatial_downsample_factors[level] * config.temporal_downsample_factors[level] > 1:
            conv(f"encoder.down_blocks.{level}.downsamplers.0.conv", channels, channels, 3)
    norm("encoder.norm_out", channels)
    moments = 2 * config.latent_channels
    conv("encoder.conv_out", channels, moments, 3)
    conv("quant_conv", moments, moments, 1)
    return shapes


def checkpoint_keys(config: VideoEncoderConfig = VideoEncoderConfig()) -> tuple[str, ...]:
    return tuple(checkpoint_shapes(config))


def _validate_weights(weights, config):
    shapes = checkpoint_shapes(config)
    if set(weights) != set(shapes):
        raise ValueError(
            "H3 tile encoder requires exactly its encoder/quant-conv checkpoint tensors"
        )
    for name, shape in shapes.items():
        value = np.asarray(weights[name])
        if value.shape != shape or value.dtype != np.float32:
            raise ValueError(
                f"H3 encoder tensor {name} requires FP32 shape {shape}, got {value.dtype} {value.shape}"
            )


def _spatial_reflect(network, value, before: int, after: int):
    if not (before or after):
        return value
    batch, channels, _, height, width = tuple(value.shape)
    if height <= max(before, after) or width <= max(before, after):
        raise ValueError("H3 reflection padding exceeds its input spatial extent")
    sizes = (batch, channels, 1, height + before + after, width + before + after)
    layer = network.add_slice(value, (0, 0, 0, -before, -before), sizes, (1, 1, 1, 1, 1))
    layer.set_input(
        2,
        op._shape_vector(
            network,
            (
                batch,
                channels,
                op._shape_dim(network, value, 2),
                height + before + after,
                width + before + after,
            ),
        ),
    )
    layer.mode = trt.SampleMode.REFLECT
    return layer.get_output(0)


def _conv(network, value, weights, prefix, *, spatial_before=0, spatial_after=0, stride=(1, 1, 1)):
    value = _spatial_reflect(network, value, spatial_before, spatial_after)
    kernel = np.ascontiguousarray(weights[prefix + ".weight"])
    bias = np.ascontiguousarray(weights[prefix + ".bias"])
    # Convolution's Weights, like native BF16 constants, borrow their buffers.
    op._WEIGHT_BUFFER_KEEPALIVE.setdefault(id(network), []).extend((kernel, bias))
    layer = network.add_convolution_nd(value, kernel.shape[0], kernel.shape[2:], kernel, bias)
    if layer is None:
        raise RuntimeError(f"TensorRT rejected H3 encoder convolution {prefix}")
    layer.name = prefix
    layer.stride_nd = stride
    layer.pre_padding = (kernel.shape[2] - 1, 0, 0)
    layer.post_padding = (0, 0, 0)
    return layer.get_output(0)


def _frame_group_norm(network, value, weights, prefix, config):
    _, channels, _, height, width = tuple(value.shape)
    groups = config.norm_num_groups
    arrange = network.add_shuffle(value)
    arrange.first_transpose = trt.Permutation((0, 2, 1, 3, 4))
    arrange.reshape_dims = (0, 0, groups, channels // groups, height, width)
    shape = (1, 1, groups, channels // groups, 1, 1)
    gamma = op.weight_constant(network, weights[prefix + ".weight"].reshape(shape))
    beta = op.weight_constant(network, weights[prefix + ".bias"].reshape(shape))
    normalization = network.add_normalization_v2(
        arrange.get_output(0), gamma, beta, (1 << 3) | (1 << 4) | (1 << 5)
    )
    if normalization is None:
        raise RuntimeError(f"TensorRT rejected H3 frame-isolated GroupNorm {prefix}")
    normalization.name = prefix
    normalization.epsilon = config.norm_eps
    restore = network.add_shuffle(normalization.get_output(0))
    restore.reshape_dims = (0, 0, channels, height, width)
    restore.second_transpose = trt.Permutation((0, 2, 1, 3, 4))
    return restore.get_output(0)


def _resnet(network, value, weights, prefix, out_channels, config):
    residual = value
    hidden = op.silu(network, _frame_group_norm(network, value, weights, prefix + ".norm1", config))
    hidden = _conv(network, hidden, weights, prefix + ".conv1", spatial_before=1, spatial_after=1)
    hidden = op.silu(
        network, _frame_group_norm(network, hidden, weights, prefix + ".norm2", config)
    )
    hidden = _conv(network, hidden, weights, prefix + ".conv2", spatial_before=1, spatial_after=1)
    if int(value.shape[1]) != out_channels:
        residual = _conv(network, residual, weights, prefix + ".conv_shortcut")
    return network.add_elementwise(residual, hidden, trt.ElementWiseOperation.SUM).get_output(0)


def build_vae_tile_encoder_engine(
    weights: dict,
    *,
    config: VideoEncoderConfig = VideoEncoderConfig(),
    tile_height: int = 256,
    tile_width: int = 256,
    max_frames: int = 17,
    workspace_bytes: int = 8 << 30,
    consume_weights: bool = False,
    verbose: bool = False,
) -> bytes:
    """Encode one original spatial tile; input T ranges from one to one clip."""

    config.validate()
    for length in (tile_height, tile_width):
        if (
            isinstance(length, bool)
            or not isinstance(length, int)
            or length < 2 * config.spatial_ratio
            or length % config.spatial_ratio
        ):
            raise ValueError(
                "H3 encoder tile sizes must be latent-aligned and at least two latent pixels"
            )
    if isinstance(max_frames, bool) or not isinstance(max_frames, int) or not 1 <= max_frames <= 17:
        raise ValueError("H3 encoder expects a single image or a clip of at most 17 frames")
    _validate_weights(weights, config)
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    options = builder.create_builder_config()
    options.builder_optimization_level = 1
    op.configure_builder(options)
    op.configure_workspace(options, workspace_bytes, default_bytes=8 << 30)
    # Keep the component's FP32 comparison independent of reduced-mantissa conv tactics.
    options.clear_flag(trt.BuilderFlag.TF32)
    pixels = network.add_input(
        "pixels", trt.float32, (1, config.in_channels, -1, tile_height, tile_width)
    )
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "pixels",
        min=(1, config.in_channels, 1, tile_height, tile_width),
        opt=(1, config.in_channels, 1, tile_height, tile_width),
        max=(1, config.in_channels, max_frames, tile_height, tile_width),
    )
    options.add_optimization_profile(profile)
    try:
        hidden = _conv(
            network, pixels, weights, "encoder.conv_in", spatial_before=1, spatial_after=1
        )
        for level, channels in enumerate(config.block_out_channels):
            for index in range(config.layers_per_block):
                hidden = _resnet(
                    network,
                    hidden,
                    weights,
                    f"encoder.down_blocks.{level}.resnets.{index}",
                    channels,
                    config,
                )
            temporal, spatial = (
                config.temporal_downsample_factors[level],
                config.spatial_downsample_factors[level],
            )
            if temporal * spatial > 1:
                hidden = _conv(
                    network,
                    hidden,
                    weights,
                    f"encoder.down_blocks.{level}.downsamplers.0.conv",
                    spatial_after=1 if spatial == 2 else 0,
                    stride=(temporal, spatial, spatial),
                )
        hidden = op.silu(
            network, _frame_group_norm(network, hidden, weights, "encoder.norm_out", config)
        )
        hidden = _conv(
            network, hidden, weights, "encoder.conv_out", spatial_before=1, spatial_after=1
        )
        moments = _conv(network, hidden, weights, "quant_conv")
        moments.name = "moments"
        network.mark_output(moments)
        op.validate_native_network(network, expected_attentions=0, label="VAE tile encoder")
        plan = builder.build_serialized_network(network, options)
    finally:
        op.release_weight_buffers(network)
        if consume_weights:
            weights.clear()
    if plan is None:
        raise RuntimeError("TensorRT failed to build H3 VAE tile encoder")
    result = bytes(plan)
    del network, options, builder, logger
    gc.collect()
    return result
