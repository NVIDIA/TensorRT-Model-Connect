# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native temporal VideoVAE tile encoder for MiniMax-H3 Ref2VA.

The released VideoVAE evaluates every non-image reference in 17-frame clips
and every 256x256 spatial tile independently.  This TensorRT-RTX plan owns one
such exact unit: causal zero padding in time, reflection padding in space,
frame-isolated group normalization, both temporal downsampling stages, and the
posterior projection.  Native C++ performs clip padding, tile blending,
posterior sampling, FP16 rounding, and latent normalization as specified by
``ref2va_contract``; no framework is needed after bundle construction.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np

from tensorrt_model_connect import trt_compat

from . import graph_ops as op
from .fl2va_contract import PlanAbi, TensorAbi, VAE_SPATIAL_COMPRESSION, VAE_TILE_SIZE
from .fl2va_vae_encoder_builder import checkpoint_keys as _visual_checkpoint_keys


trt = trt_compat.get_trt()

IN_CHANNELS = 3
MOMENT_CHANNELS = 48
BLOCK_OUT_CHANNELS = (128, 256, 256, 512, 512, 1024)
SPATIAL_DOWNSAMPLE_FACTORS = (2, 2, 2, 2, 1, 1)
TEMPORAL_DOWNSAMPLE_FACTORS = (1, 2, 2, 1, 1, 1)
LAYERS_PER_BLOCK = 2
NORM_GROUPS = 32
NORM_EPS = 1.0e-6
CLIP_FRAMES = 17
CLIP_LATENT_FRAMES = 5
REF2VA_VIDEO_ENCODER_DEFAULT_WORKSPACE_BYTES = 32 << 30


def checkpoint_keys() -> tuple[str, ...]:
    """Reuse the released visual VAE partition without duplicating its schema."""

    return _visual_checkpoint_keys()


def ref2va_video_encoder_abi() -> PlanAbi:
    """ABI of one exact temporal-clip/spatial-tile VideoVAE invocation.

    The batch is deliberately one: Diffusers evaluates tiles in a Python loop,
    and keeping that unit at the ABI both reproduces its arithmetic and bounds
    peak activation memory on unified-memory workstations.  Native C++ loops
    over the public canvas tile grid and all padded 17-frame clips.
    """

    latent_side = VAE_TILE_SIZE // VAE_SPATIAL_COMPRESSION
    return PlanAbi(
        filename="ref2va_video_vae_encoder.plan",
        inputs=(
            TensorAbi(
                "pixel_tile_clip",
                "float32",
                (1, IN_CHANNELS, CLIP_FRAMES, VAE_TILE_SIZE, VAE_TILE_SIZE),
                (1, IN_CHANNELS, CLIP_FRAMES, VAE_TILE_SIZE, VAE_TILE_SIZE),
                (1, IN_CHANNELS, CLIP_FRAMES, VAE_TILE_SIZE, VAE_TILE_SIZE),
            ),
        ),
        outputs=(
            TensorAbi(
                "posterior_parameter_tile_clip",
                "float32",
                (1, MOMENT_CHANNELS, CLIP_LATENT_FRAMES, latent_side, latent_side),
                (1, MOMENT_CHANNELS, CLIP_LATENT_FRAMES, latent_side, latent_side),
                (1, MOMENT_CHANNELS, CLIP_LATENT_FRAMES, latent_side, latent_side),
            ),
        ),
    )


def _require_weight(weights: dict[str, object], name: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        value = np.asarray(weights[name])
    except KeyError as error:
        raise ValueError(f"MiniMax-H3 Ref2VA VideoVAE checkpoint is missing {name}") from error
    if tuple(value.shape) != shape:
        raise ValueError(
            f"MiniMax-H3 Ref2VA VideoVAE tensor {name} has shape {tuple(value.shape)}, "
            f"expected {shape}"
        )
    if value.dtype != np.float32:
        raise ValueError(
            f"MiniMax-H3 Ref2VA VideoVAE tensor {name} must remain FP32, got {value.dtype}"
        )
    return np.ascontiguousarray(value)


def _spatial_reflect_pad(
    network,
    hidden,
    *,
    channels: int,
    frames: int,
    height: int,
    width: int,
    top: int = 0,
    bottom: int = 0,
    left: int = 0,
    right: int = 0,
):
    if min(top, bottom, left, right) < 0:
        raise ValueError("MiniMax-H3 Ref2VA VideoVAE reflection padding cannot be negative")
    if top >= height or bottom >= height or left >= width or right >= width:
        raise ValueError("MiniMax-H3 Ref2VA VideoVAE reflection padding exceeds its axis")
    if top == bottom == left == right == 0:
        return hidden
    layer = network.add_slice(
        hidden,
        (0, 0, 0, -top, -left),
        (1, channels, frames, height + top + bottom, width + left + right),
        (1, 1, 1, 1, 1),
    )
    if layer is None:
        raise RuntimeError("TensorRT rejected MiniMax-H3 Ref2VA VideoVAE reflection padding")
    layer.mode = trt.SampleMode.REFLECT
    return layer.get_output(0)


def _causal_zero_pad(network, hidden, temporal_padding: int):
    if temporal_padding == 0:
        return hidden
    if temporal_padding < 0:
        raise ValueError("MiniMax-H3 Ref2VA VideoVAE causal padding cannot be negative")
    shape = tuple(int(value) for value in hidden.shape)
    first = network.add_slice(
        hidden,
        (0, 0, 0, 0, 0),
        (shape[0], shape[1], 1, shape[3], shape[4]),
        (1, 1, 1, 1, 1),
    )
    if first is None:
        raise RuntimeError("TensorRT rejected MiniMax-H3 Ref2VA VideoVAE causal slice")
    zero = op.constant(network, np.zeros((1, 1, 1, 1, 1), dtype=np.float32))
    zero_frame = network.add_elementwise(
        first.get_output(0), zero, trt.ElementWiseOperation.PROD
    ).get_output(0)
    layer = network.add_concatenation([zero_frame] * temporal_padding + [hidden])
    if layer is None:
        raise RuntimeError("TensorRT rejected MiniMax-H3 Ref2VA VideoVAE causal padding")
    layer.axis = 2
    return layer.get_output(0)


def _conv3d(
    network,
    hidden,
    weights: dict[str, object],
    prefix: str,
    *,
    in_channels: int,
    out_channels: int,
    frames: int,
    height: int,
    width: int,
    kernel_size: int,
    temporal_stride: int = 1,
    spatial_stride: int = 1,
    symmetric_reflect: int = 0,
    bottom_right_reflect: int = 0,
):
    if symmetric_reflect and bottom_right_reflect:
        raise ValueError("MiniMax-H3 Ref2VA VideoVAE convolution has ambiguous padding")
    if symmetric_reflect:
        hidden = _spatial_reflect_pad(
            network,
            hidden,
            channels=in_channels,
            frames=frames,
            height=height,
            width=width,
            top=symmetric_reflect,
            bottom=symmetric_reflect,
            left=symmetric_reflect,
            right=symmetric_reflect,
        )
        height += 2 * symmetric_reflect
        width += 2 * symmetric_reflect
    elif bottom_right_reflect:
        hidden = _spatial_reflect_pad(
            network,
            hidden,
            channels=in_channels,
            frames=frames,
            height=height,
            width=width,
            bottom=bottom_right_reflect,
            right=bottom_right_reflect,
        )
        height += bottom_right_reflect
        width += bottom_right_reflect

    temporal_padding = kernel_size - 1 if kernel_size > 1 else 0
    hidden = _causal_zero_pad(network, hidden, temporal_padding)
    padded_frames = frames + temporal_padding
    kernel_shape = (out_channels, in_channels, kernel_size, kernel_size, kernel_size)
    kernel = _require_weight(weights, f"{prefix}.weight", kernel_shape)
    bias = _require_weight(weights, f"{prefix}.bias", (out_channels,))
    layer = network.add_convolution_nd(
        hidden,
        out_channels,
        (kernel_size, kernel_size, kernel_size),
        kernel,
        bias,
    )
    if layer is None:
        raise RuntimeError(f"TensorRT rejected MiniMax-H3 Ref2VA VideoVAE convolution {prefix}")
    layer.name = prefix
    layer.stride_nd = (temporal_stride, spatial_stride, spatial_stride)
    output_frames = (padded_frames - kernel_size) // temporal_stride + 1
    output_height = (height - kernel_size) // spatial_stride + 1
    output_width = (width - kernel_size) // spatial_stride + 1
    return layer.get_output(0), output_frames, output_height, output_width


def _group_norm_isolated(
    network,
    hidden,
    weights: dict[str, object],
    prefix: str,
    *,
    channels: int,
    frames: int,
    height: int,
    width: int,
):
    to_frames = network.add_shuffle(hidden)
    to_frames.first_transpose = trt.Permutation((0, 2, 1, 3, 4))
    to_frames.reshape_dims = (frames, channels, height, width)
    grouped = network.add_shuffle(to_frames.get_output(0))
    grouped.reshape_dims = (
        frames,
        NORM_GROUPS,
        channels // NORM_GROUPS,
        height,
        width,
    )
    reduction_axes = (1 << 2) | (1 << 3) | (1 << 4)
    mean = network.add_reduce(
        grouped.get_output(0), trt.ReduceOperation.AVG, reduction_axes, True
    ).get_output(0)
    centered = network.add_elementwise(
        grouped.get_output(0), mean, trt.ElementWiseOperation.SUB
    ).get_output(0)
    square = network.add_elementwise(centered, centered, trt.ElementWiseOperation.PROD).get_output(
        0
    )
    variance = network.add_reduce(square, trt.ReduceOperation.AVG, reduction_axes, True).get_output(
        0
    )
    epsilon = op.constant(network, np.full((1, 1, 1, 1, 1), NORM_EPS, dtype=np.float32))
    denominator = network.add_elementwise(
        variance, epsilon, trt.ElementWiseOperation.SUM
    ).get_output(0)
    denominator = network.add_unary(denominator, trt.UnaryOperation.SQRT).get_output(0)
    normalized = network.add_elementwise(
        centered, denominator, trt.ElementWiseOperation.DIV
    ).get_output(0)
    ungrouped = network.add_shuffle(normalized)
    ungrouped.reshape_dims = (frames, channels, height, width)
    gamma = op.weight_constant(
        network,
        _require_weight(weights, f"{prefix}.weight", (channels,)).reshape(1, channels, 1, 1),
    )
    beta = op.weight_constant(
        network,
        _require_weight(weights, f"{prefix}.bias", (channels,)).reshape(1, channels, 1, 1),
    )
    scaled = network.add_elementwise(
        ungrouped.get_output(0), gamma, trt.ElementWiseOperation.PROD
    ).get_output(0)
    affine = network.add_elementwise(scaled, beta, trt.ElementWiseOperation.SUM).get_output(0)
    restore = network.add_shuffle(affine)
    restore.reshape_dims = (1, frames, channels, height, width)
    restore.second_transpose = trt.Permutation((0, 2, 1, 3, 4))
    return restore.get_output(0)


def _resnet3d(
    network,
    hidden,
    weights: dict[str, object],
    prefix: str,
    *,
    in_channels: int,
    out_channels: int,
    frames: int,
    height: int,
    width: int,
):
    residual = hidden
    update = _group_norm_isolated(
        network,
        hidden,
        weights,
        f"{prefix}.norm1",
        channels=in_channels,
        frames=frames,
        height=height,
        width=width,
    )
    update = op.silu(network, update)
    update, out_frames, out_height, out_width = _conv3d(
        network,
        update,
        weights,
        f"{prefix}.conv1",
        in_channels=in_channels,
        out_channels=out_channels,
        frames=frames,
        height=height,
        width=width,
        kernel_size=3,
        symmetric_reflect=1,
    )
    if (out_frames, out_height, out_width) != (frames, height, width):
        raise RuntimeError("MiniMax-H3 Ref2VA VideoVAE resnet conv1 changed geometry")
    update = _group_norm_isolated(
        network,
        update,
        weights,
        f"{prefix}.norm2",
        channels=out_channels,
        frames=frames,
        height=height,
        width=width,
    )
    update = op.silu(network, update)
    update, out_frames, out_height, out_width = _conv3d(
        network,
        update,
        weights,
        f"{prefix}.conv2",
        in_channels=out_channels,
        out_channels=out_channels,
        frames=frames,
        height=height,
        width=width,
        kernel_size=3,
        symmetric_reflect=1,
    )
    if in_channels != out_channels:
        residual, shortcut_frames, shortcut_height, shortcut_width = _conv3d(
            network,
            residual,
            weights,
            f"{prefix}.conv_shortcut",
            in_channels=in_channels,
            out_channels=out_channels,
            frames=frames,
            height=height,
            width=width,
            kernel_size=1,
        )
        if (shortcut_frames, shortcut_height, shortcut_width) != (frames, height, width):
            raise RuntimeError("MiniMax-H3 Ref2VA VideoVAE shortcut changed geometry")
    result = network.add_elementwise(residual, update, trt.ElementWiseOperation.SUM)
    if result is None:
        raise RuntimeError(f"TensorRT rejected MiniMax-H3 Ref2VA VideoVAE residual {prefix}")
    return result.get_output(0), out_frames, out_height, out_width


@op.cleanup_failed_build
def build_ref2va_video_encoder_engine(
    weights: dict[str, object],
    *,
    verbose: bool = False,
    consume_weights: bool = False,
    workspace_bytes: int | None = None,
    weight_streaming: bool = False,
    output_path: str | Path | None = None,
) -> bytes | dict[str, int | str]:
    """Build the exact 17-frame/256-pixel released VideoVAE encoder unit."""

    expected = set(checkpoint_keys())
    missing = sorted(expected - set(weights))
    unexpected = sorted(set(weights) - expected)
    if missing or unexpected:
        raise ValueError(
            "MiniMax-H3 Ref2VA video encoder checkpoint partition mismatch: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    op.configure_builder(config, weight_streaming=weight_streaming)
    op.configure_workspace(
        config,
        workspace_bytes,
        default_bytes=REF2VA_VIDEO_ENCODER_DEFAULT_WORKSPACE_BYTES,
    )
    abi = ref2va_video_encoder_abi()
    pixels = network.add_input(abi.inputs[0].name, trt.float32, abi.inputs[0].opt_shape)
    hidden = pixels
    frames = CLIP_FRAMES
    height = width = VAE_TILE_SIZE
    hidden, frames, height, width = _conv3d(
        network,
        hidden,
        weights,
        "encoder.conv_in",
        in_channels=IN_CHANNELS,
        out_channels=BLOCK_OUT_CHANNELS[0],
        frames=frames,
        height=height,
        width=width,
        kernel_size=3,
        symmetric_reflect=1,
    )

    channels = BLOCK_OUT_CHANNELS[0]
    for block_index, out_channels in enumerate(BLOCK_OUT_CHANNELS):
        for layer_index in range(LAYERS_PER_BLOCK):
            hidden, frames, height, width = _resnet3d(
                network,
                hidden,
                weights,
                f"encoder.down_blocks.{block_index}.resnets.{layer_index}",
                in_channels=channels,
                out_channels=out_channels,
                frames=frames,
                height=height,
                width=width,
            )
            channels = out_channels
        temporal_stride = TEMPORAL_DOWNSAMPLE_FACTORS[block_index]
        spatial_stride = SPATIAL_DOWNSAMPLE_FACTORS[block_index]
        if temporal_stride * spatial_stride > 1:
            hidden, frames, height, width = _conv3d(
                network,
                hidden,
                weights,
                f"encoder.down_blocks.{block_index}.downsamplers.0.conv",
                in_channels=channels,
                out_channels=channels,
                frames=frames,
                height=height,
                width=width,
                kernel_size=3,
                temporal_stride=temporal_stride,
                spatial_stride=spatial_stride,
                bottom_right_reflect=int(spatial_stride == 2),
            )

    latent_side = VAE_TILE_SIZE // VAE_SPATIAL_COMPRESSION
    if (frames, height, width) != (CLIP_LATENT_FRAMES, latent_side, latent_side):
        raise RuntimeError(
            "MiniMax-H3 Ref2VA VideoVAE produced "
            f"{frames}x{height}x{width}, expected {CLIP_LATENT_FRAMES}x{latent_side}x{latent_side}"
        )
    hidden = _group_norm_isolated(
        network,
        hidden,
        weights,
        "encoder.norm_out",
        channels=channels,
        frames=frames,
        height=height,
        width=width,
    )
    hidden = op.silu(network, hidden)
    hidden, frames, height, width = _conv3d(
        network,
        hidden,
        weights,
        "encoder.conv_out",
        in_channels=channels,
        out_channels=MOMENT_CHANNELS,
        frames=frames,
        height=height,
        width=width,
        kernel_size=3,
        symmetric_reflect=1,
    )
    hidden, frames, height, width = _conv3d(
        network,
        hidden,
        weights,
        "quant_conv",
        in_channels=MOMENT_CHANNELS,
        out_channels=MOMENT_CHANNELS,
        frames=frames,
        height=height,
        width=width,
        kernel_size=1,
    )
    hidden.name = abi.outputs[0].name
    network.mark_output(hidden)
    op.validate_native_network(
        network, expected_attentions=0, label="Ref2VA temporal VideoVAE encoder"
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
        raise RuntimeError("TensorRT failed to build MiniMax-H3 Ref2VA video encoder")
    del network, config, builder, logger
    gc.collect()
    return record if record is not None else bytes(plan)
