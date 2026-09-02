# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native TensorRT encoder for MiniMax-H3 FL2VA keyframe tiles.

The released VAE always encodes keyframes as one temporal frame.  Causal 3-D
convolution therefore has one non-zero temporal tap: the last tap of every
3x3x3 kernel (or the only tap of a 1x1x1 kernel).  Expressing that exact case
as 2-D convolutions avoids manufacturing zero temporal frames while retaining
the released spatial reflection padding, isolated group norm, and FP32 math.

The plan deliberately emits posterior parameters.  Posterior sampling,
FP16-rounding, per-channel latent normalization, tile stitching, and the
request-generator draw order belong to the native runtime rather than the
engine and are stated in :mod:`fl2va_contract`.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np

from tensorrt_model_connect import trt_compat

from . import graph_ops as op
from .fl2va_contract import (
    VAE_SPATIAL_COMPRESSION,
    VAE_TILE_MAX_BATCH,
    VAE_TILE_SIZE,
    keyframe_vae_encoder_abi,
)


trt = trt_compat.get_trt()

IN_CHANNELS = 3
MOMENT_CHANNELS = 48
BLOCK_OUT_CHANNELS = (128, 256, 256, 512, 512, 1024)
SPATIAL_DOWNSAMPLE_FACTORS = (2, 2, 2, 2, 1, 1)
TEMPORAL_DOWNSAMPLE_FACTORS = (1, 2, 2, 1, 1, 1)
LAYERS_PER_BLOCK = 2
NORM_GROUPS = 32
NORM_EPS = 1.0e-6
KEYFRAME_VAE_ENCODER_DEFAULT_WORKSPACE_BYTES = 32 << 30


def checkpoint_keys() -> tuple[str, ...]:
    """Return the exhaustive VAE checkpoint partition consumed by this plan."""

    names = ["encoder.conv_in.weight", "encoder.conv_in.bias"]
    block_in_channels = (BLOCK_OUT_CHANNELS[0], *BLOCK_OUT_CHANNELS[:-1])
    for block_index, (in_channels, out_channels) in enumerate(
        zip(block_in_channels, BLOCK_OUT_CHANNELS)
    ):
        for layer_index in range(LAYERS_PER_BLOCK):
            prefix = f"encoder.down_blocks.{block_index}.resnets.{layer_index}"
            names.extend(
                [
                    f"{prefix}.norm1.weight",
                    f"{prefix}.norm1.bias",
                    f"{prefix}.conv1.weight",
                    f"{prefix}.conv1.bias",
                    f"{prefix}.norm2.weight",
                    f"{prefix}.norm2.bias",
                    f"{prefix}.conv2.weight",
                    f"{prefix}.conv2.bias",
                ]
            )
            if layer_index == 0 and in_channels != out_channels:
                names.extend([f"{prefix}.conv_shortcut.weight", f"{prefix}.conv_shortcut.bias"])
        if SPATIAL_DOWNSAMPLE_FACTORS[block_index] * TEMPORAL_DOWNSAMPLE_FACTORS[block_index] > 1:
            prefix = f"encoder.down_blocks.{block_index}.downsamplers.0.conv"
            names.extend([f"{prefix}.weight", f"{prefix}.bias"])
    names.extend(
        [
            "encoder.norm_out.weight",
            "encoder.norm_out.bias",
            "encoder.conv_out.weight",
            "encoder.conv_out.bias",
            "quant_conv.weight",
            "quant_conv.bias",
        ]
    )
    return tuple(names)


def _shape_dim(network, tensor, axis: int):
    shape = network.add_shape(tensor).get_output(0)
    return network.add_slice(shape, (axis,), (1,), (1,)).get_output(0)


def _shape_vector(network, values):
    tensors = []
    for value in values:
        if isinstance(value, int):
            tensors.append(
                op.constant(network, np.asarray([value], dtype=np.int64), dtype=np.int64)
            )
        else:
            tensors.append(value)
    if len(tensors) == 1:
        return tensors[0]
    layer = network.add_concatenation(tensors)
    layer.axis = 0
    return layer.get_output(0)


def _reflect_pad_2d(
    network,
    hidden,
    *,
    channels: int,
    height: int,
    width: int,
    top: int = 0,
    bottom: int = 0,
    left: int = 0,
    right: int = 0,
):
    """PyTorch-compatible reflection padding for static CHW and dynamic batch."""

    if min(top, bottom, left, right) < 0:
        raise ValueError("MiniMax-H3 reflect padding cannot be negative")
    if top >= height or bottom >= height or left >= width or right >= width:
        raise ValueError("MiniMax-H3 reflect padding must be smaller than its input axis")
    if top == bottom == left == right == 0:
        return hidden
    output_height = height + top + bottom
    output_width = width + left + right
    layer = network.add_slice(
        hidden,
        (0, 0, -top, -left),
        (1, channels, output_height, output_width),
        (1, 1, 1, 1),
    )
    if layer is None:
        raise RuntimeError("TensorRT rejected MiniMax-H3 VAE reflection padding")
    layer.set_input(
        2,
        _shape_vector(
            network,
            (_shape_dim(network, hidden, 0), channels, output_height, output_width),
        ),
    )
    layer.mode = trt.SampleMode.REFLECT
    return layer.get_output(0)


def _spatial_kernel(weight: np.ndarray) -> np.ndarray:
    """Select the only non-zero causal temporal tap for a one-frame clip."""

    value = np.asarray(weight)
    if value.ndim != 5 or value.shape[2] not in (1, 3):
        raise ValueError(f"MiniMax-H3 VAE convolution weight has invalid shape {value.shape}")
    return np.ascontiguousarray(value[:, :, -1])


def _conv(
    network,
    hidden,
    weights: dict[str, np.ndarray],
    prefix: str,
    *,
    in_channels: int,
    out_channels: int,
    height: int,
    width: int,
    stride: int = 1,
    symmetric_reflect: int = 0,
    bottom_right_reflect: int = 0,
):
    if symmetric_reflect and bottom_right_reflect:
        raise ValueError("MiniMax-H3 VAE convolution has ambiguous padding")
    if symmetric_reflect:
        hidden = _reflect_pad_2d(
            network,
            hidden,
            channels=in_channels,
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
        hidden = _reflect_pad_2d(
            network,
            hidden,
            channels=in_channels,
            height=height,
            width=width,
            bottom=bottom_right_reflect,
            right=bottom_right_reflect,
        )
        height += bottom_right_reflect
        width += bottom_right_reflect

    kernel = _spatial_kernel(weights[f"{prefix}.weight"])
    layer = network.add_convolution_nd(
        hidden,
        out_channels,
        tuple(kernel.shape[-2:]),
        kernel,
        np.ascontiguousarray(weights[f"{prefix}.bias"]),
    )
    if layer is None:
        raise RuntimeError(f"TensorRT rejected MiniMax-H3 VAE convolution {prefix}")
    layer.name = prefix
    layer.stride_nd = (stride, stride)
    output_height = (height - kernel.shape[-2]) // stride + 1
    output_width = (width - kernel.shape[-1]) // stride + 1
    return layer.get_output(0), output_height, output_width


def _group_norm(network, hidden, weights, prefix: str, channels: int):
    rank = len(tuple(hidden.shape))
    gamma = op.weight_constant(
        network, np.asarray(weights[f"{prefix}.weight"]).reshape(1, channels, 1, 1)
    )
    beta = op.weight_constant(
        network, np.asarray(weights[f"{prefix}.bias"]).reshape(1, channels, 1, 1)
    )
    gamma = op.cast(network, gamma, hidden.dtype)
    beta = op.cast(network, beta, hidden.dtype)
    axes = sum(1 << axis for axis in range(1, rank))
    layer = network.add_normalization_v2(hidden, gamma, beta, axes)
    if layer is None:
        raise RuntimeError(f"TensorRT rejected MiniMax-H3 VAE group norm {prefix}")
    layer.name = prefix
    layer.epsilon = NORM_EPS
    layer.num_groups = NORM_GROUPS
    return layer.get_output(0)


def _resnet(
    network,
    hidden,
    weights,
    prefix: str,
    *,
    in_channels: int,
    out_channels: int,
    height: int,
    width: int,
):
    residual = hidden
    update = _group_norm(network, hidden, weights, f"{prefix}.norm1", in_channels)
    update = op.silu(network, update)
    update, update_height, update_width = _conv(
        network,
        update,
        weights,
        f"{prefix}.conv1",
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width,
        symmetric_reflect=1,
    )
    if (update_height, update_width) != (height, width):
        raise RuntimeError("MiniMax-H3 VAE resnet conv1 changed spatial geometry")
    update = _group_norm(network, update, weights, f"{prefix}.norm2", out_channels)
    update = op.silu(network, update)
    update, update_height, update_width = _conv(
        network,
        update,
        weights,
        f"{prefix}.conv2",
        in_channels=out_channels,
        out_channels=out_channels,
        height=height,
        width=width,
        symmetric_reflect=1,
    )
    if in_channels != out_channels:
        residual, shortcut_height, shortcut_width = _conv(
            network,
            residual,
            weights,
            f"{prefix}.conv_shortcut",
            in_channels=in_channels,
            out_channels=out_channels,
            height=height,
            width=width,
        )
        if (shortcut_height, shortcut_width) != (height, width):
            raise RuntimeError("MiniMax-H3 VAE shortcut changed spatial geometry")
    result = network.add_elementwise(residual, update, trt.ElementWiseOperation.SUM)
    if result is None:
        raise RuntimeError(f"TensorRT rejected MiniMax-H3 VAE residual {prefix}")
    return result.get_output(0), update_height, update_width


@op.cleanup_failed_build
def build_keyframe_vae_encoder_engine(
    weights: dict[str, np.ndarray],
    *,
    verbose: bool = False,
    consume_weights: bool = False,
    workspace_bytes: int | None = None,
    weight_streaming: bool = False,
    output_path: str | Path | None = None,
) -> bytes | dict[str, int | str]:
    """Build the dynamic-tile FL2VA VAE encoder plan."""

    missing = sorted(set(checkpoint_keys()) - set(weights))
    unexpected = sorted(set(weights) - set(checkpoint_keys()))
    if missing or unexpected:
        raise ValueError(
            "MiniMax-H3 FL2VA VAE encoder checkpoint partition mismatch: "
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
        default_bytes=KEYFRAME_VAE_ENCODER_DEFAULT_WORKSPACE_BYTES,
    )

    abi = keyframe_vae_encoder_abi()
    binding = abi.inputs[0]
    pixels = network.add_input(
        binding.name,
        trt.float32,
        (-1, IN_CHANNELS, 1, VAE_TILE_SIZE, VAE_TILE_SIZE),
    )
    profile = builder.create_optimization_profile()
    profile.set_shape(binding.name, binding.min_shape, binding.opt_shape, binding.max_shape)
    config.add_optimization_profile(profile)

    # T=1 is removed only after the ABI boundary.  Every causal 3-D
    # convolution below selects its last temporal kernel tap.
    squeeze = network.add_shuffle(pixels)
    squeeze.reshape_dims = (-1, IN_CHANNELS, VAE_TILE_SIZE, VAE_TILE_SIZE)
    hidden = squeeze.get_output(0)
    height = width = VAE_TILE_SIZE
    hidden, height, width = _conv(
        network,
        hidden,
        weights,
        "encoder.conv_in",
        in_channels=IN_CHANNELS,
        out_channels=BLOCK_OUT_CHANNELS[0],
        height=height,
        width=width,
        symmetric_reflect=1,
    )

    current_channels = BLOCK_OUT_CHANNELS[0]
    for block_index, out_channels in enumerate(BLOCK_OUT_CHANNELS):
        for layer_index in range(LAYERS_PER_BLOCK):
            prefix = f"encoder.down_blocks.{block_index}.resnets.{layer_index}"
            hidden, height, width = _resnet(
                network,
                hidden,
                weights,
                prefix,
                in_channels=current_channels,
                out_channels=out_channels,
                height=height,
                width=width,
            )
            current_channels = out_channels
        if SPATIAL_DOWNSAMPLE_FACTORS[block_index] * TEMPORAL_DOWNSAMPLE_FACTORS[block_index] > 1:
            prefix = f"encoder.down_blocks.{block_index}.downsamplers.0.conv"
            spatial_stride = SPATIAL_DOWNSAMPLE_FACTORS[block_index]
            hidden, height, width = _conv(
                network,
                hidden,
                weights,
                prefix,
                in_channels=current_channels,
                out_channels=current_channels,
                height=height,
                width=width,
                stride=spatial_stride,
                bottom_right_reflect=int(spatial_stride == 2),
            )

    expected_side = VAE_TILE_SIZE // VAE_SPATIAL_COMPRESSION
    if (height, width) != (expected_side, expected_side):
        raise RuntimeError(
            f"MiniMax-H3 VAE encoder produced {height}x{width}, expected {expected_side}x{expected_side}"
        )
    hidden = _group_norm(network, hidden, weights, "encoder.norm_out", current_channels)
    hidden = op.silu(network, hidden)
    hidden, height, width = _conv(
        network,
        hidden,
        weights,
        "encoder.conv_out",
        in_channels=current_channels,
        out_channels=MOMENT_CHANNELS,
        height=height,
        width=width,
        symmetric_reflect=1,
    )
    hidden, height, width = _conv(
        network,
        hidden,
        weights,
        "quant_conv",
        in_channels=MOMENT_CHANNELS,
        out_channels=MOMENT_CHANNELS,
        height=height,
        width=width,
    )
    expand = network.add_shuffle(hidden)
    expand.reshape_dims = (-1, MOMENT_CHANNELS, 1, expected_side, expected_side)
    output = expand.get_output(0)
    output.name = abi.outputs[0].name
    network.mark_output(output)

    op.validate_native_network(network, expected_attentions=0, label="FL2VA keyframe VAE encoder")
    print(
        f"[minimax-h3] building native FL2VA VAE encoder: tiles=1..{VAE_TILE_MAX_BATCH}, "
        f"tile={VAE_TILE_SIZE}x{VAE_TILE_SIZE}, moments={MOMENT_CHANNELS}",
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
        raise RuntimeError("TensorRT failed to build MiniMax-H3 FL2VA VAE encoder")
    del network, config, builder
    gc.collect()
    return record if record is not None else bytes(plan)
