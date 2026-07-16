# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct TensorRT construction of the SAM3 tracker memory encoder.

The tracker session owns recurrent state and policy.  This module owns the
learned operation that turns a policy-selected mask and the current 72x72
vision feature into the next 64-channel spatial memory.  Both
supported batch sizes are fixed at engine-build time so TensorRT can select a
specialized implementation for each plan.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from tensorrt_model_connect import trt_compat

from .graph_ops import (
    add_bias_sum,
    add_constant,
    add_conv2d,
    add_gelu_erf,
    add_layer_norm_native,
    add_matmul_rhs_constant,
)


trt = trt_compat.get_trt()

_CHECKPOINT_PREFIX = "tracker_model.memory_encoder"
_VISION_CHANNELS = 256
_MEMORY_CHANNELS = 64
_SUPPORTED_MASK_SIZES = (288, 1008)
_MEMORY_MASK_SIZE = 1152
_MEMORY_HEIGHT = 72
_MEMORY_WIDTH = 72
_SPATIAL_TOKENS = _MEMORY_HEIGHT * _MEMORY_WIDTH
_LAYER_NORM_EPSILON = 1e-6
_SIGMOID_SCALE = 20.0
_SIGMOID_BIAS = -10.0


@dataclass(frozen=True)
class TrackerMemoryEncoderOutputs:
    """TensorRT tensors emitted by :func:`add_tracker_memory_encoder`.

    ``memory`` is sequence-major ``[5184, 1, 64]`` for the singleton engine
    and object-major ``[2, 5184, 64]`` for the batch-two engine.  ``position``
    always uses the same layout as ``memory``.
    """

    memory: trt.ITensor
    position: trt.ITensor


def _weight(weights: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    key = f"{_CHECKPOINT_PREFIX}.{name}"
    try:
        value = weights[key]
    except KeyError as error:
        raise KeyError(f"Missing SAM3 tracker memory weight: {key}") from error
    return np.asarray(value)


def _constant_like(
    network: trt.INetworkDefinition,
    reference: trt.ITensor,
    shape: tuple[int, ...],
    values: np.ndarray,
    *,
    dtype: np.dtype,
) -> trt.ITensor:
    constant = add_constant(network, shape, values, dtype=dtype)
    if constant.dtype != reference.dtype:
        constant = network.add_cast(constant, reference.dtype).get_output(0)
    return constant


def _validate_inputs(
    vision_features: trt.ITensor,
    mask_logits: trt.ITensor,
    object_score_logits: trt.ITensor,
    batch_size: int,
) -> None:
    if batch_size not in (1, 2):
        raise ValueError(
            f"SAM3 tracker memory plans support only fixed batch sizes 1 and 2; got {batch_size}"
        )

    vision_shape = tuple(vision_features.shape)
    if len(vision_shape) != 4 or vision_shape[1:] != (
        _VISION_CHANNELS,
        _MEMORY_HEIGHT,
        _MEMORY_WIDTH,
    ):
        raise ValueError(
            "SAM3 tracker vision features must have shape "
            f"[B, {_VISION_CHANNELS}, {_MEMORY_HEIGHT}, {_MEMORY_WIDTH}]; "
            f"got {vision_shape}"
        )
    if vision_shape[0] not in (1, batch_size):
        raise ValueError(
            "SAM3 tracker vision features must either match the memory batch "
            f"or use broadcast batch 1; got {vision_shape[0]} for batch {batch_size}"
        )

    mask_shape = tuple(mask_logits.shape)
    if (
        len(mask_shape) != 4
        or mask_shape[:2] != (batch_size, 1)
        or mask_shape[2] != mask_shape[3]
        or mask_shape[2] not in _SUPPORTED_MASK_SIZES
    ):
        raise ValueError(
            "SAM3 tracker mask logits must have shape [B, 1, S, S] with "
            f"S in {_SUPPORTED_MASK_SIZES}; got {mask_shape}"
        )

    score_shape = tuple(object_score_logits.shape)
    if int(np.prod(score_shape)) != batch_size:
        raise ValueError(
            "SAM3 tracker object score logits must contain one value per object; "
            f"got shape {score_shape} for batch {batch_size}"
        )


def _half_pixel_resize_mask(
    network: trt.INetworkDefinition,
    mask_logits: trt.ITensor,
    batch_size: int,
) -> trt.ITensor:
    resize = network.add_resize(mask_logits)
    resize.resize_mode = trt.InterpolationMode.LINEAR
    resize.coordinate_transformation = trt.ResizeCoordinateTransformation.HALF_PIXEL
    resize.shape = (batch_size, 1, _MEMORY_MASK_SIZE, _MEMORY_MASK_SIZE)
    return resize.get_output(0)


def _prepare_memory_mask(
    network: trt.INetworkDefinition,
    mask_logits: trt.ITensor,
    batch_size: int,
    *,
    hard_mask: bool,
    dtype: np.dtype,
) -> trt.ITensor:
    mask = _half_pixel_resize_mask(network, mask_logits, batch_size)
    if hard_mask:
        zero = _constant_like(
            network,
            mask,
            (1, 1, 1, 1),
            np.zeros((1, 1, 1, 1), dtype=dtype),
            dtype=dtype,
        )
        positive = network.add_elementwise(mask, zero, trt.ElementWiseOperation.GREATER).get_output(
            0
        )
        mask = network.add_cast(positive, mask.dtype).get_output(0)
    else:
        mask = network.add_activation(mask, trt.ActivationType.SIGMOID).get_output(0)

    scale = _constant_like(
        network,
        mask,
        (1, 1, 1, 1),
        np.full((1, 1, 1, 1), _SIGMOID_SCALE, dtype=dtype),
        dtype=dtype,
    )
    bias = _constant_like(
        network,
        mask,
        (1, 1, 1, 1),
        np.full((1, 1, 1, 1), _SIGMOID_BIAS, dtype=dtype),
        dtype=dtype,
    )
    scaled = network.add_elementwise(mask, scale, trt.ElementWiseOperation.PROD).get_output(0)
    return network.add_elementwise(scaled, bias, trt.ElementWiseOperation.SUM).get_output(0)


def _to_channels_last(network: trt.INetworkDefinition, inp: trt.ITensor) -> trt.ITensor:
    shuffle = network.add_shuffle(inp)
    shuffle.first_transpose = (0, 2, 3, 1)
    return shuffle.get_output(0)


def _to_channels_first(network: trt.INetworkDefinition, inp: trt.ITensor) -> trt.ITensor:
    shuffle = network.add_shuffle(inp)
    shuffle.first_transpose = (0, 3, 1, 2)
    return shuffle.get_output(0)


def _layer_norm_channels_first(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weights: Mapping[str, np.ndarray],
    prefix: str,
    channels: int,
    *,
    dtype: np.dtype,
) -> trt.ITensor:
    channels_last = _to_channels_last(network, inp)
    normalized = add_layer_norm_native(
        network,
        channels_last,
        channels,
        _weight(weights, f"{prefix}.weight"),
        _weight(weights, f"{prefix}.bias"),
        _LAYER_NORM_EPSILON,
        dtype=dtype,
    )
    return _to_channels_first(network, normalized)


def _add_mask_downsampler(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weights: Mapping[str, np.ndarray],
    *,
    dtype: np.dtype,
) -> trt.ITensor:
    hidden = inp
    channels = (4, 16, 64, 256)
    for layer_index, out_channels in enumerate(channels):
        prefix = f"mask_downsampler.layers.{layer_index}"
        hidden = add_conv2d(
            network,
            hidden,
            _weight(weights, f"{prefix}.conv.weight"),
            _weight(weights, f"{prefix}.conv.bias"),
            out_channels,
            (3, 3),
            stride=(2, 2),
            padding=(1, 1),
            dtype=dtype,
        )
        hidden = _layer_norm_channels_first(
            network,
            hidden,
            weights,
            f"{prefix}.layer_norm",
            out_channels,
            dtype=dtype,
        )
        hidden = add_gelu_erf(network, hidden, dtype=dtype)

    return add_conv2d(
        network,
        hidden,
        _weight(weights, "mask_downsampler.final_conv.weight"),
        _weight(weights, "mask_downsampler.final_conv.bias"),
        _VISION_CHANNELS,
        (1, 1),
        dtype=dtype,
    )


def _add_convnext_fuser_block(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weights: Mapping[str, np.ndarray],
    layer_index: int,
    *,
    dtype: np.dtype,
) -> trt.ITensor:
    prefix = f"memory_fuser.layers.{layer_index}"
    residual = inp
    hidden = add_conv2d(
        network,
        inp,
        _weight(weights, f"{prefix}.depthwise_conv.weight"),
        _weight(weights, f"{prefix}.depthwise_conv.bias"),
        _VISION_CHANNELS,
        (7, 7),
        padding=(3, 3),
        groups=_VISION_CHANNELS,
        dtype=dtype,
    )
    hidden = _to_channels_last(network, hidden)
    hidden = add_layer_norm_native(
        network,
        hidden,
        _VISION_CHANNELS,
        _weight(weights, f"{prefix}.layer_norm.weight"),
        _weight(weights, f"{prefix}.layer_norm.bias"),
        _LAYER_NORM_EPSILON,
        dtype=dtype,
    )
    hidden = add_matmul_rhs_constant(
        network,
        hidden,
        _VISION_CHANNELS,
        1024,
        _weight(weights, f"{prefix}.pointwise_conv1.weight").T,
        dtype=dtype,
    )
    hidden = add_bias_sum(
        network,
        hidden,
        1024,
        _weight(weights, f"{prefix}.pointwise_conv1.bias"),
        dtype=dtype,
    )
    hidden = add_gelu_erf(network, hidden, dtype=dtype)
    hidden = add_matmul_rhs_constant(
        network,
        hidden,
        1024,
        _VISION_CHANNELS,
        _weight(weights, f"{prefix}.pointwise_conv2.weight").T,
        dtype=dtype,
    )
    hidden = add_bias_sum(
        network,
        hidden,
        _VISION_CHANNELS,
        _weight(weights, f"{prefix}.pointwise_conv2.bias"),
        dtype=dtype,
    )
    scale = _constant_like(
        network,
        hidden,
        (1, 1, 1, _VISION_CHANNELS),
        _weight(weights, f"{prefix}.scale").reshape(1, 1, 1, _VISION_CHANNELS),
        dtype=dtype,
    )
    hidden = network.add_elementwise(hidden, scale, trt.ElementWiseOperation.PROD).get_output(0)
    hidden = _to_channels_first(network, hidden)
    return network.add_elementwise(residual, hidden, trt.ElementWiseOperation.SUM).get_output(0)


def _make_position_encoding(batch_size: int, *, dtype: np.dtype) -> np.ndarray:
    y = np.arange(1, _MEMORY_HEIGHT + 1, dtype=np.float32)
    x = np.arange(1, _MEMORY_WIDTH + 1, dtype=np.float32)
    y = y / np.float32(_MEMORY_HEIGHT + 1e-6) * np.float32(2.0 * np.pi)
    x = x / np.float32(_MEMORY_WIDTH + 1e-6) * np.float32(2.0 * np.pi)
    y_grid = np.broadcast_to(y[:, None], (_MEMORY_HEIGHT, _MEMORY_WIDTH))
    x_grid = np.broadcast_to(x[None, :], (_MEMORY_HEIGHT, _MEMORY_WIDTH))

    num_positional_features = _MEMORY_CHANNELS // 2
    indices = np.arange(num_positional_features, dtype=np.float32)
    exponents = 2.0 * np.floor(indices / 2.0) / num_positional_features
    dimensions = np.power(np.float32(10000.0), exponents).astype(np.float32)
    position_x = x_grid[..., None] / dimensions
    position_y = y_grid[..., None] / dimensions
    position_x = np.stack(
        (np.sin(position_x[..., 0::2]), np.cos(position_x[..., 1::2])), axis=-1
    ).reshape(_MEMORY_HEIGHT, _MEMORY_WIDTH, num_positional_features)
    position_y = np.stack(
        (np.sin(position_y[..., 0::2]), np.cos(position_y[..., 1::2])), axis=-1
    ).reshape(_MEMORY_HEIGHT, _MEMORY_WIDTH, num_positional_features)
    position = np.concatenate((position_y, position_x), axis=-1)
    position = np.broadcast_to(
        position[None, ...],
        (batch_size, _MEMORY_HEIGHT, _MEMORY_WIDTH, _MEMORY_CHANNELS),
    )
    return np.ascontiguousarray(position, dtype=dtype)


def _add_occlusion_embedding(
    network: trt.INetworkDefinition,
    memory: trt.ITensor,
    object_score_logits: trt.ITensor,
    weights: Mapping[str, np.ndarray],
    batch_size: int,
    *,
    dtype: np.dtype,
) -> trt.ITensor:
    scores = network.add_shuffle(object_score_logits)
    scores.reshape_dims = (batch_size, 1, 1, 1)
    zero = _constant_like(
        network,
        scores.get_output(0),
        (1, 1, 1, 1),
        np.zeros((1, 1, 1, 1), dtype=dtype),
        dtype=dtype,
    )
    appearing = network.add_elementwise(
        scores.get_output(0), zero, trt.ElementWiseOperation.GREATER
    ).get_output(0)
    appearing = network.add_cast(appearing, memory.dtype).get_output(0)
    one = _constant_like(
        network,
        appearing,
        (1, 1, 1, 1),
        np.ones((1, 1, 1, 1), dtype=dtype),
        dtype=dtype,
    )
    absent = network.add_elementwise(one, appearing, trt.ElementWiseOperation.SUB).get_output(0)
    embedding_key = "tracker_model.occlusion_spatial_embedding_parameter"
    try:
        embedding_value = np.asarray(weights[embedding_key])
    except KeyError as error:
        raise KeyError(f"Missing SAM3 tracker memory weight: {embedding_key}") from error
    embedding = _constant_like(
        network,
        memory,
        (1, _MEMORY_CHANNELS, 1, 1),
        embedding_value.reshape(1, _MEMORY_CHANNELS, 1, 1),
        dtype=dtype,
    )
    occlusion = network.add_elementwise(
        absent, embedding, trt.ElementWiseOperation.PROD
    ).get_output(0)
    return network.add_elementwise(memory, occlusion, trt.ElementWiseOperation.SUM).get_output(0)


def _format_outputs(
    network: trt.INetworkDefinition,
    memory: trt.ITensor,
    batch_size: int,
    *,
    dtype: np.dtype,
) -> TrackerMemoryEncoderOutputs:
    memory = _to_channels_last(network, memory)
    flattened = network.add_shuffle(memory)
    flattened.reshape_dims = (batch_size, _SPATIAL_TOKENS, _MEMORY_CHANNELS)

    position_values = _make_position_encoding(batch_size, dtype=dtype).reshape(
        batch_size, _SPATIAL_TOKENS, _MEMORY_CHANNELS
    )
    if batch_size == 1:
        sequence_major = network.add_shuffle(flattened.get_output(0))
        sequence_major.first_transpose = (1, 0, 2)
        memory_output = sequence_major.get_output(0)
        position_values = np.ascontiguousarray(position_values.transpose(1, 0, 2))
        position_shape = (_SPATIAL_TOKENS, 1, _MEMORY_CHANNELS)
    else:
        memory_output = flattened.get_output(0)
        position_shape = (batch_size, _SPATIAL_TOKENS, _MEMORY_CHANNELS)

    position_output = _constant_like(
        network,
        memory_output,
        position_shape,
        position_values,
        dtype=dtype,
    )
    return TrackerMemoryEncoderOutputs(memory_output, position_output)


def add_tracker_memory_encoder(
    network: trt.INetworkDefinition,
    vision_features: trt.ITensor,
    mask_logits: trt.ITensor,
    object_score_logits: trt.ITensor,
    weights: Mapping[str, np.ndarray],
    *,
    batch_size: int,
    hard_mask: bool,
    dtype: np.dtype = np.float32,
) -> TrackerMemoryEncoderOutputs:
    """Reconstruct the official SAM3 tracker memory encoder with TensorRT.

    Args:
        network: Strongly typed TensorRT network receiving the new layers.
        vision_features: Current frame feature map ``[1|B, 256, 72, 72]``.
        mask_logits: Policy-selected recurrent logits ``[B, 1, 288, 288]``
            or initialization logits ``[B, 1, 1008, 1008]``.
        object_score_logits: One object-presence logit per batch item.
        weights: Raw NumPy checkpoint tensors using their full checkpoint keys.
        batch_size: Fixed plan batch, either one or two.
        hard_mask: Use the point-prompt binary-mask memory rule when true;
            otherwise use the recurrent sigmoid-mask memory rule.
        dtype: TensorRT weight and constant storage dtype.
    """

    _validate_inputs(vision_features, mask_logits, object_score_logits, batch_size)
    dtype = np.dtype(dtype)
    memory_mask = _prepare_memory_mask(
        network,
        mask_logits,
        batch_size,
        hard_mask=hard_mask,
        dtype=dtype,
    )
    memory_mask = _add_mask_downsampler(network, memory_mask, weights, dtype=dtype)

    projected_features = add_conv2d(
        network,
        vision_features,
        _weight(weights, "feature_projection.weight"),
        _weight(weights, "feature_projection.bias"),
        _VISION_CHANNELS,
        (1, 1),
        dtype=dtype,
    )
    hidden = network.add_elementwise(
        projected_features, memory_mask, trt.ElementWiseOperation.SUM
    ).get_output(0)
    for layer_index in range(2):
        hidden = _add_convnext_fuser_block(network, hidden, weights, layer_index, dtype=dtype)

    memory = add_conv2d(
        network,
        hidden,
        _weight(weights, "projection.weight"),
        _weight(weights, "projection.bias"),
        _MEMORY_CHANNELS,
        (1, 1),
        dtype=dtype,
    )
    memory = _add_occlusion_embedding(
        network,
        memory,
        object_score_logits,
        weights,
        batch_size,
        dtype=dtype,
    )
    return _format_outputs(network, memory, batch_size, dtype=dtype)


__all__ = ["TrackerMemoryEncoderOutputs", "add_tracker_memory_encoder"]
