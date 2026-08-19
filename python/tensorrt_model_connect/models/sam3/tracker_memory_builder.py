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
    add_constant,
    add_gelu_erf,
)


trt = trt_compat.get_trt()

_CHECKPOINT_PREFIX = "tracker_model.memory_encoder"
_VISION_CHANNELS = 256
_MEMORY_CHANNELS = 64
_SUPPORTED_MASK_SIZES = (288, 1008)
_LOW_RES_MASK_SIZE = 288
_TRACKER_IMAGE_SIZE = 1008
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


def _cast(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    dtype: trt.DataType,
) -> trt.ITensor:
    if tensor.dtype == dtype:
        return tensor
    return network.add_cast(tensor, dtype).get_output(0)


def _bf16_constant(
    network: trt.INetworkDefinition,
    shape: tuple[int, ...],
    values: np.ndarray,
) -> trt.ITensor:
    """Create a BF16 graph constant without relying on NumPy BF16 support."""

    constant = add_constant(
        network,
        shape,
        np.asarray(values, dtype=np.float32).reshape(shape),
        dtype=np.float32,
    )
    return _cast(network, constant, trt.bfloat16)


def _add_bf16_conv2d(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weight: np.ndarray,
    bias: np.ndarray | None,
    out_channels: int,
    kernel_size: tuple[int, int],
    *,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    groups: int = 1,
) -> trt.ITensor:
    """Run Meta's autocast convolution with BF16 activations and weights."""

    bf16_input = _cast(network, inp, trt.bfloat16)
    convolution = network.add_convolution_nd(
        bf16_input,
        num_output_maps=out_channels,
        kernel_shape=kernel_size,
        kernel=trt.Weights(),
        bias=trt.Weights(),
    )
    kernel_shape = tuple(np.asarray(weight).shape)
    convolution.set_input(1, _bf16_constant(network, kernel_shape, weight))
    if bias is not None:
        convolution.set_input(
            2,
            _bf16_constant(network, (out_channels,), bias),
        )
    convolution.stride_nd = stride
    convolution.padding_nd = padding
    convolution.num_groups = groups
    return convolution.get_output(0)


def _add_bf16_linear(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weight: np.ndarray,
) -> trt.ITensor:
    """Run an Inductor-style BF16 pointwise matmul without its bias."""

    bf16_input = _cast(network, inp, trt.bfloat16)
    rank = len(tuple(bf16_input.shape))
    out_features, in_features = np.asarray(weight).shape
    rhs_shape = (1,) * (rank - 2) + (in_features, out_features)
    rhs = _bf16_constant(network, rhs_shape, np.asarray(weight).T)
    product = network.add_matrix_multiply(
        bf16_input,
        trt.MatrixOperation.NONE,
        rhs,
        trt.MatrixOperation.NONE,
    ).get_output(0)
    return product


def _add_fp32_bias_gelu_to_bf16(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    bias: np.ndarray,
) -> trt.ITensor:
    """Fuse a Linear bias and GELU in FP32, then publish BF16."""

    fp32_input = _cast(network, inp, trt.float32)
    rank = len(tuple(fp32_input.shape))
    bias_shape = (1,) * (rank - 1) + (len(bias),)
    bias_tensor = add_constant(
        network,
        bias_shape,
        np.asarray(bias, dtype=np.float32).reshape(bias_shape),
        dtype=np.float32,
    )
    biased = network.add_elementwise(
        fp32_input,
        bias_tensor,
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    fp32_output = add_gelu_erf(network, biased, dtype=np.float32)
    return _cast(network, fp32_output, trt.bfloat16)


def _add_fp32_bias_scale_residual_to_bf16(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    bias: np.ndarray,
    scale: np.ndarray,
    residual: trt.ITensor,
) -> trt.ITensor:
    """Fuse the ConvNeXt output bias, layer scale, and residual in FP32."""

    channels = len(bias)
    value = _cast(network, inp, trt.float32)
    bias_tensor = add_constant(
        network,
        (1, 1, 1, channels),
        np.asarray(bias, dtype=np.float32).reshape(1, 1, 1, channels),
        dtype=np.float32,
    )
    scale_tensor = add_constant(
        network,
        (1, 1, 1, channels),
        np.asarray(scale, dtype=np.float32).reshape(1, 1, 1, channels),
        dtype=np.float32,
    )
    value = network.add_elementwise(
        value,
        bias_tensor,
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    value = network.add_elementwise(
        value,
        scale_tensor,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    value = _to_channels_first(network, value)
    value = network.add_elementwise(
        value,
        _cast(network, residual, trt.float32),
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    return _cast(network, value, trt.bfloat16)


def _add_inductor_layer_norm_2d(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    convolution_bias: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    channels: int,
    *,
    round_mean_bf16: bool = False,
) -> trt.ITensor:
    """Fuse an FP32 convolution bias into Inductor's ``LayerNorm2d`` epilogue.

    The compiled graph executes a bias-free BF16 convolution, promotes its
    output and the original bias to FP32, and keeps the normalization plus
    affine transform in FP32.  The four-channel downsampler is the one special
    case: its normalization numerator reloads a BF16-published mean while its
    variance uses the original FP32 mean.
    """

    value = _cast(network, inp, trt.float32)
    convolution_bias_tensor = add_constant(
        network,
        (1, channels, 1, 1),
        np.asarray(convolution_bias, dtype=np.float32).reshape(1, channels, 1, 1),
        dtype=np.float32,
    )
    value = network.add_elementwise(
        value,
        convolution_bias_tensor,
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    mean = network.add_reduce(
        value,
        trt.ReduceOperation.AVG,
        1 << 1,
        keep_dims=True,
    ).get_output(0)
    centered_for_variance = network.add_elementwise(
        value,
        mean,
        trt.ElementWiseOperation.SUB,
    ).get_output(0)
    squared = network.add_elementwise(
        centered_for_variance,
        centered_for_variance,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    variance = network.add_reduce(
        squared,
        trt.ReduceOperation.AVG,
        1 << 1,
        keep_dims=True,
    ).get_output(0)
    if round_mean_bf16:
        mean = _cast(network, _cast(network, mean, trt.bfloat16), trt.float32)
    centered = network.add_elementwise(
        value,
        mean,
        trt.ElementWiseOperation.SUB,
    ).get_output(0)
    epsilon = add_constant(
        network,
        (1, 1, 1, 1),
        np.asarray([_LAYER_NORM_EPSILON], dtype=np.float32),
        dtype=np.float32,
    )
    variance_with_epsilon = network.add_elementwise(
        variance,
        epsilon,
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    denominator = network.add_unary(
        variance_with_epsilon,
        trt.UnaryOperation.SQRT,
    ).get_output(0)
    normalized = network.add_elementwise(
        centered,
        denominator,
        trt.ElementWiseOperation.DIV,
    ).get_output(0)
    gamma_tensor = add_constant(
        network,
        (1, channels, 1, 1),
        np.asarray(gamma, dtype=np.float32).reshape(1, channels, 1, 1),
        dtype=np.float32,
    )
    beta_tensor = add_constant(
        network,
        (1, channels, 1, 1),
        np.asarray(beta, dtype=np.float32).reshape(1, channels, 1, 1),
        dtype=np.float32,
    )
    scaled = network.add_elementwise(
        normalized,
        gamma_tensor,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    return network.add_elementwise(
        scaled,
        beta_tensor,
        trt.ElementWiseOperation.SUM,
    ).get_output(0)


def _validate_inputs(
    vision_features: trt.ITensor,
    mask_logits: trt.ITensor,
    object_score_logits: trt.ITensor,
    suppress_area_shrinkage: trt.ITensor | None,
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

    if suppress_area_shrinkage is not None:
        suppression_shape = tuple(suppress_area_shrinkage.shape)
        if int(np.prod(suppression_shape)) != batch_size:
            raise ValueError(
                "SAM3 tracker area-shrinkage flags must contain one value per object; "
                f"got shape {suppression_shape} for batch {batch_size}"
            )


def _half_pixel_resize_mask(
    network: trt.INetworkDefinition,
    mask_logits: trt.ITensor,
    batch_size: int,
    target_size: int,
) -> trt.ITensor:
    resize = network.add_resize(mask_logits)
    resize.resize_mode = trt.InterpolationMode.LINEAR
    resize.coordinate_transformation = trt.ResizeCoordinateTransformation.HALF_PIXEL
    resize.shape = (batch_size, 1, target_size, target_size)
    return resize.get_output(0)


def _apply_hard_mask_non_overlap(
    network: trt.INetworkDefinition,
    mask_logits: trt.ITensor,
    batch_size: int,
) -> trt.ITensor:
    """Keep Meta's highest-scoring object at each tracker-image pixel.

    Meta's predictor consolidates newly added objects before the memory encoder
    and applies its non-overlap constraint at the 1008px tracker-image grid.
    B1 is therefore an identity; B2 selects the winning row at each pixel.
    """

    if batch_size == 1:
        return mask_logits
    if batch_size != 2:
        raise ValueError(f"SAM3 hard-mask non-overlap supports only B1/B2; got {batch_size}")

    winners = network.add_topk(
        mask_logits,
        trt.TopKOperation.MAX,
        1,
        1 << 0,
    ).get_output(1)
    object_indices = _constant_like(
        network,
        winners,
        (batch_size, 1, 1, 1),
        np.arange(batch_size, dtype=np.int32).reshape(batch_size, 1, 1, 1),
        dtype=np.int32,
    )
    keep = network.add_elementwise(
        object_indices,
        winners,
        trt.ElementWiseOperation.EQUAL,
    ).get_output(0)
    losing_value = _constant_like(
        network,
        mask_logits,
        (1, 1, 1, 1),
        np.full((1, 1, 1, 1), -10.0, dtype=np.float32),
        dtype=np.float32,
    )
    return network.add_select(keep, mask_logits, losing_value).get_output(0)


def _apply_soft_area_shrinkage(
    network: trt.INetworkDefinition,
    high_res_mask_logits: trt.ITensor,
    suppress_area_shrinkage: trt.ITensor,
    batch_size: int,
) -> trt.ITensor:
    """Clamp rejected objects only after Meta's 1152-grid area decision."""

    suppression = network.add_shuffle(suppress_area_shrinkage)
    suppression.reshape_dims = (batch_size, 1, 1, 1)
    suppression = suppression.get_output(0)
    zero = _constant_like(
        network,
        suppression,
        (1, 1, 1, 1),
        np.zeros((1, 1, 1, 1), dtype=np.int32),
        dtype=np.int32,
    )
    reject = network.add_elementwise(
        suppression,
        zero,
        trt.ElementWiseOperation.GREATER,
    ).get_output(0)
    background = _constant_like(
        network,
        high_res_mask_logits,
        (1, 1, 1, 1),
        np.full((1, 1, 1, 1), -10.0, dtype=np.float32),
        dtype=np.float32,
    )
    clamped = network.add_elementwise(
        high_res_mask_logits,
        background,
        trt.ElementWiseOperation.MIN,
    ).get_output(0)
    return network.add_select(reject, clamped, high_res_mask_logits).get_output(0)


def _prepare_memory_mask(
    network: trt.INetworkDefinition,
    mask_logits: trt.ITensor,
    suppress_area_shrinkage: trt.ITensor | None,
    batch_size: int,
    *,
    hard_mask: bool,
    dtype: np.dtype,
) -> trt.ITensor:
    if hard_mask:
        if suppress_area_shrinkage is not None:
            raise ValueError("SAM3 hard memory must not receive recurrent area-shrinkage flags")
        # Exact Meta conditioning order:
        #   carved logits at 288 -> ordinary half-pixel linear at 1008
        #   -> consolidated B2 non-overlap at 1008 -> threshold > 0
        #   -> +/-10 logits
        #   -> antialiased half-pixel linear at 1152.
        # The final resize is an upsample, so PyTorch's antialias support is 1
        # and ordinary TensorRT half-pixel linear is mathematically equivalent.
        if tuple(mask_logits.shape)[-2:] == (_LOW_RES_MASK_SIZE, _LOW_RES_MASK_SIZE):
            tracker_image_logits = _half_pixel_resize_mask(
                network,
                mask_logits,
                batch_size,
                _TRACKER_IMAGE_SIZE,
            )
        else:
            tracker_image_logits = mask_logits
        tracker_image_logits = _apply_hard_mask_non_overlap(
            network,
            tracker_image_logits,
            batch_size,
        )
        zero = _constant_like(
            network,
            tracker_image_logits,
            (1, 1, 1, 1),
            np.zeros((1, 1, 1, 1), dtype=dtype),
            dtype=dtype,
        )
        positive = network.add_elementwise(
            tracker_image_logits,
            zero,
            trt.ElementWiseOperation.GREATER,
        ).get_output(0)
        mask = network.add_cast(positive, tracker_image_logits.dtype).get_output(0)
    else:
        if suppress_area_shrinkage is None:
            raise ValueError("SAM3 soft memory requires recurrent area-shrinkage flags")
        # The dense-video update bypasses the tracker's 1008px image grid. It
        # resizes recurrent 288x288 logits directly to SimpleMaskEncoder's
        # 1152x1152 interpolation size. Meta decides whether to suppress an
        # object on that high-resolution global batch, then clamps the original
        # (possibly overlapping) high-resolution logits before sigmoid.
        # This is distinct from the new-object hard-mask path above.
        if tuple(mask_logits.shape)[-2:] == (_LOW_RES_MASK_SIZE, _LOW_RES_MASK_SIZE):
            mask = _half_pixel_resize_mask(
                network,
                mask_logits,
                batch_size,
                _MEMORY_MASK_SIZE,
            )
        else:
            mask = mask_logits
        mask = _apply_soft_area_shrinkage(
            network,
            mask,
            suppress_area_shrinkage,
            batch_size,
        )
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
    prepared = network.add_elementwise(scaled, bias, trt.ElementWiseOperation.SUM).get_output(0)
    if not hard_mask:
        return prepared
    return _half_pixel_resize_mask(
        network,
        prepared,
        batch_size,
        _MEMORY_MASK_SIZE,
    )


def _to_channels_last(network: trt.INetworkDefinition, inp: trt.ITensor) -> trt.ITensor:
    shuffle = network.add_shuffle(inp)
    shuffle.first_transpose = (0, 2, 3, 1)
    return shuffle.get_output(0)


def _to_channels_first(network: trt.INetworkDefinition, inp: trt.ITensor) -> trt.ITensor:
    shuffle = network.add_shuffle(inp)
    shuffle.first_transpose = (0, 3, 1, 2)
    return shuffle.get_output(0)


def _add_mask_downsampler(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weights: Mapping[str, np.ndarray],
) -> trt.ITensor:
    hidden = inp
    channels = (4, 16, 64, 256)
    for layer_index, out_channels in enumerate(channels):
        prefix = f"mask_downsampler.layers.{layer_index}"
        hidden = _add_bf16_conv2d(
            network,
            hidden,
            _weight(weights, f"{prefix}.conv.weight"),
            None,
            out_channels,
            (3, 3),
            stride=(2, 2),
            padding=(1, 1),
        )
        hidden = _add_inductor_layer_norm_2d(
            network,
            hidden,
            _weight(weights, f"{prefix}.conv.bias"),
            _weight(weights, f"{prefix}.layer_norm.weight"),
            _weight(weights, f"{prefix}.layer_norm.bias"),
            out_channels,
            round_mean_bf16=layer_index == 0,
        )
        # Inductor fuses FP32 normalization and GELU, then materializes BF16 at
        # the input boundary of the following convolution.
        hidden = add_gelu_erf(network, hidden, dtype=np.float32)
        hidden = _cast(network, hidden, trt.bfloat16)

    return _add_bf16_conv2d(
        network,
        hidden,
        _weight(weights, "mask_downsampler.final_conv.weight"),
        _weight(weights, "mask_downsampler.final_conv.bias"),
        _VISION_CHANNELS,
        (1, 1),
    )


def _add_convnext_fuser_block(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weights: Mapping[str, np.ndarray],
    layer_index: int,
) -> trt.ITensor:
    prefix = f"memory_fuser.layers.{layer_index}"
    residual = inp
    hidden = _add_bf16_conv2d(
        network,
        inp,
        _weight(weights, f"{prefix}.depthwise_conv.weight"),
        None,
        _VISION_CHANNELS,
        (7, 7),
        padding=(3, 3),
        groups=_VISION_CHANNELS,
    )
    hidden = _add_inductor_layer_norm_2d(
        network,
        hidden,
        _weight(weights, f"{prefix}.depthwise_conv.bias"),
        _weight(weights, f"{prefix}.layer_norm.weight"),
        _weight(weights, f"{prefix}.layer_norm.bias"),
        _VISION_CHANNELS,
    )
    hidden = _cast(network, hidden, trt.bfloat16)
    hidden = _to_channels_last(network, hidden)
    hidden = _add_bf16_linear(
        network,
        hidden,
        _weight(weights, f"{prefix}.pointwise_conv1.weight"),
    )
    hidden = _add_fp32_bias_gelu_to_bf16(
        network,
        hidden,
        _weight(weights, f"{prefix}.pointwise_conv1.bias"),
    )
    hidden = _add_bf16_linear(
        network,
        hidden,
        _weight(weights, f"{prefix}.pointwise_conv2.weight"),
    )
    return _add_fp32_bias_scale_residual_to_bf16(
        network,
        hidden,
        _weight(weights, f"{prefix}.pointwise_conv2.bias"),
        _weight(weights, f"{prefix}.scale"),
        residual,
    )


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
) -> trt.ITensor:
    scores = network.add_shuffle(object_score_logits)
    scores.reshape_dims = (batch_size, 1, 1, 1)
    zero = _constant_like(
        network,
        scores.get_output(0),
        (1, 1, 1, 1),
        np.zeros((1, 1, 1, 1), dtype=np.float32),
        dtype=np.float32,
    )
    appearing = network.add_elementwise(
        scores.get_output(0), zero, trt.ElementWiseOperation.GREATER
    ).get_output(0)
    appearing = _cast(network, appearing, trt.float32)
    one = _constant_like(
        network,
        appearing,
        (1, 1, 1, 1),
        np.ones((1, 1, 1, 1), dtype=np.float32),
        dtype=np.float32,
    )
    absent = network.add_elementwise(one, appearing, trt.ElementWiseOperation.SUB).get_output(0)
    embedding_key = "tracker_model.occlusion_spatial_embedding_parameter"
    try:
        embedding_value = np.asarray(weights[embedding_key])
    except KeyError as error:
        raise KeyError(f"Missing SAM3 tracker memory weight: {embedding_key}") from error
    embedding = add_constant(
        network,
        (1, _MEMORY_CHANNELS, 1, 1),
        embedding_value.reshape(1, _MEMORY_CHANNELS, 1, 1),
        dtype=np.float32,
    )
    occlusion = network.add_elementwise(
        absent, embedding, trt.ElementWiseOperation.PROD
    ).get_output(0)
    # Meta uses ``maskmem_features += fp32_occlusion``. CUDA computes that add
    # in FP32, then rounds the in-place destination back to BF16.
    memory_fp32 = _cast(network, memory, trt.float32)
    with_occlusion = network.add_elementwise(
        memory_fp32,
        occlusion,
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    return _cast(network, with_occlusion, trt.bfloat16)


def _format_outputs(
    network: trt.INetworkDefinition,
    memory: trt.ITensor,
    batch_size: int,
) -> TrackerMemoryEncoderOutputs:
    memory = _to_channels_last(network, memory)
    flattened = network.add_shuffle(memory)
    flattened.reshape_dims = (batch_size, _SPATIAL_TOKENS, _MEMORY_CHANNELS)

    position_values = _make_position_encoding(batch_size, dtype=np.float32).reshape(
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

    # Meta publishes both SimpleMaskEncoder outputs in the feature dtype:
    # ``maskmem_features`` is already BF16 and ``maskmem_pos_enc`` explicitly
    # executes ``position_encoding(x).to(x.dtype)``.  Keep the public TensorRT
    # bindings as FP32 carriers, but round both values through BF16 at this
    # recurrent-state boundary.
    memory_output = _cast(network, memory_output, trt.float32)
    position_output = add_constant(
        network,
        position_shape,
        position_values,
        dtype=np.float32,
    )
    position_output = _cast(network, position_output, trt.bfloat16)
    position_output = _cast(network, position_output, trt.float32)
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
    suppress_area_shrinkage: trt.ITensor | None = None,
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
        suppress_area_shrinkage: One INT32 rejection flag per recurrent
            object. Required for soft memory and forbidden for hard memory.

    Notes:
        The graph always reconstructs Meta's mixed BF16/FP32 CUDA autocast
        behavior. Runtime-facing inputs and outputs remain FP32 tensors.
    """

    _validate_inputs(
        vision_features,
        mask_logits,
        object_score_logits,
        suppress_area_shrinkage,
        batch_size,
    )
    dtype = np.dtype(np.float32)
    memory_mask = _prepare_memory_mask(
        network,
        mask_logits,
        suppress_area_shrinkage,
        batch_size,
        hard_mask=hard_mask,
        dtype=dtype,
    )
    memory_mask = _add_mask_downsampler(network, memory_mask, weights)

    projected_features = _add_bf16_conv2d(
        network,
        vision_features,
        _weight(weights, "feature_projection.weight"),
        _weight(weights, "feature_projection.bias"),
        _VISION_CHANNELS,
        (1, 1),
    )
    hidden = network.add_elementwise(
        projected_features, memory_mask, trt.ElementWiseOperation.SUM
    ).get_output(0)
    hidden = _cast(network, hidden, trt.bfloat16)
    for layer_index in range(2):
        hidden = _add_convnext_fuser_block(network, hidden, weights, layer_index)

    memory = _add_bf16_conv2d(
        network,
        hidden,
        _weight(weights, "projection.weight"),
        _weight(weights, "projection.bias"),
        _MEMORY_CHANNELS,
        (1, 1),
    )
    memory = _add_occlusion_embedding(
        network,
        memory,
        object_score_logits,
        weights,
        batch_size,
    )
    return _format_outputs(network, memory, batch_size)


__all__ = ["TrackerMemoryEncoderOutputs", "add_tracker_memory_encoder"]
