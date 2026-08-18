# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT Network Definition API graph for the stereo feature extractor.

The serialized Fast Foundation Stereo checkpoint contains a pruned model, so
this graph is deliberately driven by the loaded module tree instead of by a
second, hand-written model configuration.  Every learned value below comes
from that module tree; only the fixed Fourier coordinates are generated here.
"""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

from .native_graph import NativeGraph


FEATURE_OUTPUT_NAMES = (
    "features_left_04",
    "features_left_08",
    "features_left_16",
    "features_left_32",
    "features_right_04",
    "stem_2x",
)

_DECODER_FP16_PRECONCAT_88_ENV = "TRTMC_FAST_FOUNDATION_STEREO_DECODER_FP16_PRECONCAT_88"
_DECODER_FP16_PRECONCAT_176_ENV = "TRTMC_FAST_FOUNDATION_STEREO_DECODER_FP16_PRECONCAT_176"
_DECODER_SCOPES = ("deconv32_16", "deconv16_8", "deconv8_4")
_FEATURE_INTERNAL_BATCH = 2
_DECODER_FP16_PRECONCAT_88_SCOPE = "deconv16_8"
_DECODER_FP16_PRECONCAT_88_TAIL = (96, 88, 88)
_DECODER_FP16_PRECONCAT_176_SCOPE = "deconv8_4"
_DECODER_FP16_PRECONCAT_176_TAIL = (48, 176, 176)


def _network(graph: NativeGraph):
    return graph.network


def _trt(graph: NativeGraph):
    trt = getattr(graph, "trt", None)
    if trt is None:
        from tensorrt_model_connect import trt_compat

        trt = trt_compat.get_trt()
    return trt


def _work_dtype(graph: NativeGraph, fp16: bool):
    return _trt(graph).float16 if fp16 else _trt(graph).float32


def _array(value: Any, dtype: np.dtype) -> np.ndarray:
    """Copy a torch parameter/buffer to a contiguous NumPy weight."""

    if value is None:
        return np.empty((0,), dtype=dtype)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(value, dtype=dtype)


def _constant(graph: NativeGraph, values: Any, dtype: np.dtype, shape=None):
    array = _array(values, dtype)
    if shape is not None:
        array = np.ascontiguousarray(array.reshape(shape))
    return graph.constant(array, tuple(array.shape), dtype=dtype)


def _cast(graph: NativeGraph, tensor, dtype):
    return graph.cast(tensor, dtype)


def _reshape(graph: NativeGraph, tensor, shape):
    return graph.reshape(tensor, tuple(shape))


def _transpose(graph: NativeGraph, tensor, permutation):
    return graph.transpose(tensor, tuple(permutation))


def _slice(graph: NativeGraph, tensor, start, shape):
    return graph.slice(tensor, tuple(start), tuple(shape))


def _concat(graph: NativeGraph, tensors, axis: int):
    return graph.concat(tensors, axis)


def _binary(graph: NativeGraph, lhs, rhs, operation):
    return graph.elementwise(operation, lhs, rhs)


def _conv2d(graph: NativeGraph, tensor, module, *, fp16: bool):
    """Add a checkpoint-owned Conv2d, respecting autocast at its boundary."""

    tensor = _cast(graph, tensor, _work_dtype(graph, fp16))
    return graph.conv2d(tensor, module)


def _deconv2d(graph: NativeGraph, tensor, module, *, fp16: bool):
    tensor = _cast(graph, tensor, _work_dtype(graph, fp16))
    return graph.deconv2d(tensor, module)


def _normalization(
    graph: NativeGraph,
    tensor,
    *,
    axes: tuple[int, ...],
    channels: int,
    epsilon: float,
    fp16: bool,
    weight=None,
    bias=None,
    force_fp32: bool = False,
):
    trt = _trt(graph)
    rank = len(tuple(tensor.shape))
    channel_axis = rank - 1 if axes == (rank - 1,) else 1
    param_shape = [1] * rank
    param_shape[channel_axis] = channels
    dtype = np.float32 if force_fp32 or not fp16 else np.float16
    trt_dtype = trt.float32 if dtype == np.float32 else trt.float16
    tensor = _cast(graph, tensor, trt_dtype)
    scale_values = np.ones((channels,), dtype=dtype) if weight is None else _array(weight, dtype)
    bias_values = np.zeros((channels,), dtype=dtype) if bias is None else _array(bias, dtype)
    scale = _constant(graph, scale_values, dtype, param_shape)
    shift = _constant(graph, bias_values, dtype, param_shape)
    axes_mask = sum(1 << axis for axis in axes)
    add_normalization = getattr(_network(graph), "add_normalization_v2", None)
    if add_normalization is None:
        add_normalization = _network(graph).add_normalization
    layer = add_normalization(tensor, scale, shift, axes_mask)
    layer.epsilon = float(epsilon)
    return layer.get_output(0)


def _layer_norm_last(graph: NativeGraph, tensor, module):
    rank = len(tuple(tensor.shape))
    channels = int(module.weight.numel())
    return _normalization(
        graph,
        tensor,
        axes=(rank - 1,),
        channels=channels,
        epsilon=float(module.eps),
        fp16=False,
        weight=module.weight,
        bias=module.bias,
        force_fp32=True,
    )


def _layer_norm_2d(graph: NativeGraph, tensor, module):
    channels = int(module.weight.numel())
    return _normalization(
        graph,
        tensor,
        axes=(1,),
        channels=channels,
        epsilon=float(module.eps),
        fp16=False,
        weight=module.weight,
        bias=module.bias,
        force_fp32=True,
    )


def _instance_norm_2d(graph: NativeGraph, tensor, module, *, fp16: bool):
    del fp16
    return graph.instance_norm(tensor, module)


def _use_decoder_fp16_preconcat(
    *,
    decoder_scope: str,
    target_scope: str,
    environment: str,
) -> bool:
    """Gate an FP16 pre-concat candidate by explicit decoder identity."""

    if decoder_scope not in _DECODER_SCOPES:
        raise RuntimeError(f"decoder scope must be one of {_DECODER_SCOPES}, got {decoder_scope!r}")
    return decoder_scope == target_scope and os.environ.get(environment, "1") == "1"


def _use_decoder_fp16_preconcat_88(*, decoder_scope: str) -> bool:
    """Gate the 88x88 concat candidate by explicit decoder identity."""

    return _use_decoder_fp16_preconcat(
        decoder_scope=decoder_scope,
        target_scope=_DECODER_FP16_PRECONCAT_88_SCOPE,
        environment=_DECODER_FP16_PRECONCAT_88_ENV,
    )


def _use_decoder_fp16_preconcat_176(*, decoder_scope: str) -> bool:
    """Gate the 176x176 concat candidate by explicit decoder identity."""

    return _use_decoder_fp16_preconcat(
        decoder_scope=decoder_scope,
        target_scope=_DECODER_FP16_PRECONCAT_176_SCOPE,
        environment=_DECODER_FP16_PRECONCAT_176_ENV,
    )


def _validate_decoder_fp16_preconcat_skip(
    graph: NativeGraph,
    skip,
    *,
    fp16: bool,
    spatial_size: int,
    expected_tail: tuple[int, int, int],
) -> int:
    """Validate known skip metadata before adding any decoder layers."""

    if not fp16:
        raise RuntimeError(
            f"decoder FP16 pre-concat {spatial_size} requires the FP16 feature graph"
        )
    shape = tuple(int(dimension) for dimension in skip.shape)
    expected_shape = (_FEATURE_INTERNAL_BATCH, *expected_tail)
    if shape != expected_shape:
        raise RuntimeError(
            f"decoder FP16 pre-concat {spatial_size} skip shape must be "
            f"{expected_shape}, got {shape}"
        )
    expected_dtype = _trt(graph).float32
    if skip.dtype != expected_dtype:
        raise RuntimeError(
            f"decoder FP16 pre-concat {spatial_size} skip dtype must be "
            f"TensorRT float32, got {skip.dtype!r}"
        )
    return shape[0]


def _validate_decoder_fp16_preconcat_branch(
    graph: NativeGraph,
    branch,
    *,
    internal_batch: int,
    spatial_size: int,
    expected_tail: tuple[int, int, int],
) -> None:
    """Validate the completed branch before adding the candidate concat."""

    shape = tuple(int(dimension) for dimension in branch.shape)
    expected_shape = (internal_batch, *expected_tail)
    if shape != expected_shape:
        raise RuntimeError(
            f"decoder FP16 pre-concat {spatial_size} branch shape must be "
            f"{expected_shape}, got {shape}"
        )
    expected_dtype = _trt(graph).float16
    if branch.dtype != expected_dtype:
        raise RuntimeError(
            f"decoder FP16 pre-concat {spatial_size} branch dtype must be TensorRT float16, "
            f"got {branch.dtype!r}"
        )


def _validate_decoder_fp16_preconcat_88_skip(
    graph: NativeGraph,
    skip,
    *,
    fp16: bool,
) -> int:
    return _validate_decoder_fp16_preconcat_skip(
        graph,
        skip,
        fp16=fp16,
        spatial_size=88,
        expected_tail=_DECODER_FP16_PRECONCAT_88_TAIL,
    )


def _validate_decoder_fp16_preconcat_88_branch(
    graph: NativeGraph,
    branch,
    *,
    internal_batch: int,
) -> None:
    _validate_decoder_fp16_preconcat_branch(
        graph,
        branch,
        internal_batch=internal_batch,
        spatial_size=88,
        expected_tail=_DECODER_FP16_PRECONCAT_88_TAIL,
    )


def _validate_decoder_fp16_preconcat_176_skip(
    graph: NativeGraph,
    skip,
    *,
    fp16: bool,
) -> int:
    return _validate_decoder_fp16_preconcat_skip(
        graph,
        skip,
        fp16=fp16,
        spatial_size=176,
        expected_tail=_DECODER_FP16_PRECONCAT_176_TAIL,
    )


def _validate_decoder_fp16_preconcat_176_branch(
    graph: NativeGraph,
    branch,
    *,
    internal_batch: int,
) -> None:
    _validate_decoder_fp16_preconcat_branch(
        graph,
        branch,
        internal_batch=internal_batch,
        spatial_size=176,
        expected_tail=_DECODER_FP16_PRECONCAT_176_TAIL,
    )


def _activation(graph: NativeGraph, tensor, kind: str, *, alpha: float = 0.01):
    return graph.activation(tensor, kind, alpha=float(alpha) if kind == "leaky_relu" else None)


def _linear(graph: NativeGraph, tensor, module, *, fp16: bool):
    """Apply a torch Linear to the final dimension with TRT MatMul."""

    tensor = _cast(graph, tensor, _work_dtype(graph, fp16))
    return graph.linear(tensor, module)


def _channel_scale_nhwc(graph: NativeGraph, tensor, scale, *, force_fp32: bool):
    trt = _trt(graph)
    dtype = (
        np.float32 if force_fp32 else (np.float16 if tensor.dtype == trt.float16 else np.float32)
    )
    if force_fp32:
        tensor = _cast(graph, tensor, trt.float32)
    shape = (1,) * (len(tuple(tensor.shape)) - 1) + (int(scale.numel()),)
    scale_tensor = _constant(graph, scale, dtype, shape)
    return _binary(graph, tensor, scale_tensor, trt.ElementWiseOperation.PROD)


def _conv_block(
    graph: NativeGraph,
    tensor,
    block,
    *,
    fp16: bool,
):
    """timm EdgeNeXt ConvBlock in evaluation mode."""

    trt = _trt(graph)
    shortcut = tensor
    tensor = _conv2d(graph, tensor, block.conv_dw, fp16=fp16)
    if bool(getattr(block, "shortcut_after_dw", False)):
        shortcut = tensor

    tensor = _transpose(graph, tensor, (0, 2, 3, 1))
    tensor = _layer_norm_last(graph, tensor, block.norm)
    tensor = _linear(graph, tensor, block.mlp.fc1, fp16=fp16)
    tensor = _activation(graph, tensor, "gelu")
    tensor = _linear(graph, tensor, block.mlp.fc2, fp16=fp16)
    if getattr(block, "gamma", None) is not None:
        # A float32 Parameter multiplied by an autocast float16 MLP output
        # promotes this residual branch to float32 in the source model.
        tensor = _channel_scale_nhwc(graph, tensor, block.gamma, force_fp32=fp16)
    tensor = _transpose(graph, tensor, (0, 3, 1, 2))
    if tensor.dtype != shortcut.dtype:
        shortcut = _cast(graph, shortcut, tensor.dtype)
    return _binary(graph, shortcut, tensor, trt.ElementWiseOperation.SUM)


def _l2_normalize_last(graph: NativeGraph, tensor, *, fp16: bool):
    del fp16
    trt = _trt(graph)
    tensor = _cast(graph, tensor, trt.float32)
    squared = _binary(graph, tensor, tensor, trt.ElementWiseOperation.PROD)
    axis = len(tuple(tensor.shape)) - 1
    summed = graph.reduce_sum(squared, (axis,), keep_dims=True)
    norm = graph.unary(trt.UnaryOperation.SQRT, summed)
    epsilon = _constant(
        graph,
        np.asarray([1.0e-12], dtype=np.float32),
        np.float32,
        (1,) * len(tuple(tensor.shape)),
    )
    norm = _binary(graph, norm, epsilon, trt.ElementWiseOperation.MAX)
    reciprocal = graph.unary(trt.UnaryOperation.RECIP, norm)
    return _binary(graph, tensor, reciprocal, trt.ElementWiseOperation.PROD)


def _xca(graph: NativeGraph, tensor, module, *, fp16: bool):
    """EdgeNeXt cross-covariance attention for static [B,N,C]."""

    trt = _trt(graph)
    batch, tokens, channels = (int(v) for v in tensor.shape)
    heads = int(module.num_heads)
    head_dim = channels // heads

    qkv = _linear(graph, tensor, module.qkv, fp16=fp16)
    qkv = _reshape(graph, qkv, (batch, tokens, 3, heads, head_dim))
    qkv = _transpose(graph, qkv, (2, 0, 3, 4, 1))
    pieces = []
    for index in range(3):
        piece = _slice(
            graph,
            qkv,
            (index, 0, 0, 0, 0),
            (1, batch, heads, head_dim, tokens),
        )
        pieces.append(_reshape(graph, piece, (batch, heads, head_dim, tokens)))
    query, key, value = pieces
    query = _l2_normalize_last(graph, query, fp16=fp16)
    key = _l2_normalize_last(graph, key, fp16=fp16)
    # F.normalize is FP32 under CUDA autocast; the following MatMul is
    # autocast-eligible and returns to the requested work dtype.
    query = _cast(graph, query, _work_dtype(graph, fp16))
    key = _cast(graph, key, _work_dtype(graph, fp16))
    attention = graph.matmul(
        query,
        key,
        op_rhs=trt.MatrixOperation.TRANSPOSE,
    )

    # temperature is an ordinary float32 Parameter, hence this multiply and
    # the following softmax are float32 under torch autocast.
    attention = _cast(graph, attention, trt.float32)
    temperature = _constant(
        graph,
        module.temperature,
        np.float32,
        (1, heads, 1, 1),
    )
    attention = _binary(graph, attention, temperature, trt.ElementWiseOperation.PROD)
    attention = graph.softmax(attention, axis=3)

    # MatMul is autocast-eligible: both operands return to the work dtype.
    attention = _cast(graph, attention, _work_dtype(graph, fp16))
    value = _cast(graph, value, _work_dtype(graph, fp16))
    tensor = graph.matmul(
        attention,
        value,
    )
    tensor = _transpose(graph, tensor, (0, 3, 1, 2))
    tensor = _reshape(graph, tensor, (batch, tokens, channels))
    return _linear(graph, tensor, module.proj, fp16=fp16)


def _fourier_coordinates(*, batch: int, height: int, width: int, hidden_dim: int) -> np.ndarray:
    """Generate timm PositionalEncodingFourier's deterministic input."""

    scale = 2.0 * math.pi
    y = np.arange(1, height + 1, dtype=np.float32)[None, :, None]
    y = np.broadcast_to(y, (batch, height, width))
    x = np.arange(1, width + 1, dtype=np.float32)[None, None, :]
    x = np.broadcast_to(x, (batch, height, width))
    y = y / (np.float32(height) + np.float32(1.0e-6)) * np.float32(scale)
    x = x / (np.float32(width) + np.float32(1.0e-6)) * np.float32(scale)

    indices = np.arange(hidden_dim, dtype=np.float32)
    powers = 2.0 * np.floor(indices / 2.0) / np.float32(hidden_dim)
    divisors = np.power(np.float32(10000.0), powers).astype(np.float32)
    pos_x = x[..., None] / divisors
    pos_y = y[..., None] / divisors
    pos_x = np.stack((np.sin(pos_x[..., 0::2]), np.cos(pos_x[..., 1::2])), axis=4)
    pos_y = np.stack((np.sin(pos_y[..., 0::2]), np.cos(pos_y[..., 1::2])), axis=4)
    pos_x = pos_x.reshape(batch, height, width, hidden_dim)
    pos_y = pos_y.reshape(batch, height, width, hidden_dim)
    return np.ascontiguousarray(np.concatenate((pos_y, pos_x), axis=3).transpose(0, 3, 1, 2))


def _split_transpose_block(graph: NativeGraph, tensor, block, *, fp16: bool):
    """timm SplitTransposeBlock including the optional Fourier embedding."""

    trt = _trt(graph)
    shortcut = tensor
    batch, channels, height, width = (int(v) for v in tensor.shape)
    chunks = len(block.convs) + 1
    width_per_chunk = int(block.width)
    chunk_sizes = [width_per_chunk] * (chunks - 1)
    chunk_sizes.append(channels - width_per_chunk * (chunks - 1))
    inputs = []
    start = 0
    for size in chunk_sizes:
        inputs.append(
            _slice(
                graph,
                tensor,
                (0, start, 0, 0),
                (batch, size, height, width),
            )
        )
        start += size

    outputs = []
    running = inputs[0]
    for index, convolution in enumerate(block.convs):
        if index:
            if running.dtype != inputs[index].dtype:
                running = _cast(graph, running, inputs[index].dtype)
            running = _binary(graph, running, inputs[index], trt.ElementWiseOperation.SUM)
        running = _conv2d(graph, running, convolution, fp16=fp16)
        outputs.append(running)
    outputs.append(inputs[-1])
    # cat is in autocast's promote-to-widest set.
    concat_dtype = trt.float32 if any(t.dtype == trt.float32 for t in outputs) else outputs[0].dtype
    tensor = _concat(graph, [_cast(graph, value, concat_dtype) for value in outputs], 1)

    tensor = _reshape(graph, tensor, (batch, channels, height * width))
    tensor = _transpose(graph, tensor, (0, 2, 1))
    if getattr(block, "pos_embd", None) is not None:
        coordinates = _fourier_coordinates(
            batch=batch,
            height=height,
            width=width,
            hidden_dim=int(block.pos_embd.hidden_dim),
        )
        position = _constant(graph, coordinates, np.float16 if fp16 else np.float32)
        position = _conv2d(graph, position, block.pos_embd.token_projection, fp16=fp16)
        position = _reshape(graph, position, (batch, channels, height * width))
        position = _transpose(graph, position, (0, 2, 1))
        if position.dtype != tensor.dtype:
            position = _cast(graph, position, tensor.dtype)
        tensor = _binary(graph, tensor, position, trt.ElementWiseOperation.SUM)

    attention_input = tensor
    normalized = _layer_norm_last(graph, tensor, block.norm_xca)
    attended = _xca(graph, normalized, block.xca, fp16=fp16)
    if getattr(block, "gamma_xca", None) is not None:
        attended = _channel_scale_nhwc(graph, attended, block.gamma_xca, force_fp32=fp16)
    if attention_input.dtype != attended.dtype:
        attention_input = _cast(graph, attention_input, attended.dtype)
    tensor = _binary(graph, attention_input, attended, trt.ElementWiseOperation.SUM)
    tensor = _reshape(graph, tensor, (batch, height, width, channels))

    tensor = _layer_norm_last(graph, tensor, block.norm)
    tensor = _linear(graph, tensor, block.mlp.fc1, fp16=fp16)
    tensor = _activation(graph, tensor, "gelu")
    tensor = _linear(graph, tensor, block.mlp.fc2, fp16=fp16)
    if getattr(block, "gamma", None) is not None:
        tensor = _channel_scale_nhwc(graph, tensor, block.gamma, force_fp32=fp16)
    tensor = _transpose(graph, tensor, (0, 3, 1, 2))
    if shortcut.dtype != tensor.dtype:
        shortcut = _cast(graph, shortcut, tensor.dtype)
    return _binary(graph, shortcut, tensor, trt.ElementWiseOperation.SUM)


def _edge_next(graph: NativeGraph, feature, tensor, *, fp16: bool):
    tensor = _conv2d(graph, tensor, feature.stem[0], fp16=fp16)
    tensor = _layer_norm_2d(graph, tensor, feature.stem[1])
    outputs = []
    for stage in feature.stages:
        downsample_children = list(stage.downsample.children())
        if downsample_children:
            tensor = _layer_norm_2d(graph, tensor, downsample_children[0])
            tensor = _conv2d(graph, tensor, downsample_children[1], fp16=fp16)
        for block in stage.blocks:
            if hasattr(block, "xca"):
                tensor = _split_transpose_block(graph, tensor, block, fp16=fp16)
            else:
                tensor = _conv_block(graph, tensor, block, fp16=fp16)
        outputs.append(tensor)
    return outputs


def _resnet_instance_block(
    graph: NativeGraph,
    tensor,
    block,
    *,
    fp16: bool,
):
    """Serialized decoder ResnetBasicBlock (its residual add is in-place)."""

    trt = _trt(graph)
    identity = tensor
    tensor = _conv2d(graph, tensor, block.conv1, fp16=fp16)
    tensor = _instance_norm_2d(graph, tensor, block.bn1, fp16=fp16)
    tensor = _activation(graph, tensor, "relu")
    tensor = _conv2d(graph, tensor, block.conv2, fp16=fp16)
    tensor = _instance_norm_2d(graph, tensor, block.bn2, fp16=fp16)
    # Source uses ``out += identity``.  Its float16 lhs therefore casts the
    # float32 skip in place instead of promoting the output.
    identity = _cast(graph, identity, tensor.dtype)
    tensor = _binary(graph, tensor, identity, trt.ElementWiseOperation.SUM)
    return _activation(graph, tensor, "relu")


def _decoder_block(
    graph: NativeGraph,
    tensor,
    skip,
    block,
    *,
    decoder_scope: str,
    fp16: bool,
):
    use_fp16_preconcat_88 = _use_decoder_fp16_preconcat_88(decoder_scope=decoder_scope)
    use_fp16_preconcat_176 = _use_decoder_fp16_preconcat_176(decoder_scope=decoder_scope)
    internal_batch = None
    if use_fp16_preconcat_88:
        internal_batch = _validate_decoder_fp16_preconcat_88_skip(graph, skip, fp16=fp16)
    elif use_fp16_preconcat_176:
        internal_batch = _validate_decoder_fp16_preconcat_176_skip(graph, skip, fp16=fp16)

    tensor = _deconv2d(graph, tensor, block.conv1.conv, fp16=fp16)
    tensor = _instance_norm_2d(graph, tensor, block.conv1.IN, fp16=fp16)
    tensor = _activation(
        graph,
        tensor,
        "leaky_relu",
        alpha=float(getattr(block.conv1.relu, "negative_slope", 0.01)),
    )
    if use_fp16_preconcat_88 or use_fp16_preconcat_176:
        if internal_batch is None:
            raise AssertionError("enabled decoder pre-concat must validate its internal batch")
        if use_fp16_preconcat_88:
            _validate_decoder_fp16_preconcat_88_branch(
                graph,
                tensor,
                internal_batch=internal_batch,
            )
        else:
            _validate_decoder_fp16_preconcat_176_branch(
                graph,
                tensor,
                internal_batch=internal_batch,
            )
        # This is the exact FP16 boundary already imposed by the following
        # convolution and residual path, moved before concat:
        # cast(concat(cast(branch_fp16, fp32), skip_fp32), fp16)
        # == concat(branch_fp16, cast(skip_fp32, fp16)).  The next convolution
        # input and its FP16 residual identity therefore remain bitwise equal.
        tensor = _concat(
            graph,
            [tensor, _cast(graph, skip, _trt(graph).float16)],
            axis=1,
        )
    else:
        # cat promotes to float32 because the EdgeNeXt skip is float32.
        concat_dtype = skip.dtype
        tensor = _concat(
            graph,
            [_cast(graph, tensor, concat_dtype), skip],
            axis=1,
        )
    return _resnet_instance_block(
        graph,
        tensor,
        block.conv2,
        fp16=fp16,
    )


def _stem_2x(graph: NativeGraph, stem, tensor, *, fp16: bool):
    first = stem[0]
    tensor = _conv2d(graph, tensor, first.conv, fp16=fp16)
    tensor = _instance_norm_2d(graph, tensor, first.IN, fp16=fp16)
    relu = getattr(first, "relu", True)
    if relu is True or relu.__class__.__name__ == "LeakyReLU":
        tensor = _activation(
            graph,
            tensor,
            "leaky_relu",
            alpha=float(getattr(relu, "negative_slope", 0.01)),
        )
    tensor = _conv2d(graph, tensor, stem[1], fp16=fp16)
    tensor = _instance_norm_2d(graph, tensor, stem[2], fp16=fp16)
    return _activation(graph, tensor, "relu")


def _normalize_image(graph: NativeGraph, image):
    """ImageNet normalization in the checkpoint's float32 input domain."""

    trt = _trt(graph)
    image = _cast(graph, image, trt.float32)
    value_range = _constant(
        graph, np.asarray([255.0], dtype=np.float32).reshape(1, 1, 1, 1), np.float32
    )
    mean = _constant(
        graph,
        np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1),
        np.float32,
    )
    std = _constant(
        graph,
        np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1),
        np.float32,
    )
    image = _binary(graph, image, value_range, trt.ElementWiseOperation.DIV)
    image = _binary(graph, image, mean, trt.ElementWiseOperation.SUB)
    return _binary(graph, image, std, trt.ElementWiseOperation.DIV)


def add_feature_graph(
    graph: NativeGraph,
    model,
    left,
    right,
    *,
    fp16: bool,
) -> dict[str, Any]:
    """Add the checkpoint's complete split-feature graph.

    Args:
        graph: Family-owned native TensorRT graph helper.
        model: Loaded, configured serialized Fast Foundation Stereo module.
        left/right: Float32 ``[1, 3, 704, 704]`` input tensors.
        fp16: Whether autocast-eligible work should use float16.

    Returns:
        Mapping of the six stable engine output names to TensorRT tensors.
    """

    trt = _trt(graph)
    left_normalized = _normalize_image(graph, left)
    right_normalized = _normalize_image(graph, right)
    work_dtype = _work_dtype(graph, fp16)
    stereo_batch = _concat(
        graph,
        [
            _cast(graph, left_normalized, work_dtype),
            _cast(graph, right_normalized, work_dtype),
        ],
        axis=0,
    )

    x04, x08, x16, x32 = _edge_next(graph, model.feature, stereo_batch, fp16=fp16)
    x16_decoded = _decoder_block(
        graph,
        x32,
        x16,
        model.feature.deconv32_16,
        decoder_scope="deconv32_16",
        fp16=fp16,
    )
    x08_decoded = _decoder_block(
        graph,
        x16_decoded,
        x08,
        model.feature.deconv16_8,
        decoder_scope="deconv16_8",
        fp16=fp16,
    )
    x04_decoded = _decoder_block(
        graph,
        x08_decoded,
        x04,
        model.feature.deconv8_4,
        decoder_scope="deconv8_4",
        fp16=fp16,
    )
    x04_decoded = _conv2d(graph, x04_decoded, model.feature.conv4, fp16=fp16)
    stem_2x = _stem_2x(
        graph,
        model.stem_2,
        _cast(graph, left_normalized, work_dtype),
        fp16=fp16,
    )

    batch = int(left.shape[0])
    outputs = {
        "features_left_04": _slice(graph, x04_decoded, (0, 0, 0, 0), (batch, 224, 176, 176)),
        "features_left_08": _slice(graph, x08_decoded, (0, 0, 0, 0), (batch, 192, 88, 88)),
        "features_left_16": _slice(graph, x16_decoded, (0, 0, 0, 0), (batch, 320, 44, 44)),
        "features_left_32": _slice(graph, x32, (0, 0, 0, 0), (batch, 304, 22, 22)),
        "features_right_04": _slice(graph, x04_decoded, (batch, 0, 0, 0), (batch, 224, 176, 176)),
        "stem_2x": stem_2x,
    }
    expected_dtypes = {
        "features_left_04": work_dtype,
        "features_left_08": work_dtype,
        "features_left_16": work_dtype,
        "features_left_32": trt.float32,
        "features_right_04": work_dtype,
        "stem_2x": work_dtype,
    }
    for name, tensor in tuple(outputs.items()):
        tensor = _cast(graph, tensor, expected_dtypes[name])
        tensor.name = name
        outputs[name] = tensor
    return outputs
