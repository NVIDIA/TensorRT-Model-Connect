# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned TensorRT graph operations for the Qwen-Image 2.1 engines.

Every engine is a strongly typed network with FP32 inputs and outputs. Linear layers, convolutions,
activations, residuals and attention run in the graph's compute type: BF16, the checkpoint's own
precision. Normalisation statistics and the transformer's rotary rotation run in FP32, as the
reference modules do.

A graph created with FP32 as its compute type is the same network without BF16 rounding. The
family's parity tests build it to compare the graphs with the reference modules to FP32 accuracy;
bundles are always built in BF16.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import ml_dtypes
import numpy as np
import tensorrt as trt

BF16 = trt.bfloat16
FP32 = trt.float32

# Additive mask value for excluded attention keys. Finite, so a BF16 score plus the mask stays finite.
MASKED_SCORE = -1.0e9


@dataclass
class Graph:
    """A TensorRT network plus the host arrays its constants point into."""

    builder: trt.Builder
    network: trt.INetworkDefinition
    config: trt.IBuilderConfig
    # Type of weights, activations and attention: BF16, or FP32 for the parity oracle.
    compute: trt.DataType = BF16
    # TensorRT reads constant storage through raw pointers until the plan is serialised.
    anchors: list[np.ndarray] = field(default_factory=list)


def create_graph(
    *,
    verbose: bool,
    workspace_bytes: int,
    optimization_level: int,
    compute: trt.DataType = BF16,
) -> Graph:
    """Create a strongly typed network and its builder configuration."""
    if compute not in (BF16, FP32):
        raise ValueError(f"unsupported compute type {compute}")
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.builder_optimization_level = optimization_level
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    # TF32 would round the inputs of FP32 matrix products and convolutions to 10 bits.
    config.clear_flag(trt.BuilderFlag.TF32)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    return Graph(builder=builder, network=network, config=config, compute=compute)


def serialize(graph: Graph, label: str) -> trt.IHostMemory:
    """Build the plan; the caller writes it out while ``graph`` is still alive."""
    plan = graph.builder.build_serialized_network(graph.network, graph.config)
    if plan is None:
        raise RuntimeError(f"TensorRT could not build the Qwen-Image 2.1 {label} engine")
    return plan


def shape_of(tensor: trt.ITensor) -> tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.shape)


# ---------------------------------------------------------------------------
# Constants and casts
# ---------------------------------------------------------------------------


def constant(graph: Graph, values: np.ndarray) -> trt.ITensor:
    """Add a constant in the array's own NumPy dtype (float32, int32, int64 or bool)."""
    array = np.ascontiguousarray(values)
    if array.dtype not in (np.float32, np.int32, np.int64, np.bool_):
        raise TypeError(f"unsupported constant dtype {array.dtype}")
    graph.anchors.append(array)
    return graph.network.add_constant(array.shape, trt.Weights(array)).get_output(0)


def bf16_array(values: np.ndarray) -> np.ndarray:
    """Return a contiguous BF16 array; BF16 checkpoints pass through without rounding."""
    array = np.asarray(values)
    if array.dtype != ml_dtypes.bfloat16:
        array = array.astype(np.float32).astype(ml_dtypes.bfloat16)
    return np.ascontiguousarray(array)


def compute_array(graph: Graph, values: np.ndarray) -> np.ndarray:
    """Return ``values`` as a contiguous array of the graph's compute type."""
    if graph.compute == BF16:
        return bf16_array(values)
    return np.ascontiguousarray(np.asarray(values).astype(np.float32))


def compute_weights(graph: Graph, values: np.ndarray) -> trt.Weights:
    array = compute_array(graph, values)
    graph.anchors.append(array)
    return trt.Weights(graph.compute, array.ctypes.data, array.size)


def constant_compute(graph: Graph, values: np.ndarray) -> trt.ITensor:
    """Add a constant of the graph's compute type."""
    array = compute_array(graph, values)
    graph.anchors.append(array)
    weights = trt.Weights(graph.compute, array.ctypes.data, array.size)
    return graph.network.add_constant(array.shape, weights).get_output(0)


def cast(graph: Graph, tensor: trt.ITensor, dtype: trt.DataType) -> trt.ITensor:
    if tensor.dtype == dtype:
        return tensor
    return graph.network.add_cast(tensor, dtype).get_output(0)


def scalar(graph: Graph, value: float, like: trt.ITensor) -> trt.ITensor:
    """A broadcastable scalar constant with the rank and dtype of ``like``."""
    shape = (1,) * len(like.shape)
    if like.dtype == FP32:
        return constant(graph, np.full(shape, value, dtype=np.float32))
    if like.dtype == graph.compute:
        return constant_compute(graph, np.full(shape, value, dtype=np.float32))
    raise TypeError(f"unsupported scalar dtype {like.dtype}")


# ---------------------------------------------------------------------------
# Shape manipulation
# ---------------------------------------------------------------------------


def reshape(graph: Graph, tensor: trt.ITensor, dims: tuple[int, ...]) -> trt.ITensor:
    layer = graph.network.add_shuffle(tensor)
    layer.reshape_dims = dims
    return layer.get_output(0)


def transpose(graph: Graph, tensor: trt.ITensor, permutation: tuple[int, ...]) -> trt.ITensor:
    layer = graph.network.add_shuffle(tensor)
    layer.first_transpose = trt.Permutation(list(permutation))
    return layer.get_output(0)


def slice_axis(
    graph: Graph, tensor: trt.ITensor, axis: int, start: int, length: int, stride: int = 1
) -> trt.ITensor:
    """Static slice of ``length`` elements along one axis."""
    dims = shape_of(tensor)
    starts = [0] * len(dims)
    sizes = list(dims)
    strides = [1] * len(dims)
    starts[axis] = start
    sizes[axis] = length
    strides[axis] = stride
    return graph.network.add_slice(tensor, tuple(starts), tuple(sizes), tuple(strides)).get_output(0)


def concat(graph: Graph, tensors: list[trt.ITensor], axis: int) -> trt.ITensor:
    layer = graph.network.add_concatenation(tensors)
    layer.axis = axis
    return layer.get_output(0)


# ---------------------------------------------------------------------------
# Elementwise math
# ---------------------------------------------------------------------------


def _elementwise(graph: Graph, left, right, operation) -> trt.ITensor:
    return graph.network.add_elementwise(left, right, operation).get_output(0)


def add(graph: Graph, left: trt.ITensor, right: trt.ITensor) -> trt.ITensor:
    return _elementwise(graph, left, right, trt.ElementWiseOperation.SUM)


def sub(graph: Graph, left: trt.ITensor, right: trt.ITensor) -> trt.ITensor:
    return _elementwise(graph, left, right, trt.ElementWiseOperation.SUB)


def mul(graph: Graph, left: trt.ITensor, right: trt.ITensor) -> trt.ITensor:
    return _elementwise(graph, left, right, trt.ElementWiseOperation.PROD)


def add_scalar(graph: Graph, tensor: trt.ITensor, value: float) -> trt.ITensor:
    return add(graph, tensor, scalar(graph, value, tensor))


def mul_scalar(graph: Graph, tensor: trt.ITensor, value: float) -> trt.ITensor:
    return mul(graph, tensor, scalar(graph, value, tensor))


def silu(graph: Graph, tensor: trt.ITensor) -> trt.ITensor:
    sigmoid = graph.network.add_activation(tensor, trt.ActivationType.SIGMOID).get_output(0)
    return mul(graph, tensor, sigmoid)


def tanh(graph: Graph, tensor: trt.ITensor) -> trt.ITensor:
    return graph.network.add_activation(tensor, trt.ActivationType.TANH).get_output(0)


def gelu_tanh(graph: Graph, tensor: trt.ITensor) -> trt.ITensor:
    """``0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))``."""
    cube = mul(graph, mul(graph, tensor, tensor), tensor)
    inner = add(graph, tensor, mul_scalar(graph, cube, 0.044715))
    gate = add_scalar(graph, tanh(graph, mul_scalar(graph, inner, math.sqrt(2.0 / math.pi))), 1.0)
    return mul(graph, mul_scalar(graph, tensor, 0.5), gate)


def clamp(graph: Graph, tensor: trt.ITensor, low: float, high: float) -> trt.ITensor:
    lower = _elementwise(graph, tensor, scalar(graph, low, tensor), trt.ElementWiseOperation.MAX)
    return _elementwise(graph, lower, scalar(graph, high, tensor), trt.ElementWiseOperation.MIN)


# ---------------------------------------------------------------------------
# Linear layers and normalisation
# ---------------------------------------------------------------------------


def linear(
    graph: Graph, tensor: trt.ITensor, weight: np.ndarray, bias: np.ndarray | None = None
) -> trt.ITensor:
    """``tensor @ weight.T (+ bias)`` for a checkpoint weight stored ``[out, in]``."""
    out_features, in_features = weight.shape
    rank = len(tensor.shape)
    if int(tensor.shape[-1]) != in_features:
        raise ValueError(f"linear expects {in_features} input features, got {tensor.shape[-1]}")
    lead = (1,) * (rank - 2)
    rhs = constant_compute(
        graph, compute_array(graph, weight).T.reshape(*lead, in_features, out_features)
    )
    result = graph.network.add_matrix_multiply(
        cast(graph, tensor, graph.compute), trt.MatrixOperation.NONE, rhs, trt.MatrixOperation.NONE
    ).get_output(0)
    if bias is not None:
        result = add(
            graph, result, constant_compute(graph, np.asarray(bias).reshape(*lead, 1, -1))
        )
    return result


def _inverse_rms(graph: Graph, tensor_fp32: trt.ITensor, eps: float) -> trt.ITensor:
    """``rsqrt(mean(x^2, last axis) + eps)`` in FP32."""
    axis = len(tensor_fp32.shape) - 1
    square = mul(graph, tensor_fp32, tensor_fp32)
    mean = graph.network.add_reduce(
        square, trt.ReduceOperation.AVG, 1 << axis, keep_dims=True
    ).get_output(0)
    root = graph.network.add_unary(add_scalar(graph, mean, eps), trt.UnaryOperation.SQRT)
    return graph.network.add_unary(root.get_output(0), trt.UnaryOperation.RECIP).get_output(0)


def rms_norm(graph: Graph, tensor: trt.ITensor, gamma: np.ndarray, eps: float) -> trt.ITensor:
    """RMSNorm over the last axis: FP32 statistics, then ``gamma`` in the compute type."""
    tensor_fp32 = cast(graph, tensor, FP32)
    normalized = mul(graph, tensor_fp32, _inverse_rms(graph, tensor_fp32, eps))
    shape = (1,) * (len(tensor.shape) - 1) + (-1,)
    return mul(
        graph,
        cast(graph, normalized, graph.compute),
        constant_compute(graph, np.asarray(gamma).reshape(shape)),
    )


def rms_norm_fp32_scale(
    graph: Graph, tensor: trt.ITensor, scale: np.ndarray, eps: float
) -> trt.ITensor:
    """RMSNorm whose statistics and ``scale`` product both run in FP32."""
    tensor_fp32 = cast(graph, tensor, FP32)
    normalized = mul(graph, tensor_fp32, _inverse_rms(graph, tensor_fp32, eps))
    shape = (1,) * (len(tensor.shape) - 1) + (-1,)
    scaled = mul(graph, normalized, constant(graph, np.asarray(scale, dtype=np.float32).reshape(shape)))
    return cast(graph, scaled, graph.compute)


def layer_norm(graph: Graph, tensor: trt.ITensor, eps: float) -> trt.ITensor:
    """LayerNorm over the last axis without learned parameters; FP32 statistics."""
    tensor_fp32 = cast(graph, tensor, FP32)
    rank = len(tensor.shape)
    width = int(tensor.shape[-1])
    shape = (1,) * (rank - 1) + (width,)
    ones = constant(graph, np.ones(shape, dtype=np.float32))
    zeros = constant(graph, np.zeros(shape, dtype=np.float32))
    layer = graph.network.add_normalization(tensor_fp32, ones, zeros, 1 << (rank - 1))
    layer.epsilon = eps
    return cast(graph, layer.get_output(0), graph.compute)


def channel_l2_norm(graph: Graph, tensor: trt.ITensor, gamma: np.ndarray) -> trt.ITensor:
    """``F.normalize(x, dim=1) * sqrt(C) * gamma`` for ``[1, C, H, W]``; FP32 statistics."""
    channels = int(tensor.shape[1])
    tensor_fp32 = cast(graph, tensor, FP32)
    square = mul(graph, tensor_fp32, tensor_fp32)
    total = graph.network.add_reduce(
        square, trt.ReduceOperation.SUM, 1 << 1, keep_dims=True
    ).get_output(0)
    norm = graph.network.add_unary(total, trt.UnaryOperation.SQRT).get_output(0)
    # F.normalize divides by max(norm, 1e-12).
    floor = scalar(graph, 1.0e-12, norm)
    norm = _elementwise(graph, norm, floor, trt.ElementWiseOperation.MAX)
    normalized = _elementwise(graph, tensor_fp32, norm, trt.ElementWiseOperation.DIV)
    scale = math.sqrt(channels) * np.asarray(gamma, dtype=np.float32).reshape(1, channels, 1, 1)
    return cast(graph, mul(graph, normalized, constant(graph, scale)), graph.compute)


# ---------------------------------------------------------------------------
# Attention and convolution
# ---------------------------------------------------------------------------


def attention(
    graph: Graph,
    query: trt.ITensor,
    key: trt.ITensor,
    value: trt.ITensor,
    *,
    causal: bool = False,
    mask: trt.ITensor | None = None,
) -> trt.ITensor:
    """Scaled dot-product attention on ``[1, heads, sequence, head_dim]`` tensors.

    TensorRT's attention layer applies no score scaling, so the query is pre-scaled by
    ``1 / sqrt(head_dim)``. ``mask`` is an additive mask broadcastable to the scores.
    """
    if causal and mask is not None:
        raise ValueError("causal attention does not take an explicit mask")
    head_dim = int(query.shape[-1])
    scaled = mul_scalar(graph, query, 1.0 / math.sqrt(head_dim))
    layer = graph.network.add_attention(
        scaled, key, value, trt.AttentionNormalizationOp.SOFTMAX, causal
    )
    layer.decomposable = True
    if mask is not None:
        layer.mask = cast(graph, mask, query.dtype)
    return layer.get_output(0)


def conv2d(
    graph: Graph,
    tensor: trt.ITensor,
    weight: np.ndarray,
    bias: np.ndarray | None,
    *,
    padding: int = 0,
) -> trt.ITensor:
    """Stride-1 2D convolution with a checkpoint kernel stored ``[out, in, kh, kw]``."""
    out_channels, _, kernel_h, kernel_w = weight.shape
    layer = graph.network.add_convolution_nd(
        cast(graph, tensor, graph.compute),
        num_output_maps=out_channels,
        kernel_shape=(kernel_h, kernel_w),
        kernel=compute_weights(graph, weight),
        bias=compute_weights(graph, bias) if bias is not None else trt.Weights(),
    )
    layer.padding_nd = (padding, padding)
    return layer.get_output(0)


def upsample_nearest_2x(graph: Graph, tensor: trt.ITensor) -> trt.ITensor:
    """Nearest-neighbour 2x upsampling of ``[1, C, H, W]``: output ``i`` reads input ``i // 2``."""
    batch, channels, height, width = shape_of(tensor)
    layer = graph.network.add_resize(tensor)
    layer.resize_mode = trt.InterpolationMode.NEAREST
    layer.coordinate_transformation = trt.ResizeCoordinateTransformation.ASYMMETRIC
    layer.nearest_rounding = trt.ResizeRoundMode.FLOOR
    layer.shape = (batch, channels, height * 2, width * 2)
    return layer.get_output(0)
