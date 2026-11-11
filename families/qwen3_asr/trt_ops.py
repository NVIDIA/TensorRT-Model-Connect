# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned TensorRT graph operations for the Qwen3-ASR engines.

Every engine is a strongly typed network with FP32 or int32 inputs and FP32 logits / embeddings.
Linear layers, convolutions, residual sums and the attention-value product run in the graph's
compute type: BF16, the checkpoint's own precision. Normalisation statistics and affine terms,
activation functions, attention scores and softmax run in FP32 and are rounded once, as the
reference's BF16 kernels do.

A graph built with FP32 as its compute type is the same network without BF16 rounding; the
family's parity tests and the FP32 bundle use it to show the graphs match the reference to FP32
accuracy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import ml_dtypes
import numpy as np
import tensorrt as trt

BF16 = trt.bfloat16
FP32 = trt.float32

# Excluded attention keys get this FP32 score; finite, so a fully masked row cannot become NaN.
MASKED_SCORE = -1.0e30


@dataclass
class Graph:
    """A TensorRT network plus the host arrays its constants point into."""

    builder: trt.Builder
    network: trt.INetworkDefinition
    config: trt.IBuilderConfig
    compute: trt.DataType = BF16
    # TensorRT reads constant storage through raw pointers until the plan is serialised.
    anchors: list[np.ndarray] = field(default_factory=list)


def create_graph(
    *, verbose: bool, workspace_bytes: int, optimization_level: int, compute: trt.DataType
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
        raise RuntimeError(f"TensorRT could not build the Qwen3-ASR {label} engine")
    return plan


# ---------------------------------------------------------------------------
# Constants and casts
# ---------------------------------------------------------------------------


def constant(graph: Graph, values: np.ndarray) -> trt.ITensor:
    """Add a constant in the array's own NumPy dtype (float32, int32, int64 or bool).

    The shape is taken before ``ascontiguousarray``, which turns a rank-0 scalar into rank 1.
    """
    shape = np.shape(values)
    array = np.ascontiguousarray(values)
    if array.dtype not in (np.float32, np.int32, np.int64, np.bool_):
        raise TypeError(f"unsupported constant dtype {array.dtype}")
    graph.anchors.append(array)
    return graph.network.add_constant(shape, trt.Weights(array)).get_output(0)


def compute_array(graph: Graph, values: np.ndarray) -> np.ndarray:
    """Return ``values`` as a contiguous array of the graph's compute type.

    BF16 checkpoint tensors pass through without rounding; FP32 graphs widen them exactly.
    """
    array = np.asarray(values)
    if graph.compute == BF16:
        if array.dtype != ml_dtypes.bfloat16:
            array = array.astype(np.float32).astype(ml_dtypes.bfloat16)
        return np.ascontiguousarray(array)
    return np.ascontiguousarray(array.astype(np.float32))


def compute_weights(graph: Graph, values: np.ndarray) -> trt.Weights:
    array = compute_array(graph, values)
    graph.anchors.append(array)
    return trt.Weights(graph.compute, array.ctypes.data, array.size)


def constant_compute(graph: Graph, values: np.ndarray) -> trt.ITensor:
    """Add a constant of the graph's compute type."""
    array = compute_array(graph, values)
    weights = compute_weights(graph, array)
    return graph.network.add_constant(array.shape, weights).get_output(0)


def constant_fp32(graph: Graph, values: np.ndarray) -> trt.ITensor:
    return constant(graph, np.asarray(values).astype(np.float32))


def cast(graph: Graph, tensor: trt.ITensor, dtype: trt.DataType) -> trt.ITensor:
    if tensor.dtype == dtype:
        return tensor
    return graph.network.add_cast(tensor, dtype).get_output(0)


def scalar_fp32(graph: Graph, value: float, rank: int) -> trt.ITensor:
    return constant(graph, np.full((1,) * rank, value, dtype=np.float32))


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


def reshape(graph: Graph, tensor: trt.ITensor, dims: tuple[int, ...]) -> trt.ITensor:
    """Reshape with TensorRT semantics: ``0`` copies the input dimension, ``-1`` is inferred."""
    layer = graph.network.add_shuffle(tensor)
    layer.reshape_dims = dims
    return layer.get_output(0)


def reshape_to(graph: Graph, tensor: trt.ITensor, shape: trt.ITensor) -> trt.ITensor:
    """Reshape to a runtime shape tensor."""
    layer = graph.network.add_shuffle(tensor)
    layer.set_input(1, shape)
    return layer.get_output(0)


def transpose(graph: Graph, tensor: trt.ITensor, permutation: tuple[int, ...]) -> trt.ITensor:
    layer = graph.network.add_shuffle(tensor)
    layer.first_transpose = trt.Permutation(list(permutation))
    return layer.get_output(0)


def shape_of(graph: Graph, tensor: trt.ITensor) -> trt.ITensor:
    """Runtime shape of ``tensor`` as an int64 shape tensor."""
    return graph.network.add_shape(tensor).get_output(0)


def dimension(graph: Graph, tensor: trt.ITensor, axis: int) -> trt.ITensor:
    """Runtime extent of one axis as an int64 ``[1]`` shape tensor."""
    return graph.network.add_slice(
        shape_of(graph, tensor), start=(axis,), shape=(1,), stride=(1,)
    ).get_output(0)


def concat(graph: Graph, tensors: list[trt.ITensor], axis: int) -> trt.ITensor:
    layer = graph.network.add_concatenation(tensors)
    layer.axis = axis
    return layer.get_output(0)


def slice_static(
    graph: Graph, tensor: trt.ITensor, axis: int, start: int, length: int
) -> trt.ITensor:
    """Slice ``length`` elements of one axis; every other axis is kept (dynamic extents allowed)."""
    rank = len(tensor.shape)
    starts = [0] * rank
    starts[axis] = start
    layer = graph.network.add_slice(tensor, tuple(starts), (1,) * rank, (1,) * rank)
    sizes = []
    full = shape_of(graph, tensor)
    for index in range(rank):
        if index == axis:
            sizes.append(constant(graph, np.array([length], dtype=np.int64)))
        else:
            sizes.append(
                graph.network.add_slice(full, (index,), (1,), (1,)).get_output(0)
            )
    layer.set_input(2, concat(graph, sizes, 0))
    return layer.get_output(0)


def slice_dynamic(
    graph: Graph, tensor: trt.ITensor, axis: int, start: trt.ITensor | int, length: trt.ITensor
) -> trt.ITensor:
    """Slice ``length`` (int64 ``[1]`` shape tensor) elements of one axis from ``start``."""
    rank = len(tensor.shape)
    full = shape_of(graph, tensor)
    starts = []
    sizes = []
    for index in range(rank):
        if index == axis:
            if isinstance(start, int):
                starts.append(constant(graph, np.array([start], dtype=np.int64)))
            else:
                starts.append(start)
            sizes.append(length)
        else:
            starts.append(constant(graph, np.zeros((1,), dtype=np.int64)))
            sizes.append(graph.network.add_slice(full, (index,), (1,), (1,)).get_output(0))
    layer = graph.network.add_slice(tensor, (0,) * rank, (1,) * rank, (1,) * rank)
    layer.set_input(1, concat(graph, starts, 0))
    layer.set_input(2, concat(graph, sizes, 0))
    return layer.get_output(0)


def iota(graph: Graph, length: trt.ITensor, dtype: trt.DataType = trt.int32) -> trt.ITensor:
    """``0 .. length - 1`` for an int64 ``[1]`` shape tensor ``length``."""
    if dtype == trt.int32:
        start = constant(graph, np.array(0, dtype=np.int32))
        step = constant(graph, np.array([1], dtype=np.int32))
    else:
        start = constant(graph, np.array(0, dtype=np.int64))
        step = constant(graph, np.array([1], dtype=np.int64))
    layer = graph.network.add_fill((1,), trt.FillOperation.LINSPACE, dtype)
    layer.set_input(0, length)
    layer.set_input(1, start)
    layer.set_input(2, step)
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


def less(graph: Graph, left: trt.ITensor, right: trt.ITensor) -> trt.ITensor:
    return _elementwise(graph, left, right, trt.ElementWiseOperation.LESS)


def unary(graph: Graph, tensor: trt.ITensor, operation: trt.UnaryOperation) -> trt.ITensor:
    return graph.network.add_unary(tensor, operation).get_output(0)


def select(
    graph: Graph, condition: trt.ITensor, then: trt.ITensor, otherwise: trt.ITensor
) -> trt.ITensor:
    return graph.network.add_select(condition, then, otherwise).get_output(0)


def gelu(graph: Graph, tensor: trt.ITensor) -> trt.ITensor:
    """Exact GELU, ``0.5 x (1 + erf(x / sqrt(2)))``, evaluated in FP32 and rounded once."""
    x = cast(graph, tensor, FP32)
    rank = len(tensor.shape)
    inner = unary(graph, mul(graph, x, scalar_fp32(graph, 1.0 / math.sqrt(2.0), rank)),
                  trt.UnaryOperation.ERF)
    gate = add(graph, inner, scalar_fp32(graph, 1.0, rank))
    value = mul(graph, mul(graph, x, scalar_fp32(graph, 0.5, rank)), gate)
    return cast(graph, value, graph.compute)


def silu(graph: Graph, tensor: trt.ITensor) -> trt.ITensor:
    """``x * sigmoid(x)`` evaluated in FP32 and rounded once."""
    x = cast(graph, tensor, FP32)
    sigmoid = graph.network.add_activation(x, trt.ActivationType.SIGMOID).get_output(0)
    return cast(graph, mul(graph, x, sigmoid), graph.compute)


# ---------------------------------------------------------------------------
# Linear layers, convolutions and normalisation
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
        result = add(graph, result, constant_compute(graph, np.asarray(bias).reshape(*lead, 1, -1)))
    return result


def conv2d(
    graph: Graph, tensor: trt.ITensor, weight: np.ndarray, bias: np.ndarray, *, stride: int,
    padding: int,
) -> trt.ITensor:
    """2D convolution with a checkpoint kernel stored ``[out, in, kh, kw]``."""
    out_channels, _, kernel_h, kernel_w = weight.shape
    layer = graph.network.add_convolution_nd(
        cast(graph, tensor, graph.compute),
        num_output_maps=out_channels,
        kernel_shape=(kernel_h, kernel_w),
        kernel=compute_weights(graph, weight),
        bias=compute_weights(graph, bias),
    )
    layer.stride_nd = (stride, stride)
    layer.padding_nd = (padding, padding)
    return layer.get_output(0)


def rms_norm(graph: Graph, tensor: trt.ITensor, gamma: np.ndarray, eps: float) -> trt.ITensor:
    """Qwen3 RMSNorm: FP32 statistics, rounded to the compute type, then ``gamma`` in that type."""
    x = cast(graph, tensor, FP32)
    rank = len(tensor.shape)
    square = mul(graph, x, x)
    mean = graph.network.add_reduce(
        square, trt.ReduceOperation.AVG, 1 << (rank - 1), keep_dims=True
    ).get_output(0)
    root = unary(graph, add(graph, mean, scalar_fp32(graph, eps, rank)), trt.UnaryOperation.SQRT)
    normalized = mul(graph, x, unary(graph, root, trt.UnaryOperation.RECIP))
    shape = (1,) * (rank - 1) + (-1,)
    return mul(
        graph,
        cast(graph, normalized, graph.compute),
        constant_compute(graph, np.asarray(gamma).reshape(shape)),
    )


def layer_norm(
    graph: Graph, tensor: trt.ITensor, gamma: np.ndarray, beta: np.ndarray, eps: float
) -> trt.ITensor:
    """LayerNorm over the last axis; statistics and affine terms in FP32, rounded once."""
    x = cast(graph, tensor, FP32)
    rank = len(tensor.shape)
    shape = (1,) * (rank - 1) + (-1,)
    layer = graph.network.add_normalization(
        x,
        constant_fp32(graph, np.asarray(gamma).astype(np.float32).reshape(shape)),
        constant_fp32(graph, np.asarray(beta).astype(np.float32).reshape(shape)),
        1 << (rank - 1),
    )
    layer.epsilon = eps
    return cast(graph, layer.get_output(0), graph.compute)


def softmax_last(graph: Graph, scores: trt.ITensor) -> trt.ITensor:
    layer = graph.network.add_softmax(scores)
    layer.axes = 1 << (len(scores.shape) - 1)
    return layer.get_output(0)
