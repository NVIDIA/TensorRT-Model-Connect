# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned strongly typed TensorRT graph helpers for flux3.

Activations are kept two-dimensional (``[rows, features]``) wherever possible
so that linear layers are plain matrix products against ``[out, in]`` weight
constants, with dynamic row counts for the text window.
"""

from __future__ import annotations

import math

import ml_dtypes
import numpy as np
import tensorrt as trt

_NUMPY = {
    trt.float32: np.dtype(np.float32),
    trt.bfloat16: np.dtype(ml_dtypes.bfloat16),
    trt.int32: np.dtype(np.int32),
    trt.int64: np.dtype(np.int64),
}


def new_builder_config(builder: trt.Builder, optimization_level: int) -> trt.IBuilderConfig:
    """Builder config shared by the flux3 engines: TF32 off, explicit optimization level."""

    config = builder.create_builder_config()
    config.builder_optimization_level = optimization_level
    config.clear_flag(trt.BuilderFlag.TF32)
    return config


def new_network(builder: trt.Builder) -> trt.INetworkDefinition:
    return builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))


def serialize(builder: trt.Builder, network: trt.INetworkDefinition, config) -> bytes:
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("flux3: TensorRT engine build failed")
    return bytes(plan)


class Graph:
    """Thin helper over one strongly typed network; keeps weight buffers alive."""

    def __init__(self, network: trt.INetworkDefinition, compute: trt.DataType) -> None:
        if compute not in (trt.bfloat16, trt.float32):
            raise ValueError("flux3 graphs compute in bfloat16 or float32")
        self.net = network
        self.compute = compute
        self._refs: list[np.ndarray] = []

    # -- constants and casts -------------------------------------------------

    def keep(self, *arrays: np.ndarray) -> None:
        """Keep host buffers referenced by ``trt.Weights`` alive until serialization."""

        self._refs.extend(arrays)

    def constant(self, array: np.ndarray, dtype: trt.DataType | None = None) -> trt.ITensor:
        if dtype is None:
            dtype = next(key for key, value in _NUMPY.items() if value == array.dtype)
        # np.array(order="C") keeps 0-d scalars 0-d (np.ascontiguousarray would make them 1-d).
        data = np.array(np.asarray(array).astype(_NUMPY[dtype], copy=False), order="C", copy=True)
        self._refs.append(data)
        weights = trt.Weights(dtype, data.ctypes.data, data.size)
        return self.net.add_constant(trt.Dims(data.shape), weights).get_output(0)

    def scalar(self, value: float, dtype: trt.DataType, rank: int) -> trt.ITensor:
        return self.constant(np.full((1,) * rank, value, dtype=np.float32), dtype)

    def cast(self, tensor: trt.ITensor, dtype: trt.DataType) -> trt.ITensor:
        if tensor.dtype == dtype:
            return tensor
        return self.net.add_cast(tensor, dtype).get_output(0)

    # -- elementwise -----------------------------------------------------------

    def _binary(self, a: trt.ITensor, b: trt.ITensor, op: trt.ElementWiseOperation) -> trt.ITensor:
        return self.net.add_elementwise(a, b, op).get_output(0)

    def add(self, a: trt.ITensor, b: trt.ITensor) -> trt.ITensor:
        return self._binary(a, b, trt.ElementWiseOperation.SUM)

    def sub(self, a: trt.ITensor, b: trt.ITensor) -> trt.ITensor:
        return self._binary(a, b, trt.ElementWiseOperation.SUB)

    def mul(self, a: trt.ITensor, b: trt.ITensor) -> trt.ITensor:
        return self._binary(a, b, trt.ElementWiseOperation.PROD)

    def silu(self, x: trt.ITensor) -> trt.ITensor:
        sigmoid = self.net.add_activation(x, trt.ActivationType.SIGMOID).get_output(0)
        return self.mul(x, sigmoid)

    # -- shapes ----------------------------------------------------------------

    def reshape(self, x: trt.ITensor, dims: tuple[int, ...]) -> trt.ITensor:
        layer = self.net.add_shuffle(x)
        layer.reshape_dims = trt.Dims(dims)
        return layer.get_output(0)

    def permute(self, x: trt.ITensor, order: tuple[int, ...]) -> trt.ITensor:
        layer = self.net.add_shuffle(x)
        layer.first_transpose = trt.Permutation(order)
        return layer.get_output(0)

    def concat(self, tensors: list[trt.ITensor], axis: int) -> trt.ITensor:
        layer = self.net.add_concatenation(tensors)
        layer.axis = axis
        return layer.get_output(0)

    def dim(self, x: trt.ITensor, index: int) -> trt.ITensor:
        """One dimension of ``x`` as a 1-element int64 shape tensor."""

        shape = self.net.add_shape(x).get_output(0)
        pick = self.constant(np.array([index], dtype=np.int32))
        return self.net.add_gather(shape, pick, 0).get_output(0)

    def iota(self, length: trt.ITensor, dtype: trt.DataType = trt.int32) -> trt.ITensor:
        """``[0, 1, ..., length - 1]`` for a 1-element shape tensor ``length``."""

        layer = self.net.add_fill(trt.Dims([1]), trt.FillOperation.LINSPACE, dtype)
        layer.set_input(0, length)
        layer.set_input(1, self.constant(np.array(0, dtype=_NUMPY[dtype]).reshape(()), dtype))
        layer.set_input(2, self.constant(np.array([1], dtype=_NUMPY[dtype]), dtype))
        return layer.get_output(0)

    def gather(self, table: trt.ITensor, indices: trt.ITensor, axis: int) -> trt.ITensor:
        return self.net.add_gather(table, indices, axis).get_output(0)

    def take_last(self, x: trt.ITensor, indices: np.ndarray) -> trt.ITensor:
        """Gather a fixed index list along the last axis."""

        axis = len(x.shape) - 1
        return self.gather(x, self.constant(indices.astype(np.int32)), axis)

    def chunk_last(self, x: trt.ITensor, parts: int) -> list[trt.ITensor]:
        """Split a ``[1, parts * width]`` row into ``parts`` ``[1, width]`` rows."""

        rows, width = x.shape
        size = width // parts
        return [
            self.net.add_slice(x, (0, i * size), (rows, size), (1, 1)).get_output(0)
            for i in range(parts)
        ]

    # -- layers ----------------------------------------------------------------

    def linear(self, x: trt.ITensor, weight: np.ndarray, dtype: trt.DataType | None = None):
        """``x @ weight.T`` for a ``[rows, in]`` input and an ``[out, in]`` weight."""

        dtype = dtype or self.compute
        w = self.constant(weight, dtype)
        return self.net.add_matrix_multiply(
            self.cast(x, dtype), trt.MatrixOperation.NONE, w, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)

    def layer_norm(self, x: trt.ITensor, eps: float) -> trt.ITensor:
        """LayerNorm over the last axis without affine; statistics in float32."""

        dtype = x.dtype
        x32 = self.cast(x, trt.float32)
        rank = len(x.shape)
        width = x.shape[-1]
        ones = self.constant(np.ones((1,) * (rank - 1) + (width,), dtype=np.float32))
        zeros = self.constant(np.zeros((1,) * (rank - 1) + (width,), dtype=np.float32))
        layer = self.net.add_normalization(x32, ones, zeros, 1 << (rank - 1))
        layer.epsilon = eps
        return self.cast(layer.get_output(0), dtype)

    def rms_norm32(self, x: trt.ITensor, weight: np.ndarray, eps: float) -> trt.ITensor:
        """RMSNorm over the last axis in float32; returns float32."""

        x32 = self.cast(x, trt.float32)
        rank = len(x.shape)
        squares = self.mul(x32, x32)
        mean = self.net.add_reduce(
            squares, trt.ReduceOperation.AVG, 1 << (rank - 1), keep_dims=True
        ).get_output(0)
        denom = self.net.add_unary(
            self.add(mean, self.scalar(eps, trt.float32, rank)), trt.UnaryOperation.SQRT
        ).get_output(0)
        normed = self.net.add_elementwise(x32, denom, trt.ElementWiseOperation.DIV).get_output(0)
        shape = (1,) * (rank - 1) + (weight.shape[-1],)
        return self.mul(normed, self.constant(weight.astype(np.float32).reshape(shape)))

    def attention(
        self,
        q: trt.ITensor,
        k: trt.ITensor,
        v: trt.ITensor,
        mask: trt.ITensor | None = None,
    ) -> trt.ITensor:
        """Softmax attention over ``[1, heads, rows, head_dim]`` inputs; ``q`` is pre-scaled."""

        layer = self.net.add_attention(q, k, v, trt.AttentionNormalizationOp.SOFTMAX, False)
        if layer is None:
            raise RuntimeError("flux3: TensorRT rejected the attention layer")
        layer.decomposable = True
        if mask is not None:
            layer.mask = mask
        return layer.get_output(0)

    def heads(self, x: trt.ITensor, num_heads: int, head_dim: int) -> trt.ITensor:
        """``[rows, heads * head_dim]`` → ``[rows, heads, head_dim]``."""

        return self.reshape(x, (-1, num_heads, head_dim))

    def to_attention_layout(self, x: trt.ITensor) -> trt.ITensor:
        """``[rows, heads, head_dim]`` → ``[1, heads, rows, head_dim]``."""

        heads, head_dim = x.shape[1], x.shape[2]
        layer = self.net.add_shuffle(x)
        layer.first_transpose = trt.Permutation((1, 0, 2))
        layer.reshape_dims = trt.Dims((1, heads, -1, head_dim))
        return layer.get_output(0)

    def from_attention_layout(self, x: trt.ITensor, width: int) -> trt.ITensor:
        """``[1, heads, rows, head_dim]`` → ``[rows, heads * head_dim]``."""

        layer = self.net.add_shuffle(x)
        layer.first_transpose = trt.Permutation((0, 2, 1, 3))
        layer.reshape_dims = trt.Dims((-1, width))
        return layer.get_output(0)

    def scale_query(self, q32: trt.ITensor, head_dim: int, dtype: trt.DataType) -> trt.ITensor:
        """Apply ``1 / sqrt(head_dim)`` in float32, then cast to the attention dtype."""

        rank = len(q32.shape)
        q32 = self.cast(q32, trt.float32)
        scaled = self.mul(q32, self.scalar(1.0 / math.sqrt(head_dim), trt.float32, rank))
        return self.cast(scaled, dtype)
