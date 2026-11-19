# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT network helpers with one precision policy for a strongly typed build.

Weights and activations use the build's work type (FP32 or BF16). Normalisation statistics,
softmax, SiLU, GELU and attention scores are computed in FP32 and rounded once to the work type,
which is how the reference's BF16 kernels round them.
"""

from __future__ import annotations

import ml_dtypes
import numpy as np
import tensorrt as trt

_FLOAT_TYPES = (trt.float32, trt.bfloat16)
_MASKED = -1.0e30


class Graph:
    """One network, its work type and the host buffers its constants reference until the build ends."""

    def __init__(self, network: trt.INetworkDefinition, work: trt.DataType):
        if work not in _FLOAT_TYPES:
            raise ValueError(f"granite_speech builds FP32 or BF16, not {work}")
        self.network = network
        self.work = work
        self._keep: list = []

    # ---- constants -----------------------------------------------------------------------
    def weights(self, values, dtype: trt.DataType) -> trt.Weights:
        """``trt.Weights`` for host values; the buffer stays alive with this graph."""
        if dtype == trt.bfloat16:
            array = np.array(np.asarray(values, dtype=np.float32), dtype=ml_dtypes.bfloat16, order="C")
            weights = trt.Weights(trt.bfloat16, array.ctypes.data, array.size)
        else:
            numpy_type = {trt.float32: np.float32, trt.int32: np.int32, trt.int64: np.int64}[dtype]
            # np.array keeps a 0-d value 0-d; np.ascontiguousarray would make it rank 1.
            array = np.array(values, dtype=numpy_type, order="C")
            weights = trt.Weights(array)
        self._keep.append(array)
        return weights

    def constant(self, values, dtype: trt.DataType | None = None, shape=None) -> trt.ITensor:
        dtype = self.work if dtype is None else dtype
        values = np.asarray(values)
        shape = tuple(values.shape) if shape is None else tuple(int(v) for v in shape)
        if int(np.prod(shape, dtype=np.int64)) != values.size:
            raise ValueError(f"constant of {values.size} values cannot take shape {shape}")
        return self.network.add_constant(shape, self.weights(values.reshape(shape), dtype)).get_output(0)

    def scalar(self, value: float, dtype: trt.DataType, rank: int) -> trt.ITensor:
        return self.constant(np.full((1,) * rank, value), dtype)

    def i64(self, values) -> trt.ITensor:
        return self.constant(np.asarray(values, dtype=np.int64).reshape(-1), trt.int64)

    # ---- element-wise ------------------------------------------------------------------------
    def cast(self, tensor: trt.ITensor, dtype: trt.DataType) -> trt.ITensor:
        if tensor.dtype == dtype:
            return tensor
        return self.network.add_cast(tensor, dtype).get_output(0)

    def binary(self, left, right, op: trt.ElementWiseOperation) -> trt.ITensor:
        return self.network.add_elementwise(left, right, op).get_output(0)

    def add(self, left, right):
        return self.binary(left, right, trt.ElementWiseOperation.SUM)

    def sub(self, left, right):
        return self.binary(left, right, trt.ElementWiseOperation.SUB)

    def mul(self, left, right):
        return self.binary(left, right, trt.ElementWiseOperation.PROD)

    def div(self, left, right):
        return self.binary(left, right, trt.ElementWiseOperation.DIV)

    def maximum(self, left, right):
        return self.binary(left, right, trt.ElementWiseOperation.MAX)

    def scale(self, tensor: trt.ITensor, factor: float) -> trt.ITensor:
        """``tensor * factor`` in the tensor's own type."""
        return self.mul(tensor, self.scalar(factor, tensor.dtype, len(tensor.shape)))

    def unary(self, tensor, op: trt.UnaryOperation) -> trt.ITensor:
        return self.network.add_unary(tensor, op).get_output(0)

    def activation(self, tensor, kind: trt.ActivationType) -> trt.ITensor:
        return self.network.add_activation(tensor, kind).get_output(0)

    def silu(self, tensor: trt.ITensor) -> trt.ITensor:
        source = tensor.dtype
        wide = self.cast(tensor, trt.float32)
        return self.cast(self.mul(wide, self.activation(wide, trt.ActivationType.SIGMOID)), source)

    def sigmoid(self, tensor: trt.ITensor) -> trt.ITensor:
        source = tensor.dtype
        return self.cast(self.activation(self.cast(tensor, trt.float32), trt.ActivationType.SIGMOID), source)

    def gelu(self, tensor: trt.ITensor) -> trt.ITensor:
        """Exact (erf) GELU, the reference's ``gelu``."""
        source = tensor.dtype
        return self.cast(self.activation(self.cast(tensor, trt.float32), trt.ActivationType.GELU_ERF), source)

    def softmax(self, tensor: trt.ITensor, axis: int) -> trt.ITensor:
        """Softmax over ``axis`` in FP32, returned in the input's type."""
        source = tensor.dtype
        layer = self.network.add_softmax(self.cast(tensor, trt.float32))
        layer.axes = 1 << axis
        return self.cast(layer.get_output(0), source)

    def select(self, condition, then_value, else_value) -> trt.ITensor:
        return self.network.add_select(condition, then_value, else_value).get_output(0)

    # ---- linear algebra --------------------------------------------------------------------
    def matmul(self, left, right, *, transpose_right: bool = False) -> trt.ITensor:
        op = trt.MatrixOperation.TRANSPOSE if transpose_right else trt.MatrixOperation.NONE
        return self.network.add_matrix_multiply(left, trt.MatrixOperation.NONE, right, op).get_output(0)

    def linear(self, tensor: trt.ITensor, weight: np.ndarray, bias: np.ndarray | None = None) -> trt.ITensor:
        """``tensor @ weight.T + bias`` with torch.nn.Linear's ``[out, in]`` weight layout."""
        rank = len(tensor.shape)
        weight = np.asarray(weight)
        kernel = self.constant(weight, tensor.dtype, (1,) * (rank - 2) + tuple(weight.shape))
        out = self.matmul(tensor, kernel, transpose_right=True)
        if bias is None:
            return out
        bias = np.asarray(bias)
        return self.add(out, self.constant(bias, tensor.dtype, (1,) * (rank - 1) + (bias.shape[0],)))

    def layer_norm(self, tensor, gamma, beta, eps: float) -> trt.ITensor:
        """LayerNorm over the last axis, statistics and affine in FP32."""
        source = tensor.dtype
        rank = len(tensor.shape)
        shape = (1,) * (rank - 1) + (int(np.asarray(gamma).shape[0]),)
        layer = self.network.add_normalization(
            self.cast(tensor, trt.float32), self.constant(gamma, trt.float32, shape),
            self.constant(beta, trt.float32, shape), 1 << (rank - 1))
        layer.epsilon = float(eps)
        return self.cast(layer.get_output(0), source)

    def rms_norm(self, tensor, gamma, eps: float) -> trt.ITensor:
        """Granite RMSNorm: normalise in FP32, round to the work type, then scale by the weight."""
        source = tensor.dtype
        rank = len(tensor.shape)
        wide = self.cast(tensor, trt.float32)
        mean = self.network.add_reduce(self.mul(wide, wide), trt.ReduceOperation.AVG, 1 << (rank - 1),
                                       True).get_output(0)
        root = self.unary(self.add(mean, self.scalar(eps, trt.float32, rank)), trt.UnaryOperation.SQRT)
        normed = self.cast(self.div(wide, root), source)
        width = int(np.asarray(gamma).shape[0])
        return self.mul(normed, self.constant(gamma, source, (1,) * (rank - 1) + (width,)))

    # ---- shapes ----------------------------------------------------------------------------
    def reshape(self, tensor, dims) -> trt.ITensor:
        layer = self.network.add_shuffle(tensor)
        layer.zero_is_placeholder = False
        layer.reshape_dims = tuple(int(v) for v in dims)
        return layer.get_output(0)

    def reshape_to(self, tensor, shape: trt.ITensor) -> trt.ITensor:
        layer = self.network.add_shuffle(tensor)
        layer.zero_is_placeholder = False
        layer.set_input(1, shape)
        return layer.get_output(0)

    def transpose(self, tensor, order) -> trt.ITensor:
        layer = self.network.add_shuffle(tensor)
        layer.first_transpose = tuple(int(v) for v in order)
        return layer.get_output(0)

    def shape(self, tensor) -> trt.ITensor:
        return self.network.add_shape(tensor).get_output(0)

    def dim(self, tensor, axis: int) -> trt.ITensor:
        """One dimension of ``tensor`` as a 1-element int64 shape tensor."""
        return self.network.add_gather(self.shape(tensor), self.i64([axis]), 0).get_output(0)

    def concat(self, tensors, axis: int) -> trt.ITensor:
        layer = self.network.add_concatenation(list(tensors))
        layer.axis = axis
        return layer.get_output(0)

    def slice(self, tensor, start, size, *, fill: bool = False) -> trt.ITensor:
        """Slice with static or shape-tensor ``start``/``size``; ``fill`` pads out of range with zeros."""
        rank = len(tensor.shape)
        static_start = start if isinstance(start, (tuple, list)) else (0,) * rank
        static_size = size if isinstance(size, (tuple, list)) else (1,) * rank
        layer = self.network.add_slice(tensor, tuple(static_start), tuple(static_size), (1,) * rank)
        if not isinstance(start, (tuple, list)):
            layer.set_input(1, start)
        if not isinstance(size, (tuple, list)):
            layer.set_input(2, size)
        if fill:
            layer.mode = trt.SampleMode.FILL
        return layer.get_output(0)

    def arange(self, length: int, size: trt.ITensor) -> trt.ITensor:
        """``0 .. size-1`` as int32, sliced from a constant of the profile's longest length."""
        return self.slice(self.constant(np.arange(length, dtype=np.int32), trt.int32), (0,), size)

    def masked_scores(self, valid: trt.ITensor) -> trt.ITensor:
        """0 where ``valid`` else a large negative score, FP32, shaped like ``valid``."""
        rank = len(valid.shape)
        return self.select(valid, self.scalar(0.0, trt.float32, rank), self.scalar(_MASKED, trt.float32, rank))
