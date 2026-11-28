# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT network helpers for a strongly typed build with one precision policy.

Matrix products run in the work type (FP32 or BF16), with their bias accumulated inside the product.
Everything between them is computed in FP32: normalisation, activations, the gated linear unit, the
depthwise convolution with its batch norm, attention scores, softmax and its product, the residual
stream and the logits. A BF16 engine therefore rounds only matrix-product operands and outputs, as
the reference's BF16 kernels do.
"""

from __future__ import annotations

import ml_dtypes
import numpy as np
import tensorrt as trt

_FLOAT_TYPES = (trt.float32, trt.bfloat16)
# Columns appended to fold a bias into a product: one of ones, then zeros.
_FOLD = 8
# ``linear(..., bias=FOLDED)``: the weight is a :meth:`Graph.folded` constant that already holds the bias.
FOLDED = object()


class Graph:
    """One network, its work type and the host buffers its constants reference until the build ends."""

    def __init__(self, network: trt.INetworkDefinition, work: trt.DataType):
        if work not in _FLOAT_TYPES:
            raise ValueError(f"granite_speech5 builds FP32 or BF16, not {work}")
        self.network = network
        self.work = work
        self._keep: list = []

    # ---- constants -----------------------------------------------------------------------
    def weights(self, values, dtype: trt.DataType) -> trt.Weights:
        """``trt.Weights`` for host values; the buffer stays alive with this graph."""
        if dtype == trt.bfloat16:
            with np.errstate(over="ignore"):
                array = np.array(np.asarray(values, dtype=np.float32), dtype=ml_dtypes.bfloat16, order="C")
            # A finite FP32 value above the BF16 range rounds to infinity: refuse it instead of building it.
            if not np.all(np.isfinite(array.astype(np.float32))):
                raise ValueError("granite_speech5 has a constant that BF16 cannot hold (beyond its range)")
            weights = trt.Weights(trt.bfloat16, array.ctypes.data, array.size)
        else:
            numpy_type = {trt.float32: np.float32, trt.int32: np.int32, trt.int64: np.int64}[dtype]
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

    def wide(self, tensor: trt.ITensor) -> trt.ITensor:
        return self.cast(tensor, trt.float32)

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

    def sigmoid32(self, tensor: trt.ITensor) -> trt.ITensor:
        return self.network.add_activation(self.wide(tensor), trt.ActivationType.SIGMOID).get_output(0)

    def silu32(self, tensor: trt.ITensor) -> trt.ITensor:
        """SiLU in FP32 (``x * sigmoid(x)``), FP32 out."""
        wide = self.wide(tensor)
        return self.mul(wide, self.sigmoid32(wide))

    def softmax32(self, tensor: trt.ITensor, axis: int) -> trt.ITensor:
        layer = self.network.add_softmax(self.wide(tensor))
        layer.axes = 1 << axis
        return layer.get_output(0)

    def select(self, condition, then_value, else_value) -> trt.ITensor:
        return self.network.add_select(condition, then_value, else_value).get_output(0)

    def reduce(self, tensor, op: trt.ReduceOperation, axis: int, keep: bool) -> trt.ITensor:
        return self.network.add_reduce(tensor, op, 1 << axis, keep).get_output(0)

    # ---- linear algebra --------------------------------------------------------------------
    def matmul(self, left, right, *, transpose_right: bool = False) -> trt.ITensor:
        op = trt.MatrixOperation.TRANSPOSE if transpose_right else trt.MatrixOperation.NONE
        return self.network.add_matrix_multiply(left, trt.MatrixOperation.NONE, right, op).get_output(0)

    def folded(self, weight, bias, dtype: trt.DataType) -> trt.ITensor:
        """``[out, in + 8]`` constant: the weight, then the bias as the column that meets :meth:`with_ones`' ones."""
        weight = np.asarray(weight, dtype=np.float32)
        kernel = np.zeros((weight.shape[0], weight.shape[1] + _FOLD), dtype=np.float32)
        kernel[:, : weight.shape[1]] = weight
        kernel[:, weight.shape[1]] = np.asarray(bias, dtype=np.float32)
        return self.constant(kernel, dtype)

    def with_ones(self, tensor: trt.ITensor) -> trt.ITensor:
        """``[rows, in]`` -> ``[rows, in + 8]``: a column of ones, then zeros (the width stays a multiple of 8)."""
        if len(tensor.shape) != 2:
            raise ValueError("granite_speech5 folds a bias into rank-2 products only")
        pattern = np.zeros((1, _FOLD), dtype=np.float32)
        pattern[0, 0] = 1.0
        ones = self.rows(self.constant(pattern, tensor.dtype), self.dim(tensor, 0), mode=trt.SampleMode.CLAMP)
        return self.concat([tensor, ones], 1)

    def linear(self, tensor: trt.ITensor, weight, bias=None) -> trt.ITensor:
        """``tensor @ weight.T + bias`` in the tensor's type, with torch.nn.Linear's ``[out, in]`` layout.

        A bias is accumulated with the product and rounded once, as the reference adds it in the product's
        epilogue: the input gets a column of ones and the weight the bias as its matching column
        (:meth:`folded`). A bias added after a rounded BF16 product would lose cancellation residuals.
        ``weight`` is a host array, or with ``bias=FOLDED`` a constant made by :meth:`folded`.
        """
        if bias is None:
            weight = np.asarray(weight)
            kernel = self.constant(weight, tensor.dtype, (1,) * (len(tensor.shape) - 2) + tuple(weight.shape))
            return self.matmul(tensor, kernel, transpose_right=True)
        kernel = weight if bias is FOLDED else self.folded(weight, bias, tensor.dtype)
        return self.matmul(self.with_ones(tensor), kernel, transpose_right=True)

    def layer_norm(self, tensor, gamma, beta, eps: float, out: trt.DataType) -> trt.ITensor:
        """LayerNorm over the last axis with statistics and affine in FP32, rounded once to ``out``."""
        rank = len(tensor.shape)
        shape = (1,) * (rank - 1) + (int(np.asarray(gamma).shape[0]),)
        layer = self.network.add_normalization(
            self.wide(tensor), self.constant(gamma, trt.float32, shape),
            self.constant(beta, trt.float32, shape), 1 << (rank - 1))
        layer.epsilon = float(eps)
        return self.cast(layer.get_output(0), out)

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

    def slice(self, tensor, start, size, *, mode: trt.SampleMode | None = None) -> trt.ITensor:
        """Slice with static or shape-tensor ``start`` / ``size``; ``mode`` FILL pads zeros, CLAMP repeats edges."""
        rank = len(tensor.shape)
        static_start = start if isinstance(start, (tuple, list)) else (0,) * rank
        static_size = size if isinstance(size, (tuple, list)) else (1,) * rank
        layer = self.network.add_slice(tensor, tuple(static_start), tuple(static_size), (1,) * rank)
        if not isinstance(start, (tuple, list)):
            layer.set_input(1, start)
        if not isinstance(size, (tuple, list)):
            layer.set_input(2, size)
        if mode is not None:
            layer.mode = mode
        return layer.get_output(0)

    def rows(self, tensor, count: trt.ITensor, *, offset: int = 0, mode: trt.SampleMode | None = None):
        """Rows ``offset .. offset + count`` of a rank-2 tensor (``count`` a 1-element shape tensor)."""
        width = int(tensor.shape[1])
        return self.slice(tensor, (offset, 0), self.concat([count, self.i64([width])], 0), mode=mode)

    def arange(self, length: int, size: trt.ITensor) -> trt.ITensor:
        """``0 .. size-1`` as int32, sliced from a constant of the profile's longest length."""
        return self.slice(self.constant(np.arange(length, dtype=np.int32), trt.int32), (0,), size)

    def floor_div(self, value: trt.ITensor, divisor: int) -> trt.ITensor:
        return self.binary(value, self.i64([divisor]), trt.ElementWiseOperation.FLOOR_DIV)
