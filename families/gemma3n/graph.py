# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph primitives shared by the Gemma 3n vision and text networks.

The network is strongly typed: weights are stored in the build precision and
reductions (RMS statistics, softmax, magnitudes) are computed in FP32, then
rounded back to the build precision as the reference rounds module outputs.
"""

from __future__ import annotations

import ml_dtypes
import numpy as np
import tensorrt as trt

PRECISIONS = {"fp32": trt.float32, "bf16": trt.bfloat16}
_ELEMENTWISE = trt.ElementWiseOperation


class Builder:
    """Typed TensorRT helper bound to one network and one build precision."""

    def __init__(self, network, precision: str, checkpoint):
        if precision not in PRECISIONS:
            raise ValueError(f"Gemma 3n precision must be one of {sorted(PRECISIONS)}")
        self.net = network
        self.precision = precision
        self.dtype = PRECISIONS[precision]
        self.checkpoint = checkpoint
        # TensorRT reads constant weights during serialization; keep them alive.
        self._storage = []

    # --- weights ---------------------------------------------------------
    def _weights(self, values: np.ndarray, dtype) -> trt.Weights:
        if dtype == trt.float32:
            array = np.ascontiguousarray(values, dtype=np.float32)
            self._storage.append(array)
            return trt.Weights(array)
        if dtype != trt.bfloat16:
            raise ValueError(f"unsupported Gemma 3n weight type {dtype}")
        # BF16 checkpoint values pass through bit-exactly; wider values round
        # to nearest even, as torch's .to(bfloat16) does.
        array = np.ascontiguousarray(np.asarray(values).astype(ml_dtypes.bfloat16)).view(np.uint16)
        self._storage.append(array)
        return trt.Weights(trt.DataType.BF16, array.ctypes.data, array.size)

    def weights(self, values: np.ndarray, dtype=None) -> trt.Weights:
        return self._weights(values, self.dtype if dtype is None else dtype)

    def constant(self, values: np.ndarray, dtype=None):
        values = np.asarray(values)
        dtype = self.dtype if dtype is None else dtype
        if dtype in (trt.int32, trt.int64):
            array = np.ascontiguousarray(values, dtype=np.int32 if dtype == trt.int32 else np.int64)
            self._storage.append(array)
            layer = self.net.add_constant(array.shape, trt.Weights(array))
        else:
            layer = self.net.add_constant(values.shape, self._weights(values, dtype))
        return layer.get_output(0)

    def scalar(self, value: float, rank: int, dtype=None):
        return self.constant(np.full((1,) * rank, value, np.float32), dtype)

    def tensor(self, name: str, dtype=None, shape=None):
        """A checkpoint tensor as a constant in the build (or given) precision."""
        values = self.checkpoint.get(name)
        if shape is not None and tuple(values.shape) != tuple(shape):
            raise ValueError(f"Gemma 3n tensor {name} has shape {values.shape}, expected {shape}")
        return self.constant(values, dtype)

    # --- shape and type ----------------------------------------------------
    def cast(self, tensor, dtype):
        if tensor.dtype == dtype:
            return tensor
        return self.net.add_cast(tensor, dtype).get_output(0)

    def reshape(self, tensor, shape):
        layer = self.net.add_shuffle(tensor)
        layer.reshape_dims = tuple(shape)
        return layer.get_output(0)

    def transpose(self, tensor, permutation):
        layer = self.net.add_shuffle(tensor)
        layer.first_transpose = tuple(permutation)
        return layer.get_output(0)

    def concat(self, tensors, axis: int):
        layer = self.net.add_concatenation(list(tensors))
        layer.axis = axis
        return layer.get_output(0)

    def gather(self, data, indices, axis: int):
        return self.net.add_gather(data, indices, axis).get_output(0)

    # --- arithmetic --------------------------------------------------------
    def op(self, lhs, rhs, operation):
        return self.net.add_elementwise(lhs, rhs, operation).get_output(0)

    def add(self, lhs, rhs):
        return self.op(lhs, rhs, _ELEMENTWISE.SUM)

    def sub(self, lhs, rhs):
        return self.op(lhs, rhs, _ELEMENTWISE.SUB)

    def mul(self, lhs, rhs):
        return self.op(lhs, rhs, _ELEMENTWISE.PROD)

    def div(self, lhs, rhs):
        return self.op(lhs, rhs, _ELEMENTWISE.DIV)

    def scale(self, tensor, factor: float):
        """Multiply by a scalar first rounded to the tensor's own type.

        This is the reference's ``tensor * buffer.to(tensor.dtype)``.
        """
        return self.mul(tensor, self.scalar(factor, len(tensor.shape), tensor.dtype))

    def scale_fp32(self, tensor, factor: float):
        """Multiply by an FP32 scalar and round once, as ``tensor * python_float`` does."""
        product = self.mul(self.cast(tensor, trt.float32),
                           self.scalar(factor, len(tensor.shape), trt.float32))
        return self.cast(product, tensor.dtype)

    def divide_fp32(self, tensor, divisor: float):
        """Divide by an FP32 scalar and round once, as ``tensor / python_float`` does."""
        quotient = self.div(self.cast(tensor, trt.float32),
                            self.scalar(divisor, len(tensor.shape), trt.float32))
        return self.cast(quotient, tensor.dtype)

    def unary(self, tensor, operation):
        return self.net.add_unary(tensor, operation).get_output(0)

    def activation(self, tensor, kind):
        return self.net.add_activation(tensor, kind).get_output(0)

    def mean(self, tensor, axis: int):
        axis %= len(tensor.shape)
        return self.net.add_reduce(tensor, trt.ReduceOperation.AVG, 1 << axis, True).get_output(0)

    def matmul(self, lhs, rhs, transpose_rhs: bool = False):
        return self.net.add_matrix_multiply(
            lhs, trt.MatrixOperation.NONE, rhs,
            trt.MatrixOperation.TRANSPOSE if transpose_rhs else trt.MatrixOperation.NONE,
        ).get_output(0)

    # --- model building blocks ---------------------------------------------
    def linear(self, value, name: str, shape=None):
        """``value @ W.T`` for a checkpoint ``[out, in]`` weight in the input's type."""
        weight = self.tensor(name, value.dtype, shape)
        return self.matmul(value, weight, transpose_rhs=True)

    def rms_norm(self, value, eps: float, weight: str | None = None, axis: int = -1):
        """``x * (mean(x^2) + eps) ** -0.5 [* w]`` in FP32, rounded to the input type."""
        rank = len(value.shape)
        axis %= rank
        x = self.cast(value, trt.float32)
        mean = self.mean(self.mul(x, x), axis)
        inverse = self.op(self.add(mean, self.scalar(eps, rank, trt.float32)),
                          self.scalar(-0.5, rank, trt.float32), _ELEMENTWISE.POW)
        output = self.mul(x, inverse)
        if weight is not None:
            shape = [1] * rank
            shape[axis] = value.shape[axis]
            gamma = self.reshape(self.tensor(weight, trt.float32), shape)
            output = self.mul(output, gamma)
        return self.cast(output, value.dtype)

    def gelu_tanh(self, value):
        """GELU with the tanh approximation (``gelu_pytorch_tanh``), in FP32."""
        return self.cast(self.activation(self.cast(value, trt.float32),
                                         trt.ActivationType.GELU_TANH), value.dtype)
