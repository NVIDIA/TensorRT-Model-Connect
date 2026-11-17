# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned TensorRT graph helpers for strongly typed BF16 or FP32 networks.

Checkpoint matrices keep their stored [out, in] layout and precision: a BF16 network references the
memory-mapped BF16 bytes directly, an FP32 network widens them exactly. Operations that the source
model evaluates in FP32 (normalizations, activations, routing, softmax, DeltaNet recurrence) cast to
FP32 explicitly and round back once, where the source rounds.
"""

from __future__ import annotations

import ml_dtypes
import numpy as np
import tensorrt as trt

E = trt.ElementWiseOperation


class Graph:
    """One network plus the host arrays its constants reference until the build completes."""

    def __init__(self, network: trt.INetworkDefinition, work: trt.DataType):
        if work not in (trt.bfloat16, trt.float32):
            raise ValueError("MiniCPM-V 4.7 networks compute in BF16 or FP32")
        self.network = network
        self.work = work
        self._keep: list[np.ndarray] = []

    # ----- constants -----
    def const(self, values: np.ndarray, dtype: trt.DataType | None = None) -> trt.ITensor:
        """Add a constant in ``dtype`` (default: the work dtype for floating values)."""
        values = np.asarray(values)
        if dtype is None:
            dtype = self.work if values.dtype.kind == "f" or values.dtype == ml_dtypes.bfloat16 else None
        if dtype == trt.bfloat16:
            array = values if values.dtype == ml_dtypes.bfloat16 else values.astype(np.float32).astype(
                ml_dtypes.bfloat16)
            array = np.ascontiguousarray(array)
            self._keep.append(array)
            weights = trt.Weights(trt.bfloat16, array.ctypes.data, array.size)
        elif dtype == trt.float32:
            array = np.ascontiguousarray(values, dtype=np.float32)
            self._keep.append(array)
            weights = trt.Weights(array)
        elif dtype == trt.int32 or (dtype is None and values.dtype.kind in "iu"):
            array = np.ascontiguousarray(values, dtype=np.int32)
            self._keep.append(array)
            weights = trt.Weights(array)
        else:
            raise ValueError(f"Unsupported constant dtype {dtype}")
        shape = array.shape if array.ndim else (1,)
        return self.network.add_constant(shape, weights).get_output(0)

    def scalar(self, value: float, rank: int, dtype: trt.DataType = trt.float32) -> trt.ITensor:
        return self.const(np.full((1,) * rank, value, dtype=np.float32), dtype)

    # ----- elementary layers -----
    def cast(self, tensor: trt.ITensor, dtype: trt.DataType) -> trt.ITensor:
        return tensor if tensor.dtype == dtype else self.network.add_cast(tensor, dtype).get_output(0)

    def f32(self, tensor: trt.ITensor) -> trt.ITensor:
        return self.cast(tensor, trt.float32)

    def binary(self, a: trt.ITensor, b: trt.ITensor, op: trt.ElementWiseOperation) -> trt.ITensor:
        return self.network.add_elementwise(a, b, op).get_output(0)

    def add(self, a, b):
        return self.binary(a, b, E.SUM)

    def mul(self, a, b):
        return self.binary(a, b, E.PROD)

    def reshape(self, tensor: trt.ITensor, shape, transpose=None) -> trt.ITensor:
        layer = self.network.add_shuffle(tensor)
        layer.reshape_dims = tuple(shape)
        if transpose is not None:
            layer.second_transpose = trt.Permutation(list(transpose))
        return layer.get_output(0)

    def transpose(self, tensor: trt.ITensor, order) -> trt.ITensor:
        layer = self.network.add_shuffle(tensor)
        layer.first_transpose = trt.Permutation(list(order))
        return layer.get_output(0)

    def slice(self, tensor: trt.ITensor, start, shape) -> trt.ITensor:
        return self.network.add_slice(tensor, tuple(start), tuple(shape), (1,) * len(start)).get_output(0)

    def concat(self, tensors, axis: int) -> trt.ITensor:
        layer = self.network.add_concatenation(list(tensors))
        layer.axis = axis
        return layer.get_output(0)

    def gather(self, data: trt.ITensor, indices: trt.ITensor, axis: int) -> trt.ITensor:
        return self.network.add_gather(data, indices, axis).get_output(0)

    def reduce(self, tensor: trt.ITensor, op: trt.ReduceOperation, axis: int, keep: bool = True):
        return self.network.add_reduce(tensor, op, 1 << axis, keep).get_output(0)

    def unary(self, tensor: trt.ITensor, op: trt.UnaryOperation) -> trt.ITensor:
        return self.network.add_unary(tensor, op).get_output(0)

    def sigmoid(self, tensor: trt.ITensor) -> trt.ITensor:
        return self.network.add_activation(tensor, trt.ActivationType.SIGMOID).get_output(0)

    def softmax(self, tensor: trt.ITensor, axis: int) -> trt.ITensor:
        layer = self.network.add_softmax(tensor)
        layer.axes = 1 << axis
        return layer.get_output(0)

    def matmul(self, a: trt.ITensor, b: trt.ITensor, transpose_b: bool = False) -> trt.ITensor:
        op_b = trt.MatrixOperation.TRANSPOSE if transpose_b else trt.MatrixOperation.NONE
        return self.network.add_matrix_multiply(a, trt.MatrixOperation.NONE, b, op_b).get_output(0)

    # ----- source-model blocks -----
    def linear(self, x: trt.ITensor, weight: np.ndarray, bias: np.ndarray | None = None) -> trt.ITensor:
        """``x @ weight.T (+ bias)`` with an HF [out, in] weight, in the work dtype."""
        rank = len(x.shape)
        w = self.const(weight)
        if rank != 2:
            w = self.reshape(w, (1,) * (rank - 2) + tuple(weight.shape))
        y = self.matmul(x, w, transpose_b=True)
        if bias is not None:
            b = self.const(np.asarray(bias).reshape((1,) * (rank - 1) + (-1,)))
            y = self.add(y, b)
        return y

    def silu(self, x: trt.ITensor) -> trt.ITensor:
        """SiLU evaluated in FP32 and rounded once to the input dtype."""
        x32 = self.f32(x)
        return self.cast(self.mul(x32, self.sigmoid(x32)), x.dtype)

    def gelu_tanh(self, x: trt.ITensor) -> trt.ITensor:
        """``gelu_pytorch_tanh`` in FP32, rounded once to the input dtype."""
        rank = len(x.shape)
        x32 = self.f32(x)
        cube = self.mul(self.mul(x32, x32), x32)
        inner = self.add(x32, self.mul(cube, self.scalar(0.044715, rank)))
        inner = self.mul(inner, self.scalar(float(np.sqrt(2.0 / np.pi)), rank))
        tanh = self.network.add_activation(inner, trt.ActivationType.TANH).get_output(0)
        out = self.mul(self.mul(x32, self.scalar(0.5, rank)), self.add(tanh, self.scalar(1.0, rank)))
        return self.cast(out, x.dtype)

    def gelu_erf(self, x: trt.ITensor) -> trt.ITensor:
        """Exact ``nn.GELU()`` in FP32, rounded once to the input dtype."""
        rank = len(x.shape)
        x32 = self.f32(x)
        erf = self.unary(self.mul(x32, self.scalar(float(1.0 / np.sqrt(2.0)), rank)), trt.UnaryOperation.ERF)
        out = self.mul(self.mul(x32, self.scalar(0.5, rank)), self.add(erf, self.scalar(1.0, rank)))
        return self.cast(out, x.dtype)

    def rsqrt_mean_square(self, x32: trt.ITensor, axis: int, eps: float) -> trt.ITensor:
        rank = len(x32.shape)
        mean = self.reduce(self.mul(x32, x32), trt.ReduceOperation.AVG, axis)
        root = self.unary(self.add(mean, self.scalar(eps, rank)), trt.UnaryOperation.SQRT)
        return self.unary(root, trt.UnaryOperation.RECIP)

    def rms_norm_centered(self, x: trt.ITensor, weight: np.ndarray, eps: float) -> trt.ITensor:
        """Qwen3.5 zero-centered RMSNorm over the last axis: ``(x * rsqrt(ms + eps)) * (1 + w)`` in FP32."""
        rank = len(x.shape)
        x32 = self.f32(x)
        normalized = self.mul(x32, self.rsqrt_mean_square(x32, rank - 1, eps))
        scale = 1.0 + np.asarray(weight, dtype=np.float32)
        gamma = self.const(scale.reshape((1,) * (rank - 1) + (-1,)), trt.float32)
        return self.cast(self.mul(normalized, gamma), x.dtype)

    def layer_norm(self, x: trt.ITensor, weight: np.ndarray, bias: np.ndarray, eps: float) -> trt.ITensor:
        """LayerNorm over the last axis, statistics and affine in FP32, one rounding."""
        rank = len(x.shape)
        x32 = self.f32(x)
        mean = self.reduce(x32, trt.ReduceOperation.AVG, rank - 1)
        centered = self.binary(x32, mean, E.SUB)
        normalized = self.mul(centered, self.rsqrt_mean_square(centered, rank - 1, eps))
        shape = (1,) * (rank - 1) + (-1,)
        gamma = self.const(np.asarray(weight, dtype=np.float32).reshape(shape), trt.float32)
        beta = self.const(np.asarray(bias, dtype=np.float32).reshape(shape), trt.float32)
        return self.cast(self.add(self.mul(normalized, gamma), beta), x.dtype)

    def attention(self, q: trt.ITensor, k: trt.ITensor, v: trt.ITensor, scale: float,
                  mask: trt.ITensor | None = None) -> trt.ITensor:
        """Softmax attention with FP32 scores, probabilities and context; returns FP32.

        ``q`` [..., Sq, D], ``k``/``v`` [..., Sk, D] with matching leading axes.
        """
        rank = len(q.shape)
        scores = self.matmul(self.f32(q), self.f32(k), transpose_b=True)
        scores = self.mul(scores, self.scalar(scale, rank))
        if mask is not None:
            scores = self.add(scores, mask)
        probabilities = self.softmax(scores, rank - 1)
        return self.matmul(probabilities, self.f32(v))
