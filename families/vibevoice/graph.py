# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph helpers for the VibeVoice engines.

The networks are strongly typed. ``Net.dtype`` is the compute type of the
checkpoint (bfloat16) or float32 for proof builds. Every helper follows the
reference's rounding points: matrix products and elementwise operations take
and return the compute type, while normalization statistics, activations and
softmax are evaluated in float32 and rounded once, as PyTorch does for reduced
precision tensors.
"""

from __future__ import annotations

import ml_dtypes
import numpy as np
import tensorrt as trt

Op = trt.ElementWiseOperation


class Net:
    """One strongly typed network plus the host arrays its constants borrow."""

    def __init__(self, network: trt.INetworkDefinition, dtype: trt.DataType) -> None:
        if dtype not in (trt.float32, trt.bfloat16):
            raise ValueError("vibevoice engines compute in float32 or bfloat16")
        self.network = network
        self.dtype = dtype
        self._arrays: list[np.ndarray] = []

    # -- constants -------------------------------------------------------
    def const(self, values, shape=None, dtype=None) -> trt.ITensor:
        dtype = self.dtype if dtype is None else dtype
        array = np.asarray(values)
        if shape is not None:
            array = array.reshape(shape)
        # np.array(..., order="C") keeps 0-d arrays 0-d; ascontiguousarray would make them 1-d.
        if dtype == trt.bfloat16:
            data = np.array(array.astype(np.float32).astype(ml_dtypes.bfloat16), order="C")
            weights = trt.Weights(trt.bfloat16, data.ctypes.data, data.size)
        elif dtype in (trt.float32, trt.int32, trt.int64):
            numpy_type = {trt.float32: np.float32, trt.int32: np.int32, trt.int64: np.int64}[dtype]
            data = np.array(array, dtype=numpy_type, order="C")
            weights = trt.Weights(data)
        else:
            raise ValueError(f"unsupported constant type {dtype}")
        self._arrays.append(data)
        return self.network.add_constant(tuple(int(d) for d in data.shape), weights).get_output(0)

    def scalar(self, value: float, rank: int, dtype=None) -> trt.ITensor:
        return self.const(np.full((1,) * rank, value, dtype=np.float32), dtype=dtype)

    # -- elementwise -----------------------------------------------------
    def cast(self, tensor: trt.ITensor, dtype: trt.DataType) -> trt.ITensor:
        if tensor.dtype == dtype:
            return tensor
        return self.network.add_cast(tensor, dtype).get_output(0)

    def binary(self, left: trt.ITensor, right: trt.ITensor, op: Op) -> trt.ITensor:
        if left.dtype != right.dtype:
            raise TypeError(f"elementwise operands differ in type: {left.dtype} vs {right.dtype}")
        return self.network.add_elementwise(left, right, op).get_output(0)

    def add(self, left, right):
        return self.binary(left, right, Op.SUM)

    def sub(self, left, right):
        return self.binary(left, right, Op.SUB)

    def mul(self, left, right):
        return self.binary(left, right, Op.PROD)

    def div(self, left, right):
        return self.binary(left, right, Op.DIV)

    def unary(self, tensor, op: trt.UnaryOperation):
        return self.network.add_unary(tensor, op).get_output(0)

    def silu(self, tensor: trt.ITensor) -> trt.ITensor:
        value = self.cast(tensor, trt.float32)
        gate = self.network.add_activation(value, trt.ActivationType.SIGMOID).get_output(0)
        return self.cast(self.mul(value, gate), tensor.dtype)

    def gelu(self, tensor: trt.ITensor) -> trt.ITensor:
        # The tokenizers use the exact erf GELU ("gelu"), not the tanh form.
        value = self.cast(tensor, trt.float32)
        out = self.network.add_activation(value, trt.ActivationType.GELU_ERF).get_output(0)
        return self.cast(out, tensor.dtype)

    # -- shape helpers ---------------------------------------------------
    def reshape(self, tensor, dims, first_transpose=None, second_transpose=None):
        layer = self.network.add_shuffle(tensor)
        if first_transpose is not None:
            layer.first_transpose = tuple(first_transpose)
        if dims is not None:
            layer.reshape_dims = tuple(dims)
        if second_transpose is not None:
            layer.second_transpose = tuple(second_transpose)
        return layer.get_output(0)

    def concat(self, tensors, axis: int) -> trt.ITensor:
        layer = self.network.add_concatenation(list(tensors))
        layer.axis = axis
        return layer.get_output(0)

    def slice(self, tensor, start, shape, stride=None) -> trt.ITensor:
        stride = stride or (1,) * len(start)
        return self.network.add_slice(tensor, tuple(start), tuple(shape), tuple(stride)).get_output(
            0
        )

    # -- layers ----------------------------------------------------------
    def linear(self, tensor: trt.ITensor, weight: np.ndarray, bias: np.ndarray | None = None):
        """``x @ W^T + b`` over the last axis, torch.nn.Linear layout ``[out, in]``."""
        rank = len(tensor.shape)
        weight = np.asarray(weight)
        matrix = self.const(weight, (1,) * (rank - 2) + tuple(weight.shape))
        out = self.network.add_matrix_multiply(
            tensor, trt.MatrixOperation.NONE, matrix, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
        if bias is None:
            return out
        return self.add(out, self.const(np.asarray(bias), (1,) * (rank - 1) + (weight.shape[0],)))

    def rms_norm(self, tensor: trt.ITensor, weight: np.ndarray | None, eps: float, axis: int):
        """``weight * (x * rsqrt(mean(x^2) + eps)).to(dtype)`` with float32 statistics."""
        rank = len(tensor.shape)
        axis = axis % rank
        value = self.cast(tensor, trt.float32)
        square = self.mul(value, value)
        mean = self.network.add_reduce(square, trt.ReduceOperation.AVG, 1 << axis, True).get_output(
            0
        )
        shifted = self.add(mean, self.scalar(eps, rank, trt.float32))
        inverse = self.unary(self.unary(shifted, trt.UnaryOperation.SQRT), trt.UnaryOperation.RECIP)
        normed = self.cast(self.mul(value, inverse), tensor.dtype)
        if weight is None:
            return normed
        shape = tuple(
            int(np.asarray(weight).shape[0]) if index == axis else 1 for index in range(rank)
        )
        return self.mul(self.const(np.asarray(weight), shape), normed)

    def conv(self, tensor, weight, bias, *, stride=1, groups=1, left_pad=0):
        """Conv1d over ``[1, C, 1, T]`` with torch's ``[out, in/groups, k]`` weights."""
        weight = np.asarray(weight)
        out_channels, _, kernel = weight.shape
        kernel_data = self._weights(weight[:, :, None, :])
        bias_data = (
            self._weights(np.asarray(bias)) if bias is not None else trt.Weights(self._trt_np())
        )
        layer = self.network.add_convolution_nd(
            tensor, int(out_channels), (1, int(kernel)), kernel_data, bias_data
        )
        layer.stride_nd = (1, int(stride))
        layer.pre_padding = (0, int(left_pad))
        layer.post_padding = (0, 0)
        layer.num_groups = int(groups)
        return layer.get_output(0)

    def deconv(self, tensor, weight, bias, *, stride):
        """ConvTranspose1d over ``[1, C, 1, T]`` with torch's ``[in, out, k]`` weights.

        TensorRT has no bfloat16 deconvolution, so a bfloat16 network runs this
        layer in float32: the inputs and weights are bfloat16 values, the sums
        are float32 like the reference's accumulation, and the result is rounded
        back to bfloat16 once.
        """
        weight = np.asarray(weight, dtype=np.float32)
        _, out_channels, kernel = weight.shape
        source = self.cast(tensor, trt.float32)
        kernel_data = np.ascontiguousarray(weight[:, :, None, :])
        bias_data = np.ascontiguousarray(np.asarray(bias, dtype=np.float32))
        if self.dtype == trt.bfloat16:
            # Keep exactly the checkpoint's bfloat16 values.
            kernel_data = kernel_data.astype(ml_dtypes.bfloat16).astype(np.float32)
            bias_data = bias_data.astype(ml_dtypes.bfloat16).astype(np.float32)
        self._arrays += [kernel_data, bias_data]
        layer = self.network.add_deconvolution_nd(
            source,
            int(out_channels),
            (1, int(kernel)),
            trt.Weights(kernel_data),
            trt.Weights(bias_data),
        )
        layer.stride_nd = (1, int(stride))
        layer.padding_nd = (0, 0)
        return self.cast(layer.get_output(0), self.dtype)

    def _trt_np(self):
        return np.float32 if self.dtype == trt.float32 else ml_dtypes.bfloat16

    def _weights(self, array: np.ndarray) -> trt.Weights:
        if self.dtype == trt.bfloat16:
            data = np.ascontiguousarray(
                np.asarray(array, dtype=np.float32).astype(ml_dtypes.bfloat16)
            )
            self._arrays.append(data)
            return trt.Weights(trt.bfloat16, data.ctypes.data, data.size)
        data = np.ascontiguousarray(array, dtype=np.float32)
        self._arrays.append(data)
        return trt.Weights(data)


def builder_and_network(verbose: bool):
    """A strongly typed network and a builder config without TF32."""
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    # TF32 would hide graph errors behind its 1e-3 matmul error in float32 proofs.
    config.clear_flag(trt.BuilderFlag.TF32)
    return builder, network, config


def serialize(builder, network, config, what: str) -> bytes:
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"vibevoice {what} engine build failed")
    return bytes(plan)
