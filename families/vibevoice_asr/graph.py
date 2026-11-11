# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph helpers shared by the VibeVoice-ASR encoder and decoder builders."""

from __future__ import annotations

import ml_dtypes
import numpy as np
import tensorrt as trt


class Graph:
    """A strongly typed network plus the host buffers its BF16 weights point into.

    ``trt.Weights(trt.bfloat16, pointer, count)`` does not copy, so every BF16
    array stays referenced here until the engine is serialized.
    """

    def __init__(self, network: trt.INetworkDefinition) -> None:
        self.network = network
        self._keepalive: list[np.ndarray] = []

    @staticmethod
    def _contiguous(values: np.ndarray) -> np.ndarray:
        # np.ascontiguousarray would promote a 0-d scalar to shape (1,).
        array = np.asarray(values)
        return array if array.flags.c_contiguous else array.copy(order="C")

    def weights(self, values: np.ndarray) -> trt.Weights:
        array = self._contiguous(values)
        if array.dtype == ml_dtypes.bfloat16:
            self._keepalive.append(array)
            return trt.Weights(trt.bfloat16, array.ctypes.data, array.size)
        if array.dtype not in (np.float32, np.int32, np.int64):
            raise TypeError(f"unsupported constant dtype {array.dtype}")
        self._keepalive.append(array)
        return trt.Weights(array)

    def constant(self, values: np.ndarray, shape: tuple[int, ...] | None = None) -> trt.ITensor:
        array = self._contiguous(values)
        layer = self.network.add_constant(
            tuple(array.shape if shape is None else shape), self.weights(array)
        )
        return layer.get_output(0)

    def scalar(self, value: float, dtype: trt.DataType, rank: int) -> trt.ITensor:
        shape = (1,) * rank
        if dtype == trt.bfloat16:
            return self.constant(np.full(shape, value, dtype=ml_dtypes.bfloat16))
        if dtype == trt.int32:
            return self.constant(np.full(shape, value, dtype=np.int32))
        return self.constant(np.full(shape, value, dtype=np.float32))

    def cast(self, tensor: trt.ITensor, dtype: trt.DataType) -> trt.ITensor:
        if tensor.dtype == dtype:
            return tensor
        return self.network.add_cast(tensor, dtype).get_output(0)

    def binary(self, a: trt.ITensor, b: trt.ITensor, op: trt.ElementWiseOperation) -> trt.ITensor:
        return self.network.add_elementwise(a, b, op).get_output(0)

    def add(self, a: trt.ITensor, b: trt.ITensor) -> trt.ITensor:
        return self.binary(a, b, trt.ElementWiseOperation.SUM)

    def mul(self, a: trt.ITensor, b: trt.ITensor) -> trt.ITensor:
        return self.binary(a, b, trt.ElementWiseOperation.PROD)

    def sub(self, a: trt.ITensor, b: trt.ITensor) -> trt.ITensor:
        return self.binary(a, b, trt.ElementWiseOperation.SUB)

    def unary(self, tensor: trt.ITensor, op: trt.UnaryOperation) -> trt.ITensor:
        return self.network.add_unary(tensor, op).get_output(0)

    def activation(self, tensor: trt.ITensor, kind: trt.ActivationType) -> trt.ITensor:
        return self.network.add_activation(tensor, kind).get_output(0)

    def reshape(
        self,
        tensor: trt.ITensor,
        dims: tuple[int, ...],
        transpose: tuple[int, ...] | None = None,
        *,
        transpose_first: bool = False,
    ) -> trt.ITensor:
        layer = self.network.add_shuffle(tensor)
        if transpose is not None and transpose_first:
            layer.first_transpose = transpose
        layer.reshape_dims = dims
        if transpose is not None and not transpose_first:
            layer.second_transpose = transpose
        return layer.get_output(0)

    def transpose(self, tensor: trt.ITensor, order: tuple[int, ...]) -> trt.ITensor:
        layer = self.network.add_shuffle(tensor)
        layer.first_transpose = order
        return layer.get_output(0)

    def rms_norm(
        self,
        tensor: trt.ITensor,
        weight: np.ndarray,
        eps: float,
        axis: int,
        weight_shape: tuple[int, ...],
    ) -> trt.ITensor:
        """``weight * (x.float() * rsqrt(mean(x.float()^2) + eps)).to(x.dtype)``."""
        source_dtype = tensor.dtype
        rank = len(tensor.shape)
        values = self.cast(tensor, trt.float32)
        mean = self.network.add_reduce(
            self.mul(values, values), trt.ReduceOperation.AVG, 1 << axis, True
        ).get_output(0)
        denominator = self.unary(
            self.add(mean, self.scalar(eps, trt.float32, rank)), trt.UnaryOperation.SQRT
        )
        normalized = self.binary(values, denominator, trt.ElementWiseOperation.DIV)
        return self.mul(self.cast(normalized, source_dtype), self.constant(weight, weight_shape))

    def linear(self, rows: trt.ITensor, weight: np.ndarray, bias: np.ndarray | None) -> trt.ITensor:
        """``rows @ weight.T + bias`` for a PyTorch ``[out, in]`` weight on 2-D rows."""
        out_features = int(weight.shape[0])
        product = self.network.add_matrix_multiply(
            rows, trt.MatrixOperation.NONE, self.constant(weight), trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
        if bias is None:
            return product
        return self.add(product, self.constant(bias, (1, out_features)))

    def dim(self, tensor: trt.ITensor, axis: int) -> trt.ITensor:
        """One dimension of ``tensor`` as an int64 shape tensor of shape [1]."""
        shape = self.network.add_shape(tensor).get_output(0)
        return self.network.add_slice(shape, (axis,), (1,), (1,)).get_output(0)

    def i64(self, values: list[int]) -> trt.ITensor:
        return self.constant(np.asarray(values, dtype=np.int64))

    def concat(self, tensors: list[trt.ITensor], axis: int = 0) -> trt.ITensor:
        layer = self.network.add_concatenation(tensors)
        layer.axis = axis
        return layer.get_output(0)


def new_builder(
    verbose: bool, *, optimization_level: int, workspace_bytes: int
) -> tuple[trt.Builder, trt.INetworkDefinition, trt.IBuilderConfig]:
    """Builder, strongly typed network and config with TF32 disabled.

    Every tensor type is stated by the graph. TF32 is cleared so the FP32
    attention scores and norms keep full FP32 products.
    """
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = optimization_level
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    return builder, network, config


def serialize(
    builder: trt.Builder, network: trt.INetworkDefinition, config: trt.IBuilderConfig, what: str
) -> bytes:
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"vibevoice_asr {what} engine build failed")
    return bytes(plan)
