# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Boltz-2-owned strongly typed TensorRT graph primitives."""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from typing import Any

import numpy as np


_CONSTANT_STORAGE_KEY = object()


class Graph:
    """Small family-local facade over TensorRT network-definition operations."""

    def __init__(self, network: Any, trt: Any, weights: MutableMapping[Any, Any]):
        self.network = network
        self.trt = trt
        self.weights = weights
        if not isinstance(weights, MutableMapping):
            raise TypeError("Boltz-2 TensorRT weights must provide mutable lifetime storage")
        # TensorRT may defer reading IConstantLayer host buffers until engine
        # serialization. Keep both the NumPy owner and Weights wrapper alive via
        # the builder-owned weight dictionary after this Graph instance returns.
        self._constant_storage = weights.setdefault(_CONSTANT_STORAGE_KEY, [])

    def weight(self, name: str) -> np.ndarray:
        try:
            value = self.weights[name]
        except KeyError as error:
            raise ValueError(f"Boltz-2 checkpoint tensor is missing: {name}") from error
        return np.ascontiguousarray(value, dtype=np.float32)

    def constant(
        self,
        value: np.ndarray | float | int,
        shape: Sequence[int] | None = None,
        *,
        dtype: np.dtype | type = np.float32,
    ):
        array = np.asarray(value, dtype=dtype)
        if array.ndim:
            array = np.ascontiguousarray(array)
        if shape is not None:
            array = array.reshape(tuple(shape))
        trt_weights = self.trt.Weights(array)
        self._constant_storage.append((array, trt_weights))
        return self.network.add_constant(array.shape, trt_weights).get_output(0)

    def scalar_like(self, value: float, tensor: Any):
        scalar = self.constant(np.asarray(value, dtype=np.float32), (1,) * len(tensor.shape))
        return self.cast(scalar, tensor.dtype)

    def integer_scalar_like(self, value: int, tensor: Any):
        scalar = self.constant(
            np.asarray(value, dtype=np.int32),
            (1,) * len(tensor.shape),
            dtype=np.int32,
        )
        return self.cast(scalar, tensor.dtype)

    def cast(self, tensor: Any, dtype: Any):
        if tensor.dtype == dtype:
            return tensor
        return self.network.add_cast(tensor, dtype).get_output(0)

    def elementwise(self, lhs: Any, rhs: Any, operation: Any):
        return self.network.add_elementwise(lhs, rhs, operation).get_output(0)

    def add(self, lhs: Any, rhs: Any):
        return self.elementwise(lhs, rhs, self.trt.ElementWiseOperation.SUM)

    def sub(self, lhs: Any, rhs: Any):
        return self.elementwise(lhs, rhs, self.trt.ElementWiseOperation.SUB)

    def mul(self, lhs: Any, rhs: Any):
        return self.elementwise(lhs, rhs, self.trt.ElementWiseOperation.PROD)

    def div(self, lhs: Any, rhs: Any):
        return self.elementwise(lhs, rhs, self.trt.ElementWiseOperation.DIV)

    def maximum(self, lhs: Any, rhs: Any):
        return self.elementwise(lhs, rhs, self.trt.ElementWiseOperation.MAX)

    def minimum(self, lhs: Any, rhs: Any):
        return self.elementwise(lhs, rhs, self.trt.ElementWiseOperation.MIN)

    def equal(self, lhs: Any, rhs: Any):
        return self.elementwise(lhs, rhs, self.trt.ElementWiseOperation.EQUAL)

    def unary(self, tensor: Any, operation: Any):
        return self.network.add_unary(tensor, operation).get_output(0)

    def reshape(self, tensor: Any, shape: Sequence[int]):
        layer = self.network.add_shuffle(tensor)
        layer.reshape_dims = tuple(shape)
        return layer.get_output(0)

    def transpose(self, tensor: Any, order: Sequence[int]):
        layer = self.network.add_shuffle(tensor)
        layer.first_transpose = tuple(order)
        return layer.get_output(0)

    def slice(self, tensor: Any, start: Sequence[int], shape: Sequence[int]):
        return self.network.add_slice(
            tensor,
            tuple(start),
            tuple(shape),
            (1,) * len(shape),
        ).get_output(0)

    def concatenate(self, tensors: Sequence[Any], axis: int):
        layer = self.network.add_concatenation(list(tensors))
        layer.axis = axis
        return layer.get_output(0)

    def select(self, condition: Any, then_tensor: Any, else_tensor: Any):
        return self.network.add_select(condition, then_tensor, else_tensor).get_output(0)

    def gather(self, tensor: Any, indices: Any, axis: int):
        return self.network.add_gather(tensor, indices, axis).get_output(0)

    def one_hot(self, indices: Any, depth: int, dtype: Any):
        values = self.constant(np.asarray([0.0, 1.0], dtype=np.float32))
        values = self.cast(values, dtype)
        depth_tensor = self.constant(np.asarray(depth, dtype=np.int32), dtype=np.int32)
        return self.network.add_one_hot(indices, values, depth_tensor, -1).get_output(0)

    def reduce_sum(self, tensor: Any, axis: int, *, keep_dims: bool):
        return self.network.add_reduce(
            tensor,
            self.trt.ReduceOperation.SUM,
            1 << axis,
            keep_dims,
        ).get_output(0)

    def embedding(self, indices: Any, name: str, dtype: Any):
        table = self.cast(self.constant(self.weight(f"{name}.weight")), dtype)
        return self.gather(table, indices, 0)

    def linear(self, tensor: Any, prefix: str):
        """Apply a PyTorch linear layer stored as ``[out, in]`` weights."""

        weight = self.weight(f"{prefix}.weight")
        out_features, in_features = weight.shape
        if int(tensor.shape[-1]) != in_features:
            raise ValueError(
                f"Boltz-2 linear input mismatch for {prefix}: "
                f"{int(tensor.shape[-1])} != {in_features}"
            )
        rank = len(tensor.shape)
        rhs_shape = (1,) * max(0, rank - 2) + (in_features, out_features)
        rhs = self.cast(self.constant(weight.T, rhs_shape), tensor.dtype)
        output = self.network.add_matrix_multiply(
            tensor,
            self.trt.MatrixOperation.NONE,
            rhs,
            self.trt.MatrixOperation.NONE,
        ).get_output(0)
        bias_name = f"{prefix}.bias"
        if bias_name in self.weights:
            bias = self.weight(bias_name)
            bias_shape = (1,) * (rank - 1) + (out_features,)
            output = self.add(
                output,
                self.cast(self.constant(bias, bias_shape), output.dtype),
            )
        return output

    def layer_norm(self, tensor: Any, prefix: str, *, epsilon: float = 1.0e-5):
        hidden = int(tensor.shape[-1])
        rank = len(tensor.shape)
        shape = (1,) * (rank - 1) + (hidden,)
        gamma = self.cast(self.constant(self.weight(f"{prefix}.weight"), shape), tensor.dtype)
        bias_name = f"{prefix}.bias"
        bias = self.weight(bias_name) if bias_name in self.weights else np.zeros(hidden)
        beta = self.cast(self.constant(bias, shape), tensor.dtype)
        return self.normalization(tensor, gamma, beta, epsilon=epsilon)

    def unit_layer_norm(self, tensor: Any, *, epsilon: float = 1.0e-5):
        hidden = int(tensor.shape[-1])
        rank = len(tensor.shape)
        shape = (1,) * (rank - 1) + (hidden,)
        gamma = self.cast(self.constant(np.ones(hidden), shape), tensor.dtype)
        beta = self.cast(self.constant(np.zeros(hidden), shape), tensor.dtype)
        return self.normalization(tensor, gamma, beta, epsilon=epsilon)

    def normalization(
        self,
        tensor: Any,
        gamma: Any,
        beta: Any,
        *,
        epsilon: float = 1.0e-5,
    ):
        rank = len(tensor.shape)
        layer = self.network.add_normalization_v2(
            tensor,
            gamma,
            beta,
            1 << (rank - 1),
        )
        layer.epsilon = epsilon
        if hasattr(layer, "compute_precision"):
            layer.compute_precision = self.trt.float32
        return layer.get_output(0)

    def relu(self, tensor: Any):
        return self.network.add_activation(
            tensor,
            self.trt.ActivationType.RELU,
        ).get_output(0)

    def sigmoid(self, tensor: Any):
        return self.network.add_activation(
            tensor,
            self.trt.ActivationType.SIGMOID,
        ).get_output(0)

    def silu(self, tensor: Any):
        return self.mul(tensor, self.sigmoid(tensor))

    def softmax_last(self, tensor: Any):
        layer = self.network.add_softmax(tensor)
        layer.axes = 1 << (len(tensor.shape) - 1)
        return layer.get_output(0)

    def einsum(self, tensors: Sequence[Any], equation: str):
        return self.network.add_einsum(list(tensors), equation).get_output(0)
