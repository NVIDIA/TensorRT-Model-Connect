# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ESMFold2-owned helpers over a strongly typed TensorRT network with dynamic token and atom axes.

``Net.compute`` is the dtype of weights and activations (bf16 or float32). Normalizations, softmax, the RoPE angles
and every reduction that the reference keeps in float32 run in float32 here too; the dtype of a result is stated
where it differs from its input.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import ml_dtypes
import numpy as np


def import_tensorrt() -> Any:
    import tensorrt

    return tensorrt


class Net:
    """One network under construction: constants stay alive until the plan is serialized."""

    def __init__(self, network: Any, trt: Any, compute: Any, weight: Callable[[str], np.ndarray]) -> None:
        self.network = network
        self.trt = trt
        self.compute = compute
        self.f32 = trt.float32
        self._weight = weight
        self._arrays: list[np.ndarray] = []
        # One constant per (checkpoint tensor, layout, type), shared by every use in the network.
        self._named: dict[tuple, Any] = {}

    # ----------------------------------------------------------------- constants and weights

    def weight(self, name: str) -> np.ndarray:
        return self._weight(name)

    def constant(self, value: Any, dtype: Any = None, shape: Sequence[int] | None = None) -> Any:
        """A constant of ``dtype`` (TensorRT type; float32 by default) from host values."""

        trt = self.trt
        dtype = self.f32 if dtype is None else dtype
        if dtype == trt.bfloat16:
            array = np.asarray(value, dtype=np.float32).astype(ml_dtypes.bfloat16)
        elif dtype == trt.float32:
            array = np.asarray(value, dtype=np.float32)
        elif dtype == trt.int32:
            array = np.asarray(value, dtype=np.int32)
        elif dtype == trt.int64:
            array = np.asarray(value, dtype=np.int64)
        elif dtype == trt.bool:
            array = np.asarray(value, dtype=np.bool_)
        else:
            raise ValueError(f"unsupported constant type {dtype}")
        if shape is not None:
            array = array.reshape(tuple(shape))
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        self._arrays.append(array)
        if array.dtype == np.dtype(ml_dtypes.bfloat16):
            weights = trt.Weights(trt.bfloat16, array.ctypes.data, array.size)
        else:
            weights = trt.Weights(array)
        return self.network.add_constant(array.shape, weights).get_output(0)

    def scalar(self, value: float, rank: int, dtype: Any = None) -> Any:
        return self.constant(np.full((1,) * rank, value, dtype=np.float32), dtype or self.f32)

    def int_scalar(self, value: int, rank: int, dtype: Any = None) -> Any:
        dtype = dtype or self.trt.int32
        return self.constant(np.full((1,) * rank, value), dtype)

    # ----------------------------------------------------------------- elementwise

    def cast(self, tensor: Any, dtype: Any) -> Any:
        if tensor.dtype == dtype:
            return tensor
        return self.network.add_cast(tensor, dtype).get_output(0)

    def _binary(self, a: Any, b: Any, op: Any) -> Any:
        return self.network.add_elementwise(a, b, op).get_output(0)

    def add(self, a: Any, b: Any) -> Any:
        return self._binary(a, b, self.trt.ElementWiseOperation.SUM)

    def sub(self, a: Any, b: Any) -> Any:
        return self._binary(a, b, self.trt.ElementWiseOperation.SUB)

    def mul(self, a: Any, b: Any) -> Any:
        return self._binary(a, b, self.trt.ElementWiseOperation.PROD)

    def div(self, a: Any, b: Any) -> Any:
        return self._binary(a, b, self.trt.ElementWiseOperation.DIV)

    def maximum(self, a: Any, b: Any) -> Any:
        return self._binary(a, b, self.trt.ElementWiseOperation.MAX)

    def minimum(self, a: Any, b: Any) -> Any:
        return self._binary(a, b, self.trt.ElementWiseOperation.MIN)

    def equal(self, a: Any, b: Any) -> Any:
        return self._binary(a, b, self.trt.ElementWiseOperation.EQUAL)

    def less(self, a: Any, b: Any) -> Any:
        return self._binary(a, b, self.trt.ElementWiseOperation.LESS)

    def greater(self, a: Any, b: Any) -> Any:
        return self._binary(a, b, self.trt.ElementWiseOperation.GREATER)

    def logical_and(self, a: Any, b: Any) -> Any:
        return self._binary(a, b, self.trt.ElementWiseOperation.AND)

    def logical_not(self, a: Any) -> Any:
        return self.unary(a, self.trt.UnaryOperation.NOT)

    def select(self, condition: Any, then: Any, otherwise: Any) -> Any:
        return self.network.add_select(condition, then, otherwise).get_output(0)

    def unary(self, tensor: Any, op: Any) -> Any:
        return self.network.add_unary(tensor, op).get_output(0)

    def activation(self, tensor: Any, kind: Any) -> Any:
        return self.network.add_activation(tensor, kind).get_output(0)

    def sigmoid(self, tensor: Any) -> Any:
        """Sigmoid evaluated in float32 and rounded once to the input type (as torch does for bf16)."""

        return self.cast(self.activation(self.cast(tensor, self.f32), self.trt.ActivationType.SIGMOID), tensor.dtype)

    def silu(self, tensor: Any) -> Any:
        x = self.cast(tensor, self.f32)
        return self.cast(self.mul(x, self.activation(x, self.trt.ActivationType.SIGMOID)), tensor.dtype)

    def relu(self, tensor: Any) -> Any:
        return self.activation(tensor, self.trt.ActivationType.RELU)

    def gelu(self, tensor: Any) -> Any:
        """Exact (erf) GELU in float32, rounded once to the input type."""

        x = self.cast(tensor, self.f32)
        return self.cast(self.activation(x, self.trt.ActivationType.GELU_ERF), tensor.dtype)

    # ----------------------------------------------------------------- shape manipulation

    def reshape(self, tensor: Any, dims: Sequence[int]) -> Any:
        """Reshape with TensorRT placeholders: 0 copies the input dimension, -1 is inferred."""

        layer = self.network.add_shuffle(tensor)
        layer.reshape_dims = tuple(dims)
        return layer.get_output(0)

    def reshape_to(self, tensor: Any, shape: Any) -> Any:
        """Reshape to a runtime shape tensor."""

        layer = self.network.add_shuffle(tensor)
        layer.set_input(1, shape)
        return layer.get_output(0)

    def transpose(self, tensor: Any, order: Sequence[int]) -> Any:
        layer = self.network.add_shuffle(tensor)
        layer.first_transpose = tuple(order)
        return layer.get_output(0)

    def shape(self, tensor: Any) -> Any:
        return self.network.add_shape(tensor).get_output(0)

    def dim(self, tensor: Any, axis: int) -> Any:
        """One dimension of ``tensor`` as a rank-1 int64 shape tensor of length 1."""

        return self.network.add_gather(self.shape(tensor), self.constant([axis], self.trt.int32), 0).get_output(0)

    def shape_of(self, parts: Sequence[Any]) -> Any:
        """Concatenate rank-1 int64 shape tensors and literal ints into one shape tensor."""

        tensors = [self.constant([part], self.trt.int64) if isinstance(part, int) else part for part in parts]
        layer = self.network.add_concatenation(tensors)
        layer.axis = 0
        return layer.get_output(0)

    def concat(self, tensors: Sequence[Any], axis: int) -> Any:
        layer = self.network.add_concatenation(list(tensors))
        layer.axis = axis
        return layer.get_output(0)

    def slice_last(self, tensor: Any, start: int, size: int) -> Any:
        """``tensor[..., start:start + size]`` for a static last axis and any leading shape."""

        rank = len(tensor.shape)
        layer = self.network.add_slice(tensor, (0,) * rank, (1,) * rank, (1,) * rank)
        layer.set_input(1, self.constant([0] * (rank - 1) + [start], self.trt.int64))
        leading = [self.dim(tensor, axis) for axis in range(rank - 1)]
        layer.set_input(2, self.shape_of([*leading, size]))
        return layer.get_output(0)

    def split_last(self, tensor: Any, sizes: Sequence[int]) -> list[Any]:
        parts, offset = [], 0
        for size in sizes:
            parts.append(self.slice_last(tensor, offset, size))
            offset += size
        return parts

    def gather(self, tensor: Any, indices: Any, axis: int) -> Any:
        return self.network.add_gather(tensor, indices, axis).get_output(0)

    def one_hot(self, indices: Any, depth: int, dtype: Any) -> Any:
        """One-hot along a new last axis; ``indices`` is int32."""

        values = self.constant([0.0, 1.0], dtype)
        depth_tensor = self.constant(np.asarray(depth, dtype=np.int32), self.trt.int32, shape=())
        return self.network.add_one_hot(indices, values, depth_tensor, -1).get_output(0)

    def iota(self, length: Any, dtype: Any = None) -> Any:
        """``arange(length)`` for a rank-1 int64 shape tensor ``length``; int32 unless stated."""

        dtype = dtype or self.trt.int32
        layer = self.network.add_fill((1,), self.trt.FillOperation.LINSPACE, dtype)
        layer.set_input(0, length)
        zero = 0.0 if dtype == self.f32 else 0
        layer.set_input(1, self.constant(np.asarray(zero), dtype, shape=()))
        layer.set_input(2, self.constant([1], dtype))
        return layer.get_output(0)

    # ----------------------------------------------------------------- reductions and normalization

    def reduce(self, tensor: Any, op: Any, axis: int, keep_dims: bool) -> Any:
        axis = axis % len(tensor.shape)
        return self.network.add_reduce(tensor, op, 1 << axis, keep_dims).get_output(0)

    def reduce_sum(self, tensor: Any, axis: int, keep_dims: bool = False) -> Any:
        return self.reduce(tensor, self.trt.ReduceOperation.SUM, axis, keep_dims)

    def reduce_max(self, tensor: Any, axis: int, keep_dims: bool = False) -> Any:
        return self.reduce(tensor, self.trt.ReduceOperation.MAX, axis, keep_dims)

    def reduce_mean(self, tensor: Any, axis: int, keep_dims: bool = False) -> Any:
        return self.reduce(tensor, self.trt.ReduceOperation.AVG, axis, keep_dims)

    def softmax(self, tensor: Any, axis: int = -1) -> Any:
        """Softmax in float32 (returns float32)."""

        x = self.cast(tensor, self.f32)
        layer = self.network.add_softmax(x)
        layer.axes = 1 << (axis % len(x.shape))
        return layer.get_output(0)

    def layer_norm(self, tensor: Any, prefix: str | None, *, eps: float = 1.0e-5, bias: bool = True) -> Any:
        """LayerNorm over the last axis in float32 (returns float32).

        ``prefix`` names checkpoint ``.weight`` / ``.bias``; ``None`` is the weightless norm.
        """

        x = self.cast(tensor, self.f32)
        rank, width = len(x.shape), int(x.shape[-1])
        shape = (1,) * (rank - 1) + (width,)

        def parameter(suffix: str, present: bool, fill: float) -> Any:
            key = ("norm", prefix, suffix, rank, present)
            if key not in self._named:
                values = self.weight(f"{prefix}.{suffix}") if present else np.full(width, fill, np.float32)
                self._named[key] = self.constant(values, shape=shape)
            return self._named[key]

        gamma = parameter("weight", prefix is not None, 1.0)
        beta = parameter("bias", prefix is not None and bias, 0.0)
        add = getattr(self.network, "add_normalization_v2", None) or self.network.add_normalization
        layer = add(x, gamma, beta, 1 << (rank - 1))
        layer.epsilon = eps
        if hasattr(layer, "compute_precision"):
            layer.compute_precision = self.f32
        return layer.get_output(0)

    def rms_norm(self, tensor: Any, *, eps: float) -> Any:
        """Weightless RMSNorm in float32 (returns float32)."""

        x = self.cast(tensor, self.f32)
        mean_square = self.reduce_mean(self.mul(x, x), -1, keep_dims=True)
        rank = len(x.shape)
        root = self.unary(self.add(mean_square, self.scalar(eps, rank)), self.trt.UnaryOperation.SQRT)
        return self.div(x, root)

    # ----------------------------------------------------------------- matrix products

    def matmul(self, a: Any, b: Any, *, transpose_b: bool = False) -> Any:
        op = self.trt.MatrixOperation
        return self.network.add_matrix_multiply(
            a, op.NONE, b, op.TRANSPOSE if transpose_b else op.NONE
        ).get_output(0)

    def linear(
        self,
        tensor: Any,
        prefix: str | None = None,
        *,
        weight: np.ndarray | None = None,
        bias: np.ndarray | bool = False,
        dtype: Any = None,
    ) -> Any:
        """``tensor @ W.T (+ b)`` for a PyTorch ``[out, in]`` weight, computed in ``dtype`` (default ``compute``).

        The input is cast to ``dtype``; the product is returned in ``dtype`` (accumulation is float32).
        ``bias=True`` reads ``{prefix}.bias``; an array is used as the bias.
        """

        dtype = dtype or self.compute
        rank = len(tensor.shape)
        key = ("linear", prefix, rank, str(dtype))
        if weight is not None or key not in self._named:
            if weight is None:
                if prefix is None:
                    raise ValueError("linear needs a checkpoint prefix or an explicit weight")
                weight = self.weight(f"{prefix}.weight")
            out_features, in_features = weight.shape
            rhs = self.constant(weight.T, dtype, shape=(1,) * (rank - 2) + (in_features, out_features))
            if prefix is not None:
                self._named[key] = rhs
        else:
            rhs = self._named[key]
        in_features, out_features = int(rhs.shape[-2]), int(rhs.shape[-1])
        if int(tensor.shape[-1]) != in_features:
            raise ValueError(f"linear {prefix}: input width {int(tensor.shape[-1])} != {in_features}")
        output = self.matmul(self.cast(tensor, dtype), rhs)
        if bias is True:
            if prefix is None:
                raise ValueError("bias=True needs a checkpoint prefix")
            bias_key = ("bias", prefix, rank, str(dtype))
            if bias_key not in self._named:
                self._named[bias_key] = self.constant(
                    self.weight(f"{prefix}.bias"), dtype, shape=(1,) * (rank - 1) + (out_features,)
                )
            return self.add(output, self._named[bias_key])
        if isinstance(bias, np.ndarray):
            output = self.add(output, self.constant(bias, dtype, shape=(1,) * (rank - 1) + (out_features,)))
        return output

    def mark_output(self, tensor: Any, name: str, dtype: Any = None) -> Any:
        if dtype is not None:
            tensor = self.cast(tensor, dtype)
        tensor.name = name
        self.network.mark_output(tensor)
        return tensor


def build_plan(
    trt: Any,
    define: Callable[[Net], None],
    *,
    compute: Any,
    weight: Callable[[str], np.ndarray],
    profiles: Sequence[dict[str, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]],
    verbose: bool = False,
    workspace_bytes: int | None = None,
) -> Any:
    """Define one strongly typed network and return its serialized plan (TensorRT host memory)."""

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    net = Net(network, trt, compute, weight)
    define(net)
    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    # TF32 would change the float32 graphs' arithmetic against the float32 reference.
    config.clear_flag(trt.BuilderFlag.TF32)
    if workspace_bytes is not None:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    for shapes in profiles:
        profile = builder.create_optimization_profile()
        for name, (minimum, optimum, maximum) in shapes.items():
            profile.set_shape(name, minimum, optimum, maximum)
        config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build an ESMFold2 plan")
    del net
    return serialized
