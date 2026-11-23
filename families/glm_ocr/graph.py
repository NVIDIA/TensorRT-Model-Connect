# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph primitives of the GLM-OCR family.

Every network is strongly typed. Activations run in the network dtype (FP32 or
BF16); normalizations reduce in FP32 and the vision rotary embedding rotates in
FP32, as the Transformers implementation does. Weight arrays arrive as float32
(BF16 checkpoint values are exact in float32) and are cast to the network dtype
inside the graph, where TensorRT folds the cast into the stored constant.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import tensorrt as trt

DTYPES = {"fp32": trt.float32, "bf16": trt.bfloat16}
# TensorRT keeps the first logger a process registers; every build reuses this one.
_LOGGER = trt.Logger(trt.Logger.WARNING)


class Net:
    """One strongly typed network plus the host arrays its constants reference."""

    def __init__(self, network: trt.INetworkDefinition, dtype: trt.DataType) -> None:
        self.network = network
        self.dtype = dtype
        self._keep: list[np.ndarray] = []

    # -- constants and casts ---------------------------------------------------------------

    def constant(self, values: np.ndarray, dtype: trt.DataType | None = None) -> trt.ITensor:
        """A float32 host array as a constant of ``dtype`` (the network dtype by default)."""
        array = np.ascontiguousarray(values, dtype=np.float32)
        self._keep.append(array)
        tensor = self.network.add_constant(array.shape, trt.Weights(array)).get_output(0)
        return self.cast(tensor, self.dtype if dtype is None else dtype)

    def indices(self, values) -> trt.ITensor:
        """An int32 constant; a 0-d array stays 0-d (a scalar gather index removes its axis)."""
        array = np.asarray(values, dtype=np.int32)
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        self._keep.append(array)
        return self.network.add_constant(array.shape, trt.Weights(array)).get_output(0)

    def cast(self, tensor: trt.ITensor, dtype: trt.DataType) -> trt.ITensor:
        if tensor.dtype == dtype:
            return tensor
        return self.network.add_cast(tensor, dtype).get_output(0)

    def scalar(self, value: float, rank: int, dtype: trt.DataType) -> trt.ITensor:
        return self.constant(np.full((1,) * rank, value, dtype=np.float32), dtype)

    # -- elementwise -----------------------------------------------------------------------

    def ew(self, left: trt.ITensor, right: trt.ITensor, op: trt.ElementWiseOperation) -> trt.ITensor:
        return self.network.add_elementwise(left, right, op).get_output(0)

    def add(self, left: trt.ITensor, right: trt.ITensor) -> trt.ITensor:
        return self.ew(left, right, trt.ElementWiseOperation.SUM)

    def sub(self, left: trt.ITensor, right: trt.ITensor) -> trt.ITensor:
        return self.ew(left, right, trt.ElementWiseOperation.SUB)

    def mul(self, left: trt.ITensor, right: trt.ITensor) -> trt.ITensor:
        return self.ew(left, right, trt.ElementWiseOperation.PROD)

    def div(self, left: trt.ITensor, right: trt.ITensor) -> trt.ITensor:
        return self.ew(left, right, trt.ElementWiseOperation.DIV)

    def silu(self, tensor: trt.ITensor) -> trt.ITensor:
        sigmoid = self.network.add_activation(tensor, trt.ActivationType.SIGMOID).get_output(0)
        return self.mul(tensor, sigmoid)

    def gelu(self, tensor: trt.ITensor) -> trt.ITensor:
        """Exact (erf) GELU, the torch.nn.GELU default."""
        return self.network.add_activation(tensor, trt.ActivationType.GELU_ERF).get_output(0)

    # -- shapes ----------------------------------------------------------------------------

    def reshape(self, tensor: trt.ITensor, dims: tuple[int, ...],
                second_transpose: tuple[int, ...] | None = None) -> trt.ITensor:
        layer = self.network.add_shuffle(tensor)
        layer.reshape_dims = dims
        if second_transpose is not None:
            layer.second_transpose = second_transpose
        return layer.get_output(0)

    def transpose_reshape(self, tensor: trt.ITensor, first_transpose: tuple[int, ...],
                          dims: tuple[int, ...]) -> trt.ITensor:
        layer = self.network.add_shuffle(tensor)
        layer.first_transpose = first_transpose
        layer.reshape_dims = dims
        return layer.get_output(0)

    def gather(self, tensor: trt.ITensor, indices: trt.ITensor, axis: int) -> trt.ITensor:
        return self.network.add_gather(tensor, indices, axis).get_output(0)

    def concat(self, tensors: list[trt.ITensor], axis: int) -> trt.ITensor:
        layer = self.network.add_concatenation(tensors)
        layer.axis = axis
        return layer.get_output(0)

    def last_row(self, tensor: trt.ITensor) -> trt.ITensor:
        """Rows ``[S, D]`` to the final row ``[1, D]`` for a runtime ``S``."""
        shape = self.network.add_shape(tensor).get_output(0)
        rows = self.network.add_slice(shape, (0,), (1,), (1,)).get_output(0)
        one = np.ones((1,), dtype=np.int64)
        self._keep.append(one)
        one_tensor = self.network.add_constant((1,), trt.Weights(one)).get_output(0)
        last = self.sub(rows, one_tensor)
        return self.gather(tensor, last, 0)

    # -- layers ----------------------------------------------------------------------------

    def linear(self, rows: trt.ITensor, weight: np.ndarray, bias: np.ndarray | None = None
               ) -> trt.ITensor:
        """``rows [R, in] @ weight[out, in]^T (+ bias[out])`` with the checkpoint's weight layout."""
        out_features, in_features = weight.shape
        assert rows.shape[-1] == in_features, (rows.shape, weight.shape)
        product = self.network.add_matrix_multiply(
            rows, trt.MatrixOperation.NONE, self.constant(weight), trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
        if bias is None:
            return product
        return self.add(product, self.constant(bias.reshape(1, out_features)))

    def _mean_last(self, tensor: trt.ITensor) -> trt.ITensor:
        axis = len(tensor.shape) - 1
        return self.network.add_reduce(tensor, trt.ReduceOperation.AVG, 1 << axis, True).get_output(0)

    def rms_norm(self, tensor: trt.ITensor, weight: np.ndarray, eps: float) -> trt.ITensor:
        """``weight * (x * rsqrt(mean(x^2) + eps)).to(dtype)``, reducing in FP32."""
        rank = len(tensor.shape)
        wide = self.cast(tensor, trt.float32)
        variance = self._mean_last(self.mul(wide, wide))
        variance = self.add(variance, self.scalar(eps, rank, trt.float32))
        root = self.network.add_unary(variance, trt.UnaryOperation.SQRT).get_output(0)
        normalized = self.cast(self.div(wide, root), tensor.dtype)
        scale = self.constant(weight.reshape((1,) * (rank - 1) + (weight.shape[0],)), tensor.dtype)
        return self.mul(scale, normalized)

    def layer_norm(self, tensor: trt.ITensor, weight: np.ndarray, bias: np.ndarray, eps: float
                   ) -> trt.ITensor:
        """torch.nn.LayerNorm over the last axis, computed in FP32."""
        rank = len(tensor.shape)
        wide = self.cast(tensor, trt.float32)
        centered = self.sub(wide, self._mean_last(wide))
        variance = self._mean_last(self.mul(centered, centered))
        variance = self.add(variance, self.scalar(eps, rank, trt.float32))
        root = self.network.add_unary(variance, trt.UnaryOperation.SQRT).get_output(0)
        shape = (1,) * (rank - 1) + (weight.shape[0],)
        scaled = self.mul(self.div(centered, root), self.constant(weight.reshape(shape), trt.float32))
        shifted = self.add(scaled, self.constant(bias.reshape(shape), trt.float32))
        return self.cast(shifted, tensor.dtype)

    def attention(self, query: trt.ITensor, key: trt.ITensor, value: trt.ITensor, *,
                  causal: bool, scale: float) -> trt.ITensor:
        """Softmax attention over ``[1, heads, rows, head_dim]``; the query is pre-scaled."""
        scaled = self.mul(query, self.scalar(scale, 4, query.dtype))
        layer = self.network.add_attention(
            scaled, key, value, trt.AttentionNormalizationOp.SOFTMAX, causal)
        if layer is None:
            raise RuntimeError("TensorRT refused the attention layer")
        # Use a fused kernel where one exists, otherwise its exact decomposition.
        layer.decomposable = True
        return layer.get_output(0)

    def repeat_heads(self, tensor: trt.ITensor, groups: int) -> trt.ITensor:
        """``[1, kv_heads, rows, d]`` to ``[1, kv_heads * groups, rows, d]``; head h reads h // groups."""
        if groups == 1:
            return tensor
        _, kv_heads, _, head_dim = tensor.shape
        expanded = self.reshape(tensor, (1, kv_heads, 1, -1, head_dim))
        repeated = self.concat([expanded] * groups, 2)
        return self.reshape(repeated, (1, kv_heads * groups, -1, head_dim))


def build_engine(
    populate: Callable[[Net], None],
    dtype: trt.DataType,
    profiles: list[dict[str, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]],
    *,
    verbose: bool,
) -> bytes:
    """Build one serialized engine from a network populated by ``populate``."""
    _LOGGER.min_severity = trt.Logger.VERBOSE if verbose else trt.Logger.WARNING
    builder = trt.Builder(_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    net = Net(network, dtype)
    populate(net)
    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    # FP32 graphs must compute FP32 products, not TF32 ones.
    config.clear_flag(trt.BuilderFlag.TF32)
    for shapes in profiles:
        profile = builder.create_optimization_profile()
        for name, (low, opt, high) in shapes.items():
            profile.set_shape(name, low, opt, high)
        config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT failed to build the glm_ocr engine")
    return bytes(plan)
