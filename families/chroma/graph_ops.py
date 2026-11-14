# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strongly typed TensorRT graph helpers owned by the Chroma family.

Every tensor carries an explicit type. ``Graph.dtype`` is the compute type of
the component (FP32 or BF16); normalization statistics, softmax inputs and the
rotary embedding are evaluated in FP32 and cast back, as the reference
implementation does for BF16 tensors.
"""

from __future__ import annotations

import math

import ml_dtypes
import numpy as np
import tensorrt as trt

COMPUTE_TYPES = {"fp32": trt.float32, "bf16": trt.bfloat16}


class Graph:
    """One strongly typed network plus the host arrays its constants borrow."""

    def __init__(self, network: trt.INetworkDefinition, precision: str) -> None:
        if precision not in COMPUTE_TYPES:
            raise ValueError(f"unsupported compute precision {precision!r}")
        self.network = network
        self.precision = precision
        self.dtype = COMPUTE_TYPES[precision]
        # TensorRT reads constant buffers during the build; they must stay alive.
        self._owners: list[np.ndarray] = []

    # ----------------------------------------------------------------- tensors
    def constant(self, values, dtype: trt.DataType | None = None) -> trt.ITensor:
        target = self.dtype if dtype is None else dtype
        array = np.asarray(values)
        if target == trt.bfloat16:
            host = np.ascontiguousarray(array.astype(ml_dtypes.bfloat16, copy=False))
            weights = trt.Weights(trt.bfloat16, host.ctypes.data, host.size)
        elif target == trt.float32:
            host = np.ascontiguousarray(array.astype(np.float32, copy=False))
            weights = trt.Weights(host)
        elif target == trt.int32:
            host = np.ascontiguousarray(array.astype(np.int32, copy=False))
            weights = trt.Weights(host)
        else:
            raise ValueError(f"unsupported constant type {target}")
        self._owners.append(host)
        return self.network.add_constant(tuple(array.shape), weights).get_output(0)

    def scalar(self, value: float, like: trt.ITensor) -> trt.ITensor:
        """A rank-matched scalar constant in the type of ``like``."""
        rank = len(like.shape)
        return self.constant(np.full((1,) * rank, value, dtype=np.float32), like.dtype)

    def cast(self, tensor: trt.ITensor, dtype: trt.DataType) -> trt.ITensor:
        if tensor.dtype == dtype:
            return tensor
        return self.network.add_cast(tensor, dtype).get_output(0)

    def reshape(self, tensor: trt.ITensor, shape) -> trt.ITensor:
        layer = self.network.add_shuffle(tensor)
        layer.reshape_dims = tuple(shape)
        return layer.get_output(0)

    def transpose(self, tensor: trt.ITensor, order) -> trt.ITensor:
        layer = self.network.add_shuffle(tensor)
        layer.first_transpose = trt.Permutation(list(order))
        return layer.get_output(0)

    def slice(self, tensor: trt.ITensor, start, shape) -> trt.ITensor:
        stride = (1,) * len(start)
        return self.network.add_slice(tensor, tuple(start), tuple(shape), stride).get_output(0)

    def concat(self, tensors, axis: int) -> trt.ITensor:
        layer = self.network.add_concatenation(list(tensors))
        layer.axis = axis
        return layer.get_output(0)

    def add(self, left: trt.ITensor, right: trt.ITensor) -> trt.ITensor:
        return self._binary(left, right, trt.ElementWiseOperation.SUM)

    def sub(self, left: trt.ITensor, right: trt.ITensor) -> trt.ITensor:
        return self._binary(left, right, trt.ElementWiseOperation.SUB)

    def mul(self, left: trt.ITensor, right: trt.ITensor) -> trt.ITensor:
        return self._binary(left, right, trt.ElementWiseOperation.PROD)

    def div(self, left: trt.ITensor, right: trt.ITensor) -> trt.ITensor:
        return self._binary(left, right, trt.ElementWiseOperation.DIV)

    def _binary(self, left, right, operation) -> trt.ITensor:
        if left.dtype != right.dtype:
            raise TypeError(f"element-wise operands differ in type: {left.dtype} vs {right.dtype}")
        return self.network.add_elementwise(left, right, operation).get_output(0)

    def unary(self, tensor: trt.ITensor, operation: trt.UnaryOperation) -> trt.ITensor:
        return self.network.add_unary(tensor, operation).get_output(0)

    def activation(self, tensor: trt.ITensor, kind: trt.ActivationType) -> trt.ITensor:
        return self.network.add_activation(tensor, kind).get_output(0)

    def mean(self, tensor: trt.ITensor, axis: int) -> trt.ITensor:
        axis = axis % len(tensor.shape)
        return self.network.add_reduce(tensor, trt.ReduceOperation.AVG, 1 << axis, True).get_output(0)

    # ------------------------------------------------------------------ layers
    def linear(
        self,
        tensor: trt.ITensor,
        weight: np.ndarray,
        bias: np.ndarray | None = None,
        *,
        dtype: trt.DataType | None = None,
    ) -> trt.ITensor:
        """``tensor @ weight.T + bias`` with a ``[out, in]`` weight stored as given.

        ``dtype`` selects the GEMM type; it defaults to the tensor's own type.
        """
        target = tensor.dtype if dtype is None else dtype
        tensor = self.cast(tensor, target)
        rank = len(tensor.shape)
        out_features, in_features = weight.shape
        if tensor.shape[-1] != in_features:
            raise ValueError(f"linear input width {tensor.shape[-1]} != weight {in_features}")
        matrix = self.constant(weight.reshape((1,) * (rank - 2) + weight.shape), target)
        product = self.network.add_matrix_multiply(
            tensor, trt.MatrixOperation.NONE, matrix, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
        if bias is None:
            return product
        return self.add(product, self.constant(bias.reshape((1,) * (rank - 1) + (out_features,)), target))

    def silu(self, tensor: trt.ITensor) -> trt.ITensor:
        """``x * sigmoid(x)`` in FP32, rounded once (``torch.nn.SiLU`` on CUDA)."""
        x = self.cast(tensor, trt.float32)
        return self.cast(self.mul(x, self.activation(x, trt.ActivationType.SIGMOID)), tensor.dtype)

    def gelu_tanh(self, tensor: trt.ITensor) -> trt.ITensor:
        return self.activation(tensor, trt.ActivationType.GELU_TANH)

    def layer_norm(self, tensor: trt.ITensor, eps: float) -> trt.ITensor:
        """LayerNorm over the last axis without affine parameters, statistics in FP32."""
        x = self.cast(tensor, trt.float32)
        centered = self.sub(x, self.mean(x, -1))
        variance = self.mean(self.mul(centered, centered), -1)
        inverse = self.unary(self.add(variance, self.scalar(eps, variance)), trt.UnaryOperation.SQRT)
        normalized = self.div(centered, inverse)
        return self.cast(normalized, tensor.dtype)

    def rms_norm(
        self,
        tensor: trt.ITensor,
        weight: np.ndarray | None,
        eps: float,
        *,
        scale_in_float32: bool = False,
    ) -> trt.ITensor:
        """``x * rsqrt(mean(x^2) + eps) * weight`` with FP32 statistics.

        ``scale_in_float32=False`` is the transformers T5LayerNorm order: the
        normalized value is rounded to the input type before the weight scales
        it. ``True`` is ``torch.nn.RMSNorm`` on CUDA: the weight is applied in
        FP32 and the result is rounded once. Both are identical in FP32.
        """
        x = self.cast(tensor, trt.float32)
        variance = self.mean(self.mul(x, x), -1)
        root = self.unary(self.add(variance, self.scalar(eps, variance)), trt.UnaryOperation.SQRT)
        normalized = self.div(x, root)
        if weight is None:
            return self.cast(normalized, tensor.dtype)
        rank = len(tensor.shape)
        shape = (1,) * (rank - 1) + (weight.shape[0],)
        if scale_in_float32:
            return self.cast(self.mul(normalized, self.constant(weight.reshape(shape), trt.float32)), tensor.dtype)
        return self.mul(self.cast(normalized, tensor.dtype), self.constant(weight.reshape(shape), tensor.dtype))

    def group_norm(
        self,
        tensor: trt.ITensor,
        groups: int,
        weight: np.ndarray,
        bias: np.ndarray,
        eps: float,
    ) -> trt.ITensor:
        """GroupNorm of an ``[N, C, H, W]`` tensor in FP32, affine included, rounded once (as on CUDA)."""
        n, channels, height, width = tensor.shape
        x = self.cast(tensor, trt.float32)
        grouped = self.reshape(x, (n, groups, (channels // groups) * height * width))
        centered = self.sub(grouped, self.mean(grouped, -1))
        variance = self.mean(self.mul(centered, centered), -1)
        root = self.unary(self.add(variance, self.scalar(eps, variance)), trt.UnaryOperation.SQRT)
        normalized = self.reshape(self.div(centered, root), (n, channels, height, width))
        gamma = self.constant(weight.reshape(1, channels, 1, 1), trt.float32)
        beta = self.constant(bias.reshape(1, channels, 1, 1), trt.float32)
        return self.cast(self.add(self.mul(normalized, gamma), beta), tensor.dtype)

    def conv2d(
        self,
        tensor: trt.ITensor,
        weight: np.ndarray,
        bias: np.ndarray | None,
        padding: int,
    ) -> trt.ITensor:
        out_channels, _, kernel_h, kernel_w = weight.shape
        kernel = self.constant(weight, tensor.dtype)
        layer = self.network.add_convolution_nd(
            tensor, out_channels, (kernel_h, kernel_w), trt.Weights(), trt.Weights()
        )
        layer.set_input(1, kernel)
        layer.padding_nd = (padding, padding)
        output = layer.get_output(0)
        if bias is None:
            return output
        return self.add(output, self.constant(bias.reshape(1, out_channels, 1, 1), tensor.dtype))

    def upsample_nearest_2x(self, tensor: trt.ITensor) -> trt.ITensor:
        layer = self.network.add_resize(tensor)
        layer.resize_mode = trt.InterpolationMode.NEAREST
        layer.coordinate_transformation = trt.ResizeCoordinateTransformation.ASYMMETRIC
        layer.nearest_rounding = trt.ResizeRoundMode.FLOOR
        layer.scales = [1.0, 1.0, 2.0, 2.0]
        return layer.get_output(0)

    def attention(
        self,
        query: trt.ITensor,
        key: trt.ITensor,
        value: trt.ITensor,
        mask: trt.ITensor | None,
    ) -> trt.ITensor:
        """Softmax attention of ``[B, H, S, D]`` tensors; any score scaling is pre-applied.

        ``mask`` is an additive term broadcast to ``[B, H, S, S]`` in the query type.
        """
        layer = self.network.add_attention(
            query, key, value, trt.AttentionNormalizationOp.SOFTMAX, False
        )
        layer.decomposable = True
        if mask is not None:
            layer.mask = self.cast(mask, query.dtype)
        return layer.get_output(0)


def timestep_embedding(values: np.ndarray, channels: int) -> np.ndarray:
    """``get_timestep_embedding(values, channels, flip_sin_to_cos=True, downscale_freq_shift=0)``.

    Evaluated with the reference's own float32 torch operations: arguments
    reach several hundred thousand, where a one-ulp difference in a frequency
    already moves sin/cos by about 1e-2.
    """
    import torch

    half = channels // 2
    exponent = -math.log(10000.0) * torch.arange(0, half, dtype=torch.float32)
    frequencies = torch.exp(exponent / half)
    arguments = torch.from_numpy(np.asarray(values, dtype=np.float32))[:, None] * frequencies[None, :]
    return torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=-1).numpy()
