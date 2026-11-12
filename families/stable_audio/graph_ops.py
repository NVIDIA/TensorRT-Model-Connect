# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT network helpers owned by the stable_audio family.

Every network is strongly typed: each helper computes in the dtype of its input,
and precision changes are explicit casts made by the builders.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

_NUMPY = {trt.float32: np.float32, trt.float16: np.float16, trt.int32: np.int32}


class Graph:
    """One strongly typed network plus the host arrays its constants borrow until the build.

    ``tf32_engine`` states that the engine is built with TF32 enabled (FP16 engines, see
    ``_fp32_operands``); FP32 projections then avoid the matrix-multiply layer, which TF32
    would round.
    """

    def __init__(self, network: trt.INetworkDefinition, *, tf32_engine: bool = False) -> None:
        self.network = network
        self.tf32_engine = tf32_engine
        self._arrays: list[np.ndarray] = []

    # ----------------------------------------------------------------- basics
    def constant(self, values, dtype: trt.DataType, shape: tuple[int, ...] | None = None) -> trt.ITensor:
        array = np.asarray(values, dtype=_NUMPY[dtype])
        if shape is not None:
            array = array.reshape(shape)
        if array.ndim == 0:
            raise ValueError("stable_audio graph constants carry an explicit rank")
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        self._arrays.append(array)
        return self.network.add_constant(tuple(array.shape), trt.Weights(array)).get_output(0)

    def scalar(self, value: float, like: trt.ITensor) -> trt.ITensor:
        """A broadcastable constant of ``like``'s rank and dtype."""
        return self.constant([value], like.dtype, shape=(1,) * len(like.shape))

    def cast(self, tensor: trt.ITensor, dtype: trt.DataType) -> trt.ITensor:
        if tensor.dtype == dtype:
            return tensor
        return self.network.add_cast(tensor, dtype).get_output(0)

    def _binary(self, left: trt.ITensor, right: trt.ITensor, operation) -> trt.ITensor:
        return self.network.add_elementwise(left, right, operation).get_output(0)

    def add(self, left: trt.ITensor, right: trt.ITensor) -> trt.ITensor:
        return self._binary(left, right, trt.ElementWiseOperation.SUM)

    def sub(self, left: trt.ITensor, right: trt.ITensor) -> trt.ITensor:
        return self._binary(left, right, trt.ElementWiseOperation.SUB)

    def mul(self, left: trt.ITensor, right: trt.ITensor) -> trt.ITensor:
        return self._binary(left, right, trt.ElementWiseOperation.PROD)

    def div(self, left: trt.ITensor, right: trt.ITensor) -> trt.ITensor:
        return self._binary(left, right, trt.ElementWiseOperation.DIV)

    def maximum(self, left: trt.ITensor, right: trt.ITensor) -> trt.ITensor:
        return self._binary(left, right, trt.ElementWiseOperation.MAX)

    def minimum(self, left: trt.ITensor, right: trt.ITensor) -> trt.ITensor:
        return self._binary(left, right, trt.ElementWiseOperation.MIN)

    def unary(self, tensor: trt.ITensor, operation) -> trt.ITensor:
        return self.network.add_unary(tensor, operation).get_output(0)

    def sigmoid(self, tensor: trt.ITensor) -> trt.ITensor:
        return self.network.add_activation(tensor, trt.ActivationType.SIGMOID).get_output(0)

    def relu(self, tensor: trt.ITensor) -> trt.ITensor:
        return self.network.add_activation(tensor, trt.ActivationType.RELU).get_output(0)

    def silu(self, tensor: trt.ITensor) -> trt.ITensor:
        """``x * sigmoid(x)``, torch's SiLU."""
        return self.mul(tensor, self.sigmoid(tensor))

    def reshape(self, tensor: trt.ITensor, dims, *, first=None, second=None) -> trt.ITensor:
        """Shuffle; a 0 in ``dims`` copies that input dimension (the dynamic batch)."""
        layer = self.network.add_shuffle(tensor)
        if first is not None:
            layer.first_transpose = tuple(first)
        if dims is not None:
            layer.reshape_dims = tuple(dims)
        if second is not None:
            layer.second_transpose = tuple(second)
        return layer.get_output(0)

    def concat(self, tensors, axis: int) -> trt.ITensor:
        layer = self.network.add_concatenation(list(tensors))
        layer.axis = axis
        return layer.get_output(0)

    def gather(self, tensor: trt.ITensor, indices, axis: int) -> trt.ITensor:
        index = self.constant(np.asarray(indices, dtype=np.int32), trt.int32)
        return self.network.add_gather(tensor, index, axis).get_output(0)

    def matmul(self, left: trt.ITensor, right: trt.ITensor, *, transpose_right: bool = False) -> trt.ITensor:
        return self.network.add_matrix_multiply(
            left, trt.MatrixOperation.NONE, right,
            trt.MatrixOperation.TRANSPOSE if transpose_right else trt.MatrixOperation.NONE,
        ).get_output(0)

    # ----------------------------------------------------------------- layers
    def _fp32_operands(self, tensor: trt.ITensor, weight, bias):
        """Operands of a projection or convolution computed as FP32.

        An FP16 input keeps FP16 values (the weights and bias are rounded to FP16) but is
        multiplied as FP32: TensorRT may otherwise pick FP16-accumulating tactics, and FP16
        values are exact in TF32, so the TF32 tensor cores of an FP16 engine form exact
        products and accumulate them in FP32.
        """
        weight = np.asarray(weight, dtype=np.float32)
        if tensor.dtype == trt.float16:
            weight = weight.astype(np.float16).astype(np.float32)
            if bias is not None:
                bias = np.asarray(bias, dtype=np.float32).astype(np.float16).astype(np.float32)
        return self.cast(tensor, trt.float32), weight, bias

    def _convolution_input(self, tensor: trt.ITensor) -> None:
        """Convolutions have no FP32 path that TF32 leaves exact (``linear`` has one)."""
        if tensor.dtype == trt.float32 and self.tf32_engine:
            raise ValueError("stable_audio: an FP32 convolution in a TF32 engine would be TF32-rounded")

    def linear(self, tensor: trt.ITensor, weight, bias=None, *,
               out_dtype: trt.DataType | None = None) -> trt.ITensor:
        """``x @ W.T + b`` over the last axis; ``weight`` is [out, in].

        Computed as FP32 (see ``_fp32_operands``); the result has ``out_dtype``, by default
        the input's dtype.
        """
        source, weight, bias = self._fp32_operands(tensor, weight, bias)
        out_features, in_features = (int(v) for v in weight.shape)
        rank = len(tensor.shape)
        if tensor.dtype == trt.float32 and self.tf32_engine:
            # An FP32 input in a TF32 engine: products and sum without the matrix multiply,
            # so they stay FP32. Only small projections take this path (O(in x out) per row).
            expanded = self.reshape(source, (0,) * (rank - 1) + (in_features, 1))
            rhs = self.constant(weight.T, trt.float32, shape=(1,) * (rank - 1) + (in_features, out_features))
            result = self.network.add_reduce(
                self.mul(expanded, rhs), trt.ReduceOperation.SUM, 1 << (rank - 1), False).get_output(0)
        else:
            rhs = self.constant(weight.T, trt.float32, shape=(1,) * (rank - 2) + (in_features, out_features))
            result = self.matmul(source, rhs)
        if bias is not None:
            result = self.add(result, self.constant(bias, trt.float32, shape=(1,) * (rank - 1) + (out_features,)))
        return self.cast(result, tensor.dtype if out_dtype is None else out_dtype)

    def layer_norm(self, tensor: trt.ITensor, gamma, beta, eps: float) -> trt.ITensor:
        """LayerNorm over the last axis, computed in FP32 and returned in the input dtype."""
        source = self.cast(tensor, trt.float32)
        rank = len(source.shape)
        shape = (1,) * (rank - 1) + (int(np.asarray(gamma).size),)
        layer = self.network.add_normalization(
            source, self.constant(gamma, trt.float32, shape=shape),
            self.constant(beta, trt.float32, shape=shape), 1 << (rank - 1))
        layer.epsilon = float(eps)
        return self.cast(layer.get_output(0), tensor.dtype)

    def rms_norm(self, tensor: trt.ITensor, gamma, eps: float) -> trt.ITensor:
        """T5LayerNorm: ``w * x / sqrt(mean(x^2) + eps)`` without centring, in FP32."""
        source = self.cast(tensor, trt.float32)
        rank = len(source.shape)
        mean = self.network.add_reduce(
            self.mul(source, source), trt.ReduceOperation.AVG, 1 << (rank - 1), True).get_output(0)
        root = self.unary(self.add(mean, self.scalar(eps, mean)), trt.UnaryOperation.SQRT)
        normed = self.div(source, root)
        shape = (1,) * (rank - 1) + (int(np.asarray(gamma).size),)
        return self.cast(self.mul(normed, self.constant(gamma, trt.float32, shape=shape)), tensor.dtype)

    def softmax(self, tensor: trt.ITensor, axis: int) -> trt.ITensor:
        layer = self.network.add_softmax(tensor)
        layer.axes = 1 << axis
        return layer.get_output(0)

    def attention(self, query: trt.ITensor, key: trt.ITensor, value: trt.ITensor, scale: float) -> trt.ITensor:
        """Unmasked softmax attention over [B, H, S, D] through TensorRT's IAttention.

        IAttention does not scale scores, so the query is scaled first. ``decomposable``
        lets TensorRT fall back to primitive layers where no fused kernel applies (FP32).
        """
        scaled = self.mul(query, self.scalar(scale, query))
        layer = self.network.add_attention(scaled, key, value, trt.AttentionNormalizationOp.SOFTMAX, False)
        if layer is None:
            raise RuntimeError("stable_audio could not add an attention layer")
        layer.decomposable = True
        return layer.get_output(0)

    def conv1d(self, tensor: trt.ITensor, weight, bias=None, *, dilation: int = 1,
               padding: int = 0) -> trt.ITensor:
        """Conv1d on a [N, C, 1, L] tensor, computed as FP32 (see ``_fp32_operands``) and
        returned in the input's dtype; ``weight`` is [out, in, k]."""
        self._convolution_input(tensor)
        source, weight, bias = self._fp32_operands(tensor, weight, bias)
        out_channels, _, kernel = (int(v) for v in weight.shape)
        kernel_array = np.ascontiguousarray(weight.reshape(out_channels, -1, 1, kernel), dtype=np.float32)
        self._arrays.append(kernel_array)
        bias_weights = trt.Weights(trt.float32)  # empty: no bias
        if bias is not None:
            bias_array = np.ascontiguousarray(bias, dtype=np.float32)
            self._arrays.append(bias_array)
            bias_weights = trt.Weights(bias_array)
        layer = self.network.add_convolution_nd(
            source, out_channels, (1, kernel), trt.Weights(kernel_array), bias_weights)
        layer.stride_nd = (1, 1)
        layer.padding_nd = (0, padding)
        layer.dilation_nd = (1, dilation)
        return self.cast(layer.get_output(0), tensor.dtype)

    def conv_transpose1d(self, tensor: trt.ITensor, weight, bias, *, stride: int, padding: int) -> trt.ITensor:
        """ConvTranspose1d on a [N, C, 1, L] tensor, computed as FP32 (see ``_fp32_operands``)
        and returned in the input's dtype; ``weight`` is [in, out, k]."""
        self._convolution_input(tensor)
        source, weight, bias = self._fp32_operands(tensor, weight, bias)
        in_channels, out_channels, kernel = (int(v) for v in weight.shape)
        kernel_array = np.ascontiguousarray(weight.reshape(in_channels, out_channels, 1, kernel), dtype=np.float32)
        bias_array = np.ascontiguousarray(bias, dtype=np.float32)
        self._arrays.extend((kernel_array, bias_array))
        layer = self.network.add_deconvolution_nd(
            source, out_channels, (1, kernel), trt.Weights(kernel_array), trt.Weights(bias_array))
        layer.stride_nd = (1, stride)
        layer.padding_nd = (0, padding)
        return self.cast(layer.get_output(0), tensor.dtype)
