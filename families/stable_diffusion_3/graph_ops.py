# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph helpers for the Stable Diffusion 3 engines.

Every engine is a strongly typed network with one work dtype, FP32 or BF16.
Matrix products, convolutions and attention run in the work dtype with the
checkpoint's weights; residual streams, normalizations, modulations and
activations run in FP32 between them. In an FP32 build the two coincide.
"""

from __future__ import annotations

import ml_dtypes
import numpy as np
import tensorrt as trt

FP32 = trt.float32
BF16 = trt.bfloat16
_WORK = {"fp32": FP32, "bf16": BF16}


def work_dtype(precision: str):
    """The TensorRT dtype of a precision name, or a clear error."""
    if precision not in _WORK:
        raise ValueError(f"stable_diffusion_3 builds fp32 or bf16 engines, not {precision!r}")
    return _WORK[precision]


class Graph:
    """One strongly typed network plus the host arrays its constants borrow.

    TensorRT reads constant weights when the engine is built, not when the
    layer is added, so every array handed to it is kept alive here.
    """

    def __init__(self, network, precision: str) -> None:
        self.network = network
        self.work = work_dtype(precision)
        self._arrays: list[np.ndarray] = []

    # -- constants and casts -------------------------------------------------

    def _weights(self, values, dtype=None):
        """``trt.Weights`` over a kept-alive host copy of ``values`` in ``dtype``."""
        dtype = self.work if dtype is None else dtype
        source = np.asarray(values)
        if dtype == FP32:
            array = np.ascontiguousarray(source, dtype=np.float32)
            weights = trt.Weights(array)
        elif dtype == BF16:
            # The (dtype, pointer, count) form does not copy the buffer.
            array = np.ascontiguousarray(source, dtype=ml_dtypes.bfloat16)
            weights = trt.Weights(BF16, array.ctypes.data, array.size)
        elif dtype == trt.int32:
            array = np.ascontiguousarray(source, dtype=np.int32)
            weights = trt.Weights(array)
        else:
            raise ValueError(f"unsupported constant dtype {dtype}")
        self._arrays.append(array)
        return weights

    def constant(self, values, dtype=None):
        """A constant tensor stored in ``dtype`` (default: the work dtype)."""
        shape = tuple(int(v) for v in np.shape(values))
        return self.network.add_constant(shape, self._weights(values, dtype)).get_output(0)

    def scalar(self, value: float, like, dtype=None):
        """A broadcastable scalar with the rank of ``like``."""
        rank = len(tuple(like.shape))
        dtype = like.dtype if dtype is None else dtype
        return self.constant(np.full((1,) * rank, value, dtype=np.float32), dtype=dtype)

    def cast(self, tensor, dtype):
        if tensor.dtype == dtype:
            return tensor
        return self.network.add_cast(tensor, dtype).get_output(0)

    def fp32(self, tensor):
        return self.cast(tensor, FP32)

    # -- elementwise ---------------------------------------------------------

    def _binary(self, left, right, operation):
        return self.network.add_elementwise(left, right, operation).get_output(0)

    def add(self, left, right):
        return self._binary(left, right, trt.ElementWiseOperation.SUM)

    def sub(self, left, right):
        return self._binary(left, right, trt.ElementWiseOperation.SUB)

    def mul(self, left, right):
        return self._binary(left, right, trt.ElementWiseOperation.PROD)

    def div(self, left, right):
        return self._binary(left, right, trt.ElementWiseOperation.DIV)

    def silu(self, tensor):
        tensor = self.fp32(tensor)
        gate = self.network.add_activation(tensor, trt.ActivationType.SIGMOID).get_output(0)
        return self.mul(tensor, gate)

    def gelu_tanh(self, tensor):
        """torch ``gelu(approximate="tanh")``, which transformers calls ``gelu_new``."""
        tensor = self.fp32(tensor)
        return self.network.add_activation(tensor, trt.ActivationType.GELU_TANH).get_output(0)

    def gelu_erf(self, tensor):
        tensor = self.fp32(tensor)
        return self.network.add_activation(tensor, trt.ActivationType.GELU_ERF).get_output(0)

    def quick_gelu(self, tensor):
        """CLIP's ``quick_gelu``: x * sigmoid(1.702 * x)."""
        tensor = self.fp32(tensor)
        scaled = self.mul(tensor, self.scalar(1.702, tensor))
        gate = self.network.add_activation(scaled, trt.ActivationType.SIGMOID).get_output(0)
        return self.mul(tensor, gate)

    # -- shape ---------------------------------------------------------------

    def reshape(self, tensor, dims, *, first_transpose=None, second_transpose=None):
        layer = self.network.add_shuffle(tensor)
        if first_transpose is not None:
            layer.first_transpose = tuple(first_transpose)
        if dims is not None:
            layer.reshape_dims = tuple(dims)
        if second_transpose is not None:
            layer.second_transpose = tuple(second_transpose)
        return layer.get_output(0)

    def concat(self, tensors, axis: int):
        layer = self.network.add_concatenation(list(tensors))
        layer.axis = axis
        return layer.get_output(0)

    def slice_last(self, tensor, start: int, size: int):
        """``tensor[..., start:start + size]`` for a tensor whose batch may be dynamic."""
        return self.slice_axis(tensor, len(tuple(tensor.shape)) - 1, start, size)

    def slice_axis(self, tensor, axis: int, start: int, size: int):
        """``tensor`` restricted to ``[start, start + size)`` along one static axis."""
        rank = len(tuple(tensor.shape))
        starts = [0] * rank
        starts[axis] = int(start)
        layer = self.network.add_slice(tensor, tuple(starts), (1,) * rank, (1,) * rank)
        # The size follows the runtime shape on every axis but the sliced one,
        # so a dynamic batch axis passes through unchanged.
        shape = self.network.add_shape(tensor).get_output(0)
        sizes = [
            self._int64(np.array([size], dtype=np.int64)) if index == axis
            else self._shape_element(shape, index)
            for index in range(rank)
        ]
        layer.set_input(2, self.concat(sizes, 0))
        return layer.get_output(0)

    def _int64(self, values):
        array = np.ascontiguousarray(values, dtype=np.int64)
        self._arrays.append(array)
        return self.network.add_constant(array.shape, trt.Weights(array)).get_output(0)

    def _shape_element(self, shape, index: int):
        indices = self._int64(np.array([index], dtype=np.int64))
        return self.network.add_gather(shape, indices, 0).get_output(0)

    # -- dense layers --------------------------------------------------------

    def linear(self, tensor, weight, bias=None, *, dtype=None):
        """``tensor @ weight.T + bias`` with ``weight`` stored as [out, in].

        The product runs in ``dtype`` (default: the work dtype) and the result
        is returned in FP32.
        """
        dtype = self.work if dtype is None else dtype
        weight = np.asarray(weight)
        out_features, in_features = (int(v) for v in weight.shape)
        rank = len(tuple(tensor.shape))
        if int(tuple(tensor.shape)[-1]) != in_features:
            raise ValueError(
                f"linear input width {tuple(tensor.shape)[-1]} does not match {in_features}")
        rhs = self.constant(weight.reshape((1,) * (rank - 2) + (out_features, in_features)),
                            dtype=dtype)
        product = self.network.add_matrix_multiply(
            self.cast(tensor, dtype), trt.MatrixOperation.NONE,
            rhs, trt.MatrixOperation.TRANSPOSE,
        ).get_output(0)
        product = self.fp32(product)
        if bias is None:
            return product
        bias_t = self.constant(np.asarray(bias).reshape((1,) * (rank - 1) + (out_features,)),
                               dtype=FP32)
        return self.add(product, bias_t)

    # -- normalization -------------------------------------------------------

    def layer_norm(self, tensor, eps: float, weight=None, bias=None):
        """LayerNorm over the last axis, computed and returned in FP32."""
        tensor = self.fp32(tensor)
        rank = len(tuple(tensor.shape))
        width = int(tuple(tensor.shape)[-1])
        shape = (1,) * (rank - 1) + (width,)
        gamma = np.ones(shape, np.float32) if weight is None else np.asarray(weight).reshape(shape)
        beta = np.zeros(shape, np.float32) if bias is None else np.asarray(bias).reshape(shape)
        layer = self.network.add_normalization(
            tensor, self.constant(gamma, dtype=FP32), self.constant(beta, dtype=FP32),
            1 << (rank - 1))
        layer.epsilon = float(eps)
        return layer.get_output(0)

    def rms_norm(self, tensor, eps: float, weight=None):
        """``x * rsqrt(mean(x^2) + eps) * weight`` over the last axis, in FP32."""
        tensor = self.fp32(tensor)
        rank = len(tuple(tensor.shape))
        squared = self.mul(tensor, tensor)
        mean = self.network.add_reduce(
            squared, trt.ReduceOperation.AVG, 1 << (rank - 1), True).get_output(0)
        root = self.network.add_unary(
            self.add(mean, self.scalar(eps, mean)), trt.UnaryOperation.SQRT).get_output(0)
        normed = self.div(tensor, root)
        if weight is None:
            return normed
        width = int(tuple(tensor.shape)[-1])
        gamma = np.asarray(weight).reshape((1,) * (rank - 1) + (width,))
        return self.mul(normed, self.constant(gamma, dtype=FP32))

    def group_norm(self, tensor, groups: int, eps: float, weight, bias):
        """GroupNorm over a static [N, C, H, W] tensor, computed and returned in FP32.

        The mean and the variance of the centred values are reduced in two
        passes: a VAE group at full resolution holds millions of values, and a
        single-pass variance loses precision there.
        """
        tensor = self.fp32(tensor)
        batch, channels, height, width = (int(v) for v in tuple(tensor.shape))
        if channels % groups:
            raise ValueError(f"{groups} groups do not divide {channels} channels")
        grouped = self.reshape(tensor, (batch, groups, (channels // groups) * height * width))
        mean = self.network.add_reduce(grouped, trt.ReduceOperation.AVG, 1 << 2, True).get_output(0)
        centred = self.sub(grouped, mean)
        variance = self.network.add_reduce(
            self.mul(centred, centred), trt.ReduceOperation.AVG, 1 << 2, True).get_output(0)
        root = self.network.add_unary(
            self.add(variance, self.scalar(eps, variance)), trt.UnaryOperation.SQRT).get_output(0)
        normed = self.reshape(self.div(centred, root), (batch, channels, height, width))
        shape = (1, channels, 1, 1)
        normed = self.mul(normed, self.constant(np.asarray(weight).reshape(shape), dtype=FP32))
        return self.add(normed, self.constant(np.asarray(bias).reshape(shape), dtype=FP32))

    # -- attention -----------------------------------------------------------

    def split_heads(self, tensor, heads: int, head_dim: int):
        """[B, S, H * D] -> [B, H, S, D]."""
        return self.reshape(tensor, (0, 0, heads, head_dim), second_transpose=(0, 2, 1, 3))

    def merge_heads(self, tensor, heads: int, head_dim: int):
        """[B, H, S, D] -> [B, S, H * D]."""
        return self.reshape(tensor, (0, 0, heads * head_dim), first_transpose=(0, 2, 1, 3))

    def attention(self, query, key, value, *, scale: float, causal: bool = False, bias=None):
        """softmax(q k^T * scale + bias) v over [B, H, S, D] tensors, in the work dtype.

        TensorRT's attention layer applies no score scale, so the query is
        scaled first, in FP32 before it is rounded to the work dtype: one
        rounding, and none at all for the power-of-two scales of the text
        encoders and the denoiser.
        """
        if scale != 1.0:
            query = self.fp32(query)
            query = self.mul(query, self.scalar(scale, query))
        query = self.cast(query, self.work)
        key = self.cast(key, self.work)
        value = self.cast(value, self.work)
        layer = self.network.add_attention(
            query, key, value, trt.AttentionNormalizationOp.SOFTMAX, causal)
        if layer is None:
            raise RuntimeError("TensorRT rejected the attention layer")
        if bias is not None:
            layer.mask = self.cast(bias, self.work)
        layer.decomposable = True
        return layer.get_output(0)

    # -- convolution ---------------------------------------------------------

    def conv2d(self, tensor, weight, bias, *, stride=1, padding=0):
        """2-D convolution in the work dtype; ``weight`` is [out, in, kh, kw]. Returns FP32."""
        weight = np.asarray(weight)
        out_channels, _, kernel_h, kernel_w = (int(v) for v in weight.shape)
        tensor = self.cast(tensor, self.work)
        layer = self.network.add_convolution_nd(
            tensor, out_channels, (kernel_h, kernel_w), self._weights(weight),
            trt.Weights() if bias is None else self._weights(np.asarray(bias)))
        layer.stride_nd = (stride, stride)
        layer.padding_nd = (padding, padding)
        return self.fp32(layer.get_output(0))

    def upsample_nearest2x(self, tensor):
        """``F.interpolate(x, scale_factor=2, mode="nearest")`` on [N, C, H, W]."""
        layer = self.network.add_resize(tensor)
        layer.resize_mode = trt.InterpolationMode.NEAREST
        layer.coordinate_transformation = trt.ResizeCoordinateTransformation.ASYMMETRIC
        layer.nearest_rounding = trt.ResizeRoundMode.FLOOR
        batch, channels, height, width = (int(v) for v in tuple(tensor.shape))
        layer.shape = (batch, channels, height * 2, width * 2)
        return layer.get_output(0)

    # -- outputs -------------------------------------------------------------

    def output(self, tensor, name: str) -> None:
        tensor = self.fp32(tensor)
        tensor.name = name
        self.network.mark_output(tensor)


def build_engine(populate, *, precision: str, verbose: bool, profile=None) -> trt.IHostMemory:
    """Build one strongly typed engine whose layers ``populate(graph)`` adds."""
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    # FP32 tensors are real FP32: TF32 would round them to 10 mantissa bits.
    config.clear_flag(trt.BuilderFlag.TF32)
    graph = Graph(network, precision)
    populate(graph)
    if profile is not None:
        optimization_profile = builder.create_optimization_profile()
        for name, (minimum, optimum, maximum) in profile.items():
            optimization_profile.set_shape(name, minimum, optimum, maximum)
        config.add_optimization_profile(optimization_profile)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("stable_diffusion_3 TensorRT engine build failed")
    return plan
