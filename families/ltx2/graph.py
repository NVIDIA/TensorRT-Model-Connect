# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned TensorRT network helpers for LTX-2.5 engine builds.

Every network is strongly typed. Activations are bf16 (the precision LTX-2.5 and
its Gemma 4 text encoder are trained and run in); normalization statistics,
RoPE, timestep sinusoids and softmax-free reductions run in fp32 islands.

Linear weights stay in the checkpoint's ``[out, in]`` layout and are consumed by
a transposed matrix multiply, so multi-GB checkpoints are neither transposed nor
copied on the host. bf16 constants are passed to TensorRT without a copy and kept
alive by the :class:`Graph` that created them.
"""

from __future__ import annotations

import math
from typing import Sequence

import ml_dtypes
import numpy as np
import tensorrt as trt

BF16 = ml_dtypes.bfloat16


def np_dtype_for(dtype: "trt.DataType"):
    if dtype == trt.bfloat16:
        return BF16
    if dtype == trt.float16:
        return np.float16
    if dtype == trt.float32:
        return np.float32
    if dtype == trt.int32:
        return np.int32
    raise ValueError(f"unsupported constant dtype {dtype}")


class Graph:
    """Thin stateful wrapper around one ``INetworkDefinition``."""

    def __init__(self, network: "trt.INetworkDefinition"):
        self.net = network
        self._keepalive: list[np.ndarray] = []

    # ------------------------------------------------------------------ constants

    def const(self, values, dtype: "trt.DataType" = trt.float32, shape: Sequence[int] | None = None):
        """Constant tensor of ``dtype``; ``values`` is converted (round-to-nearest-even) if needed."""
        arr = np.asarray(values)
        target = np_dtype_for(dtype)
        if arr.dtype != target:
            arr = arr.astype(target)
        arr = np.ascontiguousarray(arr)
        if shape is None:
            shape = arr.shape if arr.ndim else (1,)
        # The explicit (type, pointer, count) form: implicit NumPy dtype detection differs per
        # platform (e.g. int32 on Windows) and does not know bf16. TensorRT does not copy.
        weights = trt.Weights(dtype, arr.ctypes.data, arr.size)
        self._keepalive.append(arr)
        return self.net.add_constant(tuple(int(s) for s in shape), weights).get_output(0)

    def weights(self, values, dtype: "trt.DataType") -> "trt.Weights":
        """``trt.Weights`` in ``dtype`` for layer parameters (kept alive with the graph)."""
        arr = np.ascontiguousarray(np.asarray(values).astype(np_dtype_for(dtype)))
        self._keepalive.append(arr)
        return trt.Weights(dtype, arr.ctypes.data, arr.size)

    def scalar(self, value: float, dtype: "trt.DataType", rank: int):
        return self.const(np.full((1,) * rank, value, dtype=np.float32), dtype)

    # ------------------------------------------------------------------ basic ops

    def cast(self, x, dtype: "trt.DataType"):
        if x.dtype == dtype:
            return x
        return self.net.add_cast(x, dtype).get_output(0)

    def ew(self, a, b, op):
        return self.net.add_elementwise(a, b, op).get_output(0)

    def add(self, a, b):
        return self.ew(a, b, trt.ElementWiseOperation.SUM)

    def sub(self, a, b):
        return self.ew(a, b, trt.ElementWiseOperation.SUB)

    def mul(self, a, b):
        return self.ew(a, b, trt.ElementWiseOperation.PROD)

    def div(self, a, b):
        return self.ew(a, b, trt.ElementWiseOperation.DIV)

    def maximum(self, a, b):
        return self.ew(a, b, trt.ElementWiseOperation.MAX)

    def minimum(self, a, b):
        return self.ew(a, b, trt.ElementWiseOperation.MIN)

    def unary(self, x, op):
        return self.net.add_unary(x, op).get_output(0)

    def reduce(self, x, op, axis: int, keep_dims: bool = True):
        rank = len(x.shape)
        axis = axis % rank
        return self.net.add_reduce(x, op, 1 << axis, keep_dims).get_output(0)

    def reshape(self, x, shape: Sequence[int], *, first: Sequence[int] | None = None,
                second: Sequence[int] | None = None):
        layer = self.net.add_shuffle(x)
        if first is not None:
            layer.first_transpose = trt.Permutation(list(first))
        layer.reshape_dims = tuple(int(s) for s in shape)
        if second is not None:
            layer.second_transpose = trt.Permutation(list(second))
        return layer.get_output(0)

    def transpose(self, x, perm: Sequence[int]):
        layer = self.net.add_shuffle(x)
        layer.first_transpose = trt.Permutation(list(perm))
        return layer.get_output(0)

    def slice(self, x, start: Sequence[int], size: Sequence[int], stride: Sequence[int] | None = None):
        if stride is None:
            stride = (1,) * len(start)
        return self.net.add_slice(x, tuple(start), tuple(size), tuple(stride)).get_output(0)

    def concat(self, xs, axis: int):
        layer = self.net.add_concatenation(list(xs))
        layer.axis = axis
        return layer.get_output(0)

    def gather(self, x, indices, axis: int):
        return self.net.add_gather(x, indices, axis).get_output(0)

    def select(self, cond, a, b):
        return self.net.add_select(cond, a, b).get_output(0)

    def mark_output(self, x, name: str, dtype: "trt.DataType | None" = None):
        if dtype is not None:
            x = self.cast(x, dtype)
        x.name = name
        self.net.mark_output(x)
        return x

    # ------------------------------------------------------------------ layers

    def linear(self, x, weight: np.ndarray, bias: np.ndarray | None = None):
        """``x @ weight.T + bias`` with ``weight`` in checkpoint ``[out, in]`` layout."""
        rank = len(x.shape)
        out_f, in_f = (int(s) for s in weight.shape)
        w = self.const(weight, x.dtype, shape=(1,) * (rank - 2) + (out_f, in_f))
        y = self.net.add_matrix_multiply(
            x, trt.MatrixOperation.NONE, w, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
        if bias is not None:
            b = self.const(bias, x.dtype, shape=(1,) * (rank - 1) + (out_f,))
            y = self.add(y, b)
        return y

    def rms_norm(self, x, weight: np.ndarray | None, eps: float, *, out_dtype=None):
        """RMSNorm over the last axis with fp32 statistics (``x * rsqrt(mean(x^2) + eps) * w``)."""
        out_dtype = out_dtype or x.dtype
        rank = len(x.shape)
        xf = self.cast(x, trt.float32)
        ms = self.reduce(self.mul(xf, xf), trt.ReduceOperation.AVG, -1)
        inv = self.unary(self.unary(self.add(ms, self.scalar(eps, trt.float32, rank)),
                                    trt.UnaryOperation.SQRT), trt.UnaryOperation.RECIP)
        y = self.mul(xf, inv)
        if weight is not None:
            y = self.mul(y, self.const(np.asarray(weight, dtype=np.float32), trt.float32,
                                       shape=(1,) * (rank - 1) + (int(weight.shape[-1]),)))
        return self.cast(y, out_dtype)

    def layer_norm(self, x, eps: float, *, out_dtype=None):
        """Non-affine LayerNorm over the last axis in fp32."""
        out_dtype = out_dtype or x.dtype
        rank = len(x.shape)
        xf = self.cast(x, trt.float32)
        mean = self.reduce(xf, trt.ReduceOperation.AVG, -1)
        centered = self.sub(xf, mean)
        var = self.reduce(self.mul(centered, centered), trt.ReduceOperation.AVG, -1)
        inv = self.unary(self.unary(self.add(var, self.scalar(eps, trt.float32, rank)),
                                    trt.UnaryOperation.SQRT), trt.UnaryOperation.RECIP)
        return self.cast(self.mul(centered, inv), out_dtype)

    def gelu_tanh(self, x):
        """GELU, tanh approximation (``gelu_pytorch_tanh`` / ``gelu-approximate``).

        Evaluated in fp32 and rounded once to x.dtype, like torch's bf16 kernel.
        """
        rank = len(x.shape)
        out_dtype = x.dtype
        x = self.cast(x, trt.float32)
        c = lambda v: self.scalar(v, trt.float32, rank)  # noqa: E731
        inner = self.mul(c(math.sqrt(2.0 / math.pi)),
                         self.add(x, self.mul(c(0.044715), self.mul(self.mul(x, x), x))))
        t = self.net.add_activation(inner, trt.ActivationType.TANH).get_output(0)
        return self.cast(self.mul(self.mul(c(0.5), x), self.add(c(1.0), t)), out_dtype)

    def silu(self, x):
        """SiLU evaluated in fp32 and rounded once to x.dtype."""
        out_dtype = x.dtype
        x = self.cast(x, trt.float32)
        s = self.net.add_activation(x, trt.ActivationType.SIGMOID).get_output(0)
        return self.cast(self.mul(x, s), out_dtype)

    def sigmoid(self, x):
        return self.net.add_activation(x, trt.ActivationType.SIGMOID).get_output(0)

    def attention(self, q, k, v, *, scale: float | None = None, mask=None):
        """Softmax attention over ``[B, H, S, D]`` tensors (TensorRT IAttention).

        IAttention computes raw ``Q @ K^T``; Q is pre-scaled by ``scale``
        (default ``1/sqrt(D)``; ``scale=1.0`` skips the multiply).
        """
        if scale is None:
            scale = 1.0 / math.sqrt(int(q.shape[-1]))
        if scale != 1.0:
            q = self.mul(q, self.scalar(scale, q.dtype, 4))
        layer = self.net.add_attention(q, k, v, trt.AttentionNormalizationOp.SOFTMAX, False)
        if layer is None:
            raise RuntimeError("TensorRT rejected the attention layer")
        layer.decomposable = True
        if mask is not None:
            layer.mask = mask
        return layer.get_output(0)

    def timestep_sinusoid(self, t, dim: int = 256, max_period: float = 10000.0):
        """diffusers ``Timesteps(dim, flip_sin_to_cos=True, downscale_freq_shift=0)``: [B] -> [B, dim] fp32."""
        half = dim // 2
        freqs = np.exp(-math.log(max_period) * np.arange(half, dtype=np.float32) / half).astype(np.float32)
        b = int(t.shape[0])
        t2 = self.reshape(self.cast(t, trt.float32), (b, 1))
        args = self.mul(t2, self.const(freqs.reshape(1, half), trt.float32))
        return self.concat([self.unary(args, trt.UnaryOperation.COS),
                            self.unary(args, trt.UnaryOperation.SIN)], axis=1)


def new_network(logger: "trt.ILogger"):
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    return builder, network


def build_plan(builder, network, *, label: str = "engine", tf32: bool = True):
    """Serialized plan (a bytes-like ``IHostMemory``; multi-GB plans are not copied again).

    ``tf32=False`` keeps fp32 convolutions / matrix multiplies in full fp32 (TensorRT allows
    TF32 for fp32 layers by default).
    """
    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    if not tf32:
        config.clear_flag(trt.BuilderFlag.TF32)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"TensorRT failed to build the LTX-2.5 {label}")
    return plan


_LOGGERS: dict[bool, "trt.ILogger"] = {}


def make_logger(verbose: bool = False):
    """One TensorRT logger per process and verbosity."""
    if verbose not in _LOGGERS:
        _LOGGERS[verbose] = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    return _LOGGERS[verbose]
