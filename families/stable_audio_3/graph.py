# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph helpers for the Stable Audio 3 engines.

Every network is strongly typed. ``Graph.work`` is the trunk dtype; the places
where the reference computes in FP32 whatever the model dtype is (RMSNorm,
rotary angles and rotation, Fourier features) cast to FP32 explicitly here.
"""

from __future__ import annotations

import math

import ml_dtypes
import numpy as np
import tensorrt as trt

_NUMPY = {
    trt.float32: np.float32,
    trt.float16: np.float16,
    trt.bfloat16: ml_dtypes.bfloat16,
    trt.int32: np.int32,
    trt.int64: np.int64,
}
# Large enough to remove a key from the softmax, small enough for FP16.
MASKED = -60000.0
LOW_PRECISION = (trt.float16, trt.bfloat16)


class Graph:
    """One network plus the numpy buffers its constants point at."""

    def __init__(self, network: trt.INetworkDefinition, work: trt.DataType) -> None:
        self.n = network
        self.work = work
        self._anchors: list[np.ndarray] = []

    # -- constants -----------------------------------------------------------
    def const(self, values, dtype: trt.DataType | None = None, shape=None) -> trt.ITensor:
        dtype = self.work if dtype is None else dtype
        array = np.asarray(values)
        if shape is not None:
            array = array.reshape(shape)
        array = np.ascontiguousarray(array.astype(_NUMPY[dtype]))
        self._anchors.append(array)
        if dtype == trt.bfloat16:
            weights = trt.Weights(trt.bfloat16, array.ctypes.data, array.size)
        else:
            weights = trt.Weights(array)
        return self.n.add_constant(tuple(array.shape), weights).get_output(0)

    def scalar(self, value: float, like: trt.ITensor) -> trt.ITensor:
        return self.const(np.full((1,) * len(like.shape), value, dtype=np.float32), like.dtype)

    # -- dtype and elementwise -------------------------------------------------
    def cast(self, x: trt.ITensor, dtype: trt.DataType) -> trt.ITensor:
        if x.dtype == dtype:
            return x
        return self.n.add_cast(x, dtype).get_output(0)

    def _binary(self, a, b, op):
        return self.n.add_elementwise(a, b, op).get_output(0)

    def add(self, a, b):
        return self._binary(a, b, trt.ElementWiseOperation.SUM)

    def sub(self, a, b):
        return self._binary(a, b, trt.ElementWiseOperation.SUB)

    def mul(self, a, b):
        return self._binary(a, b, trt.ElementWiseOperation.PROD)

    def div(self, a, b):
        return self._binary(a, b, trt.ElementWiseOperation.DIV)

    def unary(self, x, op):
        return self.n.add_unary(x, op).get_output(0)

    def act(self, x, kind):
        return self.n.add_activation(x, kind).get_output(0)

    def silu(self, x):
        return self.mul(x, self.act(x, trt.ActivationType.SIGMOID))

    def mul_scalar(self, x, value: float):
        return self.mul(x, self.scalar(value, x))

    def add_scalar(self, x, value: float):
        return self.add(x, self.scalar(value, x))

    # -- shapes ----------------------------------------------------------------
    def shuffle(self, x, reshape=None, first=None, second=None, shape_tensor=None):
        layer = self.n.add_shuffle(x)
        if first is not None:
            layer.first_transpose = tuple(first)
        if shape_tensor is not None:
            layer.set_input(1, shape_tensor)
        elif reshape is not None:
            layer.reshape_dims = tuple(reshape)
        if second is not None:
            layer.second_transpose = tuple(second)
        return layer.get_output(0)

    def transpose(self, x, order):
        return self.shuffle(x, first=order)

    def concat(self, tensors, axis: int):
        layer = self.n.add_concatenation(list(tensors))
        layer.axis = axis
        return layer.get_output(0)

    def shape_of(self, x) -> trt.ITensor:
        return self.cast(self.n.add_shape(x).get_output(0), trt.int64)

    def dim(self, x, index: int) -> trt.ITensor:
        """One dimension of ``x`` as a 1-element int64 shape tensor."""
        indices = self.const(np.array([index], dtype=np.int32), trt.int32)
        return self.n.add_gather(self.shape_of(x), indices, 0).get_output(0)

    def shape_vector(self, parts) -> trt.ITensor:
        """Concatenate ints and 1-element int64 tensors into one shape tensor."""
        tensors = [part if isinstance(part, trt.ITensor)
                   else self.const(np.array([part], dtype=np.int64), trt.int64) for part in parts]
        return self.concat(tensors, 0)

    def int_mul(self, a, value: int):
        return self.mul(a, self.const(np.array([value], dtype=np.int64), trt.int64))

    def int_add(self, a, value: int):
        return self.add(a, self.const(np.array([value], dtype=np.int64), trt.int64))

    def reshape(self, x, parts):
        """Reshape with a mix of static ints and dynamic 1-element dims."""
        if all(isinstance(part, int) for part in parts):
            return self.shuffle(x, reshape=parts)
        return self.shuffle(x, shape_tensor=self.shape_vector(parts))

    def slice(self, x, start, size, stride=None):
        """Static or dynamic slice; ``start``/``size`` items may be shape tensors."""
        rank = len(x.shape)
        stride = stride or (1,) * rank
        dynamic = any(isinstance(item, trt.ITensor) for item in (*start, *size))
        if not dynamic:
            return self.n.add_slice(x, tuple(start), tuple(size), tuple(stride)).get_output(0)
        layer = self.n.add_slice(x, (0,) * rank, (1,) * rank, tuple(stride))
        layer.set_input(1, self.shape_vector(start))
        layer.set_input(2, self.shape_vector(size))
        return layer.get_output(0)

    def split_last(self, x, parts: int):
        """Split the last axis into ``parts`` equal chunks (``torch.chunk``)."""
        rank = len(x.shape)
        width = int(x.shape[-1]) // parts
        lead = [self.dim(x, axis) if int(x.shape[axis]) < 0 else int(x.shape[axis]) for axis in range(rank - 1)]
        return [self.slice(x, (0,) * (rank - 1) + (index * width,), (*lead, width)) for index in range(parts)]

    # -- layers ----------------------------------------------------------------
    def matmul(self, a, b):
        """``a @ b``. FP16 / BF16 operands are multiplied in FP32 and the product is rounded once.

        The reference's cuBLAS GEMMs accumulate low-precision products in FP32; a low-precision TensorRT matrix
        multiply may not. The operands keep their low-precision values, which TF32 represents exactly
        (``model.products_exact_in_tf32``).
        """
        low = a.dtype
        if low not in LOW_PRECISION:
            return self.n.add_matrix_multiply(a, trt.MatrixOperation.NONE, b, trt.MatrixOperation.NONE).get_output(0)
        product = self.n.add_matrix_multiply(self.cast(a, trt.float32), trt.MatrixOperation.NONE,
                                             self.cast(b, trt.float32), trt.MatrixOperation.NONE).get_output(0)
        return self.cast(product, low)

    def linear(self, x, weight: np.ndarray, bias: np.ndarray | None = None):
        """``x @ weight.T + bias`` over the last axis; ``weight`` is [out, in], stored in ``x``'s dtype.

        Low-precision inputs accumulate in FP32 with the bias added in FP32 and one final rounding (see ``matmul``).
        """
        compute = trt.float32 if x.dtype in LOW_PRECISION else x.dtype
        weight = np.asarray(weight, dtype=np.float32)
        out_features, in_features = weight.shape
        rank = len(x.shape)
        rhs = self.const(weight.T, x.dtype, shape=(1,) * (rank - 2) + (in_features, out_features))
        y = self.n.add_matrix_multiply(self.cast(x, compute), trt.MatrixOperation.NONE, self.cast(rhs, compute),
                                       trt.MatrixOperation.NONE).get_output(0)
        if bias is not None:
            bias = self.const(np.asarray(bias, dtype=np.float32), x.dtype, shape=(1,) * (rank - 1) + (out_features,))
            y = self.add(y, self.cast(bias, compute))
        return self.cast(y, x.dtype)

    def rms_norm(self, x, gamma: np.ndarray | None, eps: float, offset: float = 0.0):
        """RMSNorm over the last axis, computed in FP32 and returned in ``x``'s dtype.

        ``offset`` = 1 gives the Gemma form ``x_hat * (1 + w)``.
        """
        out_dtype = x.dtype
        x32 = self.cast(x, trt.float32)
        rank = len(x.shape)
        mean = self.n.add_reduce(self.mul(x32, x32), trt.ReduceOperation.AVG, 1 << (rank - 1), True).get_output(0)
        denom = self.unary(self.add_scalar(mean, eps), trt.UnaryOperation.SQRT)
        y = self.div(x32, denom)
        if gamma is not None:
            scale = np.asarray(gamma, dtype=np.float32) + np.float32(offset)
            y = self.mul(y, self.const(scale, trt.float32, shape=(1,) * (rank - 1) + (scale.size,)))
        return self.cast(y, out_dtype)

    def dynamic_tanh(self, x, alpha: np.ndarray, gamma: np.ndarray, beta: np.ndarray):
        """``gamma * tanh(alpha * x) + beta`` (DynamicTanh) in the trunk dtype."""
        rank = len(x.shape)
        width = int(np.asarray(gamma).size)
        shape = (1,) * (rank - 1) + (width,)
        y = self.mul(x, self.const(np.asarray(alpha, dtype=np.float32), x.dtype, shape=(1,) * rank))
        y = self.act(y, trt.ActivationType.TANH)
        y = self.mul(self.const(gamma, x.dtype, shape=shape), y)
        return self.add(y, self.const(beta, x.dtype, shape=shape))

    def attention(self, q, k, v, scale: float, mask=None):
        """``softmax(q k^T * scale + mask) v`` on [B, H, S, D] through IAttention."""
        q = self.mul_scalar(q, scale)
        layer = self.n.add_attention(q, k, v, trt.AttentionNormalizationOp.SOFTMAX, False)
        layer.decomposable = True
        if mask is not None:
            layer.mask = self.cast(mask, q.dtype)
        return layer.get_output(0)

    # -- accurate trigonometry for large FP32 arguments -------------------------
    def cos_sin(self, angle):
        """cos and sin of an FP32 angle after an exact-enough reduction to [-pi, pi].

        The Fourier features reach 2*pi*1e4 rad; reducing first keeps the result
        independent of how a backend approximates cos/sin of large arguments.
        """
        two_pi_hi = 6.28125  # 8 significant bits: k * hi is exact for |k| < 2**16
        two_pi_mid = float(np.float32(2.0 * math.pi - two_pi_hi))
        two_pi_lo = 2.0 * math.pi - two_pi_hi - two_pi_mid
        turns = self.unary(self.mul_scalar(angle, 1.0 / (2.0 * math.pi)), trt.UnaryOperation.ROUND)
        reduced = self.sub(angle, self.mul_scalar(turns, two_pi_hi))
        reduced = self.sub(reduced, self.mul_scalar(turns, two_pi_mid))
        reduced = self.sub(reduced, self.mul_scalar(turns, two_pi_lo))
        return self.unary(reduced, trt.UnaryOperation.COS), self.unary(reduced, trt.UnaryOperation.SIN)


def torch_linspace(start: float, end: float, steps: int) -> np.ndarray:
    """``torch.linspace`` in float32, with its symmetric two-sided evaluation."""
    if steps == 1:
        return np.array([start], dtype=np.float32)
    start32, end32 = np.float32(start), np.float32(end)
    step = np.float32((end32 - start32) / np.float32(steps - 1))
    index = np.arange(steps)
    values = np.where(index < steps // 2,
                      start32 + step * index.astype(np.float32),
                      end32 - step * (steps - index - 1).astype(np.float32))
    return values.astype(np.float32)


def expo_fourier_frequencies(features: int, min_freq: float, max_freq: float) -> np.ndarray:
    """The FP32 frequency ramp of the reference ``ExpoFourierFeatures``."""
    ramp = torch_linspace(0.0, 1.0, features // 2)
    log_min = np.float32(math.log(min_freq))
    log_span = np.float32(math.log(max_freq) - math.log(min_freq))
    exponent = (ramp * log_span + log_min).astype(np.float32)
    # Correctly rounded float32 exp; an ulp at 1e4 Hz is 6e-3 rad at t = 1.
    return np.exp(exponent.astype(np.float64)).astype(np.float32)


def expo_fourier_features(g: Graph, value, frequencies: np.ndarray):
    """``cat(cos, sin)(value * freqs * 2 * pi)`` in FP32 for a [1, 1] FP32 input."""
    value = g.cast(value, trt.float32)
    freqs = g.const(frequencies, trt.float32, shape=(1, frequencies.size))
    angle = g.mul_scalar(g.mul_scalar(g.mul(value, freqs), 2.0), math.pi)
    cos, sin = g.cos_sin(angle)
    return g.concat([cos, sin], 1)


def rotary_tables(g: Graph, inv_freq: np.ndarray, positions: int):
    """FP32 cos/sin tables [positions, 2 * len(inv_freq)] of the reference RotaryEmbedding."""
    t = np.arange(positions, dtype=np.float32)
    freqs = np.outer(t, np.asarray(inv_freq, dtype=np.float32)).astype(np.float32)
    freqs = np.concatenate([freqs, freqs], axis=-1)
    angles = freqs.astype(np.float64)
    return (g.const(np.cos(angles), trt.float32), g.const(np.sin(angles), trt.float32))


def apply_rotary(g: Graph, x, cos, sin, rot_dim: int, head_dim: int, out_dtype):
    """Rotate the first ``rot_dim`` features of [B, H, S, D] in FP32 (``rotate_half`` layout).

    ``cos``/``sin`` are FP32 [S, rot_dim] tables already cut to the sequence.
    """
    x32 = g.cast(x, trt.float32)
    rank = len(x.shape)
    lead = [g.dim(x, axis) if int(x.shape[axis]) < 0 else int(x.shape[axis]) for axis in range(rank - 1)]
    half = rot_dim // 2
    first = g.slice(x32, (0,) * rank, (*lead, half))
    second = g.slice(x32, (0,) * (rank - 1) + (half,), (*lead, half))
    rotated_half = g.concat([g.mul_scalar(second, -1.0), first], rank - 1)
    rotated = g.slice(x32, (0,) * rank, (*lead, rot_dim))
    out = g.add(g.mul(rotated, cos), g.mul(rotated_half, sin))
    if rot_dim < head_dim:
        rest = g.slice(x32, (0,) * (rank - 1) + (rot_dim,), (*lead, head_dim - rot_dim))
        out = g.concat([out, rest], rank - 1)
    return g.cast(out, out_dtype)
