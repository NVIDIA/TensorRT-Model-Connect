# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned TensorRT graph helpers with explicit precision boundaries.

Activations are stored in the build precision (``work``). Matrix products always accumulate in FP32. In FP16 builds
the linear layers and the decoder attention multiply FP16 values cast to FP32, which TF32 tensor cores multiply
exactly; the vision attention's scaled queries and softmax probabilities are FP32 values that TF32 may round to a
10-bit mantissa, as FP16 attention kernels would. Normalizations, softmax inputs and the Gated DeltaNet recurrence
run in FP32, as the reference does.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

_EW = trt.ElementWiseOperation


class Graph:
    """A strongly typed network plus the build precision."""

    def __init__(self, network: trt.INetworkDefinition, work: trt.DataType) -> None:
        if work not in (trt.float16, trt.float32):
            raise ValueError("minicpm_ocr graphs are built in fp16 or fp32")
        self.network = network
        self.work = work
        self.np_work = np.float16 if work == trt.float16 else np.float32

    # -- basic tensors --------------------------------------------------------------------------------------------
    def constant(self, value, dtype=np.float32) -> trt.ITensor:
        array = np.asarray(value, dtype=dtype)
        if array.ndim == 0:
            array = array.reshape(1)
        array = np.ascontiguousarray(array)
        return self.network.add_constant(array.shape, trt.Weights(array)).get_output(0)

    def rounded(self, value) -> np.ndarray:
        """A parameter as the reference holds it in the build precision, widened to FP32."""
        return np.asarray(value, dtype=np.float32).astype(self.np_work).astype(np.float32)

    def cast(self, x: trt.ITensor, dtype: trt.DataType) -> trt.ITensor:
        return x if x.dtype == dtype else self.network.add_cast(x, dtype).get_output(0)

    def to_work(self, x: trt.ITensor) -> trt.ITensor:
        return self.cast(x, self.work)

    def fp32(self, x: trt.ITensor) -> trt.ITensor:
        return self.cast(x, trt.float32)

    def reshape(self, x: trt.ITensor, shape) -> trt.ITensor:
        layer = self.network.add_shuffle(x)
        layer.reshape_dims = tuple(shape)
        return layer.get_output(0)

    def transpose(self, x: trt.ITensor, order) -> trt.ITensor:
        layer = self.network.add_shuffle(x)
        layer.first_transpose = trt.Permutation(list(order))
        return layer.get_output(0)

    def binary(self, a: trt.ITensor, b: trt.ITensor, op) -> trt.ITensor:
        return self.network.add_elementwise(a, b, op).get_output(0)

    def add(self, a, b):
        return self.binary(a, b, _EW.SUM)

    def sub(self, a, b):
        return self.binary(a, b, _EW.SUB)

    def mul(self, a, b):
        return self.binary(a, b, _EW.PROD)

    def div(self, a, b):
        return self.binary(a, b, _EW.DIV)

    def unary(self, x: trt.ITensor, op) -> trt.ITensor:
        return self.network.add_unary(x, op).get_output(0)

    def activation(self, x: trt.ITensor, kind) -> trt.ITensor:
        return self.network.add_activation(x, kind).get_output(0)

    def scalar(self, value: float, dtype: trt.DataType, rank: int) -> trt.ITensor:
        return self.cast(self.constant(np.full((1,) * rank, value, dtype=np.float32)), dtype)

    def slice(self, x: trt.ITensor, start, shape) -> trt.ITensor:
        return self.network.add_slice(x, tuple(start), tuple(shape), (1,) * len(start)).get_output(0)

    def concat(self, tensors, axis: int) -> trt.ITensor:
        layer = self.network.add_concatenation(list(tensors))
        layer.axis = axis
        return layer.get_output(0)

    def reduce(self, x: trt.ITensor, op, axis: int) -> trt.ITensor:
        return self.network.add_reduce(x, op, 1 << axis, keep_dims=True).get_output(0)

    # -- model building blocks ------------------------------------------------------------------------------------
    def linear(self, x: trt.ITensor, weight_in_out: np.ndarray, bias: np.ndarray | None = None) -> trt.ITensor:
        """x [.., in] @ W [in, out] (+ bias) with FP32 accumulation; result in the build precision."""
        rank = len(tuple(x.shape))
        width_in, width_out = weight_in_out.shape
        weight = self.constant(self.rounded(weight_in_out).reshape((1,) * (rank - 2) + (width_in, width_out)))
        y = self.network.add_matrix_multiply(self.fp32(x), trt.MatrixOperation.NONE, weight,
                                             trt.MatrixOperation.NONE).get_output(0)
        if bias is not None:
            y = self.add(y, self.constant(self.rounded(bias).reshape((1,) * (rank - 1) + (width_out,))))
        return self.to_work(y)

    def rms_norm(self, x: trt.ITensor, scale: np.ndarray, eps: float, axis: int = -1) -> trt.ITensor:
        """x * rsqrt(mean(x^2) + eps) * scale in FP32; ``scale`` already includes any (1 + w) centring."""
        rank = len(tuple(x.shape))
        axis = axis % rank
        x32 = self.fp32(x)
        mean = self.reduce(self.mul(x32, x32), trt.ReduceOperation.AVG, axis)
        inv = self.unary(self.unary(self.add(mean, self.scalar(eps, trt.float32, rank)), trt.UnaryOperation.SQRT),
                         trt.UnaryOperation.RECIP)
        shape = [1] * rank
        shape[axis] = scale.shape[-1]
        if scale.ndim == 2:  # per-head scale table [heads, dim] for a [.., heads, dim] input
            shape[axis - 1] = scale.shape[0]
        weight = self.constant(np.asarray(scale, dtype=np.float32).reshape(shape))
        return self.to_work(self.mul(self.mul(x32, inv), weight))

    def layer_norm(self, x: trt.ITensor, gamma: np.ndarray, beta: np.ndarray, eps: float) -> trt.ITensor:
        rank = len(tuple(x.shape))
        shape = (1,) * (rank - 1) + (gamma.shape[0],)
        layer = self.network.add_normalization_v2(
            self.fp32(x), self.constant(self.rounded(gamma).reshape(shape)),
            self.constant(self.rounded(beta).reshape(shape)), 1 << (rank - 1))
        layer.epsilon = eps
        return self.to_work(layer.get_output(0))

    def silu32(self, x: trt.ITensor) -> trt.ITensor:
        """SiLU evaluated in FP32 (the reference activation computes internally in FP32)."""
        x = self.fp32(x)
        return self.mul(x, self.activation(x, trt.ActivationType.SIGMOID))

    def silu(self, x: trt.ITensor) -> trt.ITensor:
        return self.to_work(self.silu32(x))

    def gelu(self, x: trt.ITensor, approximate: bool) -> trt.ITensor:
        kind = trt.ActivationType.GELU_TANH if approximate else trt.ActivationType.GELU_ERF
        return self.to_work(self.activation(self.fp32(x), kind))

    def attention(self, q: trt.ITensor, k: trt.ITensor, v: trt.ITensor, scale: float) -> trt.ITensor:
        """Unmasked softmax attention on [1, heads, n, dim] with FP32 inputs and softmax; result in the build precision.

        Products accumulate in FP32 over the whole sequence. In FP16 builds TF32 may round the scaled queries and the
        probabilities to a 10-bit mantissa, as FP16 attention kernels would.
        """
        q = self.mul(self.fp32(q), self.scalar(scale, trt.float32, 4))
        layer = self.network.add_attention(q, self.fp32(k), self.fp32(v), trt.AttentionNormalizationOp.SOFTMAX,
                                           False)
        if layer is None:
            raise RuntimeError("minicpm_ocr: TensorRT rejected the attention layer")
        layer.decomposable = True
        return self.to_work(layer.get_output(0))

    def grouped_query_attention(self, q: trt.ITensor, k: trt.ITensor, v: trt.ITensor, mask: trt.ITensor,
                                scale: float) -> trt.ITensor:
        """One query row per head against a masked key/value window, written out as explicit operations.

        q [heads, dim]; k, v [kv_heads, window, dim]; additive mask [window]. Query head h reads key/value head
        h // (heads / kv_heads). Scores, the additive mask and the softmax are FP32; the probabilities are rounded to
        the build precision before the value product (as half-precision attention kernels do) and every product
        accumulates in FP32. The operations are explicit so that FP32 and FP16 engines apply the mask identically.
        """
        heads, dim = q.shape
        kv_heads = k.shape[0]
        group = heads // kv_heads
        query = self.mul(self.reshape(self.fp32(q), (kv_heads, group, dim)), self.scalar(scale, trt.float32, 3))
        scores = self.network.add_matrix_multiply(query, trt.MatrixOperation.NONE, self.fp32(k),
                                                  trt.MatrixOperation.TRANSPOSE).get_output(0)
        scores = self.add(scores, self.reshape(self.fp32(mask), (1, 1, -1)))
        softmax = self.network.add_softmax(scores)
        softmax.axes = 1 << 2
        probabilities = self.fp32(self.to_work(softmax.get_output(0)))
        context = self.network.add_matrix_multiply(probabilities, trt.MatrixOperation.NONE, self.fp32(v),
                                                   trt.MatrixOperation.NONE).get_output(0)
        return self.to_work(self.reshape(context, (1, heads * dim)))
