# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph primitives of this family.

Engines are strongly typed. The work precision is FP32 or the checkpoint's own
BF16; operations Hugging Face evaluates in FP32 before rounding once (norms,
GELU, SiLU, and the decoder's attention scores and softmax) are expressed with
an explicit FP32 island, so the BF16 graph rounds where the reference model
rounds. The encoder's attention is TensorRT's attention layer at the work
precision.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from .checkpoint import Checkpoint

PRECISIONS = ("fp32", "bf16")


def float32_to_bfloat16_bits(values: np.ndarray) -> np.ndarray:
    """Round float32 values to bfloat16 (nearest, ties to even) bit patterns."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounding = ((bits >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
    return ((bits + rounding) >> np.uint32(16)).astype(np.uint16)


class Graph:
    """One strongly typed network plus the host arrays its constants borrow."""

    def __init__(self, network: trt.INetworkDefinition, checkpoint: Checkpoint,
                 precision: str) -> None:
        if precision not in PRECISIONS:
            raise ValueError(f"unsupported work precision {precision!r}")
        self.net = network
        self.checkpoint = checkpoint
        self.precision = precision
        self.work = trt.float32 if precision == "fp32" else trt.bfloat16
        # TensorRT keeps raw pointers into these arrays until serialization.
        self._keep: list[np.ndarray] = []

    # ---- constants -------------------------------------------------------

    def _constant(self, shape: tuple[int, ...], array: np.ndarray,
                  dtype: trt.DataType) -> trt.ITensor:
        array = np.ascontiguousarray(array)
        self._keep.append(array)
        if dtype == trt.bfloat16:
            if array.dtype != np.uint16:
                raise TypeError("bfloat16 constants take uint16 bit patterns")
            weights = trt.Weights(trt.DataType.BF16, array.ctypes.data, array.size)
        else:
            weights = trt.Weights(array)
        layer = self.net.add_constant(tuple(int(d) for d in shape), weights)
        if layer is None:
            raise RuntimeError("TensorRT rejected a constant")
        return layer.get_output(0)

    def f32(self, values: np.ndarray, shape: tuple[int, ...] | None = None) -> trt.ITensor:
        array = np.asarray(values, dtype=np.float32)
        return self._constant(array.shape if shape is None else shape, array, trt.float32)

    def i32(self, values: np.ndarray, shape: tuple[int, ...] | None = None) -> trt.ITensor:
        array = np.asarray(values, dtype=np.int32)
        return self._constant(array.shape if shape is None else shape, array, trt.int32)

    def i64(self, values: np.ndarray) -> trt.ITensor:
        array = np.asarray(values, dtype=np.int64)
        return self._constant(array.shape, array, trt.int64)

    def work_scalar(self, value: float, rank: int) -> trt.ITensor:
        """A rank-``rank`` broadcast scalar in the work precision."""
        shape = (1,) * rank
        if self.work == trt.float32:
            return self.f32(np.full(shape, value, dtype=np.float32))
        bits = float32_to_bfloat16_bits(np.full(shape, value, dtype=np.float32))
        return self._constant(shape, bits, trt.bfloat16)

    def weight(self, name: str, shape: tuple[int, ...] | None = None) -> trt.ITensor:
        """A checkpoint tensor as a constant in the work precision."""
        stored_shape = self.checkpoint.shape(name)
        target = stored_shape if shape is None else shape
        if int(np.prod(target)) != int(np.prod(stored_shape)):
            raise ValueError(f"cannot view {name} {stored_shape} as {target}")
        if self.work == trt.float32:
            return self._constant(target, self.checkpoint.float32(name), trt.float32)
        return self._constant(target, self.checkpoint.bfloat16_bits(name), trt.bfloat16)

    def weight_f32(self, name: str, shape: tuple[int, ...] | None = None) -> trt.ITensor:
        values = self.checkpoint.float32(name)
        return self._constant(values.shape if shape is None else shape, values, trt.float32)

    def raw_weights(self, name: str) -> trt.Weights:
        """A checkpoint tensor as layer weights in the work precision."""
        if self.work == trt.float32:
            values = self.checkpoint.float32(name)
            self._keep.append(values)
            return trt.Weights(values)
        bits = self.checkpoint.bfloat16_bits(name)
        self._keep.append(bits)
        return trt.Weights(trt.DataType.BF16, bits.ctypes.data, bits.size)

    # ---- elementwise -----------------------------------------------------

    def cast(self, tensor: trt.ITensor, dtype: trt.DataType) -> trt.ITensor:
        if tensor.dtype == dtype:
            return tensor
        return self.net.add_cast(tensor, dtype).get_output(0)

    def binary(self, a: trt.ITensor, b: trt.ITensor,
               op: trt.ElementWiseOperation) -> trt.ITensor:
        return self.net.add_elementwise(a, b, op).get_output(0)

    def add(self, a: trt.ITensor, b: trt.ITensor) -> trt.ITensor:
        return self.binary(a, b, trt.ElementWiseOperation.SUM)

    def mul(self, a: trt.ITensor, b: trt.ITensor) -> trt.ITensor:
        return self.binary(a, b, trt.ElementWiseOperation.PROD)

    def unary(self, x: trt.ITensor, op: trt.UnaryOperation) -> trt.ITensor:
        return self.net.add_unary(x, op).get_output(0)

    def reshape(self, x: trt.ITensor, dims: tuple[int, ...]) -> trt.ITensor:
        layer = self.net.add_shuffle(x)
        layer.reshape_dims = dims
        return layer.get_output(0)

    def transpose(self, x: trt.ITensor, order: tuple[int, ...]) -> trt.ITensor:
        layer = self.net.add_shuffle(x)
        layer.first_transpose = trt.Permutation(list(order))
        return layer.get_output(0)

    # ---- model operations ------------------------------------------------

    def linear(self, x: trt.ITensor, weight: str, bias: str | None = None) -> trt.ITensor:
        """``x @ W.T (+ b)`` for a rank-2 ``x`` and a Hugging Face ``[out, in]`` weight."""
        w = self.weight(weight)
        out = self.net.add_matrix_multiply(
            x, trt.MatrixOperation.NONE, w, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
        if bias is None:
            return out
        width = self.checkpoint.shape(bias)[0]
        return self.add(out, self.weight(bias, (1, width)))

    def rms_norm(self, x: trt.ITensor, gamma: str, eps: float) -> trt.ITensor:
        """Qwen3RMSNorm: normalize in FP32, round, then scale in the work precision."""
        rank = len(x.shape)
        xf = self.cast(x, trt.float32)
        mean = self.net.add_reduce(
            self.mul(xf, xf), trt.ReduceOperation.AVG, 1 << (rank - 1), keep_dims=True
        ).get_output(0)
        denom = self.unary(
            self.add(mean, self.f32(np.full((1,) * rank, eps, dtype=np.float32))),
            trt.UnaryOperation.SQRT,
        )
        normalized = self.binary(xf, denom, trt.ElementWiseOperation.DIV)
        width = self.checkpoint.shape(gamma)[0]
        scale = self.weight(gamma, (1,) * (rank - 1) + (width,))
        return self.mul(self.cast(normalized, self.work), scale)

    def layer_norm(self, x: trt.ITensor, gamma: str, beta: str, eps: float) -> trt.ITensor:
        """nn.LayerNorm evaluated in FP32 and rounded once, as PyTorch does."""
        rank = len(x.shape)
        width = self.checkpoint.shape(gamma)[0]
        shape = (1,) * (rank - 1) + (width,)
        layer = self.net.add_normalization_v2(
            self.cast(x, trt.float32),
            self.weight_f32(gamma, shape),
            self.weight_f32(beta, shape),
            1 << (rank - 1),
        )
        layer.epsilon = float(eps)
        return self.cast(layer.get_output(0), self.work)

    def gelu(self, x: trt.ITensor) -> trt.ITensor:
        """Exact (erf) GELU in FP32, rounded once."""
        out = self.net.add_activation(self.cast(x, trt.float32), trt.ActivationType.GELU_ERF)
        return self.cast(out.get_output(0), self.work)

    def silu(self, x: trt.ITensor) -> trt.ITensor:
        """``x * sigmoid(x)`` in FP32, rounded once."""
        xf = self.cast(x, trt.float32)
        gate = self.net.add_activation(xf, trt.ActivationType.SIGMOID).get_output(0)
        return self.cast(self.mul(xf, gate), self.work)


def builder_config(builder: trt.Builder) -> trt.IBuilderConfig:
    """Builder settings shared by every engine of this family.

    TF32 is disabled: an FP32 engine must compute FP32 products, and the BF16
    engine's explicit FP32 islands (attention scores) must stay FP32 too.
    """
    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    config.clear_flag(trt.BuilderFlag.TF32)
    return config


def new_network(builder: trt.Builder) -> trt.INetworkDefinition:
    return builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
