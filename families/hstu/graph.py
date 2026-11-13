# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strongly typed TensorRT expressions owned by the HSTU graph.

The work dtype is BF16 (the reference's) or FP32 (the structure validation).
Checkpoint weights are BF16 bits; a BF16 graph uses them as stored and an
FP32 graph widens them exactly. A GEMM's output is in the work dtype; the
epilogues after it (bias, LayerNorm, residual, L2 norm) compute in FP32 and
round once to the work dtype. The attention's pointwise ops between its two
GEMMs stay in the work dtype (``model._hstu_layer`` says why).
"""

from __future__ import annotations

import numpy as np

import tensorrt as trt

from .weights import bf16_bits_to_fp32


def new_builder(verbose: bool):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    # TensorRT's default optimization level; one stream (the runtime runs the
    # engines in order on one stream); FP32 GEMMs stay FP32 (the structure
    # validation compares them against the FP32 reference).
    config.builder_optimization_level = 3
    config.max_aux_streams = 0
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    return builder, network, config


class Graph:
    def __init__(self, network, work_dtype):
        if work_dtype not in (trt.bfloat16, trt.float32):
            raise ValueError("HSTU work dtype is bf16 or fp32")
        self.n = network
        self.work = work_dtype
        self.keep = []  # host arrays TensorRT reads at build time

    # --- constants -------------------------------------------------------
    def _constant(self, array: np.ndarray, dtype) -> trt.ITensor:
        array = np.ascontiguousarray(array)
        self.keep.append(array)
        weights = trt.Weights(dtype, array.ctypes.data, array.size)
        self.keep.append(weights)
        return self.n.add_constant(array.shape, weights).get_output(0)

    def weight(self, bits: np.ndarray) -> trt.ITensor:
        """A checkpoint tensor (BF16 bits) in the work dtype."""
        if self.work == trt.bfloat16:
            return self._constant(np.asarray(bits, dtype=np.uint16), trt.bfloat16)
        return self._constant(bf16_bits_to_fp32(bits), trt.float32)

    def weight_f32(self, bits: np.ndarray, shape: tuple[int, ...]) -> trt.ITensor:
        """A checkpoint tensor widened to FP32 (epilogue operands)."""
        return self._constant(bf16_bits_to_fp32(bits).reshape(shape), trt.float32)

    def f32(self, values, shape: tuple[int, ...]) -> trt.ITensor:
        return self._constant(np.asarray(values, dtype=np.float32).reshape(shape), trt.float32)

    def i32(self, values, shape: tuple[int, ...]) -> trt.ITensor:
        return self._constant(np.asarray(values, dtype=np.int32).reshape(shape), trt.int32)

    # --- expressions -----------------------------------------------------
    def cast(self, x, dtype):
        return x if x.dtype == dtype else self.n.add_cast(x, dtype).get_output(0)

    def to_f32(self, x):
        return self.cast(x, trt.float32)

    def to_work(self, x):
        return self.cast(x, self.work)

    def binary(self, a, b, op):
        return self.n.add_elementwise(a, b, op).get_output(0)

    def add(self, a, b):
        return self.binary(a, b, trt.ElementWiseOperation.SUM)

    def mul(self, a, b):
        return self.binary(a, b, trt.ElementWiseOperation.PROD)

    def div(self, a, b):
        return self.binary(a, b, trt.ElementWiseOperation.DIV)

    def maximum(self, a, b):
        return self.binary(a, b, trt.ElementWiseOperation.MAX)

    def scalar(self, value: float, rank: int):
        return self.f32([value], (1,) * rank)

    def work_scalar(self, value: float, rank: int):
        """A scalar in the work dtype (BF16-rounded in a BF16 graph)."""
        return self.to_work(self.scalar(value, rank))

    def silu(self, x):
        sigmoid = self.n.add_activation(x, trt.ActivationType.SIGMOID).get_output(0)
        return self.mul(x, sigmoid)

    def relu(self, x):
        return self.n.add_activation(x, trt.ActivationType.RELU).get_output(0)

    def reshape(self, x, shape):
        layer = self.n.add_shuffle(x)
        layer.reshape_dims = shape
        return layer.get_output(0)

    def heads(self, x, tokens: int, heads: int, head_dim: int):
        """[tokens, heads * head_dim] -> [heads, tokens, head_dim]."""
        layer = self.n.add_shuffle(x)
        layer.reshape_dims = (tokens, heads, head_dim)
        layer.second_transpose = (1, 0, 2)
        return layer.get_output(0)

    def merge_heads(self, x, tokens: int, width: int):
        """[heads, tokens, head_dim] -> [tokens, heads * head_dim]."""
        layer = self.n.add_shuffle(x)
        layer.first_transpose = (1, 0, 2)
        layer.reshape_dims = (tokens, width)
        return layer.get_output(0)

    def concat(self, tensors, axis: int):
        layer = self.n.add_concatenation(list(tensors))
        layer.axis = axis
        return layer.get_output(0)

    def rows(self, x, start: int, count: int):
        width = x.shape[1]
        return self.n.add_slice(x, (start, 0), (count, width), (1, 1)).get_output(0)

    def columns(self, x, start: int, count: int):
        tokens = x.shape[0]
        return self.n.add_slice(x, (0, start), (tokens, count), (1, 1)).get_output(0)

    def gather(self, table, indices):
        return self.n.add_gather(table, indices, 0).get_output(0)

    def matmul(self, a, b, *, transpose_b: bool = False):
        op_b = trt.MatrixOperation.TRANSPOSE if transpose_b else trt.MatrixOperation.NONE
        return self.n.add_matrix_multiply(a, trt.MatrixOperation.NONE, b, op_b).get_output(0)

    def linear(self, x, weight_bits: np.ndarray):
        """``x @ W^T`` for a PyTorch [out, in] weight, in the work dtype."""
        return self.matmul(x, self.weight(np.ascontiguousarray(np.asarray(weight_bits).T)))

    def layer_norm_f32(self, x, weight_bits, bias_bits, eps: float):
        """LayerNorm over the last axis of a rank-2 tensor, in FP32 (FP32 out)."""
        width = x.shape[1]
        layer = self.n.add_normalization_v2(
            self.to_f32(x),
            self.weight_f32(weight_bits, (1, width)),
            self.weight_f32(bias_bits, (1, width)),
            1 << 1,
        )
        layer.epsilon = eps
        return layer.get_output(0)

    def l2_normalize_f32(self, x, floor: float):
        """``x / max(||x||_2, floor)`` over the last axis, in FP32."""
        x = self.to_f32(x)
        squares = self.mul(x, x)
        total = self.n.add_reduce(squares, trt.ReduceOperation.SUM, 1 << 1, True).get_output(0)
        norm = self.n.add_unary(total, trt.UnaryOperation.SQRT).get_output(0)
        return self.div(x, self.maximum(norm, self.scalar(floor, 2)))

    def compare(self, a, b, op):
        return self.n.add_elementwise(a, b, op).get_output(0)

    def logical_not(self, x):
        return self.n.add_unary(x, trt.UnaryOperation.NOT).get_output(0)

    def select(self, condition, then, otherwise):
        return self.n.add_select(condition, then, otherwise).get_output(0)

    def output(self, x, name: str):
        x.name = name
        self.n.mark_output(x)
