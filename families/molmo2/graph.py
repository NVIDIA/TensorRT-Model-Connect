# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Family-local strongly typed equations for the Molmo2 decoder and vision tower."""

from __future__ import annotations

import numpy as np
import tensorrt as trt


class Graph:
    def __init__(self, precision, verbose=False):
        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
        self.builder = trt.Builder(self.logger)
        self.net = self.builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        )
        self.dtype = {"fp32": trt.float32, "bf16": trt.bfloat16}[precision]
        self.constants = []
        self.profile = self.builder.create_optimization_profile()

    def constant(self, value, dtype=None):
        array = np.ascontiguousarray(value)
        if array.dtype.kind == "f":
            array = array.astype(np.float32)
        self.constants.append(array)
        tensor = self.net.add_constant(array.shape, array).get_output(0)
        return self.cast(tensor, dtype) if dtype is not None else tensor

    def cast(self, x, dtype):
        return x if x.dtype == dtype else self.net.add_cast(x, dtype).get_output(0)

    def input(self, name, dtype, minimum, optimum, maximum):
        shape = tuple(a if a == c else -1 for a, c in zip(minimum, maximum))
        tensor = self.net.add_input(name, dtype, shape)
        if -1 in shape:
            self.profile.set_shape(name, minimum, optimum, maximum)
        return tensor

    def output(self, x, name, dtype=None):
        x = self.cast(x, dtype) if dtype is not None else x
        x.name = name
        self.net.mark_output(x)

    def reshape(self, x, shape, transpose=None):
        layer = self.net.add_shuffle(x)
        layer.reshape_dims = shape
        if transpose is not None:
            layer.second_transpose = transpose
        return layer.get_output(0)

    def transpose(self, x, order):
        layer = self.net.add_shuffle(x)
        layer.first_transpose = order
        return layer.get_output(0)

    def ew(self, a, b, op):
        return self.net.add_elementwise(a, b, op).get_output(0)

    def add(self, a, b):
        return self.ew(a, b, trt.ElementWiseOperation.SUM)

    def mul(self, a, b):
        return self.ew(a, b, trt.ElementWiseOperation.PROD)

    def scalar(self, value, rank, dtype=trt.float32):
        return self.constant(np.full((1,) * rank, value, np.float32), dtype)

    def linear(self, x, weight, bias=None):
        rank = len(x.shape)
        w = self.constant(weight.reshape((1,) * (rank - 2) + weight.shape), x.dtype)
        y = self.net.add_matrix_multiply(
            x, trt.MatrixOperation.NONE, w, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
        if bias is not None:
            y = self.add(y, self.constant(bias.reshape((1,) * (rank - 1) + (-1,)), x.dtype))
        return y

    def reduce(self, x, operation, axis=-1, keep=True):
        axis %= len(x.shape)
        return self.net.add_reduce(x, operation, 1 << axis, keep).get_output(0)

    def rms(self, x, weight, eps):
        f = self.cast(x, trt.float32)
        variance = self.reduce(self.mul(f, f), trt.ReduceOperation.AVG)
        scale = self.net.add_unary(
            self.add(variance, self.scalar(eps, len(x.shape))), trt.UnaryOperation.SQRT
        ).get_output(0)
        normalized = self.ew(f, scale, trt.ElementWiseOperation.DIV)
        # Upstream rounds the normalized value before the affine multiply.
        return self.mul(
            self.cast(normalized, x.dtype),
            self.constant(weight.reshape((1,) * (len(x.shape) - 1) + (-1,)), x.dtype),
        )

    def norm(self, x, weight, bias, eps):
        f = self.cast(x, trt.float32)
        mean = self.reduce(f, trt.ReduceOperation.AVG)
        center = self.ew(f, mean, trt.ElementWiseOperation.SUB)
        var = self.reduce(self.mul(center, center), trt.ReduceOperation.AVG)
        denominator = self.net.add_unary(
            self.add(var, self.scalar(eps, len(x.shape))), trt.UnaryOperation.SQRT
        ).get_output(0)
        y = self.ew(center, denominator, trt.ElementWiseOperation.DIV)
        y = self.add(
            self.mul(y, self.constant(weight.reshape((1,) * (len(x.shape) - 1) + (-1,)))),
            self.constant(bias.reshape((1,) * (len(x.shape) - 1) + (-1,))),
        )
        return self.cast(y, x.dtype)

    def silu(self, x):
        sigmoid = self.net.add_activation(x, trt.ActivationType.SIGMOID).get_output(0)
        return self.mul(x, sigmoid)

    def gelu(self, x):
        f = self.cast(x, trt.float32)
        cube = self.mul(self.mul(f, f), f)
        z = self.mul(
            self.add(f, self.mul(cube, self.scalar(0.044715, len(x.shape)))),
            self.scalar(np.sqrt(2 / np.pi), len(x.shape)),
        )
        z = self.net.add_activation(z, trt.ActivationType.TANH).get_output(0)
        y = self.mul(
            self.mul(f, self.scalar(0.5, len(x.shape))), self.add(z, self.scalar(1, len(x.shape)))
        )
        return self.cast(y, x.dtype)

    def concat(self, tensors, axis):
        layer = self.net.add_concatenation(tensors)
        layer.axis = axis
        return layer.get_output(0)

    def gather(self, x, indices, axis):
        return self.net.add_gather(x, indices, axis).get_output(0)

    def attention(self, q, k, v, mask=None):
        q32, k32 = self.cast(q, trt.float32), self.cast(k, trt.float32)
        q32 = self.mul(q32, self.scalar(q.shape[-1] ** -0.5, len(q.shape)))
        scores = self.net.add_matrix_multiply(
            q32, trt.MatrixOperation.NONE, k32, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
        if mask is not None:
            scores = self.add(scores, mask)
        softmax = self.net.add_softmax(scores)
        softmax.axes = 1 << (len(scores.shape) - 1)
        # Upstream's float32 SDPA also multiplies values in FP32.
        y = self.net.add_matrix_multiply(
            softmax.get_output(0),
            trt.MatrixOperation.NONE,
            self.cast(v, trt.float32),
            trt.MatrixOperation.NONE,
        ).get_output(0)
        return self.cast(y, v.dtype)

    def finish(self):
        config = self.builder.create_builder_config()
        config.builder_optimization_level = 3
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
        config.clear_flag(trt.BuilderFlag.TF32)
        config.add_optimization_profile(self.profile)
        plan = self.builder.build_serialized_network(self.net, config)
        if plan is None:
            raise RuntimeError("Molmo2 TensorRT engine construction failed")
        return bytes(plan)
