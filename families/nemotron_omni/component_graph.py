# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Omni-local graph mechanics for the visual and acoustic encoders."""

import numpy as np
import tensorrt as trt


class _Graph:
    def __init__(self, precision: str, verbose: bool):
        types = {"fp32": trt.float32, "fp16": trt.float16, "bf16": trt.bfloat16}
        if precision not in types:
            raise ValueError("Nemotron Omni precision must be fp32, fp16, or bf16")
        self.dtype = types[precision]
        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        self.builder = trt.Builder(self.logger)
        self.network = self.builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        )
        self.config = self.builder.create_builder_config()
        self.config.builder_optimization_level = 3
        self.config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
        if precision == "fp32":
            self.config.clear_flag(trt.BuilderFlag.TF32)
        self.profile = self.builder.create_optimization_profile()
        self.constants = []

    def cast(self, value, dtype):
        return value if value.dtype == dtype else self.network.add_cast(value, dtype).get_output(0)

    def constant(self, value, dtype=None):
        dtype = self.dtype if dtype is None else dtype
        storage = (
            np.int32 if dtype == trt.int32 else np.float16 if dtype == trt.float16 else np.float32
        )
        array = np.ascontiguousarray(value, dtype=storage)
        self.constants.append(array)
        return self.cast(self.network.add_constant(array.shape, array).get_output(0), dtype)

    def scalar(self, value, rank, dtype=None):
        return self.constant(np.full((1,) * rank, value), dtype)

    def reshape(self, value, shape, permutation=None):
        layer = self.network.add_shuffle(value)
        layer.reshape_dims = shape
        if permutation is not None:
            layer.second_transpose = permutation
        return layer.get_output(0)

    def binary(self, left, right, operation):
        return self.network.add_elementwise(left, right, operation).get_output(0)

    def add(self, left, right):
        return self.binary(left, right, trt.ElementWiseOperation.SUM)

    def mul(self, left, right):
        return self.binary(left, right, trt.ElementWiseOperation.PROD)

    def linear(self, value, weights, prefix, bias=False):
        matrix = self.constant(weights[prefix + ".weight"])
        result = self.network.add_matrix_multiply(
            value, trt.MatrixOperation.NONE, matrix, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
        if bias:
            result = self.add(result, self.constant(weights[prefix + ".bias"][None, :]))
        return result

    def norm(self, value, weights, prefix, eps, centered=False):
        x = self.cast(value, trt.float32)
        if centered:
            mean = self.network.add_reduce(x, trt.ReduceOperation.AVG, 2, True).get_output(0)
            x = self.binary(x, mean, trt.ElementWiseOperation.SUB)
        mean_square = self.network.add_reduce(
            self.mul(x, x), trt.ReduceOperation.AVG, 2, True
        ).get_output(0)
        denominator = self.network.add_unary(
            self.add(mean_square, self.scalar(eps, 2, trt.float32)), trt.UnaryOperation.SQRT
        ).get_output(0)
        x = self.binary(x, denominator, trt.ElementWiseOperation.DIV)
        if centered:
            x = self.mul(x, self.constant(weights[prefix + ".weight"][None, :], trt.float32))
            x = self.add(x, self.constant(weights[prefix + ".bias"][None, :], trt.float32))
            return self.cast(x, self.dtype)
        return self.cast(
            self.mul(x, self.constant(weights[prefix + ".weight"][None, :], trt.float32)),
            self.dtype,
        )

    def gelu(self, value, approximate=False):
        x = value
        if approximate:
            cube = self.mul(self.mul(x, x), x)
            inner = self.mul(
                self.add(x, self.mul(cube, self.scalar(0.044715, 2))),
                self.scalar(np.sqrt(2 / np.pi), 2),
            )
            sigmoid = self.network.add_activation(inner, trt.ActivationType.TANH).get_output(0)
        else:
            scaled = self.mul(x, self.scalar(1 / np.sqrt(2), 2))
            sigmoid = self.network.add_unary(scaled, trt.UnaryOperation.ERF).get_output(0)
        return self.mul(self.mul(x, self.scalar(0.5, 2)), self.add(sigmoid, self.scalar(1, 2)))

    def attention(self, query, key, value, head_dim, mask=None):
        scores = self.network.add_matrix_multiply(
            self.cast(query, trt.float32),
            trt.MatrixOperation.NONE,
            self.cast(key, trt.float32),
            trt.MatrixOperation.TRANSPOSE,
        ).get_output(0)
        scores = self.mul(scores, self.scalar(head_dim**-0.5, 3, trt.float32))
        if mask is not None:
            scores = self.add(scores, mask)
        softmax = self.network.add_softmax(scores)
        softmax.axes = 4
        probabilities = self.cast(softmax.get_output(0), self.dtype)
        return self.network.add_matrix_multiply(
            probabilities, trt.MatrixOperation.NONE, value, trt.MatrixOperation.NONE
        ).get_output(0)

    def output(self, value, name, dtype=None):
        result = self.cast(value, dtype) if dtype is not None else value
        result.name = name
        self.network.mark_output(result)

    def finish(self):
        self.config.add_optimization_profile(self.profile)
        plan = self.builder.build_serialized_network(self.network, self.config)
        if plan is None:
            raise RuntimeError("Nemotron Omni native TensorRT graph build failed")
        return bytes(plan)
