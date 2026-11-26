# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SeamlessM4T v2 text graph operations."""

import numpy as np
import tensorrt as trt

from . import graph as g


class Graph:
    def __init__(self, weights, verbose, quant=None):
        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        self.builder = trt.Builder(self.logger)
        self.net = self.builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        )
        self.options = self.builder.create_builder_config()
        self.options.builder_optimization_level = 3
        self.options.clear_flag(trt.BuilderFlag.TF32)
        self.weights = weights
        self.quant = quant
        if quant is not None:
            quant.configure(self.options)

    def const(self, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        return self.net.add_constant(x.shape, x).get_output(0)

    def reshape(self, x, shape):
        layer = self.net.add_shuffle(x)
        layer.reshape_dims = shape
        return layer.get_output(0)

    def add(self, a, b):
        return self.net.add_elementwise(a, b, trt.ElementWiseOperation.SUM).get_output(0)

    def mul(self, a, b):
        return self.net.add_elementwise(a, b, trt.ElementWiseOperation.PROD).get_output(0)

    def linear(self, x, name):
        weight = self.weights[name + ".weight"]
        matmul = None
        if self.quant is not None and self.quant.covers(name):

            def matmul(network, value):
                return self.quant.matmul(network, value, name, weight)

        return g.add_linear(self.net, x, weight, self.weights.get(name + ".bias"), matmul=matmul)

    def norm(self, x, name, eps):
        return g.add_layer_norm(
            self.net, x, self.weights[name + ".weight"], self.weights[name + ".bias"], eps
        )

    def attention(self, q, k, v, heads, mask=None):
        qlen, width = q.shape
        klen = k.shape[0]
        dim = width // heads

        def split(x, length):
            layer = self.net.add_shuffle(x)
            layer.reshape_dims = (1, length, heads, dim)
            layer.second_transpose = (0, 2, 1, 3)
            return layer.get_output(0)

        scores = self.net.add_matrix_multiply(
            split(q, qlen), trt.MatrixOperation.NONE, split(k, klen), trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
        scores = self.mul(scores, self.const(np.full((1, 1, 1, 1), dim**-0.5)))
        if mask is not None:
            scores = self.add(scores, mask)
        softmax = self.net.add_softmax(scores)
        softmax.axes = 8
        context = self.net.add_matrix_multiply(
            softmax.get_output(0),
            trt.MatrixOperation.NONE,
            split(v, klen),
            trt.MatrixOperation.NONE,
        ).get_output(0)
        layer = self.net.add_shuffle(context)
        layer.first_transpose = (0, 2, 1, 3)
        layer.reshape_dims = (qlen, width)
        return layer.get_output(0)

    def finish(self):
        plan = self.builder.build_serialized_network(self.net, self.options)
        if plan is None:
            raise RuntimeError("SeamlessM4T v2 engine build failed")
        return bytes(plan)
