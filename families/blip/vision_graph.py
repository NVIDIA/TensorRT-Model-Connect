# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BLIP vision math with native FP32 primitive bindings."""

import math

import tensorrt as trt

from .graph import Graph
from .native_kernels import add_fp32_kernel


class VisionGraph(Graph):
    def norm(self, x, prefix, eps):
        shape = (1,) * (len(x.shape) - 1) + (x.shape[-1],)
        return add_fp32_kernel(
            self.n,
            "norm",
            [
                x,
                self.constant(self.w[prefix + ".weight"], shape),
                self.constant(self.w[prefix + ".bias"], shape),
            ],
            x.shape,
            [{"type": "float", "value": eps}],
        )

    def linear(self, x, prefix):
        weight = self.w[prefix + ".weight"]
        return add_fp32_kernel(
            self.n,
            "linear",
            [x, self.constant(weight), self.constant(self.w[prefix + ".bias"])],
            tuple(x.shape[:-1]) + (weight.shape[0],),
        )

    def gelu(self, x):
        return add_fp32_kernel(self.n, "gelu", [x], x.shape)

    def attention(self, q, k, v, heads, mask=None):
        width = q.shape[-1]
        depth = width // heads
        q, k, v = [self.reshape(x, (-1, heads, depth), (1, 0, 2)) for x in (q, k, v)]
        scores = self.n.add_matrix_multiply(
            q, trt.MatrixOperation.NONE, k, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
        scores = self.scale(scores, 1 / math.sqrt(depth))
        if mask is not None:
            scores = self.op(scores, mask)
        probabilities = add_fp32_kernel(self.n, "softmax", [scores], scores.shape)
        context = self.n.add_matrix_multiply(
            probabilities, trt.MatrixOperation.NONE, v, trt.MatrixOperation.NONE
        ).get_output(0)
        transpose = self.n.add_shuffle(context)
        transpose.first_transpose = (1, 0, 2)
        transpose.reshape_dims = (-1, width)
        return transpose.get_output(0)
