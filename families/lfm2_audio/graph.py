# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LFM audio-owned strongly typed network construction mechanics."""

from __future__ import annotations

import numpy as np
import tensorrt as trt


class Graph:
    def __init__(self, weights, precision="bf16", verbose=False):
        self.weights = weights
        self.dtype = {"bf16": trt.bfloat16, "fp16": trt.float16, "fp32": trt.float32}[precision]
        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        self.builder = trt.Builder(self.logger)
        self.net = self.builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        )
        self.config = self.builder.create_builder_config()
        self.config.builder_optimization_level = 1
        self.config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
        self.keep = []

    def cast(self, x, dtype):
        return x if x.dtype == dtype else self.net.add_cast(x, dtype).get_output(0)

    def constant(self, value, *, dtype=None):
        value = np.ascontiguousarray(value)
        self.keep.append(value)
        out = self.net.add_constant(value.shape, trt.Weights(value)).get_output(0)
        return self.cast(out, dtype) if dtype is not None else out

    def weight(self, name, shape=None):
        w = np.asarray(self.weights[name], dtype=np.float32)
        return self.constant(w if shape is None else w.reshape(shape), dtype=self.dtype)

    def scalar(self, value, rank, dtype):
        return self.constant(np.full((1,) * rank, value, np.float32), dtype=dtype)

    def op(self, a, b, op):
        return self.net.add_elementwise(a, b, op).get_output(0)

    def add(self, a, b):
        return self.op(a, b, trt.ElementWiseOperation.SUM)

    def mul(self, a, b):
        return self.op(a, b, trt.ElementWiseOperation.PROD)

    def scale(self, a, scale):
        return self.mul(a, self.scalar(scale, len(a.shape), a.dtype))

    def unary(self, x, op):
        return self.net.add_unary(x, op).get_output(0)

    def activation(self, x, kind):
        return self.net.add_activation(x, kind).get_output(0)

    def silu(self, x):
        fp32 = self.cast(x, trt.float32)
        return self.cast(self.mul(fp32, self.activation(fp32, trt.ActivationType.SIGMOID)), x.dtype)

    def glu(self, a, b):
        return self.mul(a, self.activation(b, trt.ActivationType.SIGMOID))

    def shape(self, *dims):
        parts = [self.constant(np.array([d], np.int64)) if isinstance(d, int) else d for d in dims]
        c = self.net.add_concatenation(parts)
        c.axis = 0
        return c.get_output(0)

    def dim(self, x, i):
        shape = self.net.add_shape(x).get_output(0)
        return self.net.add_slice(shape, (i,), (1,), (1,)).get_output(0)

    def reshape(self, x, shape):
        s = self.net.add_shuffle(x)
        if isinstance(shape, tuple):
            s.reshape_dims = shape
        else:
            s.set_input(1, shape)
        return s.get_output(0)

    def transpose(self, x, order):
        s = self.net.add_shuffle(x)
        s.first_transpose = order
        return s.get_output(0)

    def slice(self, x, start, shape):
        layer = self.net.add_slice(
            x, tuple(0 for _ in x.shape), tuple(1 for _ in x.shape), tuple(1 for _ in x.shape)
        )
        if isinstance(start, tuple):
            layer.start = start
        else:
            layer.set_input(1, start)
        if isinstance(shape, tuple):
            layer.shape = shape
        else:
            layer.set_input(2, shape)
        return layer.get_output(0)

    def linear(self, x, name, bias=True):
        w = np.asarray(self.weights[name + ".weight"], np.float32)
        w = w.reshape(w.shape[0], -1)
        wt = self.constant(w.T.reshape((1,) * (len(x.shape) - 2) + w.T.shape), dtype=x.dtype)
        y = self.matmul(x, wt)
        if bias and name + ".bias" in self.weights:
            y = self.add(y, self.weight(name + ".bias", (1,) * (len(x.shape) - 1) + (w.shape[0],)))
        return y

    def matmul(self, x, y, transpose=False):
        return self.net.add_matrix_multiply(
            x,
            trt.MatrixOperation.NONE,
            y,
            trt.MatrixOperation.TRANSPOSE if transpose else trt.MatrixOperation.NONE,
        ).get_output(0)

    def norm(self, x, name, epsilon=1e-5, rms=False):
        dtype = x.dtype
        fp = self.cast(x, trt.float32)
        axes = 1 << (len(x.shape) - 1)
        if not rms:
            mean = self.net.add_reduce(fp, trt.ReduceOperation.AVG, axes, True).get_output(0)
            fp = self.op(fp, mean, trt.ElementWiseOperation.SUB)
        variance = self.net.add_reduce(
            self.mul(fp, fp), trt.ReduceOperation.AVG, axes, True
        ).get_output(0)
        denom = self.unary(
            self.add(variance, self.scalar(epsilon, len(x.shape), trt.float32)),
            trt.UnaryOperation.SQRT,
        )
        y = self.op(fp, denom, trt.ElementWiseOperation.DIV)
        shape = (1,) * (len(x.shape) - 1) + (x.shape[-1],)
        y = self.mul(y, self.cast(self.weight(name + ".weight", shape), trt.float32))
        if not rms:
            y = self.add(y, self.cast(self.weight(name + ".bias", shape), trt.float32))
        return self.cast(y, dtype)

    def conv(self, x, name, stride=(1, 1), padding=(0, 0), groups=1):
        w = np.asarray(self.weights[name + ".weight"], np.float32)
        if w.ndim == 3:
            w = w[..., None]
        b = np.asarray(self.weights[name + ".bias"], np.float32)
        # Dynamic weight inputs preserve strong typing for BF16 convolution.
        layer = self.net.add_convolution_nd(
            x, w.shape[0], w.shape[-2:], trt.Weights(), trt.Weights()
        )
        layer.set_input(1, self.constant(w, dtype=x.dtype))
        layer.set_input(2, self.constant(b, dtype=x.dtype))
        layer.stride_nd = stride
        layer.padding_nd = padding
        layer.num_groups = groups
        layer.name = name
        return layer.get_output(0)

    def output(self, x, name, dtype=trt.float32):
        out = self.net.add_identity(self.cast(x, dtype)).get_output(0)
        out.name = name
        self.net.mark_output(out)

    def finish(self, profiles=None):
        if profiles:
            p = self.builder.create_optimization_profile()
            for name, shapes in profiles.items():
                p.set_shape(name, *shapes)
            self.config.add_optimization_profile(p)
        result = self.builder.build_serialized_network(self.net, self.config)
        if result is None:
            raise RuntimeError("TensorRT failed to build an LFM audio engine")
        return bytes(result)
