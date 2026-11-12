# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MusicGen's FP32 mathematical graph primitives; TensorRT owns lowering."""

import numpy as np
import tensorrt as trt


class Graph:
    def __init__(self, verbose=False):
        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        self.builder = trt.Builder(self.logger)
        self.net = self.builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        )
        self.weights = []

    def const(self, data, dtype=np.float32):
        data = np.ascontiguousarray(data, dtype=dtype)
        self.weights.append(data)
        return self.net.add_constant(data.shape, trt.Weights(data)).get_output(0)

    def reshape(self, x, shape, permutation=None):
        layer = self.net.add_shuffle(x)
        layer.reshape_dims = shape
        if permutation is not None:
            layer.second_transpose = permutation
        return layer.get_output(0)

    def op(self, x, y, operation=trt.ElementWiseOperation.SUM):
        return self.net.add_elementwise(x, y, operation).get_output(0)

    def scalar(self, x, value):
        return self.const(np.full((1,) * len(x.shape), value))

    def linear(self, x, weight, bias=None):
        w = np.asarray(weight)
        w = self.const(w.T.reshape((1,) * (len(x.shape) - 2) + w.T.shape))
        y = self.net.add_matrix_multiply(
            x, trt.MatrixOperation.NONE, w, trt.MatrixOperation.NONE
        ).get_output(0)
        if bias is not None:
            y = self.op(y, self.const(np.asarray(bias).reshape((1,) * (len(x.shape) - 1) + (-1,))))
        return y

    def norm(self, x, weight, bias=None, epsilon=1e-6):
        axes = 1 << (len(x.shape) - 1)
        if bias is not None:
            mean = self.net.add_reduce(x, trt.ReduceOperation.AVG, axes, True).get_output(0)
            x = self.op(x, mean, trt.ElementWiseOperation.SUB)
        square = self.op(x, x, trt.ElementWiseOperation.PROD)
        mean = self.net.add_reduce(square, trt.ReduceOperation.AVG, axes, True).get_output(0)
        variance = self.op(mean, self.scalar(mean, epsilon))
        denom = self.net.add_unary(variance, trt.UnaryOperation.SQRT).get_output(0)
        x = self.op(x, denom, trt.ElementWiseOperation.DIV)
        shape = (1,) * (len(x.shape) - 1) + (-1,)
        x = self.op(x, self.const(np.asarray(weight).reshape(shape)), trt.ElementWiseOperation.PROD)
        return x if bias is None else self.op(x, self.const(np.asarray(bias).reshape(shape)))

    def attention(self, q, k, v, heads, mask, scale):
        batch, length, width = q.shape
        source = k.shape[1]
        depth = width // heads
        q = self.reshape(q, (batch, length, heads, depth), (0, 2, 1, 3))
        k = self.reshape(k, (batch, source, heads, depth), (0, 2, 1, 3))
        v = self.reshape(v, (batch, source, heads, depth), (0, 2, 1, 3))
        q = self.op(q, self.scalar(q, scale), trt.ElementWiseOperation.PROD)
        score = self.net.add_matrix_multiply(
            q, trt.MatrixOperation.NONE, k, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
        score = self.op(score, mask)
        softmax = self.net.add_softmax(score)
        softmax.axes = 1 << 3
        out = self.net.add_matrix_multiply(
            softmax.get_output(0), trt.MatrixOperation.NONE, v, trt.MatrixOperation.NONE
        ).get_output(0)
        layer = self.net.add_shuffle(out)
        layer.first_transpose = (0, 2, 1, 3)
        layer.reshape_dims = (batch, length, width)
        return layer.get_output(0)

    def finish(self, output, name):
        output.name = name
        self.net.mark_output(output)
        config = self.builder.create_builder_config()
        config.builder_optimization_level = 1
        config.clear_flag(trt.BuilderFlag.TF32)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
        plan = self.builder.build_serialized_network(self.net, config)
        if plan is None:
            raise RuntimeError(f"MusicGen TensorRT build failed: {name}")
        return bytes(plan)

    def pad(self, x, left, right):
        length = x.shape[-1]
        indices = np.pad(np.arange(length), (left, right), mode="reflect").astype(np.int32)
        return self.net.add_gather(x, self.const(indices, np.int32), 2).get_output(0)

    def conv(self, x, weight, bias, stride=1, transpose=False):
        batch, channels, length = x.shape
        kernel = weight.shape[-1]
        x = self.reshape(x, (batch, channels, 1, length))
        output_channels = weight.shape[1] if transpose else weight.shape[0]
        w = np.ascontiguousarray(weight[:, :, None, :], dtype=np.float32)
        b = np.ascontiguousarray(bias, dtype=np.float32)
        self.weights.extend((w, b))
        add = self.net.add_deconvolution_nd if transpose else self.net.add_convolution_nd
        layer = add(x, output_channels, (1, kernel), trt.Weights(w), trt.Weights(b))
        layer.stride_nd = (1, stride)
        y = layer.get_output(0)
        return self.reshape(y, (batch, output_channels, y.shape[-1]))

    def lstm(self, x, w_ih, w_hh, b_ih, b_hh):
        _, length, hidden = x.shape
        loop = self.net.add_loop()
        count = self.net.add_constant((), trt.Weights(np.array(length, np.int32))).get_output(0)
        loop.add_trip_limit(count, trt.TripLimit.COUNT)
        zero = self.const(np.zeros((1, hidden)))
        h, c = loop.add_recurrence(zero), loop.add_recurrence(zero)
        item = loop.add_iterator(x, 1).get_output(0)
        gates = self.op(self.linear(item, w_ih, b_ih + b_hh), self.linear(h.get_output(0), w_hh))
        values = []
        for i in range(4):
            gate = self.net.add_slice(gates, (0, i * hidden), (1, hidden), (1, 1)).get_output(0)
            activation = trt.ActivationType.TANH if i == 2 else trt.ActivationType.SIGMOID
            values.append(self.net.add_activation(gate, activation).get_output(0))
        i, f, candidate, o = values
        cell = self.op(
            self.op(f, c.get_output(0), trt.ElementWiseOperation.PROD),
            self.op(i, candidate, trt.ElementWiseOperation.PROD),
        )
        hidden_out = self.op(
            o,
            self.net.add_activation(cell, trt.ActivationType.TANH).get_output(0),
            trt.ElementWiseOperation.PROD,
        )
        h.set_input(1, hidden_out)
        c.set_input(1, cell)
        output = loop.add_loop_output(hidden_out, trt.LoopOutput.CONCATENATE, 1)
        output.set_input(1, count)
        return output.get_output(0)
