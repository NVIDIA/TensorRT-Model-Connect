# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strongly typed TensorRT expressions owned by the Clef graph."""

import numpy as np
import tensorrt as trt


class Graph:
    def __init__(self, network, weights, *, fp32_sigmoid=True):
        self.n = network
        self.weights = weights
        self.keep = []
        self.fp32_sigmoid = fp32_sigmoid

    def cast(self, x, dtype):
        return x if x.dtype == dtype else self.n.add_cast(x, dtype).get_output(0)

    def const(self, value, dtype=trt.float32):
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().numpy()
        a = np.asarray(value, dtype=np.int32 if dtype == trt.int32 else np.float32)
        a = np.array(a, copy=True, order="C")
        if dtype == trt.bfloat16:
            import ml_dtypes

            a = a.astype(ml_dtypes.bfloat16).view(np.uint16)
            w = trt.Weights(trt.bfloat16, a.ctypes.data, a.size)
        else:
            w = trt.Weights(a)
        self.keep.extend([a, w])
        return self.n.add_constant(a.shape, w).get_output(0)

    def scalar(self, value, x):
        return self.const(np.full((1,) * len(x.shape), value), x.dtype)

    def binary(self, a, b, op):
        if not isinstance(b, trt.ITensor):
            b = self.scalar(b, a)
        return self.n.add_elementwise(a, b, op).get_output(0)

    def add(self, a, b):
        return self.binary(a, b, trt.ElementWiseOperation.SUM)

    def sub(self, a, b):
        return self.binary(a, b, trt.ElementWiseOperation.SUB)

    def mul(self, a, b):
        return self.binary(a, b, trt.ElementWiseOperation.PROD)

    def div(self, a, b):
        return self.binary(a, b, trt.ElementWiseOperation.DIV)

    def maximum(self, a, b):
        return self.binary(a, b, trt.ElementWiseOperation.MAX)

    def unary(self, x, op):
        return self.n.add_unary(x, op).get_output(0)

    def sqrt(self, x):
        return self.unary(x, trt.UnaryOperation.SQRT)

    def exp(self, x):
        return self.unary(x, trt.UnaryOperation.EXP)

    def abs(self, x):
        return self.unary(x, trt.UnaryOperation.ABS)

    def reshape(self, x, shape):
        layer = self.n.add_shuffle(x)
        layer.reshape_dims = shape
        return layer.get_output(0)

    def permute(self, x, order):
        layer = self.n.add_shuffle(x)
        layer.first_transpose = order
        return layer.get_output(0)

    def concat(self, xs, axis=-1):
        layer = self.n.add_concatenation(xs)
        layer.axis = axis % len(xs[0].shape)
        return layer.get_output(0)

    def mm(self, a, b, transpose=False):
        return self.n.add_matrix_multiply(
            a,
            trt.MatrixOperation.NONE,
            b,
            trt.MatrixOperation.TRANSPOSE if transpose else trt.MatrixOperation.NONE,
        ).get_output(0)

    def linear(self, x, name, weight=None, bias=None):
        weight = self.weights[name + ".weight"] if weight is None else weight
        if bias is None:
            bias = self.weights.get(name + ".bias")
        w = self.const(weight, x.dtype)
        if len(x.shape) > 2:
            w = self.reshape(w, (1,) * (len(x.shape) - 2) + tuple(w.shape))
        result = self.mm(x, w, True)
        if bias is not None:
            shape = (1,) * (len(x.shape) - 1) + (len(bias),)
            result = self.add(result, self.reshape(self.const(bias, x.dtype), shape))
        return result

    def reduce(self, x, op=trt.ReduceOperation.SUM, axis=-1, keep=True):
        return self.n.add_reduce(x, op, 1 << (axis % len(x.shape)), keep).get_output(0)

    def mean_pool(self, mask, values):
        return self.cast(self.mm(mask, self.cast(values, trt.float32)), values.dtype)

    def norm(self, x, name, eps=1e-5):
        dtype = x.dtype
        if getattr(self, "precise_norm", False):
            x = self.cast(x, trt.float32)
        shape = (1,) * (len(x.shape) - 1) + (x.shape[-1],)
        scale = self.const(self.weights[name + ".weight"], x.dtype)
        bias = self.const(self.weights[name + ".bias"], x.dtype)
        layer = self.n.add_normalization(
            x, self.reshape(scale, shape), self.reshape(bias, shape), 1 << (len(x.shape) - 1)
        )
        layer.epsilon = eps
        return self.cast(layer.get_output(0), dtype)

    def rms(self, x, name, eps=1e-6, centered=True):
        f = self.cast(x, trt.float32)
        variance = self.reduce(self.mul(f, f), trt.ReduceOperation.AVG)
        f = self.div(f, self.sqrt(self.add(variance, eps)))
        weight = self.weights[name + ".weight"].float()
        if centered:
            weight = weight + 1
        shape = (1,) * (len(x.shape) - 1) + (x.shape[-1],)
        return self.cast(self.mul(f, self.reshape(self.const(weight), shape)), x.dtype)

    def unit(self, x, eps):
        # torch.linalg.vector_norm accumulates in FP32 and returns input dtype.
        f = self.cast(x, trt.float32)
        norm = self.cast(self.sqrt(self.reduce(self.mul(f, f))), x.dtype)
        return self.div(x, self.maximum(norm, eps))

    def activation(self, x, kind):
        return self.n.add_activation(x, kind).get_output(0)

    def silu(self, x):
        f = self.cast(x, trt.float32)
        return self.cast(self.mul(f, self.activation(f, trt.ActivationType.SIGMOID)), x.dtype)

    def sigmoid(self, x):
        if not self.fp32_sigmoid:
            return self.activation(x, trt.ActivationType.SIGMOID)
        # Torch's BF16 sigmoid evaluates in FP32 and rounds only its result.
        return self.cast(
            self.activation(self.cast(x, trt.float32), trt.ActivationType.SIGMOID), x.dtype
        )

    def gelu(self, x):
        return self.cast(
            self.activation(self.cast(x, trt.float32), trt.ActivationType.GELU_ERF), x.dtype
        )

    def softmax(self, x, axis=-1):
        layer = self.n.add_softmax(x)
        layer.axes = 1 << (axis % len(x.shape))
        return layer.get_output(0)

    def attention(self, q, k, v, heads, causal=False, fp32_accumulation=False):
        dim = q.shape[-1] // heads
        dtype = q.dtype
        if fp32_accumulation:
            q, k, v = [self.cast(x, trt.float32) for x in (q, k, v)]

        def layout(x):
            return self.permute(self.reshape(x, (1, -1, heads, dim)), (0, 2, 1, 3))

        # TensorRT IAttention has no implicit 1/sqrt(D) factor.
        q = layout(q)
        if fp32_accumulation:
            if causal:
                raise ValueError("explicit FP32 attention requires an unmasked vision sequence")
            attended = self.stream_attention(q, layout(k), layout(v), heads, dim)
        else:
            q = self.mul(q, dim**-0.5)
            layer = self.n.add_attention(
                q, layout(k), layout(v), trt.AttentionNormalizationOp.SOFTMAX, causal
            )
            layer.decomposable = True
            attended = layer.get_output(0)
        return self.cast(
            self.reshape(self.permute(attended, (0, 2, 1, 3)), (-1, heads * dim)), dtype
        )

    def stream_attention(self, q, k, v, heads, dim):
        """Online softmax with linear workspace and explicit reference precision.

        Scores, softmax state, and value products accumulate in FP32; only the
        final attention output is rounded back to the model dtype. The
        matrix/reduction/loop graph avoids an S-by-S intermediate for large
        images. TensorRT lowers every operation.
        """
        block = 128
        q, k, v = [self.reshape(x, (heads, -1, dim)) for x in (q, k, v)]
        shape = self.cast(self.n.add_shape(k).get_output(0), trt.int32)
        length = self.n.add_gather(shape, self.const(1, trt.int32), 0).get_output(0)
        count = self.binary(self.add(length, block - 1), block, trt.ElementWiseOperation.FLOOR_DIV)
        padded_length = self.mul(count, block)
        sizes = self.concat(
            [
                self.const([heads], trt.int32),
                self.reshape(padded_length, (1,)),
                self.const([dim], trt.int32),
            ],
            0,
        )

        def pad(x):
            layer = self.n.add_slice(x, (0, 0, 0), (heads, 1, dim), (1, 1, 1))
            layer.mode = trt.SampleMode.FILL
            layer.set_input(2, sizes)
            layer.set_input(4, self.const(0.0))
            return self.reshape(layer.get_output(0), (heads, -1, block, dim))

        k, v = pad(k), pad(v)
        row_zero = self.mul(self.reduce(q), 0)
        loop = self.n.add_loop()
        loop.add_trip_limit(count, trt.TripLimit.COUNT)
        # An early memory block can contain no keys for a later frame. A finite
        # sentinel plus an explicit validity mask keeps that row at zero until
        # its first matching block, avoiding -inf - -inf.
        maximum = loop.add_recurrence(self.add(row_zero, -1e30))
        denominator = loop.add_recurrence(row_zero)
        numerator = loop.add_recurrence(self.mul(q, 0))
        index = loop.add_recurrence(self.const(0, trt.int32))
        key = loop.add_iterator(k, 1).get_output(0)
        value = loop.add_iterator(v, 1).get_output(0)
        scores = self.mul(self.mm(q, key, True), dim**-0.5)
        positions = self.add(
            self.const(np.arange(block).reshape(1, 1, block), trt.int32),
            self.reshape(self.mul(index.get_output(0), block), (1, 1, 1)),
        )
        valid = self.binary(
            positions, self.reshape(length, (1, 1, 1)), trt.ElementWiseOperation.LESS
        )
        if hasattr(self, "attention_groups"):
            groups = self.attention_groups
            group_pad = self.n.add_slice(groups, (0,), (1,), (1,))
            group_pad.mode = trt.SampleMode.FILL
            group_pad.set_input(2, self.reshape(padded_length, (1,)))
            group_pad.set_input(4, self.const(-1, trt.int32))
            blocks = self.reshape(group_pad.get_output(0), (-1, block))
            key_groups = loop.add_iterator(blocks, 0).get_output(0)
            same_group = self.binary(
                self.reshape(groups, (1, -1, 1)),
                self.reshape(key_groups, (1, 1, block)),
                trt.ElementWiseOperation.EQUAL,
            )
            valid = self.binary(valid, same_group, trt.ElementWiseOperation.AND)
        scores = self.n.add_select(valid, scores, self.scalar(-1e30, scores)).get_output(0)
        next_max = self.maximum(maximum.get_output(0), self.reduce(scores, trt.ReduceOperation.MAX))
        rescale = self.exp(self.sub(maximum.get_output(0), next_max))
        probabilities = self.exp(self.sub(scores, next_max))
        probabilities = self.n.add_select(
            valid, probabilities, self.scalar(0, probabilities)
        ).get_output(0)
        denominator.set_input(
            1, self.add(self.mul(denominator.get_output(0), rescale), self.reduce(probabilities))
        )
        numerator.set_input(
            1, self.add(self.mul(numerator.get_output(0), rescale), self.mm(probabilities, value))
        )
        maximum.set_input(1, next_max)
        index.set_input(1, self.add(index.get_output(0), 1))
        total = loop.add_loop_output(
            numerator.get_output(0), trt.LoopOutput.LAST_VALUE, 0
        ).get_output(0)
        normalizer = loop.add_loop_output(
            denominator.get_output(0), trt.LoopOutput.LAST_VALUE, 0
        ).get_output(0)
        return self.reshape(self.div(total, normalizer), (1, heads, -1, dim))

    def mha(self, q, memory, name, heads):
        w, b = self.weights[name + ".in_proj_weight"], self.weights[name + ".in_proj_bias"]
        qs, ks, vs = w.chunk(3, 0)
        qb, kb, vb = b.chunk(3, 0)
        q = self.linear(q, name, qs, qb)
        k = self.linear(memory, name, ks, kb)
        v = self.linear(memory, name, vs, vb)
        return self.linear(self.attention(q, k, v, heads), name + ".out_proj")

    def slice(self, x, axis, start, size):
        rank = len(x.shape)
        axis %= rank
        starts, sizes = [0] * rank, list(x.shape)
        starts[axis], sizes[axis] = start, size
        layer = self.n.add_slice(x, starts, [max(1, d) for d in sizes], [1] * rank)
        if any(d == -1 for d in sizes):
            shape = self.cast(self.n.add_shape(x).get_output(0), trt.int32)
            values = [
                self.const([d], trt.int32)
                if d != -1
                else self.n.add_gather(shape, self.const([i], trt.int32), 0).get_output(0)
                for i, d in enumerate(sizes)
            ]
            layer.set_input(2, self.concat(values, 0))
        return layer.get_output(0)
