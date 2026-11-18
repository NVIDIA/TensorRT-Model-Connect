# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned strongly typed Conformer and autoregressive decoder graphs."""

from __future__ import annotations
import math
import numpy as np
import tensorrt as trt


class Graph:
    def __init__(self, network, weights):
        self.net = network
        self.weights = weights
        self.storage = []

    def constant(self, value, dtype=np.float32):
        value = np.ascontiguousarray(value, dtype=dtype)
        self.storage.append(value)
        return self.net.add_constant(value.shape, trt.Weights(value)).get_output(0)

    def reshape(self, x, shape, transpose=None):
        layer = self.net.add_shuffle(x)
        if transpose is not None:
            layer.first_transpose = trt.Permutation(transpose)
        layer.reshape_dims = shape
        return layer.get_output(0)

    def dim(self, x, axis):
        return self.net.add_slice(
            self.net.add_shape(x).get_output(0), (axis,), (1,), (1,)
        ).get_output(0)

    def shape(self, *parts):
        tensors = [self.constant([p], np.int64) if isinstance(p, int) else p for p in parts]
        tensors = [
            self.net.add_cast(p, trt.int64).get_output(0) if p.dtype != trt.int64 else p
            for p in tensors
        ]
        return self.net.add_concatenation(tensors).get_output(0)

    def op(self, a, b, kind=trt.ElementWiseOperation.SUM):
        return self.net.add_elementwise(a, b, kind).get_output(0)

    def scale(self, x, scale):
        return self.op(
            x, self.constant(np.full((1,) * len(x.shape), scale)), trt.ElementWiseOperation.PROD
        )

    def linear(self, x, key):
        w = self.weights[key + ".weight"]
        rank = len(x.shape)
        rhs = self.constant(w.T.reshape((1,) * (rank - 2) + w.T.shape))
        result = self.net.add_matrix_multiply(
            x, trt.MatrixOperation.NONE, rhs, trt.MatrixOperation.NONE
        ).get_output(0)
        if key + ".bias" in self.weights:
            bias = self.weights[key + ".bias"]
            result = self.op(result, self.constant(bias.reshape((1,) * (rank - 1) + (-1,))))
        return result

    def norm(self, x, key):
        shape = (1,) * (len(x.shape) - 1) + (int(x.shape[-1]),)
        norm = self.net.add_normalization_v2(
            x,
            self.constant(self.weights[key + ".weight"].reshape(shape)),
            self.constant(self.weights[key + ".bias"].reshape(shape)),
            1 << (len(shape) - 1),
        )
        norm.epsilon = 1e-5
        return norm.get_output(0)

    def activation(self, x, kind="silu"):
        if kind == "relu":
            return self.net.add_activation(x, trt.ActivationType.RELU).get_output(0)
        sigmoid = self.net.add_activation(x, trt.ActivationType.SIGMOID).get_output(0)
        return self.op(x, sigmoid, trt.ElementWiseOperation.PROD)

    def conv(self, x, key, stride=1, pad=0, groups=1):
        w = self.weights[key + ".weight"]
        bias = self.weights[key + ".bias"]
        one_dimensional = len(w.shape) == 3
        if one_dimensional:
            x = self.reshape(x, (1, int(x.shape[1]), 1, -1))
            w = w.reshape(w.shape[0], w.shape[1], 1, w.shape[2])
        w = np.ascontiguousarray(w)
        bias = np.ascontiguousarray(bias)
        self.storage.extend([w, bias])
        layer = self.net.add_convolution_nd(
            x, int(w.shape[0]), w.shape[2:], trt.Weights(w), trt.Weights(bias)
        )
        layer.stride_nd = (1, stride) if one_dimensional else (stride,) * (len(w.shape) - 2)
        layer.padding_nd = (0, pad) if one_dimensional else (pad,) * (len(w.shape) - 2)
        layer.num_groups = groups
        result = layer.get_output(0)
        return self.reshape(result, (1, int(w.shape[0]), -1)) if one_dimensional else result

    def heads(self, x, heads, dimension):
        return self.reshape(
            self.reshape(x, (1, -1, heads, dimension)), (1, heads, -1, dimension), (0, 2, 1, 3)
        )

    def rows(self, x, hidden):
        return self.reshape(x, (1, -1, hidden), (0, 2, 1, 3))

    def mm(self, a, b, transpose=False):
        return self.net.add_matrix_multiply(
            a,
            trt.MatrixOperation.NONE,
            b,
            trt.MatrixOperation.TRANSPOSE if transpose else trt.MatrixOperation.NONE,
        ).get_output(0)

    def softmax(self, x):
        layer = self.net.add_softmax(x)
        layer.axes = 1 << (len(x.shape) - 1)
        return layer.get_output(0)

    def fill(self, shape, start, step, dtype=trt.int32):
        layer = self.net.add_fill((1,) * len(step), trt.FillOperation.LINSPACE, dtype)
        layer.set_input(0, shape)
        if start.dtype != dtype:
            start = self.net.add_cast(start, dtype).get_output(0)
        start = self.reshape(start, ())
        layer.set_input(1, start)
        layer.set_input(2, self.constant(step, np.int32 if dtype == trt.int32 else np.float32))
        return layer.get_output(0)

    def relative_attention(self, x, pos, key, hidden, heads):
        dimension = hidden // heads
        q = self.heads(self.linear(x, key + ".linear_q"), heads, dimension)
        k = self.heads(self.linear(x, key + ".linear_k"), heads, dimension)
        v = self.heads(self.linear(x, key + ".linear_v"), heads, dimension)
        p = self.heads(self.linear(pos, key + ".linear_pos"), heads, dimension)
        qu = self.op(
            q, self.constant(self.weights[key + ".pos_bias_u"].reshape(1, heads, 1, dimension))
        )
        qv = self.op(
            q, self.constant(self.weights[key + ".pos_bias_v"].reshape(1, heads, 1, dimension))
        )
        content = self.mm(qu, k, True)
        relative = self.mm(qv, p, True)
        length = self.dim(x, 1)
        first = self.op(length, self.constant([-1], np.int64))
        indices = self.fill(self.shape(1, heads, length, length), first, [0, 0, -1, 1])
        gather = self.net.add_gather_v2(relative, indices, trt.GatherMode.ELEMENT)
        gather.axis = 3
        scores = self.scale(self.op(content, gather.get_output(0)), 1 / math.sqrt(dimension))
        return self.linear(self.rows(self.mm(self.softmax(scores), v), hidden), key + ".linear_out")

    def conformer(self, x, pos, index, hidden, heads):
        key = f"encoder.layers.{index}"
        y = self.norm(x, key + ".norm_feed_forward1")
        y = self.linear(
            self.activation(self.linear(y, key + ".feed_forward1.linear1")),
            key + ".feed_forward1.linear2",
        )
        x = self.op(x, self.scale(y, 0.5))
        x = self.op(
            x,
            self.relative_attention(
                self.norm(x, key + ".norm_self_att"), pos, key + ".self_attn", hidden, heads
            ),
        )
        y = self.norm(x, key + ".norm_conv")
        y = self.reshape(y, (1, hidden, -1), (0, 2, 1))
        y = self.conv(y, key + ".conv.pointwise_conv1")
        a = self.net.add_slice(y, (0, 0, 0), (1, hidden, 1), (1, 1, 1))
        a.set_input(2, self.shape(1, hidden, self.dim(y, 2)))
        b = self.net.add_slice(y, (0, hidden, 0), (1, hidden, 1), (1, 1, 1))
        b.set_input(2, self.shape(1, hidden, self.dim(y, 2)))
        y = self.op(
            a.get_output(0),
            self.net.add_activation(b.get_output(0), trt.ActivationType.SIGMOID).get_output(0),
            trt.ElementWiseOperation.PROD,
        )
        y = self.conv(y, key + ".conv.depthwise_conv", pad=4, groups=hidden)
        gamma = self.weights[key + ".conv.batch_norm.weight"]
        variance = self.weights[key + ".conv.batch_norm.running_var"]
        scale = gamma / np.sqrt(variance + 1e-5)
        shift = (
            self.weights[key + ".conv.batch_norm.bias"]
            - self.weights[key + ".conv.batch_norm.running_mean"] * scale
        )
        y = self.op(
            self.op(y, self.constant(scale.reshape(1, -1, 1)), trt.ElementWiseOperation.PROD),
            self.constant(shift.reshape(1, -1, 1)),
        )
        y = self.conv(self.activation(y), key + ".conv.pointwise_conv2")
        x = self.op(x, self.reshape(y, (1, -1, hidden), (0, 2, 1)))
        y = self.norm(x, key + ".norm_feed_forward2")
        y = self.linear(
            self.activation(self.linear(y, key + ".feed_forward2.linear1")),
            key + ".feed_forward2.linear2",
        )
        return self.norm(self.op(x, self.scale(y, 0.5)), key + ".norm_out")

    def encoder(self, features, config, max_frames):
        hidden = int(config["d_model"])
        channels = int(config["subsampling_conv_channels"])
        x = self.reshape(features, (1, 1, -1, 128), (0, 2, 1))
        for index in (0, 2, 3, 5, 6):
            x = self.conv(
                x,
                f"encoder.pre_encode.conv.{index}",
                stride=2 if index in (0, 2, 5) else 1,
                pad=1 if index in (0, 2, 5) else 0,
                groups=channels if index in (2, 5) else 1,
            )
            if index in (0, 3, 6):
                x = self.activation(x, "relu")
        x = self.reshape(x, (1, -1, channels * 16), (0, 2, 1, 3))
        x = self.linear(x, "encoder.pre_encode.out")
        max_sequence = (max_frames + 7) // 8
        import torch

        positions = torch.arange(max_sequence - 1, -max_sequence, -1, dtype=torch.float32)[:, None]
        divisor = torch.exp(
            torch.arange(0, hidden, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden)
        )
        pe = torch.empty((2 * max_sequence - 1, hidden), dtype=torch.float32)
        pe[:, 0::2] = torch.sin(positions * divisor)
        pe[:, 1::2] = torch.cos(positions * divisor)
        length = self.dim(x, 1)
        start = self.op(
            self.constant([max_sequence], np.int64), length, trt.ElementWiseOperation.SUB
        )
        width = self.op(
            self.op(length, self.constant([2], np.int64), trt.ElementWiseOperation.PROD),
            self.constant([-1], np.int64),
        )
        layer = self.net.add_slice(self.constant(pe.numpy()), (0, 0), (1, hidden), (1, 1))
        layer.set_input(1, self.shape(start, 0))
        layer.set_input(2, self.shape(width, hidden))
        pos = self.reshape(layer.get_output(0), (1, -1, hidden))
        for index in range(int(config["n_layers"])):
            x = self.conformer(x, pos, index, hidden, int(config["n_heads"]))
        return x

    def decoder_attention(self, x, context, key, hidden, heads, mask=None):
        dimension = hidden // heads
        q = self.heads(self.linear(x, key + ".query_net"), heads, dimension)
        k = self.heads(self.linear(context, key + ".key_net"), heads, dimension)
        v = self.heads(self.linear(context, key + ".value_net"), heads, dimension)
        scores = self.scale(self.mm(q, k, True), 1 / math.sqrt(dimension))
        if mask is not None:
            scores = self.op(scores, mask)
        return self.linear(
            self.rows(self.mm(self.softmax(scores), v), hidden), key + ".out_projection"
        )

    def decoder(self, ids, encoded, config):
        hidden = int(config["hidden_size"])
        heads = int(config["num_attention_heads"])
        token_emb = self.constant(self.weights["transf_decoder._embedding.token_embedding.weight"])
        tokens = self.net.add_gather(token_emb, ids, 0).get_output(0)
        length = self.dim(ids, 1)
        position = self.fill(self.shape(length), self.constant([0], np.int32), [1])
        positions = self.net.add_gather(
            self.constant(self.weights["transf_decoder._embedding.position_embedding.pos_enc"]),
            position,
            0,
        ).get_output(0)
        x = self.norm(
            self.op(tokens, self.reshape(positions, (1, -1, hidden))),
            "transf_decoder._embedding.layer_norm",
        )
        encoded = self.linear(encoded, "encoder_decoder_proj")
        delta = self.fill(
            self.shape(1, 1, length, length), self.constant([0], np.int32), [0, 0, -1, 1]
        )
        future = self.op(
            delta, self.constant(np.zeros((1, 1, 1, 1)), np.int32), trt.ElementWiseOperation.GREATER
        )
        mask = self.net.add_select(
            future,
            self.constant(np.full((1, 1, 1, 1), -1e9)),
            self.constant(np.zeros((1, 1, 1, 1))),
        ).get_output(0)
        for index in range(int(config["num_layers"])):
            key = f"transf_decoder._decoder.layers.{index}"
            y = self.norm(x, key + ".layer_norm_1")
            x = self.op(
                x, self.decoder_attention(y, y, key + ".first_sub_layer", hidden, heads, mask)
            )
            x = self.op(
                x,
                self.decoder_attention(
                    self.norm(x, key + ".layer_norm_2"),
                    encoded,
                    key + ".second_sub_layer",
                    hidden,
                    heads,
                ),
            )
            y = self.norm(x, key + ".layer_norm_3")
            y = self.linear(
                self.activation(self.linear(y, key + ".third_sub_layer.dense_in"), "relu"),
                key + ".third_sub_layer.dense_out",
            )
            x = self.op(x, y)
        x = self.norm(x, "transf_decoder._decoder.final_layer_norm")
        last = self.net.add_slice(x, (0, 0, 0), (1, 1, hidden), (1, 1, 1))
        last.set_input(1, self.shape(0, self.op(length, self.constant([-1], np.int64)), 0))
        logits = self.linear(last.get_output(0), "log_softmax.mlp.layer0")
        maximum = self.net.add_reduce(logits, trt.ReduceOperation.MAX, 1 << 2, True).get_output(0)
        centered = self.op(logits, maximum, trt.ElementWiseOperation.SUB)
        exponent = self.net.add_unary(centered, trt.UnaryOperation.EXP).get_output(0)
        total = self.net.add_reduce(exponent, trt.ReduceOperation.SUM, 1 << 2, True).get_output(0)
        logarithm = self.net.add_unary(total, trt.UnaryOperation.LOG).get_output(0)
        return self.op(centered, logarithm, trt.ElementWiseOperation.SUB)
