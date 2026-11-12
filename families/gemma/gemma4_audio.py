# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gemma4 audio encoder and projector using native TensorRT graph operations."""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from .checkpoint_mapper import _has_tensor, _load_tensor


class AudioGraph:
    """Bounded mel capacity with explicit validity, preserving all admitted frames."""

    def __init__(self, network, raw, readers, precision, max_frames, mel_bins=128):
        self.net, self.raw, self.readers = network, raw, readers
        self.dtype = {"fp16": trt.float16, "bf16": trt.bfloat16, "fp32": trt.float32}[precision]
        self.storage = np.float16 if precision == "fp16" else np.float32
        self.frames, self.mel_bins = int(max_frames), int(mel_bins)
        if self.frames <= 0 or self.frames % 4 or raw["hidden_size"] % raw["num_attention_heads"]:
            raise ValueError("Gemma4 audio has invalid frame or attention geometry")
        if mel_bins != raw["subsampling_conv_channels"][0] or raw["hidden_act"] != "silu":
            raise NotImplementedError("Gemma4 audio requires its original mel geometry and SiLU")
        self.steps = self.frames // 4
        self.chunk = int(raw["attention_chunk_size"])
        self.context = self.chunk + raw["attention_context_left"] - 1 + raw["attention_context_right"]
        self.blocks = (self.steps + self.chunk - 1) // self.chunk
        self.buffers = []

    def cast(self, value, dtype):
        return value if value.dtype == dtype else self.net.add_cast(value, dtype).get_output(0)

    def const(self, value, dtype=None):
        value = np.ascontiguousarray(value)
        self.buffers.append(value)
        tensor = self.net.add_constant(value.shape, trt.Weights(value)).get_output(0)
        return self.cast(tensor, dtype) if dtype is not None else tensor

    def reshape(self, value, shape):
        layer = self.net.add_shuffle(value)
        layer.reshape_dims = shape
        return layer.get_output(0)

    def op(self, lhs, rhs, operation):
        return self.net.add_elementwise(lhs, rhs, operation).get_output(0)

    def weight(self, name, dtype=None):
        value = _load_tensor(self.readers, name)
        return self.const(value.astype(self.storage if dtype is None else np.float32),
                          self.dtype if dtype is None else dtype)

    def scalar(self, value, rank, dtype):
        return self.const(np.full((1,) * rank, value, np.float32), dtype)

    def scale(self, value, factor):
        return self.op(value, self.scalar(factor, len(value.shape), value.dtype),
                       trt.ElementWiseOperation.PROD)

    def linear(self, value, name):
        weight = self.weight(name + ".weight")
        output = self.net.add_matrix_multiply(value, trt.MatrixOperation.NONE,
                                             self.cast(weight, value.dtype),
                                             trt.MatrixOperation.TRANSPOSE).get_output(0)
        if _has_tensor(self.readers, name + ".bias"):
            bias = self.reshape(self.weight(name + ".bias"), (1, -1))
            output = self.op(output, self.cast(bias, output.dtype), trt.ElementWiseOperation.SUM)
        return output

    def clip(self, value, name, direction):
        if not self.raw.get("use_clipped_linears", False):
            return value
        for suffix, operation, neutral in (("min", trt.ElementWiseOperation.MAX, -np.inf),
                                            ("max", trt.ElementWiseOperation.MIN, np.inf)):
            bound = _load_tensor(self.readers, f"{name}.{direction}_{suffix}")
            if not np.all(bound == neutral):
                value = self.op(value, self.const(np.asarray(bound).reshape(
                    (1,) * len(value.shape)), value.dtype), operation)
        return value

    def clipped_linear(self, value, name):
        value = self.clip(value, name, "input")
        return self.clip(self.linear(value, name + ".linear"), name, "output")

    def norm(self, value, name=None):
        dtype, rank = value.dtype, len(value.shape)
        value = self.cast(value, trt.float32)
        square = self.op(value, value, trt.ElementWiseOperation.PROD)
        mean = self.net.add_reduce(square, trt.ReduceOperation.AVG, 1 << (rank - 1), True).get_output(0)
        mean = self.op(mean, self.scalar(self.raw["rms_norm_eps"], rank, trt.float32),
                       trt.ElementWiseOperation.SUM)
        inv = self.op(mean, self.scalar(-0.5, rank, trt.float32), trt.ElementWiseOperation.POW)
        output = self.op(value, inv, trt.ElementWiseOperation.PROD)
        if name is not None:
            weight = self.reshape(self.weight(name + ".weight", trt.float32),
                                  (1,) * (rank - 1) + (value.shape[-1],))
            output = self.op(output, weight, trt.ElementWiseOperation.PROD)
        return self.cast(output, dtype)

    def transpose(self, value, order):
        layer = self.net.add_shuffle(value)
        layer.first_transpose = order
        return layer.get_output(0)

    def slice(self, value, start, shape):
        return self.net.add_slice(value, start, shape, (1,) * len(shape)).get_output(0)

    def bounded(self, value):
        maximum = 65504.0 if self.dtype == trt.float16 else np.finfo(np.float32).max
        bound = min(float(self.raw["gradient_clipping"]), maximum)
        value = self.op(value, self.scalar(-bound, len(value.shape), value.dtype),
                        trt.ElementWiseOperation.MAX)
        return self.op(value, self.scalar(bound, len(value.shape), value.dtype),
                       trt.ElementWiseOperation.MIN)

    def silu(self, value):
        original = value.dtype
        value = self.cast(value, trt.float32)
        sigmoid = self.net.add_activation(value, trt.ActivationType.SIGMOID).get_output(0)
        return self.cast(self.op(value, sigmoid, trt.ElementWiseOperation.PROD), original)

    def convolution(self, value, name, stride, before, after, groups=1):
        weight = self.weight(name + ".weight")
        if len(weight.shape) == 3:
            weight = self.reshape(weight, tuple(weight.shape) + (1,))
        layer = self.net.add_convolution_nd(value, weight.shape[0], tuple(weight.shape[2:]),
                                            trt.Weights())
        layer.set_input(1, weight)
        layer.stride_nd = stride
        layer.pre_padding = before
        layer.post_padding = after
        layer.num_groups = groups
        return layer.get_output(0)

    def channel_norm(self, value, name):
        dtype = value.dtype
        shape = tuple(value.shape)  # N,C,T,F
        rows = self.reshape(self.transpose(value, (0, 2, 3, 1)), (-1, shape[1]))
        rows = self.cast(rows, trt.float32)
        mean = self.net.add_reduce(rows, trt.ReduceOperation.AVG, 2, True).get_output(0)
        centered = self.op(rows, mean, trt.ElementWiseOperation.SUB)
        square = self.op(centered, centered, trt.ElementWiseOperation.PROD)
        variance = self.net.add_reduce(square, trt.ReduceOperation.AVG, 2, True).get_output(0)
        variance = self.op(variance, self.scalar(self.raw["rms_norm_eps"], 2, trt.float32),
                           trt.ElementWiseOperation.SUM)
        inv = self.op(variance, self.scalar(-0.5, 2, trt.float32), trt.ElementWiseOperation.POW)
        rows = self.op(centered, inv, trt.ElementWiseOperation.PROD)
        weight = self.reshape(self.weight(name + ".weight", trt.float32), (1, shape[1]))
        rows = self.cast(self.op(rows, weight, trt.ElementWiseOperation.PROD), dtype)
        return self.transpose(self.reshape(rows, (shape[0], shape[2], shape[3], shape[1])),
                              (0, 3, 1, 2))

    def feed_forward(self, state, name):
        value = self.norm(self.bounded(state), name + ".pre_layer_norm")
        value = self.silu(self.clipped_linear(value, name + ".ffw_layer_1"))
        value = self.clipped_linear(value, name + ".ffw_layer_2")
        value = self.norm(self.bounded(value), name + ".post_layer_norm")
        return self.op(state, self.scale(value, self.raw["residual_weight"]),
                       trt.ElementWiseOperation.SUM)

    def light_convolution(self, state, name):
        hidden, steps = self.raw["hidden_size"], self.steps
        value = self.clipped_linear(self.norm(state, name + ".pre_layer_norm"),
                                    name + ".linear_start")
        value = self.cast(value, trt.float32)
        first = self.slice(value, (0, 0), (steps, hidden))
        gate = self.slice(value, (0, hidden), (steps, hidden))
        gate = self.net.add_activation(gate, trt.ActivationType.SIGMOID).get_output(0)
        value = self.cast(self.op(first, gate, trt.ElementWiseOperation.PROD), self.dtype)
        value = self.reshape(self.transpose(value, (1, 0)), (1, hidden, steps, 1))
        value = self.convolution(value, name + ".depthwise_conv1d", (1, 1),
                                 (self.raw["conv_kernel_size"] - 1, 0), (0, 0), hidden)
        value = self.transpose(self.reshape(value, (hidden, steps)), (1, 0))
        value = self.silu(self.norm(self.bounded(value), name + ".conv_norm"))
        return self.op(state, self.clipped_linear(value, name + ".linear_end"),
                       trt.ElementWiseOperation.SUM)

    def attention(self, value, name, mask, query_indices, key_indices, position_embeddings):
        n, raw = self.net, self.raw
        heads, hidden = int(raw["num_attention_heads"]), int(raw["hidden_size"])
        dim, chunk, context, blocks = hidden // heads, self.chunk, self.context, self.blocks
        q = self.cast(self.clipped_linear(value, name + ".q_proj"), trt.float32)
        q = self.reshape(q, (self.steps, heads, dim))
        q = self.scale(q, dim**-0.5 / np.log(2.0))
        # The original softplus result rounds to the parameter's working dtype
        # before multiplying FP32 queries.
        learned = _load_tensor(self.readers, name + ".per_dim_scale").astype(self.storage)
        scale = np.logaddexp(np.float32(0), learned.astype(np.float32)).astype(self.storage)
        scale = self.cast(self.const(scale.reshape(1, 1, dim), self.dtype), trt.float32)
        q = self.reshape(self.op(q, scale, trt.ElementWiseOperation.PROD),
                         (self.steps, hidden))
        k = self.scale(self.cast(self.clipped_linear(value, name + ".k_proj"), trt.float32),
                       np.log1p(np.e) / np.log(2.0))
        v = self.cast(self.clipped_linear(value, name + ".v_proj"), trt.float32)
        blocked = []
        for item, indices, count in ((q, query_indices, chunk), (k, key_indices, context),
                                     (v, key_indices, context)):
            joined = n.add_concatenation([item, self.const(np.zeros((1, hidden), np.float32))])
            joined.axis = 0
            item = n.add_gather(joined.get_output(0), indices, 0).get_output(0)
            item = self.reshape(item, (blocks, count, heads, dim))
            blocked.append(self.transpose(item, (2, 0, 1, 3)))
        q, k, v = blocked
        scores = n.add_matrix_multiply(q, trt.MatrixOperation.NONE, k,
                                       trt.MatrixOperation.TRANSPOSE).get_output(0)
        relative = self.linear(position_embeddings, name + ".relative_k_proj")
        length = relative.shape[0]
        relative = self.reshape(relative, (length, heads, dim))
        relative = self.reshape(self.transpose(relative, (1, 2, 0)), (heads, 1, dim, length))
        relative_scores = n.add_matrix_multiply(q, trt.MatrixOperation.NONE,
                                                self.cast(relative, trt.float32),
                                                trt.MatrixOperation.NONE).get_output(0)
        # Preserve the provider's blocked relative shift exactly, including its
        # padded zeros, rather than replacing it with an assumed distance lookup.
        relative_scores = n.add_padding_nd(relative_scores, (0, 0),
                                            (0, context + 1 - length)).get_output(0)
        relative_scores = self.reshape(relative_scores, (heads, blocks, chunk * (context + 1)))
        relative_scores = self.slice(relative_scores, (0, 0, 0), (heads, blocks, chunk * context))
        relative_scores = self.reshape(relative_scores, (heads, blocks, chunk, context))
        scores = self.scale(self.op(scores, relative_scores, trt.ElementWiseOperation.SUM),
                             1 / raw["attention_logit_cap"])
        scores = n.add_activation(scores, trt.ActivationType.TANH).get_output(0)
        scores = self.scale(scores, raw["attention_logit_cap"])
        scores = n.add_select(mask, scores,
                              self.scalar(raw["attention_invalid_logits_value"], 4, trt.float32)).get_output(0)
        probabilities = n.add_softmax(scores)
        probabilities.axes = 1 << 3
        value = n.add_matrix_multiply(probabilities.get_output(0), trt.MatrixOperation.NONE,
                                      v, trt.MatrixOperation.NONE).get_output(0)
        value = self.reshape(self.transpose(value, (1, 2, 0, 3)), (blocks * chunk, hidden))
        value = self.cast(self.slice(value, (0, 0), (self.steps, hidden)), self.dtype)
        return self.clipped_linear(value, name + ".post")

    def build(self):
        n, raw = self.net, self.raw
        hidden = int(raw["hidden_size"])
        prefix = "model.audio_tower"
        features = n.add_input("input_features", trt.float32, (self.frames, self.mel_bins))
        validity = n.add_input("input_features_mask", trt.int32, (self.frames,))
        state = self.reshape(features, (1, 1, self.frames, self.mel_bins))
        length = self.frames
        mask = validity
        for index in (0, 1):
            name = f"{prefix}.subsample_conv_projection.layer{index}"
            state = self.op(state, self.reshape(self.cast(mask, state.dtype), (1, 1, length, 1)),
                             trt.ElementWiseOperation.PROD)
            state = self.convolution(self.cast(state, self.dtype), name + ".conv",
                                     (2, 2), (1, 1), (1, 1))
            state = self.channel_norm(state, name + ".norm")
            state = n.add_activation(state, trt.ActivationType.RELU).get_output(0)
            indices = self.const(np.arange(0, length, 2, dtype=np.int32))
            mask = n.add_gather(mask, indices, 0).get_output(0)
            length = (length + 1) // 2
        state = self.reshape(self.transpose(state, (0, 2, 3, 1)), (self.steps, -1))
        state = self.linear(state, prefix + ".subsample_conv_projection.input_proj_linear")

        positions = np.arange(self.context // 2, -1, -1, dtype=np.float32)[:, None]
        inverse = np.exp(np.arange(hidden // 2, dtype=np.float32) *
                         np.float32(-np.log(10000.0) / max(hidden // 2 - 1, 1)))
        angles = positions * inverse
        positional = self.const(np.concatenate((np.sin(angles), np.cos(angles)), axis=-1)
                                .astype(self.storage), self.dtype)
        starts = np.arange(self.blocks, dtype=np.int32)[:, None] * self.chunk
        queries = starts + np.arange(self.chunk, dtype=np.int32)[None, :]
        keys = starts + np.arange(self.context, dtype=np.int32)[None, :] - (raw["attention_context_left"] - 1)
        qindices = self.const(np.where(queries < self.steps, queries, self.steps).astype(np.int32))
        kindices = self.const(np.where((keys >= 0) & (keys < self.steps), keys, self.steps).astype(np.int32))
        distance = queries[:, :, None] - keys[:, None, :]
        allowed = (((distance >= 0) & (distance < raw["attention_context_left"] - 1)) |
                   ((distance < 0) & (-distance < raw["attention_context_right"])))
        allowed &= (queries[:, :, None] < self.steps) & (keys[:, None, :] >= 0) & (keys[:, None, :] < self.steps)
        padded_mask = n.add_concatenation([mask, self.const(np.zeros(1, np.int32))])
        padded_mask.axis = 0
        valid_keys = n.add_gather(padded_mask.get_output(0), kindices, 0).get_output(0)
        valid_keys = self.cast(self.reshape(valid_keys, (1, self.blocks, 1, self.context)), trt.bool)
        attention_mask = self.op(valid_keys, self.const(allowed.astype(np.int32)[None], trt.bool),
                                 trt.ElementWiseOperation.AND)
        for index in range(int(raw["num_hidden_layers"])):
            name = f"{prefix}.layers.{index}"
            state = self.feed_forward(state, name + ".feed_forward1")
            attended = self.attention(self.norm(self.bounded(state), name + ".norm_pre_attn"),
                                      name + ".self_attn", attention_mask, qindices, kindices, positional)
            attended = self.norm(self.bounded(attended), name + ".norm_post_attn")
            state = self.op(state, attended, trt.ElementWiseOperation.SUM)
            state = self.light_convolution(state, name + ".lconv1d")
            state = self.feed_forward(state, name + ".feed_forward2")
            state = self.norm(self.bounded(state), name + ".norm_out")
        state = self.linear(state, prefix + ".output_proj")
        state = self.linear(self.norm(state), "model.embed_audio.embedding_projection")
        state = self.cast(state, trt.float32)
        state.name = "audio_features"
        n.mark_output(state)
        mask.name = "audio_feature_mask"
        n.mark_output(mask)
