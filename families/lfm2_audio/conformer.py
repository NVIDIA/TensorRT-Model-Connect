# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastConformer and audio adapter expressed as TensorRT network topology."""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from .graph import Graph


def build_encoder(weights, config, max_mel_frames, *, precision="bf16", verbose=False):
    g = Graph(weights, precision, verbose)
    mel = g.net.add_input("mel", trt.float32, (1, 1, -1, 128))
    valid_frames = g.net.add_input("valid_frames", trt.int32, (1,))
    x = g.cast(mel, g.dtype)
    for i, stride, groups in ((0, 2, 1), (2, 2, 256), (3, 1, 1), (5, 2, 256), (6, 1, 1)):
        x = g.conv(
            x,
            f"conformer.pre_encode.conv.{i}",
            (stride, stride),
            (1, 1) if stride == 2 else (0, 0),
            groups,
        )
        if i in (0, 3, 6):
            x = g.activation(x, trt.ActivationType.RELU)
    x = g.transpose(x, (0, 2, 1, 3))
    x = g.reshape(x, (1, -1, 4096))
    x = g.linear(x, "conformer.pre_encode.out")
    length = g.dim(x, 1)
    twice = g.add(length, length)
    relative_length = g.op(twice, g.shape(1), trt.ElementWiseOperation.SUB)
    max_length = (max_mel_frames + 7) // 8
    indices = g.slice(g.constant(np.arange(max_length, dtype=np.int32)), (0,), g.shape(length))
    valid = g.op(indices, valid_frames, trt.ElementWiseOperation.LESS)
    valid_rows = g.reshape(valid, g.shape(1, length, 1))
    keys = g.reshape(valid, g.shape(1, 1, 1, length))
    queries = g.reshape(valid, g.shape(1, 1, length, 1))
    attention_mask = g.op(keys, queries, trt.ElementWiseOperation.AND)
    positions = np.arange(max_length - 1, -max_length, -1, dtype=np.float32)[:, None]
    frequencies = np.exp(
        np.arange(0, 512, 2, dtype=np.float32) * np.float32(-np.log(10000.0) / 512)
    )
    pe = np.empty((1, 2 * max_length - 1, 512), np.float32)
    pe[0, :, 0::2] = np.sin(positions * frequencies)
    pe[0, :, 1::2] = np.cos(positions * frequencies)
    start = g.op(g.shape(max_length), length, trt.ElementWiseOperation.SUB)
    pos = g.slice(
        g.constant(pe, dtype=g.dtype), g.shape(0, start, 0), g.shape(1, relative_length, 512)
    )

    def heads(t):
        return g.transpose(g.reshape(t, (1, -1, 8, 64)), (0, 2, 1, 3))

    def ff(t, prefix, number):
        t = g.norm(t, prefix + f".norm_feed_forward{number}")
        t = g.linear(t, prefix + f".feed_forward{number}.linear1")
        t = g.silu(t)
        return g.scale(g.linear(t, prefix + f".feed_forward{number}.linear2"), 0.5)

    for i in range(config["n_layers"]):
        prefix = f"conformer.layers.{i}"
        x = g.add(x, ff(x, prefix, 1))
        h = g.norm(x, prefix + ".norm_self_att")
        att = prefix + ".self_attn"
        q, k, v = (heads(g.linear(h, att + ".linear_" + p)) for p in ("q", "k", "v"))
        p = heads(g.linear(pos, att + ".linear_pos", bias=False))
        qu = g.add(q, g.weight(att + ".pos_bias_u", (1, 8, 1, 64)))
        qv = g.add(q, g.weight(att + ".pos_bias_v", (1, 8, 1, 64)))
        ac = g.matmul(qu, k, transpose=True)
        bd = g.matmul(qv, p, transpose=True)
        # Transformer-XL relative shift: pad left, reshape, remove one row,
        # reshape back, and retain the key-length columns.
        padding = g.net.add_slice(bd, (0, 0, 0, -1), (1, 8, 1, 1), (1, 1, 1, 1))
        padding.set_input(2, g.shape(1, 8, length, twice))
        padding.mode = trt.SampleMode.FILL
        padding.set_input(4, g.scalar(0, 0, g.dtype))
        bd = g.reshape(padding.get_output(0), g.shape(1, 8, twice, length))
        bd = g.slice(bd, (0, 0, 1, 0), g.shape(1, 8, relative_length, length))
        bd = g.reshape(bd, g.shape(1, 8, length, relative_length))
        bd = g.slice(bd, (0, 0, 0, 0), g.shape(1, 8, length, length))
        scores = g.scale(g.add(ac, bd), 0.125)
        scores = g.net.add_select(attention_mask, scores, g.scalar(-10000, 4, g.dtype)).get_output(
            0
        )
        probs = g.net.add_softmax(scores)
        probs.axes = 1 << 3
        probabilities = g.net.add_select(
            attention_mask, probs.get_output(0), g.scalar(0, 4, g.dtype)
        ).get_output(0)
        h = g.matmul(probabilities, v)
        h = g.reshape(g.transpose(h, (0, 2, 1, 3)), (1, -1, 512))
        x = g.add(x, g.linear(h, att + ".linear_out"))
        h = g.norm(x, prefix + ".norm_conv")
        h = g.linear(h, prefix + ".conv.pointwise_conv1")
        a = g.slice(h, (0, 0, 0), g.shape(1, length, 512))
        b = g.slice(h, (0, 0, 512), g.shape(1, length, 512))
        h = g.glu(a, b)
        h = g.net.add_select(valid_rows, h, g.scalar(0, 3, g.dtype)).get_output(0)
        h = g.reshape(g.transpose(h, (0, 2, 1)), g.shape(1, 512, length, 1))
        h = g.conv(h, prefix + ".conv.depthwise_conv", padding=(4, 0), groups=512)
        h = g.transpose(g.reshape(h, g.shape(1, 512, length)), (0, 2, 1))
        bn = prefix + ".conv.batch_norm"
        fp = g.cast(h, trt.float32)
        mean = g.cast(g.weight(bn + ".running_mean", (1, 1, 512)), trt.float32)
        var = g.cast(g.weight(bn + ".running_var", (1, 1, 512)), trt.float32)
        fp = g.op(fp, mean, trt.ElementWiseOperation.SUB)
        fp = g.op(
            fp,
            g.unary(g.add(var, g.scalar(1e-5, 3, trt.float32)), trt.UnaryOperation.SQRT),
            trt.ElementWiseOperation.DIV,
        )
        fp = g.mul(fp, g.cast(g.weight(bn + ".weight", (1, 1, 512)), trt.float32))
        fp = g.add(fp, g.cast(g.weight(bn + ".bias", (1, 1, 512)), trt.float32))
        h = g.silu(g.cast(fp, g.dtype))
        x = g.add(x, g.linear(h, prefix + ".conv.pointwise_conv2"))
        x = g.add(x, ff(x, prefix, 2))
        x = g.norm(x, prefix + ".norm_out")
    g.output(x, "encoder_features")
    x = g.norm(x, "audio_adapter.model.0")
    x = g.linear(x, "audio_adapter.model.1")
    x = g.activation(x, trt.ActivationType.GELU_ERF)
    x = g.linear(x, "audio_adapter.model.3")
    g.output(x, "audio_embeddings")
    return g.finish(
        {
            "mel": (
                (1, 1, 9, 128),
                (1, 1, min(max_mel_frames, 801), 128),
                (1, 1, max_mel_frames, 128),
            )
        }
    )
