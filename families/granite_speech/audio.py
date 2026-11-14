# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audio engine: log-mel front end, Conformer CTC encoder and Q-Former projector.

The encoder attends only inside blocks of ``context_size`` frames. Shaw relative positions add
``q · E[i - j + max_pos_emb]`` to every score; the product with the ``2C - 1`` usable rows of ``E``
is turned into the ``C x C`` block by the relative-shift reshape instead of materialising
``E[i - j]`` for every pair. Rows that pad the last block are hidden from the keys, which is what
the reference's mask does for every real query row.

The projector pads the encoder output with zero frames to whole windows and does not mask them, as
the reference does; every window yields ``window_size / downsample_rate`` query outputs.
"""

from __future__ import annotations

import math

import numpy as np
import tensorrt as trt

from .checkpoint import Checkpoint
from .config import FrontEnd, GraniteSpeechConfig
from .constants import dft_basis, mel_filterbank, relative_table
from .graph import Graph

_ENCODER_NORM_EPS = 1e-5  # torch.nn.LayerNorm / BatchNorm1d defaults used by the encoder


def relative_shift(g: Graph, scores: trt.ITensor, context: int, heads: int) -> trt.ITensor:
    """``[nb, H, C, 2C - 1]`` with column ``j`` ↔ offset ``r - c + C - 1`` → ``[nb, H, C, C]``.

    Padding one zero column and re-reading the flat rows with stride ``2C - 1`` after dropping the
    first ``C - 1`` values moves row ``c`` left by ``c`` columns, which is the shift that turns the
    offset-indexed product into the position-indexed block.
    """
    batch = g.dim(scores, 0)
    padded = g.slice(scores, (0, 0, 0, 0), g.concat([batch, g.i64([heads, context, 2 * context])], 0),
                     fill=True)
    flat = g.reshape(padded, (-1, heads, 2 * context * context))
    span = context * (2 * context - 1)
    trimmed = g.slice(flat, (0, 0, context - 1), g.concat([batch, g.i64([heads, span])], 0))
    grid = g.reshape(trimmed, (-1, heads, context, 2 * context - 1))
    return g.slice(grid, (0, 0, 0, 0), g.concat([batch, g.i64([heads, context, context])], 0))


def add_front_end(g: Graph, waveform: trt.ITensor, fe: FrontEnd) -> trt.ITensor:
    """``waveform [L]`` (reflect padded) → stacked log-mel features ``[T, 2 * n_mels]`` in FP32.

    ``log10(max(mel, 1e-10))``, clamped at the clip's maximum minus 8 and mapped ``x / 4 + 1``;
    an odd last frame is dropped (after the maximum is taken) and frame pairs are stacked.
    """
    bins = fe.n_fft // 2 + 1
    signal = g.reshape(waveform, (1, 1, 1, -1))
    basis = dft_basis(fe)
    conv = g.network.add_convolution_nd(
        signal, 2 * bins, (1, fe.n_fft),
        g.weights(basis.reshape(2 * bins, 1, 1, fe.n_fft), trt.float32), trt.Weights(trt.float32))
    conv.stride_nd = (1, fe.hop_length)
    spectrum = conv.get_output(0)  # [1, 2 * bins, 1, F]
    frames = g.dim(spectrum, 3)
    real = g.slice(spectrum, (0, 0, 0, 0), g.concat([g.i64([1, bins, 1]), frames], 0))
    imag = g.slice(spectrum, (0, bins, 0, 0), g.concat([g.i64([1, bins, 1]), frames], 0))
    power = g.add(g.mul(real, real), g.mul(imag, imag))
    rows = g.transpose(g.reshape(power, (bins, -1)), (1, 0))  # [F, bins]
    mel = g.matmul(rows, g.constant(mel_filterbank(fe), trt.float32))  # [F, n_mels]
    floor = g.maximum(mel, g.scalar(1.0e-10, trt.float32, 2))
    log10 = g.scale(g.unary(floor, trt.UnaryOperation.LOG), 1.0 / math.log(10.0))
    peak = g.network.add_reduce(log10, trt.ReduceOperation.MAX, 0b11, True).get_output(0)
    clamped = g.maximum(log10, g.sub(peak, g.scalar(8.0, trt.float32, 2)))
    scaled = g.add(g.scale(clamped, 0.25), g.scalar(1.0, trt.float32, 2))
    two = g.i64([2])
    pairs = g.binary(frames, two, trt.ElementWiseOperation.FLOOR_DIV)
    even = g.slice(scaled, (0, 0), g.concat([g.mul(pairs, two), g.i64([fe.n_mels])], 0))
    return g.reshape(even, (-1, 2 * fe.n_mels))


def _feed_forward(g: Graph, ckpt: Checkpoint, x: trt.ITensor, base: str) -> trt.ITensor:
    normed = g.layer_norm(x, ckpt.tensor(base + "pre_norm.weight"), ckpt.tensor(base + "pre_norm.bias"),
                          _ENCODER_NORM_EPS)
    inner = g.silu(g.linear(normed, ckpt.tensor(base + "up_proj.weight"), ckpt.tensor(base + "up_proj.bias")))
    return g.linear(inner, ckpt.tensor(base + "down_proj.weight"), ckpt.tensor(base + "down_proj.bias"))


def _attention(g: Graph, ckpt: Checkpoint, cfg: GraniteSpeechConfig, x: trt.ITensor, base: str,
               key_mask: trt.ITensor, padded_rows: trt.ITensor, frames: trt.ITensor) -> trt.ITensor:
    e = cfg.encoder
    context, heads, width = e.context_size, e.num_heads, e.dim_head
    inner = heads * width
    normed = g.layer_norm(x, ckpt.tensor(base + "pre_norm.weight"), ckpt.tensor(base + "pre_norm.bias"),
                          _ENCODER_NORM_EPS)
    padded = g.slice(normed, (0, 0), g.concat([padded_rows, g.i64([e.hidden_dim])], 0), fill=True)
    query = g.linear(padded, ckpt.tensor(base + "to_q.weight"))
    kv = g.linear(padded, ckpt.tensor(base + "to_kv.weight"))
    key = g.slice(kv, (0, 0), g.concat([padded_rows, g.i64([inner])], 0))
    value = g.slice(kv, (0, inner), g.concat([padded_rows, g.i64([inner])], 0))

    def blocks(tensor):
        return g.transpose(g.reshape(tensor, (-1, context, heads, width)), (0, 2, 1, 3))

    q, k, v = blocks(query), blocks(key), blocks(value)
    content = g.matmul(q, k, transpose_right=True)  # [nb, H, C, C]
    table = relative_table(ckpt.tensor(base + "rel_pos_emb.weight"), context, e.max_pos_emb)
    offsets = g.matmul(q, g.constant(table, q.dtype, (1, 1) + table.shape))  # [nb, H, C, 2C - 1]
    scores = g.add(content, relative_shift(g, offsets, context, heads))
    scores = g.add(g.scale(g.cast(scores, trt.float32), width ** -0.5), key_mask)
    probs = g.softmax(scores, 3)
    attended = g.matmul(g.cast(probs, q.dtype), v)  # [nb, H, C, D]
    rows = g.reshape(g.transpose(attended, (0, 2, 1, 3)), (-1, inner))
    rows = g.slice(rows, (0, 0), g.concat([frames, g.i64([inner])], 0))
    return g.linear(rows, ckpt.tensor(base + "to_out.weight"), ckpt.tensor(base + "to_out.bias"))


def _convolution(g: Graph, ckpt: Checkpoint, cfg: GraniteSpeechConfig, x: trt.ITensor, base: str) -> trt.ITensor:
    e = cfg.encoder
    channels = e.hidden_dim * e.conv_expansion_factor
    normed = g.layer_norm(x, ckpt.tensor(base + "norm.weight"), ckpt.tensor(base + "norm.bias"), _ENCODER_NORM_EPS)
    up = g.linear(normed, ckpt.tensor(base + "up_conv.weight")[:, :, 0], ckpt.tensor(base + "up_conv.bias"))
    frames = g.dim(up, 0)
    gate_size = g.concat([frames, g.i64([channels])], 0)
    gated = g.mul(g.slice(up, (0, 0), gate_size), g.sigmoid(g.slice(up, (0, channels), gate_size)))
    planes = g.reshape(g.transpose(gated, (1, 0)), (1, channels, 1, -1))
    kernel = ckpt.tensor(base + "depth_conv.conv.weight")  # [channels, 1, k]
    size = e.conv_kernel_size
    depth = g.network.add_convolution_nd(
        planes, channels, (1, size), g.weights(kernel.reshape(channels, 1, 1, size), planes.dtype),
        trt.Weights(planes.dtype))
    depth.num_groups = channels
    # The reference pads (k // 2, k // 2 - (k + 1) % 2) with zeros over the whole clip.
    depth.pre_padding = (0, size // 2)
    depth.post_padding = (0, size // 2 - (size + 1) % 2)
    mean = ckpt.tensor(base + "batch_norm.running_mean").astype(np.float64)
    var = ckpt.tensor(base + "batch_norm.running_var").astype(np.float64)
    gamma = ckpt.tensor(base + "batch_norm.weight").astype(np.float64)
    beta = ckpt.tensor(base + "batch_norm.bias").astype(np.float64)
    factor = gamma / np.sqrt(var + _ENCODER_NORM_EPS)
    shift = beta - mean * factor
    normed_planes = g.add(g.mul(depth.get_output(0), g.constant(factor, planes.dtype, (1, channels, 1, 1))),
                          g.constant(shift, planes.dtype, (1, channels, 1, 1)))
    activated = g.silu(normed_planes)
    rows = g.transpose(g.reshape(activated, (channels, -1)), (1, 0))
    return g.linear(rows, ckpt.tensor(base + "down_conv.weight")[:, :, 0], ckpt.tensor(base + "down_conv.bias"))


def add_encoder(g: Graph, ckpt: Checkpoint, cfg: GraniteSpeechConfig, features: trt.ITensor,
                max_frames: int) -> trt.ITensor:
    """``features [T, input_dim]`` (work type) → ``[T, hidden_dim]``."""
    e = cfg.encoder
    context = e.context_size
    x = g.linear(features, ckpt.tensor("encoder.input_linear.weight"), ckpt.tensor("encoder.input_linear.bias"))
    frames = g.dim(x, 0)
    blocks = g.binary(g.add(frames, g.i64([context - 1])), g.i64([context]), trt.ElementWiseOperation.FLOOR_DIV)
    padded_rows = g.mul(blocks, g.i64([context]))
    longest = -(-max_frames // context) * context
    positions = g.arange(longest, padded_rows)
    valid = g.binary(positions, g.cast(frames, trt.int32), trt.ElementWiseOperation.LESS)
    key_mask = g.reshape(g.masked_scores(valid), (-1, 1, 1, context))
    for index in range(e.num_layers):
        base = f"encoder.layers.{index}."
        x = g.add(x, g.scale(_feed_forward(g, ckpt, x, base + "ff1."), 0.5))
        x = g.add(x, _attention(g, ckpt, cfg, x, base + "attn.", key_mask, padded_rows, frames))
        x = g.add(x, _convolution(g, ckpt, cfg, x, base + "conv."))
        x = g.add(x, g.scale(_feed_forward(g, ckpt, x, base + "ff2."), 0.5))
        x = g.layer_norm(x, ckpt.tensor(base + "post_norm.weight"), ckpt.tensor(base + "post_norm.bias"),
                         _ENCODER_NORM_EPS)
        if index + 1 == e.num_layers // 2:
            logits = g.linear(x, ckpt.tensor("encoder.out.weight"), ckpt.tensor("encoder.out.bias"))
            x = g.add(x, g.linear(g.softmax(logits, 1), ckpt.tensor("encoder.out_mid.weight"),
                                  ckpt.tensor("encoder.out_mid.bias")))
    return x


def _qformer_attention(g: Graph, ckpt: Checkpoint, base: str, queries: trt.ITensor, source: trt.ITensor,
                       heads: int) -> trt.ITensor:
    width = int(queries.shape[-1])
    head = width // heads
    count = int(queries.shape[1])

    def split(tensor, length):
        return g.transpose(g.reshape(tensor, (-1, length, heads, head)), (0, 2, 1, 3))

    q = split(g.linear(queries, ckpt.tensor(base + "query.weight"), ckpt.tensor(base + "query.bias")), count)
    length = int(source.shape[1])
    k = split(g.linear(source, ckpt.tensor(base + "key.weight"), ckpt.tensor(base + "key.bias")), length)
    v = split(g.linear(source, ckpt.tensor(base + "value.weight"), ckpt.tensor(base + "value.bias")), length)
    scores = g.scale(g.cast(g.matmul(q, k, transpose_right=True), trt.float32), head ** -0.5)
    attended = g.matmul(g.cast(g.softmax(scores, 3), q.dtype), v)
    return g.reshape(g.transpose(attended, (0, 2, 1, 3)), (-1, count, width))


def _qformer_residual(g: Graph, ckpt: Checkpoint, base: str, update: trt.ITensor, residual: trt.ITensor,
                      eps: float) -> trt.ITensor:
    dense = g.linear(update, ckpt.tensor(base + "dense.weight"), ckpt.tensor(base + "dense.bias"))
    return g.layer_norm(g.add(dense, residual), ckpt.tensor(base + "LayerNorm.weight"),
                        ckpt.tensor(base + "LayerNorm.bias"), eps)


def add_projector(g: Graph, ckpt: Checkpoint, cfg: GraniteSpeechConfig, encoded: trt.ITensor) -> trt.ITensor:
    """``[T, hidden_dim]`` → ``[windows * num_queries, text_hidden]`` audio embeddings."""
    p = cfg.projector
    window = p.window_size
    frames = g.dim(encoded, 0)
    windows = g.binary(g.add(frames, g.i64([window - 1])), g.i64([window]), trt.ElementWiseOperation.FLOOR_DIV)
    padded = g.slice(encoded, (0, 0), g.concat([g.mul(windows, g.i64([window])), g.i64([cfg.encoder.hidden_dim])], 0),
                     fill=True)
    source = g.reshape(padded, (-1, window, cfg.encoder.hidden_dim))
    query = g.constant(ckpt.tensor("projector.query"), g.work)  # [1, Q, width]
    hidden = g.layer_norm(query, ckpt.tensor("projector.qformer.layernorm.weight"),
                          ckpt.tensor("projector.qformer.layernorm.bias"), p.layer_norm_eps)
    for index in range(p.num_layers):
        base = f"projector.qformer.encoder.layer.{index}."
        attended = _qformer_attention(g, ckpt, base + "attention.attention.", hidden, hidden, p.num_heads)
        hidden = _qformer_residual(g, ckpt, base + "attention.output.", attended, hidden, p.layer_norm_eps)
        if p.has_cross_attention(index):
            attended = _qformer_attention(g, ckpt, base + "crossattention.attention.", hidden, source, p.num_heads)
            hidden = _qformer_residual(g, ckpt, base + "crossattention.output.", attended, hidden, p.layer_norm_eps)
        inner = g.gelu(g.linear(hidden, ckpt.tensor(base + "intermediate_query.dense.weight"),
                                ckpt.tensor(base + "intermediate_query.dense.bias")))
        hidden = _qformer_residual(g, ckpt, base + "output_query.", inner, hidden, p.layer_norm_eps)
    rows = g.reshape(hidden, (-1, p.hidden_size))
    return g.linear(rows, ckpt.tensor("projector.linear.weight"), ckpt.tensor("projector.linear.bias"))


def build_audio_network(g: Graph, ckpt: Checkpoint, cfg: GraniteSpeechConfig, max_frames: int) -> None:
    """Mark ``waveform [L]`` FP32 in and ``audio_embeds [A, text_hidden]`` (work type) out."""
    waveform = g.network.add_input("waveform", trt.float32, (-1,))
    features = add_front_end(g, waveform, cfg.front_end)
    encoded = add_encoder(g, ckpt, cfg, g.cast(features, g.work), max_frames)
    embeds = add_projector(g, ckpt, cfg, encoded)
    embeds.name = "audio_embeds"
    g.network.mark_output(embeds)
