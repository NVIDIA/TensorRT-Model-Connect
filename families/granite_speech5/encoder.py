# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One engine: log-mel and delta front end, Conformer CTC encoder and greedy CTC frame ids.

Front end (``GraniteSpeech5FeatureExtractor``): the runtime pads the clip as the reference does
(:func:`.constants.pad_waveform`); the graph keeps an even number of frames, takes
``log10(max(mel, 1e-10))``, clamps it at the clip's maximum minus ``logmel_floor_db``, maps it
``x / 4 + 1``, appends the deltas (edge frames repeated) and stacks frame pairs.

Encoder (``GraniteSpeech5Encoder``): Conformer blocks with block-wise attention over ``context_size``
frames and Shaw relative positions; a subsampling block halves the frame rate with a stride-2
depthwise convolution next to a mean-pooled residual; the CTC posteriors of the middle layer are
fed back through ``out_mid``. The CTC head is ``encoder.out``; the engine returns the index of
each frame's first largest logit, as ``torch.argmax`` does.
"""

from __future__ import annotations

import math

import numpy as np
import tensorrt as trt

from .checkpoint import Checkpoint
from .config import FRAME_STACKING, NORM_EPS, FrontEnd, GraniteSpeech5Config
from .constants import delta_weights, dft_basis, mel_filterbank, relative_table
from .graph import FOLDED, Graph

LOG_FLOOR = 1.0e-10  # clamp_min of the reference before log10


def relative_shift(g: Graph, offsets: trt.ITensor, context: int, heads: int) -> trt.ITensor:
    """``[nb, H, C, 2C - 1]`` with column ``m`` for offset ``i - j = C - 1 - m`` -> ``[nb, H, C, C]``.

    Row ``i`` needs columns ``C - 1 - i .. 2C - 2 - i``. Padding each row to ``2C`` columns and
    reading the flat rows with stride ``2C - 1`` after dropping the first ``C - 1`` values starts
    row ``i`` exactly there.
    """
    batch = g.dim(offsets, 0)
    padded = g.slice(offsets, (0, 0, 0, 0), g.concat([batch, g.i64([heads, context, 2 * context])], 0),
                     mode=trt.SampleMode.FILL)
    flat = g.reshape(padded, (-1, heads, 2 * context * context))
    span = context * (2 * context - 1)
    trimmed = g.slice(flat, (0, 0, context - 1), g.concat([batch, g.i64([heads, span])], 0))
    grid = g.reshape(trimmed, (-1, heads, context, 2 * context - 1))
    return g.slice(grid, (0, 0, 0, 0), g.concat([batch, g.i64([heads, context, context])], 0))


def add_front_end(g: Graph, waveform: trt.ITensor, fe: FrontEnd) -> trt.ITensor:
    """``waveform [P]`` (padded by the runtime) -> stacked features ``[T, 4 * n_mels]`` in FP32."""
    bins = fe.bins
    signal = g.reshape(waveform, (1, 1, 1, -1))
    conv = g.network.add_convolution_nd(
        signal, 2 * bins, (1, fe.n_fft),
        g.weights(dft_basis(fe).reshape(2 * bins, 1, 1, fe.n_fft), trt.float32), trt.Weights(trt.float32))
    conv.stride_nd = (1, fe.hop_length)
    spectrum = conv.get_output(0)  # [1, 2 * bins, 1, F]
    frames = g.dim(spectrum, 3)
    real = g.slice(spectrum, (0, 0, 0, 0), g.concat([g.i64([1, bins, 1]), frames], 0))
    imag = g.slice(spectrum, (0, bins, 0, 0), g.concat([g.i64([1, bins, 1]), frames], 0))
    power = g.add(g.mul(real, real), g.mul(imag, imag))
    rows = g.transpose(g.reshape(power, (bins, -1)), (1, 0))  # [F, bins]
    # The reference keeps an even number of frames before taking the log or the clip maximum.
    kept = g.mul(g.floor_div(frames, FRAME_STACKING), g.i64([FRAME_STACKING]))
    rows = g.rows(rows, kept)
    mel = g.matmul(rows, g.constant(mel_filterbank(fe), trt.float32))  # [T2, n_mels]
    floor = g.maximum(mel, g.scalar(LOG_FLOOR, trt.float32, 2))
    log10 = g.scale(g.unary(floor, trt.UnaryOperation.LOG), 1.0 / math.log(10.0))
    peak = g.network.add_reduce(log10, trt.ReduceOperation.MAX, 0b11, True).get_output(0)
    clamped = g.maximum(log10, g.sub(peak, g.scalar(fe.floor_db, trt.float32, 2)))
    logmel = g.add(g.scale(clamped, 0.25), g.scalar(1.0, trt.float32, 2))
    taps, denominator = delta_weights(fe)
    total = None
    for tap in taps:
        later = g.rows(logmel, kept, offset=tap, mode=trt.SampleMode.CLAMP)
        earlier = g.rows(logmel, kept, offset=-tap, mode=trt.SampleMode.CLAMP)
        term = g.sub(later, earlier) if tap == 1 else g.scale(g.sub(later, earlier), float(tap))
        total = term if total is None else g.add(total, term)
    delta = g.div(total, g.scalar(denominator, trt.float32, 2))
    frame = g.concat([logmel, delta], 1)  # [T2, 2 * n_mels]
    return g.reshape(frame, (-1, FRAME_STACKING * 2 * fe.n_mels))


def _norm(g: Graph, ckpt: Checkpoint, x: trt.ITensor, name: str, out: trt.DataType) -> trt.ITensor:
    return g.layer_norm(x, ckpt.tensor(name + ".weight"), ckpt.tensor(name + ".bias"), NORM_EPS, out)


def _bias(ckpt: Checkpoint, name: str, present: bool):
    return ckpt.tensor(name) if present else None


def _feed_forward(g: Graph, ckpt: Checkpoint, cfg: GraniteSpeech5Config, normed: trt.ITensor,
                  base: str) -> trt.ITensor:
    bias = cfg.encoder.feed_forward_bias
    inner = g.linear(normed, ckpt.tensor(base + ".linear1.weight"), _bias(ckpt, base + ".linear1.bias", bias))
    activated = g.cast(g.silu32(inner), g.work)
    return g.linear(activated, ckpt.tensor(base + ".linear2.weight"), _bias(ckpt, base + ".linear2.bias", bias))


def attend(g: Graph, query: trt.ITensor, key: trt.ITensor, value: trt.ITensor, table: trt.ITensor,
           valid_keys: trt.ITensor) -> trt.ITensor:
    """FP32 block attention ``softmax(q k / sqrt(D) + q E[i - j] / sqrt(D)) v`` over the valid keys of ``[nb, H, C, D]``.

    The queries are scaled before both products, as the reference scales the position table, so a sum
    whose scaled value fits FP32 cannot overflow first. ``table`` is :func:`relative_table`'s
    ``[1, 1, D, 2C - 1]``. ``valid_keys`` (bool, ``[nb, 1, 1, C]``) replaces the summed scores of padded
    keys with negative infinity, so they lose to any valid score; every block holds a valid key.
    """
    heads, context, width = int(query.shape[1]), int(query.shape[2]), int(query.shape[3])
    query = g.scale(query, width ** -0.5)
    content = g.matmul(query, key, transpose_right=True)  # [nb, H, C, C]
    offsets = g.matmul(query, table)  # [nb, H, C, 2C - 1]
    scores = g.add(content, relative_shift(g, offsets, context, heads))
    scores = g.select(valid_keys, scores, g.scalar(-np.inf, trt.float32, 4))
    return g.matmul(g.softmax32(scores, 3), value)  # [nb, H, C, D]


def _attention(g: Graph, ckpt: Checkpoint, cfg: GraniteSpeech5Config, normed: trt.ITensor, base: str,
               frames: trt.ITensor, longest: int) -> trt.ITensor:
    """Block-wise attention of ``frames`` rows; padded key rows are hidden from every query."""
    e = cfg.encoder
    context, heads, width = e.context_size, e.num_heads, e.head_dim
    blocks = g.floor_div(g.add(frames, g.i64([context - 1])), context)
    padded_rows = g.mul(blocks, g.i64([context]))

    def split(name: str) -> trt.ITensor:
        projected = g.wide(g.linear(normed, ckpt.tensor(f"{base}.{name}.weight")))
        rows = g.rows(projected, padded_rows, mode=trt.SampleMode.FILL)
        return g.transpose(g.reshape(rows, (-1, context, heads, width)), (0, 2, 1, 3))

    query, key, value = split("q_proj"), split("k_proj"), split("v_proj")
    table = relative_table(ckpt.tensor(base + ".rel_pos_emb.weight"), context, e.max_position_embeddings)
    positions = g.arange(longest, padded_rows)
    valid = g.binary(positions, g.cast(frames, trt.int32), trt.ElementWiseOperation.LESS)
    attended = attend(g, query, key, value, g.constant(table, trt.float32, (1, 1) + table.shape),
                      g.reshape(valid, (-1, 1, 1, context)))
    rows = g.reshape(g.transpose(attended, (0, 2, 1, 3)), (-1, e.attention_width))
    rows = g.rows(rows, frames)
    return g.linear(g.cast(rows, g.work), ckpt.tensor(base + ".o_proj.weight"), ckpt.tensor(base + ".o_proj.bias"))


def _convolution(g: Graph, ckpt: Checkpoint, cfg: GraniteSpeech5Config, normed: trt.ITensor, base: str,
                 stride: int) -> trt.ITensor:
    """GLU, depthwise convolution (zero padded by ``(k - 1) / 2``), batch norm and SiLU, all in FP32."""
    e = cfg.encoder
    channels, size = e.conv_channels, e.conv_kernel_size
    up = g.wide(g.linear(normed, ckpt.tensor(base + ".pointwise_lin1.weight"),
                         ckpt.tensor(base + ".pointwise_lin1.bias")))
    frames = g.dim(up, 0)
    half = g.concat([frames, g.i64([channels])], 0)
    gated = g.mul(g.slice(up, (0, 0), half), g.sigmoid32(g.slice(up, (0, channels), half)))
    planes = g.reshape(g.transpose(gated, (1, 0)), (1, channels, 1, -1))
    kernel = ckpt.tensor(base + ".depthwise_conv.weight").reshape(channels, 1, 1, size)
    depth = g.network.add_convolution_nd(planes, channels, (1, size), g.weights(kernel, trt.float32),
                                         trt.Weights(trt.float32))
    depth.num_groups = channels
    depth.stride_nd = (1, stride)
    depth.pre_padding = (0, (size - 1) // 2)
    depth.post_padding = (0, (size - 1) // 2)
    mean = ckpt.tensor(base + ".norm.running_mean").astype(np.float64)
    var = ckpt.tensor(base + ".norm.running_var").astype(np.float64)
    if np.any(var + NORM_EPS <= 0.0):
        raise ValueError(f"granite_speech5 {base}.norm.running_var is below -eps")
    factor = ckpt.tensor(base + ".norm.weight").astype(np.float64) / np.sqrt(var + NORM_EPS)
    shift = ckpt.tensor(base + ".norm.bias").astype(np.float64) - mean * factor
    for name, values in (("scale", factor), ("shift", shift)):
        if not np.all(np.isfinite(values.astype(np.float32))):
            raise ValueError(f"granite_speech5 {base}.norm folds to a non-finite float32 {name}")
    normed_planes = g.add(g.mul(depth.get_output(0), g.constant(factor, trt.float32, (1, channels, 1, 1))),
                          g.constant(shift, trt.float32, (1, channels, 1, 1)))
    rows = g.transpose(g.reshape(g.silu32(normed_planes), (channels, -1)), (1, 0))
    return g.linear(g.cast(rows, g.work), ckpt.tensor(base + ".pointwise_lin2.weight"),
                    ckpt.tensor(base + ".pointwise_lin2.bias"))


def _block(g: Graph, ckpt: Checkpoint, cfg: GraniteSpeech5Config, x: trt.ITensor, base: str,
           frames: trt.ITensor, longest: int, subsample: bool) -> trt.ITensor:
    """One Conformer block on the FP32 residual stream ``x [frames, hidden]``."""
    work = g.work
    ff1 = _feed_forward(g, ckpt, cfg, _norm(g, ckpt, x, base + "norm_feed_forward1", work), base + "feed_forward1")
    x = g.add(x, g.scale(g.wide(ff1), 0.5))
    attended = _attention(g, ckpt, cfg, _norm(g, ckpt, x, base + "norm_self_att", work), base + "self_attn",
                          frames, longest)
    x = g.add(x, g.wide(attended))
    conv = g.wide(_convolution(g, ckpt, cfg, _norm(g, ckpt, x, base + "norm_conv", work), base + "conv",
                               2 if subsample else 1))
    if subsample:
        # Mean of each frame pair (an odd last frame dropped) plus the first floor(T / 2) strided outputs.
        halved = g.floor_div(frames, 2)
        pairs = g.reshape(g.rows(x, g.mul(halved, g.i64([2]))), (-1, 2, cfg.encoder.hidden_size))
        pooled = g.reduce(pairs, trt.ReduceOperation.AVG, 1, False)
        x = g.add(pooled, g.rows(conv, halved))
    else:
        x = g.add(x, conv)
    ff2 = _feed_forward(g, ckpt, cfg, _norm(g, ckpt, x, base + "norm_feed_forward2", work), base + "feed_forward2")
    x = g.add(x, g.scale(g.wide(ff2), 0.5))
    return _norm(g, ckpt, x, base + "norm_out", trt.float32)


def self_condition(g: Graph, ckpt: Checkpoint, x: trt.ITensor, head: trt.ITensor) -> trt.ITensor:
    """``x + out_mid(softmax(out(x)))`` on the FP32 stream ``x``; ``head`` is the folded FP32 ``encoder.out``."""
    posteriors = g.softmax32(g.linear(x, head, FOLDED), 1)
    injected = g.linear(g.cast(posteriors, g.work), ckpt.tensor("encoder.out_mid.weight"),
                        ckpt.tensor("encoder.out_mid.bias"))
    return g.add(x, g.wide(injected))


def first_argmax(g: Graph, logits: trt.ITensor) -> trt.ITensor:
    """``[T, V]`` FP32 -> ``[T]`` int32: the lowest index holding each row's maximum.

    A row holding a non-finite logit (an overflow: NaN or either infinity) gets the sentinel ``V``, outside
    the vocabulary, so the runtime refuses it instead of decoding it.
    """
    vocab = int(logits.shape[1])
    peak = g.reduce(logits, trt.ReduceOperation.MAX, 1, True)
    hit = g.binary(logits, peak, trt.ElementWiseOperation.EQUAL)
    index = g.constant(np.arange(vocab, dtype=np.int32), trt.int32, (1, vocab))
    sentinel = g.constant(np.array([[vocab]], dtype=np.int32), trt.int32)
    first = g.reduce(g.select(hit, index, sentinel), trt.ReduceOperation.MIN, 1, True)  # [T, 1]
    # |x| < inf is false for NaN and both infinities.
    finite = g.binary(g.unary(logits, trt.UnaryOperation.ABS), g.scalar(np.inf, trt.float32, 2),
                      trt.ElementWiseOperation.LESS)
    zero = g.constant(np.zeros((1, 1), dtype=np.int32), trt.int32)
    overflowed = g.reduce(g.select(finite, zero, g.constant(np.ones((1, 1), dtype=np.int32), trt.int32)),
                          trt.ReduceOperation.MAX, 1, True)  # [T, 1]: 1 when the row holds a non-finite logit
    ids = g.select(g.binary(overflowed, zero, trt.ElementWiseOperation.EQUAL), first, sentinel)
    return g.reshape(ids, (-1,))


def longest_rows(cfg: GraniteSpeech5Config, max_waveform: int) -> list[int]:
    """Block-padded row count of every layer's attention at the profile's longest waveform."""
    fe = cfg.front_end
    frames = 1 + (max_waveform - fe.n_fft) // fe.hop_length
    stacked = (frames // FRAME_STACKING * FRAME_STACKING) // FRAME_STACKING
    context = cfg.encoder.context_size
    return [-(-count // context) * context for count in cfg.layer_frames(stacked)[:-1]]


def build_network(g: Graph, ckpt: Checkpoint, cfg: GraniteSpeech5Config, max_waveform: int) -> None:
    """Mark ``waveform [P]`` FP32 in and ``frame_ids [frames]`` int32 out."""
    e = cfg.encoder
    waveform = g.network.add_input("waveform", trt.float32, (-1,))
    features = add_front_end(g, waveform, cfg.front_end)
    x = g.wide(g.linear(g.cast(features, g.work), ckpt.tensor("encoder.input_linear.weight"),
                        ckpt.tensor("encoder.input_linear.bias")))
    frames = g.dim(x, 0)
    longest = longest_rows(cfg, max_waveform)
    # The CTC head is read once and shared by the self-conditioning and the output in FP32.
    head = g.folded(ckpt.tensor("encoder.out.weight"), ckpt.tensor("encoder.out.bias"), trt.float32)
    for index in range(e.num_layers):
        subsample = index in e.subsample_layers
        x = _block(g, ckpt, cfg, x, f"encoder.layers.{index}.", frames, longest[index], subsample)
        if subsample:
            frames = g.floor_div(frames, 2)
        if e.self_conditioning_after(index):
            x = self_condition(g, ckpt, x, head)
    ids = first_argmax(g, g.linear(x, head, FOLDED))
    ids.name = "frame_ids"
    g.network.mark_output(ids)
