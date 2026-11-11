# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT engine for the Qwen3-ASR audio encoder and its projector.

``Qwen3ASREncoder`` cuts the mel spectrogram into chunks of ``2 * n_window`` frames, runs three
stride-2 convolutions on every chunk, adds sinusoid positions that restart in every chunk, keeps
the valid tokens, and attends inside windows of ``n_window_infer / (2 * n_window)`` chunks.

The runtime pads the clip's mel frames with zeros to whole chunks and the chunk count to whole
windows, so the engine sees a dense ``[windows, chunks per window, mel bins, chunk frames]``
tensor. Only the last real chunk can be partial, so the reference's packed valid tokens are a
prefix of this chunk-major grid and every reference window is one engine window. The per-chunk
token counts become the key mask; rows at invalid tokens are computed but meaningless, and the
runtime keeps only the valid prefix.

Engine contract:
  ``input_features``  [W, window_chunks, num_mel_bins, chunk_frames] FP32   log-mel, zero padded
  ``chunk_tokens``    [W, window_chunks] int32                              valid tokens per chunk
  ``audio_embeds``    [W, window_tokens, output_dim] FP32                   projector output
"""

from __future__ import annotations

from typing import BinaryIO

import numpy as np
import tensorrt as trt

from . import trt_ops
from .config import EncoderConfig
from .frontend import sinusoid_positions
from .trt_ops import BF16, FP32, Graph
from .weights import take

_OPTIMIZATION_LEVEL = 3
_WORKSPACE_BYTES = 8 << 30
_TOWER = "model.audio_tower"
_PROJECTOR = "model.multi_modal_projector"
# PyTorch's nn.LayerNorm default; the checkpoint declares no other value.
_LAYER_NORM_EPS = 1.0e-5
OPT_WINDOWS = 4


def _windowed_attention(
    graph: Graph,
    config: EncoderConfig,
    weights: dict,
    prefix: str,
    hidden_states: trt.ITensor,
    key_mask: trt.ITensor,
) -> trt.ITensor:
    """Multi-head attention inside each window; ``key_mask`` is BOOL ``[W, 1, 1, window_tokens]``."""
    width = config.d_model
    heads = config.heads
    head_dim = config.head_dim
    tokens = config.window_tokens

    def projection(name: str) -> trt.ITensor:
        value = trt_ops.linear(
            graph,
            hidden_states,
            take(weights, f"{prefix}.{name}.weight", (width, width)),
            take(weights, f"{prefix}.{name}.bias", (width,)),
        )
        value = trt_ops.reshape(graph, value, (-1, tokens, heads, head_dim))
        return trt_ops.transpose(graph, value, (0, 2, 1, 3))

    query = projection("q_proj")
    key = projection("k_proj")
    value = projection("v_proj")
    scores = graph.network.add_matrix_multiply(
        trt_ops.cast(graph, query, FP32),
        trt.MatrixOperation.NONE,
        trt_ops.cast(graph, key, FP32),
        trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)
    scores = trt_ops.mul(graph, scores, trt_ops.scalar_fp32(graph, head_dim**-0.5, 4))
    scores = trt_ops.select(graph, key_mask, scores, trt_ops.scalar_fp32(graph, trt_ops.MASKED_SCORE, 4))
    probabilities = trt_ops.cast(graph, trt_ops.softmax_last(graph, scores), graph.compute)
    context = graph.network.add_matrix_multiply(
        probabilities, trt.MatrixOperation.NONE, value, trt.MatrixOperation.NONE
    ).get_output(0)
    context = trt_ops.reshape(graph, trt_ops.transpose(graph, context, (0, 2, 1, 3)), (-1, tokens, width))
    return trt_ops.linear(
        graph,
        context,
        take(weights, f"{prefix}.out_proj.weight", (width, width)),
        take(weights, f"{prefix}.out_proj.bias", (width,)),
    )


def _encoder_layer(
    graph: Graph,
    config: EncoderConfig,
    weights: dict,
    index: int,
    hidden_states: trt.ITensor,
    key_mask: trt.ITensor,
) -> trt.ITensor:
    prefix = f"{_TOWER}.layers.{index}"
    width = config.d_model

    def norm(name: str, tensor: trt.ITensor) -> trt.ITensor:
        return trt_ops.layer_norm(
            graph,
            tensor,
            take(weights, f"{prefix}.{name}.weight", (width,)),
            take(weights, f"{prefix}.{name}.bias", (width,)),
            _LAYER_NORM_EPS,
        )

    attended = _windowed_attention(
        graph, config, weights, f"{prefix}.self_attn",
        norm("self_attn_layer_norm", hidden_states), key_mask,
    )
    hidden_states = trt_ops.add(graph, hidden_states, attended)
    inner = trt_ops.gelu(
        graph,
        trt_ops.linear(
            graph,
            norm("final_layer_norm", hidden_states),
            take(weights, f"{prefix}.fc1.weight", (config.ffn, width)),
            take(weights, f"{prefix}.fc1.bias", (config.ffn,)),
        ),
    )
    feed_forward = trt_ops.linear(
        graph,
        inner,
        take(weights, f"{prefix}.fc2.weight", (width, config.ffn)),
        take(weights, f"{prefix}.fc2.bias", (width,)),
    )
    return trt_ops.add(graph, hidden_states, feed_forward)


def _convolution_front(
    graph: Graph, config: EncoderConfig, weights: dict, features: trt.ITensor
) -> trt.ITensor:
    """``[W, chunks, mel, frames]`` -> chunk tokens ``[W * chunks, tokens_per_chunk, d_model]``."""
    channels = config.conv_channels
    mels = config.num_mel_bins
    frames = config.chunk_frames
    hidden = trt_ops.reshape(graph, features, (-1, 1, mels, frames))
    for index, in_channels in ((1, 1), (2, channels), (3, channels)):
        hidden = trt_ops.conv2d(
            graph,
            hidden,
            take(weights, f"{_TOWER}.conv2d{index}.weight", (channels, in_channels, 3, 3)),
            take(weights, f"{_TOWER}.conv2d{index}.bias", (channels,)),
            stride=2,
            padding=1,
        )
        hidden = trt_ops.gelu(graph, hidden)
    # [N, C, F, T] -> [N, T, C * F], the reference's permute(0, 3, 1, 2) then view.
    freq = config.freq_bins_after_cnn
    tokens = config.tokens_per_chunk
    hidden = trt_ops.transpose(graph, hidden, (0, 3, 1, 2))
    hidden = trt_ops.reshape(graph, hidden, (-1, tokens, channels * freq))
    hidden = trt_ops.linear(
        graph, hidden, take(weights, f"{_TOWER}.conv_out.weight", (config.d_model, channels * freq))
    )
    positions = sinusoid_positions(config.max_positions, config.d_model)[:tokens]
    return trt_ops.add(
        graph, hidden, trt_ops.constant_compute(graph, positions.reshape(1, tokens, config.d_model))
    )


def _key_mask(graph: Graph, config: EncoderConfig, chunk_tokens: trt.ITensor) -> trt.ITensor:
    """BOOL ``[W, 1, 1, window_tokens]``: token ``p`` of a chunk is valid when ``p < chunk_tokens``."""
    tokens = config.tokens_per_chunk
    positions = trt_ops.constant(graph, np.arange(tokens, dtype=np.int32).reshape(1, 1, tokens))
    counts = trt_ops.reshape(graph, chunk_tokens, (0, 0, 1))
    valid = trt_ops.less(graph, positions, counts)
    return trt_ops.reshape(graph, valid, (-1, 1, 1, config.window_tokens))


def build_encoder_plan(
    config: EncoderConfig,
    weights: dict,
    output: BinaryIO,
    *,
    max_windows: int,
    verbose: bool,
    compute: trt.DataType = BF16,
) -> None:
    """Build the audio encoder for ``1 .. max_windows`` attention windows; write the plan."""
    if max_windows < 1:
        raise ValueError("the audio encoder needs at least one attention window")
    graph = trt_ops.create_graph(
        verbose=verbose,
        workspace_bytes=_WORKSPACE_BYTES,
        optimization_level=_OPTIMIZATION_LEVEL,
        compute=compute,
    )
    network = graph.network
    chunks = config.window_chunks
    features = network.add_input(
        "input_features", FP32, (-1, chunks, config.num_mel_bins, config.chunk_frames)
    )
    chunk_tokens = network.add_input("chunk_tokens", trt.int32, (-1, chunks))
    profile = graph.builder.create_optimization_profile()
    opt = min(OPT_WINDOWS, max_windows)
    profile.set_shape(
        "input_features",
        (1, chunks, config.num_mel_bins, config.chunk_frames),
        (opt, chunks, config.num_mel_bins, config.chunk_frames),
        (max_windows, chunks, config.num_mel_bins, config.chunk_frames),
    )
    profile.set_shape("chunk_tokens", (1, chunks), (opt, chunks), (max_windows, chunks))
    graph.config.add_optimization_profile(profile)

    hidden_states = _convolution_front(graph, config, weights, features)
    hidden_states = trt_ops.reshape(graph, hidden_states, (-1, config.window_tokens, config.d_model))
    key_mask = _key_mask(graph, config, chunk_tokens)
    for index in range(config.layers):
        hidden_states = _encoder_layer(graph, config, weights, index, hidden_states, key_mask)
    width = config.d_model
    hidden_states = trt_ops.layer_norm(
        graph,
        hidden_states,
        take(weights, f"{_TOWER}.ln_post.weight", (width,)),
        take(weights, f"{_TOWER}.ln_post.bias", (width,)),
        _LAYER_NORM_EPS,
    )
    projected = trt_ops.gelu(
        graph,
        trt_ops.linear(
            graph,
            hidden_states,
            take(weights, f"{_PROJECTOR}.linear_1.weight", (width, width)),
            take(weights, f"{_PROJECTOR}.linear_1.bias", (width,)),
        ),
    )
    projected = trt_ops.linear(
        graph,
        projected,
        take(weights, f"{_PROJECTOR}.linear_2.weight", (config.output_dim, width)),
        take(weights, f"{_PROJECTOR}.linear_2.bias", (config.output_dim,)),
    )
    audio_embeds = trt_ops.cast(graph, projected, FP32)
    audio_embeds.name = "audio_embeds"
    network.mark_output(audio_embeds)
    output.write(memoryview(trt_ops.serialize(graph, "audio encoder")))
