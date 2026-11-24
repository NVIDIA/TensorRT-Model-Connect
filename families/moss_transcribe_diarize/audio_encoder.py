# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audio engine: Whisper encoder, 4x frame merge and the VQ adaptor, one chunk per call.

    input_features [1, mel_bins, mel_frames]   FP32 log-mel of one 30 s chunk
      conv1 (k3 s1 p1) + GELU, conv2 (k3 s2 p1) + GELU
      + learned positions                      [frames, d_model]
      pre-norm encoder layers, bidirectional attention
      final LayerNorm
      merge consecutive frames                 [frames / merge, d_model * merge]
      Linear + SiLU + Linear + LayerNorm       [tokens_per_chunk, hidden]
    audio_embeds [tokens_per_chunk, hidden]     work precision

The checkpoint keeps only the first ``token_len * merge`` encoder frames of each
chunk and merges them after concatenating the chunks of one audio. Every chunk
contributes a multiple of ``merge`` frames and the adaptor acts on each merged
frame independently, so merging inside the engine and keeping the leading
``token_len`` rows per chunk is the same computation.
"""

from __future__ import annotations

import sys

import tensorrt as trt

from .checkpoint import Checkpoint
from .config import WHISPER_LAYER_NORM_EPS, MossConfig
from .graph import Graph, builder_config, new_network

_ENCODER = "model.whisper_encoder."
_ADAPTOR = "model.vq_adaptor.layers."

INPUT_NAME = "input_features"
OUTPUT_NAME = "audio_embeds"


def _conv_gelu(g: Graph, x: trt.ITensor, name: str, out_channels: int,
               stride: int) -> trt.ITensor:
    kernel = g.checkpoint.shape(f"{name}.weight")
    if len(kernel) != 3 or kernel[2] != 3 or kernel[0] != out_channels:
        raise ValueError(f"{name}.weight must be a [{out_channels}, in, 3] Conv1d kernel")
    conv = g.net.add_convolution_nd(
        x,
        num_output_maps=out_channels,
        kernel_shape=(1, 3),
        kernel=g.raw_weights(f"{name}.weight"),
        bias=g.raw_weights(f"{name}.bias"),
    )
    conv.stride_nd = (1, stride)
    conv.padding_nd = (0, 1)
    return g.gelu(conv.get_output(0))


def _attention(g: Graph, x: trt.ITensor, prefix: str, frames: int, heads: int,
               head_dim: int) -> trt.ITensor:
    def as_heads(rows: trt.ITensor) -> trt.ITensor:
        split = g.reshape(rows, (frames, heads, head_dim))
        return g.reshape(g.transpose(split, (1, 0, 2)), (1, heads, frames, head_dim))

    # WhisperAttention scales the projected query before attention; the scale is
    # a power of two, so the scaled values are exact in every precision.
    query = g.linear(x, f"{prefix}.q_proj.weight", f"{prefix}.q_proj.bias")
    query = g.mul(as_heads(query), g.work_scalar(head_dim ** -0.5, 4))
    key = as_heads(g.linear(x, f"{prefix}.k_proj.weight"))
    value = as_heads(g.linear(x, f"{prefix}.v_proj.weight", f"{prefix}.v_proj.bias"))
    attention = g.net.add_attention(
        query, key, value, trt.AttentionNormalizationOp.SOFTMAX, False
    )
    if attention is None:
        raise RuntimeError("TensorRT rejected the encoder attention")
    attention.decomposable = True
    context = g.transpose(attention.get_output(0), (0, 2, 1, 3))
    context = g.reshape(context, (frames, heads * head_dim))
    return g.linear(context, f"{prefix}.out_proj.weight", f"{prefix}.out_proj.bias")


def _encoder_layer(g: Graph, hidden: trt.ITensor, index: int, config: MossConfig) -> trt.ITensor:
    audio = config.audio
    prefix = f"{_ENCODER}layers.{index}"
    normed = g.layer_norm(
        hidden,
        f"{prefix}.self_attn_layer_norm.weight",
        f"{prefix}.self_attn_layer_norm.bias",
        WHISPER_LAYER_NORM_EPS,
    )
    attended = _attention(
        g, normed, f"{prefix}.self_attn", config.encoder_frames, audio.num_heads, audio.head_dim
    )
    hidden = g.add(hidden, attended)
    normed = g.layer_norm(
        hidden,
        f"{prefix}.final_layer_norm.weight",
        f"{prefix}.final_layer_norm.bias",
        WHISPER_LAYER_NORM_EPS,
    )
    ffn = g.gelu(g.linear(normed, f"{prefix}.fc1.weight", f"{prefix}.fc1.bias"))
    ffn = g.linear(ffn, f"{prefix}.fc2.weight", f"{prefix}.fc2.bias")
    return g.add(hidden, ffn)


def build_audio_engine(checkpoint: Checkpoint, config: MossConfig, precision: str,
                       *, verbose: bool = False) -> bytes:
    """Serialize the one-chunk audio engine."""
    audio = config.audio
    frames = config.encoder_frames
    mel_frames = config.frontend.mel_frames
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = new_network(builder)
    g = Graph(network, checkpoint, precision)

    features = network.add_input(INPUT_NAME, trt.float32, (1, audio.num_mel_bins, mel_frames))
    x = g.reshape(g.cast(features, g.work), (1, audio.num_mel_bins, 1, mel_frames))
    x = _conv_gelu(g, x, f"{_ENCODER}conv1", audio.d_model, 1)
    x = _conv_gelu(g, x, f"{_ENCODER}conv2", audio.d_model, 2)
    hidden = g.transpose(g.reshape(x, (audio.d_model, frames)), (1, 0))
    if checkpoint.shape(f"{_ENCODER}embed_positions.weight") != (frames, audio.d_model):
        raise ValueError("the Whisper position table must cover every encoder frame")
    hidden = g.add(hidden, g.weight(f"{_ENCODER}embed_positions.weight"))

    for index in range(audio.num_layers):
        hidden = _encoder_layer(g, hidden, index, config)
    hidden = g.layer_norm(
        hidden, f"{_ENCODER}layer_norm.weight", f"{_ENCODER}layer_norm.bias",
        WHISPER_LAYER_NORM_EPS,
    )

    merged = g.reshape(hidden, (config.tokens_per_chunk, config.adaptor_input_dim))
    adapted = g.silu(g.linear(merged, f"{_ADAPTOR}0.weight", f"{_ADAPTOR}0.bias"))
    adapted = g.linear(adapted, f"{_ADAPTOR}2.weight", f"{_ADAPTOR}2.bias")
    # VQAdaptor builds its LayerNorm with the text config's rms_norm_eps.
    adapted = g.layer_norm(
        adapted, f"{_ADAPTOR}3.weight", f"{_ADAPTOR}3.bias", config.text.rms_norm_eps
    )
    adapted.name = OUTPUT_NAME
    network.mark_output(adapted)

    if verbose:
        print(
            f"[trtmc build] moss_transcribe_diarize audio engine: {audio.num_layers} layers, "
            f"{frames} frames -> {config.tokens_per_chunk} tokens, precision={precision}",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, builder_config(builder))
    if plan is None:
        raise RuntimeError("moss_transcribe_diarize audio engine build failed")
    return bytes(plan)


def audio_weight_names(config: MossConfig) -> list[str]:
    """Every checkpoint tensor the audio engine reads."""
    names = [
        f"{_ENCODER}conv1.weight", f"{_ENCODER}conv1.bias",
        f"{_ENCODER}conv2.weight", f"{_ENCODER}conv2.bias",
        f"{_ENCODER}embed_positions.weight",
        f"{_ENCODER}layer_norm.weight", f"{_ENCODER}layer_norm.bias",
    ]
    for index in range(config.audio.num_layers):
        prefix = f"{_ENCODER}layers.{index}"
        for part in ("q_proj", "v_proj", "out_proj"):
            names += [f"{prefix}.self_attn.{part}.weight", f"{prefix}.self_attn.{part}.bias"]
        names.append(f"{prefix}.self_attn.k_proj.weight")
        for part in ("self_attn_layer_norm", "final_layer_norm", "fc1", "fc2"):
            names += [f"{prefix}.{part}.weight", f"{prefix}.{part}.bias"]
    for layer in (0, 2, 3):
        names += [f"{_ADAPTOR}{layer}.weight", f"{_ADAPTOR}{layer}.bias"]
    return names


__all__ = ["INPUT_NAME", "OUTPUT_NAME", "audio_weight_names", "build_audio_engine"]
