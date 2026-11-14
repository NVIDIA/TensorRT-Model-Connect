# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Acoustic and semantic sigma-VAE encoders plus both speech connectors, as one BF16 engine.

Input ``audio`` is 24 kHz mono PCM ``[1, 1, 1, T]`` (FP32, cast to BF16 like the
reference's ``speech_tensors.to(dtype)``). Output ``audio_features`` is
``[ceil(T / hop), hidden]`` BF16: ``acoustic_connector(acoustic_mean) +
semantic_connector(semantic_mean)``, the rows the decoder substitutes for the
speech placeholder tokens.

Every convolution is causal like ``SConv1d._forward_non_streaming``: left zero
padding ``(kernel - 1) - (stride - 1)`` and, for strided layers, right zero
padding up to a whole stride so the frame count is ``ceil(length / stride)``.
The acoustic latent is the sigma-VAE mean; the reference's Gaussian sampling
noise is not part of the graph.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from .checkpoint import EncoderConfig, VibeVoiceAsrConfig, Weights
from .graph import Graph, new_builder, serialize

CONNECTOR_EPS = 1e-6


def _conv_kernel(weight: np.ndarray) -> np.ndarray:
    """``[out, in / groups, k]`` -> ``[out, in / groups, 1, k]``."""
    return weight.reshape(weight.shape[0], weight.shape[1], 1, weight.shape[2])


def _pad_right_to_stride(
    graph: Graph, tensor: trt.ITensor, channels: int, stride: int
) -> trt.ITensor:
    """Zero-extend the time axis of ``[1, C, 1, T]`` to ``ceil(T / stride) * stride``."""
    length = graph.dim(tensor, 3)
    stride_t = graph.i64([stride])
    blocks = graph.binary(
        graph.add(length, graph.i64([stride - 1])), stride_t, trt.ElementWiseOperation.FLOOR_DIV
    )
    padded = graph.mul(blocks, stride_t)
    size = graph.concat([graph.i64([1, channels, 1]), padded])
    layer = graph.network.add_slice(tensor, (0, 0, 0, 0), (1, channels, 1, 1), (1, 1, 1, 1))
    layer.set_input(2, size)
    layer.mode = trt.SampleMode.FILL
    return layer.get_output(0)


def _causal_conv(
    graph: Graph,
    tensor: trt.ITensor,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    stride: int = 1,
    groups: int = 1,
) -> trt.ITensor:
    out_channels, _, kernel = weight.shape
    if stride > 1:
        tensor = _pad_right_to_stride(graph, tensor, int(tensor.shape[1]), stride)
    layer = graph.network.add_convolution_nd(
        tensor,
        int(out_channels),
        (1, int(kernel)),
        graph.weights(_conv_kernel(weight)),
        graph.weights(bias),
    )
    layer.stride_nd = (1, stride)
    layer.pre_padding = (0, int(kernel) - stride)
    layer.post_padding = (0, 0)
    layer.num_groups = groups
    return layer.get_output(0)


def _pointwise(
    graph: Graph, tensor: trt.ITensor, weight: np.ndarray, bias: np.ndarray
) -> trt.ITensor:
    """``nn.Linear`` over the channel axis of ``[1, C, 1, T]``."""
    kernel = weight.reshape(weight.shape[0], weight.shape[1], 1, 1)
    layer = graph.network.add_convolution_nd(
        tensor, int(weight.shape[0]), (1, 1), graph.weights(kernel), graph.weights(bias)
    )
    return layer.get_output(0)


def _block(
    graph: Graph,
    tensor: trt.ITensor,
    weights: Weights,
    prefix: str,
    channels: int,
    cfg: EncoderConfig,
) -> trt.ITensor:
    """``Block1D``: depthwise-conv mixer and GELU FFN, each pre-normed and layer-scaled."""
    shape = (1, channels, 1, 1)
    vector = (channels,)
    mixed = graph.rms_norm(tensor, weights.bf16(f"{prefix}.norm.weight", vector), cfg.eps, 1, shape)
    mixed = _causal_conv(
        graph,
        mixed,
        weights.bf16(f"{prefix}.mixer.conv.conv.conv.weight", (channels, 1, cfg.kernel_size)),
        weights.bf16(f"{prefix}.mixer.conv.conv.conv.bias", vector),
        groups=channels,
    )
    mixed = graph.mul(mixed, graph.constant(weights.bf16(f"{prefix}.gamma", vector), shape))
    tensor = graph.add(tensor, mixed)

    hidden = graph.rms_norm(
        tensor, weights.bf16(f"{prefix}.ffn_norm.weight", vector), cfg.eps, 1, shape
    )
    hidden = _pointwise(
        graph,
        hidden,
        weights.bf16(f"{prefix}.ffn.linear1.weight", (4 * channels, channels)),
        weights.bf16(f"{prefix}.ffn.linear1.bias", (4 * channels,)),
    )
    hidden = graph.activation(hidden, trt.ActivationType.GELU_ERF)
    hidden = _pointwise(
        graph,
        hidden,
        weights.bf16(f"{prefix}.ffn.linear2.weight", (channels, 4 * channels)),
        weights.bf16(f"{prefix}.ffn.linear2.bias", vector),
    )
    hidden = graph.mul(hidden, graph.constant(weights.bf16(f"{prefix}.ffn_gamma", vector), shape))
    return graph.add(tensor, hidden)


def _encoder(graph: Graph, audio: trt.ITensor, weights: Weights, cfg: EncoderConfig) -> trt.ITensor:
    """``TokenizerEncoder.forward`` -> latent mean ``[1, vae_dim, 1, frames]``."""
    prefix = cfg.prefix
    channels = cfg.stage_channels(0)
    hidden = _causal_conv(
        graph,
        audio,
        weights.bf16(
            f"{prefix}.downsample_layers.0.0.conv.conv.weight", (channels, 1, cfg.kernel_size)
        ),
        weights.bf16(f"{prefix}.downsample_layers.0.0.conv.conv.bias", (channels,)),
    )
    for stage, depth in enumerate(cfg.depths):
        if stage > 0:
            ratio = cfg.ratios[stage - 1]
            previous = cfg.stage_channels(stage - 1)
            channels = cfg.stage_channels(stage)
            hidden = _causal_conv(
                graph,
                hidden,
                weights.bf16(
                    f"{prefix}.downsample_layers.{stage}.0.conv.conv.weight",
                    (channels, previous, 2 * ratio),
                ),
                weights.bf16(f"{prefix}.downsample_layers.{stage}.0.conv.conv.bias", (channels,)),
                stride=ratio,
            )
        for index in range(depth):
            hidden = _block(
                graph, hidden, weights, f"{prefix}.stages.{stage}.{index}", channels, cfg
            )
    return _causal_conv(
        graph,
        hidden,
        weights.bf16(f"{prefix}.head.conv.conv.weight", (cfg.vae_dim, channels, cfg.kernel_size)),
        weights.bf16(f"{prefix}.head.conv.conv.bias", (cfg.vae_dim,)),
    )


def _connector(
    graph: Graph,
    latents: trt.ITensor,
    weights: Weights,
    prefix: str,
    vae_dim: int,
    hidden_size: int,
) -> trt.ITensor:
    """``SpeechConnector``: fc1 -> LlamaRMSNorm -> fc2 on ``[frames, vae_dim]`` rows."""
    rows = graph.reshape(latents, (-1, vae_dim), (0, 3, 1, 2), transpose_first=True)
    hidden = graph.linear(
        rows,
        weights.bf16(f"{prefix}.fc1.weight", (hidden_size, vae_dim)),
        weights.bf16(f"{prefix}.fc1.bias", (hidden_size,)),
    )
    hidden = graph.rms_norm(
        hidden,
        weights.bf16(f"{prefix}.norm.weight", (hidden_size,)),
        CONNECTOR_EPS,
        1,
        (1, hidden_size),
    )
    return graph.linear(
        hidden,
        weights.bf16(f"{prefix}.fc2.weight", (hidden_size, hidden_size)),
        weights.bf16(f"{prefix}.fc2.bias", (hidden_size,)),
    )


def build_audio_encoder_engine(
    cfg: VibeVoiceAsrConfig,
    weights: Weights,
    *,
    max_samples: int,
    opt_samples: int,
    verbose: bool = False,
) -> bytes:
    builder, network, config = new_builder(verbose, optimization_level=3, workspace_bytes=8 << 30)
    graph = Graph(network)
    audio = network.add_input("audio", trt.float32, (1, 1, 1, -1))
    samples = graph.cast(audio, trt.bfloat16)
    hidden_size = cfg.decoder.hidden_size
    acoustic = _connector(
        graph,
        _encoder(graph, samples, weights, cfg.acoustic),
        weights,
        "model.acoustic_connector",
        cfg.acoustic.vae_dim,
        hidden_size,
    )
    semantic = _connector(
        graph,
        _encoder(graph, samples, weights, cfg.semantic),
        weights,
        "model.semantic_connector",
        cfg.semantic.vae_dim,
        hidden_size,
    )
    features = graph.add(acoustic, semantic)
    features.name = "audio_features"
    network.mark_output(features)

    profile = builder.create_optimization_profile()
    profile.set_shape(
        "audio", (1, 1, 1, 1), (1, 1, 1, min(opt_samples, max_samples)), (1, 1, 1, max_samples)
    )
    config.add_optimization_profile(profile)
    return serialize(builder, network, config, "audio encoder")
