# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The decoder half of the SD3 ``AutoencoderKL`` as one engine.

The engine takes latents already mapped back to the VAE's space (the
runtime divides by ``scaling_factor`` and adds ``shift_factor``) and returns
the decoded image in [-1, 1] as NCHW.
"""

from __future__ import annotations

import tensorrt as trt

from .graph_ops import Graph, build_engine
from .weights import take

EPS = 1e-6


def _resnet(graph: Graph, weights: dict, prefix: str, tensor, groups: int):
    hidden = graph.silu(graph.group_norm(tensor, groups, EPS, take(weights, prefix + "norm1.weight"),
                                         take(weights, prefix + "norm1.bias")))
    hidden = graph.conv2d(hidden, take(weights, prefix + "conv1.weight"),
                          take(weights, prefix + "conv1.bias"), padding=1)
    hidden = graph.silu(graph.group_norm(hidden, groups, EPS, take(weights, prefix + "norm2.weight"),
                                         take(weights, prefix + "norm2.bias")))
    hidden = graph.conv2d(hidden, take(weights, prefix + "conv2.weight"),
                          take(weights, prefix + "conv2.bias"), padding=1)
    if prefix + "conv_shortcut.weight" in weights:
        tensor = graph.conv2d(tensor, take(weights, prefix + "conv_shortcut.weight"),
                              take(weights, prefix + "conv_shortcut.bias"))
    return graph.add(graph.fp32(tensor), hidden)


def _mid_attention(graph: Graph, weights: dict, prefix: str, tensor, groups: int):
    """Single-head self-attention over all spatial positions, with a residual."""
    batch, channels, height, width = (int(v) for v in tuple(tensor.shape))
    normed = graph.group_norm(tensor, groups, EPS, take(weights, prefix + "group_norm.weight"),
                              take(weights, prefix + "group_norm.bias"))
    tokens = graph.reshape(normed, (batch, channels, height * width), second_transpose=(0, 2, 1))

    def project(name):
        projected = graph.linear(tokens, take(weights, prefix + name + ".weight"),
                                 take(weights, prefix + name + ".bias"))
        return graph.split_heads(projected, 1, channels)

    context = graph.attention(project("to_q"), project("to_k"), project("to_v"),
                              scale=channels ** -0.5)
    context = graph.linear(graph.merge_heads(context, 1, channels),
                           take(weights, prefix + "to_out.0.weight"),
                           take(weights, prefix + "to_out.0.bias"))
    spatial = graph.reshape(context, (batch, height, width, channels), second_transpose=(0, 3, 1, 2))
    return graph.add(graph.fp32(tensor), spatial)


def decode(graph: Graph, latents, weights: dict, cfg: dict):
    groups = cfg["norm_num_groups"]
    layers = cfg["layers_per_block"] + 1
    blocks = len(cfg["block_out_channels"])

    hidden = graph.conv2d(latents, take(weights, "decoder.conv_in.weight"),
                          take(weights, "decoder.conv_in.bias"), padding=1)
    hidden = _resnet(graph, weights, "decoder.mid_block.resnets.0.", hidden, groups)
    hidden = _mid_attention(graph, weights, "decoder.mid_block.attentions.0.", hidden, groups)
    hidden = _resnet(graph, weights, "decoder.mid_block.resnets.1.", hidden, groups)
    for block in range(blocks):
        for layer in range(layers):
            hidden = _resnet(graph, weights, f"decoder.up_blocks.{block}.resnets.{layer}.", hidden,
                             groups)
        if block != blocks - 1:
            hidden = graph.conv2d(graph.upsample_nearest2x(hidden),
                                  take(weights, f"decoder.up_blocks.{block}.upsamplers.0.conv.weight"),
                                  take(weights, f"decoder.up_blocks.{block}.upsamplers.0.conv.bias"),
                                  padding=1)
    hidden = graph.silu(graph.group_norm(hidden, groups, EPS,
                                         take(weights, "decoder.conv_norm_out.weight"),
                                         take(weights, "decoder.conv_norm_out.bias")))
    return graph.conv2d(hidden, take(weights, "decoder.conv_out.weight"),
                        take(weights, "decoder.conv_out.bias"), padding=1)


def build_vae_engine(weights: dict, cfg: dict, *, latent_h: int, latent_w: int, precision: str,
                     verbose: bool):
    def populate(graph: Graph) -> None:
        latents = graph.network.add_input(
            "latents", trt.float32, (1, cfg["latent_channels"], latent_h, latent_w))
        graph.output(decode(graph, latents, weights, cfg), "image")

    return build_engine(populate, precision=precision, verbose=verbose)
