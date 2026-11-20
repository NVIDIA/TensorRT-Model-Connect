# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``AutoencoderKL`` decoder graph (the FLUX.1 VAE that Chroma ships).

Engine I/O: ``latents`` ``[1, latent_channels, h, w]`` float32 (already
divided by the scaling factor and shifted) in, ``image`` ``[1, 3, 8h, 8w]``
float32 in ``[-1, 1]`` out.
"""

from __future__ import annotations

import tensorrt as trt

from .checkpoint import SafetensorsDirectory, VaeConfig
from .graph_ops import Graph

EPS = 1e-6


def _resnet(g: Graph, w: SafetensorsDirectory, x: trt.ITensor, name: str, groups: int) -> trt.ITensor:
    h = g.silu(g.group_norm(x, groups, w[f"{name}.norm1.weight"], w[f"{name}.norm1.bias"], EPS))
    h = g.conv2d(h, w[f"{name}.conv1.weight"], w[f"{name}.conv1.bias"], 1)
    h = g.silu(g.group_norm(h, groups, w[f"{name}.norm2.weight"], w[f"{name}.norm2.bias"], EPS))
    h = g.conv2d(h, w[f"{name}.conv2.weight"], w[f"{name}.conv2.bias"], 1)
    if f"{name}.conv_shortcut.weight" in w:
        x = g.conv2d(x, w[f"{name}.conv_shortcut.weight"], w[f"{name}.conv_shortcut.bias"], 0)
    return g.add(x, h)


def _attention(g: Graph, w: SafetensorsDirectory, x: trt.ITensor, name: str, groups: int) -> trt.ITensor:
    _, channels, height, width = x.shape
    tokens = height * width
    h = g.group_norm(x, groups, w[f"{name}.group_norm.weight"], w[f"{name}.group_norm.bias"], EPS)
    h = g.transpose(g.reshape(h, (1, channels, tokens)), (0, 2, 1))
    q = g.linear(h, w[f"{name}.to_q.weight"], w[f"{name}.to_q.bias"])
    k = g.linear(h, w[f"{name}.to_k.weight"], w[f"{name}.to_k.bias"])
    v = g.linear(h, w[f"{name}.to_v.weight"], w[f"{name}.to_v.bias"])
    # One head of width C; the 1/sqrt(C) score scale is applied to the query in FP32.
    q32 = g.cast(q, trt.float32)
    q = g.cast(g.mul(q32, g.scalar(float(channels) ** -0.5, q32)), q.dtype)
    q, k, v = (g.reshape(item, (1, 1, tokens, channels)) for item in (q, k, v))
    context = g.reshape(g.attention(q, k, v, None), (1, tokens, channels))
    out = g.linear(context, w[f"{name}.to_out.0.weight"], w[f"{name}.to_out.0.bias"])
    out = g.reshape(g.transpose(out, (0, 2, 1)), (1, channels, height, width))
    return g.add(out, x)


def build_vae_decoder_network(
    graph: Graph,
    weights: SafetensorsDirectory,
    config: VaeConfig,
    latent_height: int,
    latent_width: int,
) -> None:
    g, w = graph, weights
    groups = config.groups
    latents = g.network.add_input("latents", trt.float32, (1, config.latent_channels, latent_height, latent_width))
    x = g.conv2d(g.cast(latents, g.dtype), w["decoder.conv_in.weight"], w["decoder.conv_in.bias"], 1)
    x = _resnet(g, w, x, "decoder.mid_block.resnets.0", groups)
    x = _attention(g, w, x, "decoder.mid_block.attentions.0", groups)
    x = _resnet(g, w, x, "decoder.mid_block.resnets.1", groups)
    blocks = len(config.block_out_channels)
    for block in range(blocks):
        for layer in range(config.layers_per_block + 1):
            x = _resnet(g, w, x, f"decoder.up_blocks.{block}.resnets.{layer}", groups)
        if block != blocks - 1:
            name = f"decoder.up_blocks.{block}.upsamplers.0.conv"
            x = g.conv2d(g.upsample_nearest_2x(x), w[f"{name}.weight"], w[f"{name}.bias"], 1)
    x = g.silu(g.group_norm(x, groups, w["decoder.conv_norm_out.weight"], w["decoder.conv_norm_out.bias"], EPS))
    x = g.conv2d(x, w["decoder.conv_out.weight"], w["decoder.conv_out.bias"], 1)
    image = g.cast(x, trt.float32)
    image.name = "image"
    g.network.mark_output(image)
