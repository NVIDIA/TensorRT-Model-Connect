# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the Stable Diffusion VAE decoder.

`AutoencoderKL`, structurally the same decoder PixArt-Sigma uses: identical
`block_out_channels`, `layers_per_block`, `latent_channels` and `norm_num_groups`.

One difference that is not visible from the config: this checkpoint stores the
mid-block attention under the **legacy** `query`/`key`/`value`/`proj_attn` names
rather than `to_q`/`to_k`/`to_v`/`to_out.0`. Newer diffusers renames them on
load, but the raw safetensors keep the old spelling, so both are accepted here.
"""

from __future__ import annotations

import math


from . import graph as g

_SCALING_FACTOR = 0.18215


def _pick(weights, prefix: str, *names: str):
    """Return the first present spelling of a weight, so both namings work."""
    for name in names:
        if f"{prefix}.{name}.weight" in weights:
            return (weights[f"{prefix}.{name}.weight"],
                    weights.get(f"{prefix}.{name}.bias"))
    raise KeyError(f"VAE checkpoint has none of {names} under {prefix}")


def _resnet(network, x, weights, prefix, groups, eps, dtype):
    """The VAE resnet: no timestep term, unlike the UNet's."""
    residual = x
    h = g.add_group_norm(network, x, weights[f"{prefix}.norm1.weight"],
                         weights[f"{prefix}.norm1.bias"], groups, eps, dtype=dtype)
    h = g.add_silu(network, h)
    h = g.add_conv2d(network, h, weights[f"{prefix}.conv1.weight"],
                     weights[f"{prefix}.conv1.bias"], padding=(1, 1), dtype=dtype)
    h = g.add_group_norm(network, h, weights[f"{prefix}.norm2.weight"],
                         weights[f"{prefix}.norm2.bias"], groups, eps, dtype=dtype)
    h = g.add_silu(network, h)
    h = g.add_conv2d(network, h, weights[f"{prefix}.conv2.weight"],
                     weights[f"{prefix}.conv2.bias"], padding=(1, 1), dtype=dtype)
    if f"{prefix}.conv_shortcut.weight" in weights:
        residual = g.add_conv2d(network, residual, weights[f"{prefix}.conv_shortcut.weight"],
                                weights.get(f"{prefix}.conv_shortcut.bias"), dtype=dtype)
    return g.add_sum(network, h, residual)


def _mid_attention(network, x, weights, prefix, groups, eps, dtype):
    """Single-head spatial self-attention over the feature map."""
    residual = x
    shape = tuple(int(v) for v in x.shape)
    channels, height, width = shape[1], shape[2], shape[3]
    tokens = height * width

    h = g.add_group_norm(network, x, weights[f"{prefix}.group_norm.weight"],
                         weights[f"{prefix}.group_norm.bias"], groups, eps, dtype=dtype)
    h = g.spatial_to_tokens(network, h)

    qw, qb = _pick(weights, prefix, "to_q", "query")
    kw, kb = _pick(weights, prefix, "to_k", "key")
    vw, vb = _pick(weights, prefix, "to_v", "value")
    ow, ob = _pick(weights, prefix, "to_out.0", "proj_attn")

    query = g.split_heads(network, g.add_linear(network, h, qw, qb, dtype=dtype), tokens, 1, channels)
    key = g.split_heads(network, g.add_linear(network, h, kw, kb, dtype=dtype), tokens, 1, channels)
    value = g.split_heads(network, g.add_linear(network, h, vw, vb, dtype=dtype), tokens, 1, channels)
    attended = g.merge_heads(
        network, g.add_attention(network, query, key, value, 1.0 / math.sqrt(channels)),
        tokens, channels)
    attended = g.add_linear(network, attended, ow, ob, dtype=dtype)
    return g.add_sum(network, g.tokens_to_spatial(network, attended, height, width), residual)


def build_decoder(network, latents, weights, cfg, dtype):
    """Latents in, pixels out. Returns the decoded tensor."""
    groups = cfg["norm_num_groups"]
    eps = 1e-6
    channels = list(cfg["block_out_channels"])
    layers = cfg["layers_per_block"]

    h = latents
    if "post_quant_conv.weight" in weights:
        h = g.add_conv2d(network, h, weights["post_quant_conv.weight"],
                         weights.get("post_quant_conv.bias"), dtype=dtype)
    h = g.add_conv2d(network, h, weights["decoder.conv_in.weight"],
                     weights["decoder.conv_in.bias"], padding=(1, 1), dtype=dtype)

    h = _resnet(network, h, weights, "decoder.mid_block.resnets.0", groups, eps, dtype)
    h = _mid_attention(network, h, weights, "decoder.mid_block.attentions.0", groups, eps, dtype)
    h = _resnet(network, h, weights, "decoder.mid_block.resnets.1", groups, eps, dtype)

    # The decoder walks its blocks from the deepest channel count outward, so the
    # up_blocks are indexed against reversed(block_out_channels).
    for block in range(len(channels)):
        for layer in range(layers + 1):
            h = _resnet(network, h, weights,
                        f"decoder.up_blocks.{block}.resnets.{layer}", groups, eps, dtype)
        if f"decoder.up_blocks.{block}.upsamplers.0.conv.weight" in weights:
            shape = tuple(int(v) for v in h.shape)
            h = g.add_resize_nearest(network, h, (shape[2] * 2, shape[3] * 2))
            h = g.add_conv2d(
                network, h, weights[f"decoder.up_blocks.{block}.upsamplers.0.conv.weight"],
                weights[f"decoder.up_blocks.{block}.upsamplers.0.conv.bias"],
                padding=(1, 1), dtype=dtype)

    h = g.add_group_norm(network, h, weights["decoder.conv_norm_out.weight"],
                         weights["decoder.conv_norm_out.bias"], groups, eps, dtype=dtype)
    h = g.add_silu(network, h)
    return g.add_conv2d(network, h, weights["decoder.conv_out.weight"],
                        weights["decoder.conv_out.bias"], padding=(1, 1), dtype=dtype)
