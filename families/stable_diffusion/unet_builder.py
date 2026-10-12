# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the Stable Diffusion UNet denoiser.

This is the repository's first UNet: every other diffusion family builds a DiT.
The shape that matters, and that no DiT has, is the skip stack. Read from
``diffusers`` rather than inferred:

* ``conv_in``'s output is pushed as the **first** skip, before any block runs.
* each down block pushes one skip per resnet, and one more after its downsampler.
* each up block pops exactly ``len(resnets)`` skips, and inside the block pops
  them **from the end**, concatenating on the channel axis before each resnet.

For SD 1.5 that is 1 + 3 + 3 + 3 + 2 = 12 pushed and 4 x 3 = 12 popped. The
builder asserts that balance rather than trusting it.
"""

from __future__ import annotations

import math

import numpy as np
import tensorrt as trt

from . import graph as g


def _timestep_embedding(network, timestep, channels: int, max_period: float = 10000.0,
                        dtype=np.float32):
    """Sinusoidal embedding, cos first then sin, matching diffusers' flip_sin_to_cos."""
    half = channels // 2
    frequencies = np.exp(
        -math.log(max_period) * np.arange(half, dtype=np.float32) / half
    ).reshape(1, half)
    table = g.add_constant(network, (1, half), frequencies, dtype=dtype)
    angles = network.add_elementwise(
        timestep, g._retype(network, table, timestep.dtype), trt.ElementWiseOperation.PROD
    ).get_output(0)
    cos = network.add_unary(angles, trt.UnaryOperation.COS).get_output(0)
    sin = network.add_unary(angles, trt.UnaryOperation.SIN).get_output(0)
    return g.concat(network, [cos, sin], axis=1)


def _resnet(network, x, temb, weights, prefix, groups, eps, dtype):
    """ResnetBlock2D, including the timestep term a VAE resnet does not have."""
    residual = x
    h = g.add_group_norm(network, x, weights[f"{prefix}.norm1.weight"],
                         weights[f"{prefix}.norm1.bias"], groups, eps, dtype=dtype)
    h = g.add_silu(network, h)
    h = g.add_conv2d(network, h, weights[f"{prefix}.conv1.weight"],
                     weights[f"{prefix}.conv1.bias"], padding=(1, 1), dtype=dtype)

    # The time embedding enters here, broadcast over height and width.
    scale = g.add_linear(network, g.add_silu(network, temb),
                         weights[f"{prefix}.time_emb_proj.weight"],
                         weights[f"{prefix}.time_emb_proj.bias"], dtype=dtype)
    channels = int(np.asarray(weights[f"{prefix}.time_emb_proj.weight"]).shape[0])
    reshaped = network.add_shuffle(scale)
    reshaped.reshape_dims = (1, channels, 1, 1)
    h = g.add_sum(network, h, reshaped.get_output(0))

    h = g.add_group_norm(network, h, weights[f"{prefix}.norm2.weight"],
                         weights[f"{prefix}.norm2.bias"], groups, eps, dtype=dtype)
    h = g.add_silu(network, h)
    h = g.add_conv2d(network, h, weights[f"{prefix}.conv2.weight"],
                     weights[f"{prefix}.conv2.bias"], padding=(1, 1), dtype=dtype)

    if f"{prefix}.conv_shortcut.weight" in weights:
        residual = g.add_conv2d(network, residual, weights[f"{prefix}.conv_shortcut.weight"],
                                weights.get(f"{prefix}.conv_shortcut.bias"), dtype=dtype)
    return g.add_sum(network, h, residual)


def _attention(network, x, context, weights, prefix, heads, groups, eps, dtype):
    """Transformer2DModel: GroupNorm, 1x1 in, transformer blocks, 1x1 out, residual."""
    residual = x
    shape = tuple(int(v) for v in x.shape)
    height, width = shape[2], shape[3]

    h = g.add_group_norm(network, x, weights[f"{prefix}.norm.weight"],
                         weights[f"{prefix}.norm.bias"], groups, eps, dtype=dtype)
    h = g.add_conv2d(network, h, weights[f"{prefix}.proj_in.weight"],
                     weights[f"{prefix}.proj_in.bias"], dtype=dtype)
    h = g.spatial_to_tokens(network, h)

    index = 0
    while f"{prefix}.transformer_blocks.{index}.norm1.weight" in weights:
        block = f"{prefix}.transformer_blocks.{index}"
        channels = int(np.asarray(weights[f"{block}.attn1.to_q.weight"]).shape[0])
        head_dim = channels // heads
        tokens = height * width

        normed = g.add_layer_norm(network, h, weights[f"{block}.norm1.weight"],
                                  weights[f"{block}.norm1.bias"], 1e-5, dtype=dtype)
        h = g.add_sum(network, _cross_attention(
            network, normed, normed, weights, f"{block}.attn1", heads, head_dim,
            tokens, tokens, dtype), h)

        normed = g.add_layer_norm(network, h, weights[f"{block}.norm2.weight"],
                                  weights[f"{block}.norm2.bias"], 1e-5, dtype=dtype)
        context_len = int(tuple(context.shape)[1])
        h = g.add_sum(network, _cross_attention(
            network, normed, context, weights, f"{block}.attn2", heads, head_dim,
            tokens, context_len, dtype), h)

        normed = g.add_layer_norm(network, h, weights[f"{block}.norm3.weight"],
                                  weights[f"{block}.norm3.bias"], 1e-5, dtype=dtype)
        inner = g.add_geglu(network, normed, weights[f"{block}.ff.net.0.proj.weight"],
                            weights[f"{block}.ff.net.0.proj.bias"], dtype=dtype)
        h = g.add_sum(network, g.add_linear(
            network, inner, weights[f"{block}.ff.net.2.weight"],
            weights[f"{block}.ff.net.2.bias"], dtype=dtype), h)
        index += 1

    h = g.tokens_to_spatial(network, h, height, width)
    h = g.add_conv2d(network, h, weights[f"{prefix}.proj_out.weight"],
                     weights[f"{prefix}.proj_out.bias"], dtype=dtype)
    return g.add_sum(network, h, residual)


def _cross_attention(network, query_source, kv_source, weights, prefix, heads, head_dim,
                     query_len, kv_len, dtype):
    """attn1 passes the same tensor twice; attn2 keys and values on the text context."""
    query = g.split_heads(network, g.add_linear(
        network, query_source, weights[f"{prefix}.to_q.weight"], None, dtype=dtype),
        query_len, heads, head_dim)
    key = g.split_heads(network, g.add_linear(
        network, kv_source, weights[f"{prefix}.to_k.weight"], None, dtype=dtype),
        kv_len, heads, head_dim)
    value = g.split_heads(network, g.add_linear(
        network, kv_source, weights[f"{prefix}.to_v.weight"], None, dtype=dtype),
        kv_len, heads, head_dim)
    context = g.merge_heads(
        network, g.add_attention(network, query, key, value, 1.0 / math.sqrt(head_dim)),
        query_len, heads * head_dim)
    return g.add_linear(network, context, weights[f"{prefix}.to_out.0.weight"],
                        weights[f"{prefix}.to_out.0.bias"], dtype=dtype)


def _has(weights, prefix: str) -> bool:
    return any(key.startswith(prefix) for key in weights)


def build_unet(network, weights, cfg, dtype, work_trt):
    """Assemble the denoiser and return its output tensor."""
    groups = cfg["norm_num_groups"]
    eps = 1e-5
    heads = cfg["attention_head_dim"]
    channels = cfg["block_out_channels"]
    layers = cfg["layers_per_block"]
    latent = cfg["sample_size"]

    sample = network.add_input("sample", trt.float32, (1, cfg["in_channels"], latent, latent))
    timestep = network.add_input("timestep", trt.float32, (1, 1))
    context = network.add_input(
        "encoder_hidden_states", trt.float32, (1, cfg["context_length"], cfg["cross_attention_dim"]))
    for tensor in (sample, timestep, context):
        pass
    x = sample if sample.dtype == work_trt else network.add_cast(sample, work_trt).get_output(0)
    ctx = context if context.dtype == work_trt else network.add_cast(
        context, work_trt).get_output(0)
    t = timestep if timestep.dtype == work_trt else network.add_cast(
        timestep, work_trt).get_output(0)

    temb = _timestep_embedding(network, t, channels[0], dtype=dtype)
    temb = g.add_linear(network, temb, weights["time_embedding.linear_1.weight"],
                        weights["time_embedding.linear_1.bias"], dtype=dtype)
    temb = g.add_silu(network, temb)
    temb = g.add_linear(network, temb, weights["time_embedding.linear_2.weight"],
                        weights["time_embedding.linear_2.bias"], dtype=dtype)

    h = g.add_conv2d(network, x, weights["conv_in.weight"], weights["conv_in.bias"],
                     padding=(1, 1), dtype=dtype)
    # conv_in's output is the first skip, before any block runs.
    skips = [h]

    for block in range(len(channels)):
        for layer in range(layers):
            h = _resnet(network, h, temb, weights, f"down_blocks.{block}.resnets.{layer}",
                        groups, eps, dtype)
            if _has(weights, f"down_blocks.{block}.attentions.{layer}."):
                h = _attention(network, h, ctx, weights,
                               f"down_blocks.{block}.attentions.{layer}", heads, groups, eps, dtype)
            skips.append(h)
        if _has(weights, f"down_blocks.{block}.downsamplers.0."):
            h = g.add_conv2d(network, h, weights[f"down_blocks.{block}.downsamplers.0.conv.weight"],
                             weights[f"down_blocks.{block}.downsamplers.0.conv.bias"],
                             stride=(2, 2), padding=(1, 1), dtype=dtype)
            skips.append(h)

    h = _resnet(network, h, temb, weights, "mid_block.resnets.0", groups, eps, dtype)
    h = _attention(network, h, ctx, weights, "mid_block.attentions.0", heads, groups, eps, dtype)
    h = _resnet(network, h, temb, weights, "mid_block.resnets.1", groups, eps, dtype)

    pushed = len(skips)
    popped = 0
    for block in range(len(channels)):
        take = layers + 1
        available = skips[-take:]
        skips = skips[:-take]
        popped += take
        for layer in range(take):
            # Inside the block the skips are consumed from the end.
            h = g.concat(network, [h, available.pop()], axis=1)
            h = _resnet(network, h, temb, weights, f"up_blocks.{block}.resnets.{layer}",
                        groups, eps, dtype)
            if _has(weights, f"up_blocks.{block}.attentions.{layer}."):
                h = _attention(network, h, ctx, weights,
                               f"up_blocks.{block}.attentions.{layer}", heads, groups, eps, dtype)
        if _has(weights, f"up_blocks.{block}.upsamplers.0."):
            shape = tuple(int(v) for v in h.shape)
            # diffusers upsamples nearest, not bilinear.
            h = g.add_resize_nearest(network, h, (shape[2] * 2, shape[3] * 2))
            h = g.add_conv2d(network, h, weights[f"up_blocks.{block}.upsamplers.0.conv.weight"],
                             weights[f"up_blocks.{block}.upsamplers.0.conv.bias"],
                             padding=(1, 1), dtype=dtype)
    if skips or popped != pushed:
        raise ValueError(
            f"UNet skip stack did not balance: pushed {pushed}, popped {popped}, "
            f"{len(skips)} left over")

    h = g.add_group_norm(network, h, weights["conv_norm_out.weight"],
                         weights["conv_norm_out.bias"], groups, eps, dtype=dtype)
    h = g.add_silu(network, h)
    h = g.add_conv2d(network, h, weights["conv_out.weight"], weights["conv_out.bias"],
                     padding=(1, 1), dtype=dtype)
    if h.dtype != trt.float32:
        h = network.add_cast(h, trt.float32).get_output(0)
    h.name = "out_sample"
    network.mark_output(h)
    return h
