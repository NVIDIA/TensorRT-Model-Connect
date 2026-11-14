# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""UNetSpatioTemporalConditionModel for one classifier-free-guidance pair.

The engine denoises the unconditional and the conditional sample together
(batch 2) for every frame. Its conditioning is the CLIP image embedding: one key
per sample. Softmax over a single key is exactly one, so each cross-attention
returns its projected value for every query; the builder emits that value once
per sample and adds it to every token instead of building the query and key
projections whose result cannot change the output.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from . import blocks
from . import graph as g

CFG_BATCH = 2
_CROSS_ATTENTION_DOWN = "CrossAttnDownBlockSpatioTemporal"
_PLAIN_DOWN = "DownBlockSpatioTemporal"
_CROSS_ATTENTION_UP = "CrossAttnUpBlockSpatioTemporal"
_PLAIN_UP = "UpBlockSpatioTemporal"
# Epsilons the diffusers blocks hard-code; resnet_eps passed by the UNet only
# reaches the 3D-UNet block types, not these.
_DOWN_EPS = {_CROSS_ATTENTION_DOWN: 1e-6, _PLAIN_DOWN: 1e-5}
_UP_EPS = 1e-6
_MID_EPS = 1e-5
_OUT_EPS = 1e-5
_TRANSFORMER_NORM_EPS = 1e-6
_LAYER_NORM_EPS = 1e-5


def _cross_attention_value(network, context, weights, prefix: str, work):
    """to_out(to_v(context)) for the [B, 1, cross_dim] context: the attention output."""
    value = g.linear(network, context, weights[f"{prefix}.to_v.weight"], None, work)
    return g.linear(network, value, weights[f"{prefix}.to_out.0.weight"],
                    weights[f"{prefix}.to_out.0.bias"], work)


def _add_per_sample(network, tokens, per_sample, batch: int):
    """[batch * k, T, C] tokens + [batch, 1, C] values, sample-major."""
    rows, length, channels = g.shape_of(tokens)
    grouped = g.reshape(network, tokens, (batch, (rows // batch) * length, channels))
    return g.reshape(network, g.add(network, grouped, per_sample), (rows, length, channels))


def _basic_block(network, tokens, weights, prefix: str, heads: int, cross, batch: int, work):
    """diffusers BasicTransformerBlock (layer_norm, geglu) on [B * F, H * W, C]."""
    normed = g.layer_norm(network, tokens, weights[f"{prefix}.norm1.weight"],
                          weights[f"{prefix}.norm1.bias"], _LAYER_NORM_EPS, work)
    tokens = g.add(network, g.self_attention(network, normed, weights, f"{prefix}.attn1", heads,
                                             work, bias=False), tokens)
    tokens = _add_per_sample(network, tokens, cross, batch)
    normed = g.layer_norm(network, tokens, weights[f"{prefix}.norm3.weight"],
                          weights[f"{prefix}.norm3.bias"], _LAYER_NORM_EPS, work)
    return g.add(network, g.geglu_feed_forward(network, normed, weights, f"{prefix}.ff", work),
                 tokens)


def _temporal_block(network, tokens, weights, prefix: str, heads: int, cross, batch: int,
                    frames: int, work):
    """diffusers TemporalBasicTransformerBlock: attention across frames per position."""
    rows, length, channels = g.shape_of(tokens)
    layer = network.add_shuffle(tokens)
    layer.reshape_dims = (batch, frames, length, channels)
    layer.second_transpose = (0, 2, 1, 3)
    per_position = g.reshape(network, layer.get_output(0), (batch * length, frames, channels))

    normed = g.layer_norm(network, per_position, weights[f"{prefix}.norm_in.weight"],
                          weights[f"{prefix}.norm_in.bias"], _LAYER_NORM_EPS, work)
    x = g.add(network, g.geglu_feed_forward(network, normed, weights, f"{prefix}.ff_in", work),
              per_position)
    normed = g.layer_norm(network, x, weights[f"{prefix}.norm1.weight"],
                          weights[f"{prefix}.norm1.bias"], _LAYER_NORM_EPS, work)
    x = g.add(network, g.self_attention(network, normed, weights, f"{prefix}.attn1", heads, work,
                                        bias=False), x)
    x = _add_per_sample(network, x, cross, batch)
    normed = g.layer_norm(network, x, weights[f"{prefix}.norm3.weight"],
                          weights[f"{prefix}.norm3.bias"], _LAYER_NORM_EPS, work)
    x = g.add(network, g.geglu_feed_forward(network, normed, weights, f"{prefix}.ff", work), x)

    layer = network.add_shuffle(x)
    layer.reshape_dims = (batch, length, frames, channels)
    layer.second_transpose = (0, 2, 1, 3)
    return g.reshape(network, layer.get_output(0), (rows, length, channels))


def _frame_embedding(network, weights, prefix: str, frames: int, channels: int, work):
    """TransformerSpatioTemporalModel.time_pos_embed of the frame index, [1, F, 1, C]."""
    table = g.constant(network, g.sinusoid_table(np.arange(frames), channels), g.numpy_dtype(work))
    hidden = g.linear(network, table, weights[f"{prefix}.time_pos_embed.linear_1.weight"],
                      weights[f"{prefix}.time_pos_embed.linear_1.bias"], work)
    embedding = g.linear(network, g.silu(network, hidden),
                         weights[f"{prefix}.time_pos_embed.linear_2.weight"],
                         weights[f"{prefix}.time_pos_embed.linear_2.bias"], work)
    return g.reshape(network, embedding, (1, frames, 1, channels))


def _transformer(network, x, weights, prefix: str, heads: int, context, batch: int, frames: int,
                 work):
    """diffusers TransformerSpatioTemporalModel with one spatial and one temporal block."""
    if f"{prefix}.transformer_blocks.1.norm1.weight" in weights:
        raise ValueError("stable_video_diffusion builds one transformer layer per block")
    rows, channels, height, width = g.shape_of(x)
    length = height * width
    normed = g.group_norm(network, x, weights[f"{prefix}.norm.weight"],
                          weights[f"{prefix}.norm.bias"], blocks.GROUPS, _TRANSFORMER_NORM_EPS,
                          work)
    tokens = g.linear(network, blocks.spatial_tokens(network, normed),
                      weights[f"{prefix}.proj_in.weight"], weights[f"{prefix}.proj_in.bias"], work)

    spatial_prefix = f"{prefix}.transformer_blocks.0"
    temporal_prefix = f"{prefix}.temporal_transformer_blocks.0"
    spatial_cross = _cross_attention_value(network, context, weights, f"{spatial_prefix}.attn2", work)
    temporal_cross = _cross_attention_value(network, context, weights, f"{temporal_prefix}.attn2",
                                            work)
    tokens = _basic_block(network, tokens, weights, spatial_prefix, heads, spatial_cross, batch,
                          work)

    positioned = g.add(network, g.reshape(network, tokens, (batch, frames, length, channels)),
                       _frame_embedding(network, weights, prefix, frames, channels, work))
    mixed = _temporal_block(network, g.reshape(network, positioned, (rows, length, channels)),
                            weights, temporal_prefix, heads, temporal_cross, batch, frames, work)
    mixing = g.blend_constants(weights[f"{prefix}.time_mixer.mix_factor"], work,
                               spatial_is_sigmoid=True)
    tokens = g.blend(network, tokens, mixed, mixing, work)
    tokens = g.linear(network, tokens, weights[f"{prefix}.proj_out.weight"],
                      weights[f"{prefix}.proj_out.bias"], work)
    return g.add(network, blocks.spatial_map(network, tokens, height, width), x)


def _st_resnet(network, x, weights, prefix: str, eps: float, batch: int, frames: int, work,
               temb_silu):
    return blocks.spatio_temporal_resnet(
        network, x, weights, prefix, batch=batch, frames=frames, eps=eps, temporal_eps=eps,
        spatial_is_sigmoid=True, work=work, temb_silu=temb_silu)


def _time_embedding(network, timestep, time_ids, weights, cfg: dict, work):
    """time_embedding(time_proj(t)) + add_embedding(add_time_proj(ids)), [1, 4 * C0]."""
    first = cfg["block_out_channels"][0]
    features = g.sinusoid_embedding(network, g.reshape(network, timestep, (1, 1)), first)
    hidden = g.linear(network, g.cast(network, features, work),
                      weights["time_embedding.linear_1.weight"],
                      weights["time_embedding.linear_1.bias"], work)
    temb = g.linear(network, g.silu(network, hidden), weights["time_embedding.linear_2.weight"],
                    weights["time_embedding.linear_2.bias"], work)

    count = cfg["added_time_ids"]
    width = cfg["addition_time_embed_dim"]
    id_features = g.sinusoid_embedding(network, g.reshape(network, time_ids, (count, 1)), width)
    id_features = g.reshape(network, id_features, (1, count * width))
    hidden = g.linear(network, g.cast(network, id_features, work),
                      weights["add_embedding.linear_1.weight"],
                      weights["add_embedding.linear_1.bias"], work)
    augment = g.linear(network, g.silu(network, hidden), weights["add_embedding.linear_2.weight"],
                       weights["add_embedding.linear_2.bias"], work)
    return g.add(network, temb, augment)


def build_unet(network, weights, cfg: dict, frames: int, latent_height: int, latent_width: int,
               work) -> None:
    """Mark ``noise_pred`` [2, F, out, h, w] for the CFG pair's ``sample`` [2, F, in, h, w]."""
    batch = CFG_BATCH
    heads = cfg["num_attention_heads"]
    layers = cfg["layers_per_block"]
    down_types = cfg["down_block_types"]
    up_types = cfg["up_block_types"]

    sample = network.add_input(
        "sample", trt.float32, (batch, frames, cfg["in_channels"], latent_height, latent_width))
    timestep = network.add_input("timestep", trt.float32, (1,))
    context_input = network.add_input(
        "encoder_hidden_states", trt.float32, (batch, 1, cfg["cross_attention_dim"]))
    time_ids = network.add_input("added_time_ids", trt.float32, (1, cfg["added_time_ids"]))

    x = g.cast(network, g.reshape(
        network, sample, (batch * frames, cfg["in_channels"], latent_height, latent_width)), work)
    context = g.cast(network, context_input, work)
    temb_silu = g.silu(network, _time_embedding(network, timestep, time_ids, weights, cfg, work))

    x = g.conv(network, x, weights["conv_in.weight"], weights["conv_in.bias"], work, padding=(1, 1))
    skips = [x]
    for level, kind in enumerate(down_types):
        eps = _DOWN_EPS[kind]
        for layer in range(layers):
            x = _st_resnet(network, x, weights, f"down_blocks.{level}.resnets.{layer}", eps, batch,
                           frames, work, temb_silu)
            if kind == _CROSS_ATTENTION_DOWN:
                x = _transformer(network, x, weights, f"down_blocks.{level}.attentions.{layer}",
                                 heads[level], context, batch, frames, work)
            skips.append(x)
        if level < len(down_types) - 1:
            x = g.conv(network, x, weights[f"down_blocks.{level}.downsamplers.0.conv.weight"],
                       weights[f"down_blocks.{level}.downsamplers.0.conv.bias"], work,
                       stride=(2, 2), padding=(1, 1))
            skips.append(x)

    x = _st_resnet(network, x, weights, "mid_block.resnets.0", _MID_EPS, batch, frames, work,
                   temb_silu)
    x = _transformer(network, x, weights, "mid_block.attentions.0", heads[-1], context, batch,
                     frames, work)
    x = _st_resnet(network, x, weights, "mid_block.resnets.1", _MID_EPS, batch, frames, work,
                   temb_silu)

    up_heads = list(reversed(heads))
    pushed = len(skips)
    for level, kind in enumerate(up_types):
        take = layers + 1
        pending = skips[-take:]
        skips = skips[:-take]
        for layer in range(take):
            # Each up resnet consumes the most recent remaining skip.
            x = g.concat(network, [x, pending.pop()], axis=1)
            x = _st_resnet(network, x, weights, f"up_blocks.{level}.resnets.{layer}", _UP_EPS,
                           batch, frames, work, temb_silu)
            if kind == _CROSS_ATTENTION_UP:
                x = _transformer(network, x, weights, f"up_blocks.{level}.attentions.{layer}",
                                 up_heads[level], context, batch, frames, work)
        if level < len(up_types) - 1:
            x = g.conv(network, blocks.upsample_nearest(network, x),
                       weights[f"up_blocks.{level}.upsamplers.0.conv.weight"],
                       weights[f"up_blocks.{level}.upsamplers.0.conv.bias"], work, padding=(1, 1))
    if skips or pushed != len(up_types) * (layers + 1):
        raise ValueError("UNet skip connections did not balance")

    x = g.group_norm(network, x, weights["conv_norm_out.weight"], weights["conv_norm_out.bias"],
                     blocks.GROUPS, _OUT_EPS, work)
    x = g.conv(network, g.silu(network, x), weights["conv_out.weight"], weights["conv_out.bias"],
               work, padding=(1, 1))
    out = g.cast(network, g.reshape(
        network, x, (batch, frames, cfg["out_channels"], latent_height, latent_width)), trt.float32)
    out.name = "noise_pred"
    network.mark_output(out)
