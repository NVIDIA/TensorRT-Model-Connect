# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AutoencoderKLTemporalDecoder: the 2D image encoder and the temporal video decoder.

The encoder returns the posterior mode (the mean half of ``quant_conv``); the
pipeline does not sample it and does not scale it. The decoder runs every frame
of the clip in one call, which is the pipeline's default ``decode_chunk_size``:
its temporal layers mix frames only within one call.
"""

from __future__ import annotations

import tensorrt as trt

from . import blocks
from . import graph as g

_EPS = 1e-6
_TEMPORAL_EPS = 1e-5


def build_encoder(network, weights, cfg: dict, height: int, width: int, work) -> None:
    """Mark ``image_latent`` [1, latent, H/8, W/8] for ``image`` [1, 3, H, W] in [-1, 1]."""
    channels = cfg["block_out_channels"]
    layers = cfg["layers_per_block"]
    image = network.add_input("image", trt.float32, (1, cfg["in_channels"], height, width))
    x = g.conv(network, g.cast(network, image, work), weights["encoder.conv_in.weight"],
               weights["encoder.conv_in.bias"], work, padding=(1, 1))
    for level in range(len(channels)):
        for layer in range(layers):
            x = blocks.resnet2d(network, x, weights,
                                f"encoder.down_blocks.{level}.resnets.{layer}", _EPS, work)
        if level < len(channels) - 1:
            x = blocks.downsample_asymmetric(
                network, x, weights, f"encoder.down_blocks.{level}.downsamplers.0.conv", work)
    x = blocks.resnet2d(network, x, weights, "encoder.mid_block.resnets.0", _EPS, work)
    x = blocks.map_attention(network, x, weights, "encoder.mid_block.attentions.0", _EPS, work)
    x = blocks.resnet2d(network, x, weights, "encoder.mid_block.resnets.1", _EPS, work)
    x = g.group_norm(network, x, weights["encoder.conv_norm_out.weight"],
                     weights["encoder.conv_norm_out.bias"], blocks.GROUPS, _EPS, work)
    x = g.conv(network, g.silu(network, x), weights["encoder.conv_out.weight"],
               weights["encoder.conv_out.bias"], work, padding=(1, 1))
    moments = g.conv(network, x, weights["quant_conv.weight"], weights["quant_conv.bias"], work)
    mean = g.slice_axis(network, moments, 1, 0, cfg["latent_channels"])
    mean = g.cast(network, mean, trt.float32)
    mean.name = "image_latent"
    network.mark_output(mean)


def _decoder_resnet(network, x, weights, prefix: str, frames: int, work):
    return blocks.spatio_temporal_resnet(
        network, x, weights, prefix, batch=1, frames=frames, eps=_EPS,
        temporal_eps=_TEMPORAL_EPS, spatial_is_sigmoid=False, work=work)


def build_decoder(network, weights, cfg: dict, frames: int, latent_height: int,
                  latent_width: int, work) -> None:
    """Mark ``frames`` [F, 3, H, W] for ``latents`` [F, latent, h, w] already divided by the scale."""
    channels = cfg["block_out_channels"]
    layers = cfg["layers_per_block"]
    latents = network.add_input(
        "latents", trt.float32, (frames, cfg["latent_channels"], latent_height, latent_width))
    x = g.conv(network, g.cast(network, latents, work), weights["decoder.conv_in.weight"],
               weights["decoder.conv_in.bias"], work, padding=(1, 1))
    # MidBlockTemporalDecoder: resnets[0], then zip(resnets[1:], attentions) with one attention.
    x = _decoder_resnet(network, x, weights, "decoder.mid_block.resnets.0", frames, work)
    if layers > 1:
        x = blocks.map_attention(network, x, weights, "decoder.mid_block.attentions.0", _EPS, work)
        x = _decoder_resnet(network, x, weights, "decoder.mid_block.resnets.1", frames, work)
    for level in range(len(channels)):
        for layer in range(layers + 1):
            x = _decoder_resnet(network, x, weights, f"decoder.up_blocks.{level}.resnets.{layer}",
                                frames, work)
        if level < len(channels) - 1:
            x = g.conv(network, blocks.upsample_nearest(network, x),
                       weights[f"decoder.up_blocks.{level}.upsamplers.0.conv.weight"],
                       weights[f"decoder.up_blocks.{level}.upsamplers.0.conv.bias"], work,
                       padding=(1, 1))
    x = g.group_norm(network, x, weights["decoder.conv_norm_out.weight"],
                     weights["decoder.conv_norm_out.bias"], blocks.GROUPS, _EPS, work)
    x = g.conv(network, g.silu(network, x), weights["decoder.conv_out.weight"],
               weights["decoder.conv_out.bias"], work, padding=(1, 1))
    x = blocks.to_clip_layout(network, x, 1, frames)
    x = g.conv(network, x, weights["decoder.time_conv_out.weight"],
               weights["decoder.time_conv_out.bias"], work, padding=(1, 0, 0))
    x = g.cast(network, blocks.to_frame_layout(network, x), trt.float32)
    x.name = "frames"
    network.mark_output(x)
