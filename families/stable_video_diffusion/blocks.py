# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Residual, sampling and attention blocks shared by this family's UNet and VAE.

Spatial tensors are [batch * frames, C, H, W] with the frame index inner, as
diffusers flattens them. The temporal resnet runs on the [batch, C, frames, H, W]
view, so its GroupNorm statistics span all frames of a sample.
"""

from __future__ import annotations

import tensorrt as trt

from . import graph as g

GROUPS = 32


def resnet2d(network, x, weights, prefix: str, eps: float, work, temb_silu=None):
    """diffusers ResnetBlock2D with the default time-embedding addition."""
    h = g.group_norm(network, x, weights[f"{prefix}.norm1.weight"], weights[f"{prefix}.norm1.bias"],
                     GROUPS, eps, work)
    h = g.conv(network, g.silu(network, h), weights[f"{prefix}.conv1.weight"],
               weights[f"{prefix}.conv1.bias"], work, padding=(1, 1))
    if temb_silu is not None:
        projected = g.linear(network, temb_silu, weights[f"{prefix}.time_emb_proj.weight"],
                             weights[f"{prefix}.time_emb_proj.bias"], work)
        channels = g.shape_of(projected)[-1]
        h = g.add(network, h, g.reshape(network, projected, (1, channels, 1, 1)))
    h = g.group_norm(network, h, weights[f"{prefix}.norm2.weight"], weights[f"{prefix}.norm2.bias"],
                     GROUPS, eps, work)
    h = g.conv(network, g.silu(network, h), weights[f"{prefix}.conv2.weight"],
               weights[f"{prefix}.conv2.bias"], work, padding=(1, 1))
    shortcut = x
    if f"{prefix}.conv_shortcut.weight" in weights:
        shortcut = g.conv(network, x, weights[f"{prefix}.conv_shortcut.weight"],
                          weights[f"{prefix}.conv_shortcut.bias"], work)
    return g.add(network, shortcut, h)


def _temporal_resnet(network, x, weights, prefix: str, eps: float, work, temb_silu=None):
    """diffusers TemporalResnetBlock on [B, C, F, H, W]: (3, 1, 1) convolutions over frames."""
    h = g.group_norm(network, x, weights[f"{prefix}.norm1.weight"], weights[f"{prefix}.norm1.bias"],
                     GROUPS, eps, work)
    h = g.conv(network, g.silu(network, h), weights[f"{prefix}.conv1.weight"],
               weights[f"{prefix}.conv1.bias"], work, padding=(1, 0, 0))
    if temb_silu is not None:
        projected = g.linear(network, temb_silu, weights[f"{prefix}.time_emb_proj.weight"],
                             weights[f"{prefix}.time_emb_proj.bias"], work)
        channels = g.shape_of(projected)[-1]
        h = g.add(network, h, g.reshape(network, projected, (1, channels, 1, 1, 1)))
    h = g.group_norm(network, h, weights[f"{prefix}.norm2.weight"], weights[f"{prefix}.norm2.bias"],
                     GROUPS, eps, work)
    h = g.conv(network, g.silu(network, h), weights[f"{prefix}.conv2.weight"],
               weights[f"{prefix}.conv2.bias"], work, padding=(1, 0, 0))
    if f"{prefix}.conv_shortcut.weight" in weights:
        raise ValueError(f"{prefix} has a temporal shortcut, which this family does not build")
    return g.add(network, x, h)


def to_clip_layout(network, x, batch: int, frames: int):
    """[B * F, C, H, W] -> [B, C, F, H, W]."""
    _, channels, height, width = g.shape_of(x)
    layer = network.add_shuffle(x)
    layer.reshape_dims = (batch, frames, channels, height, width)
    layer.second_transpose = (0, 2, 1, 3, 4)
    return layer.get_output(0)


def to_frame_layout(network, x):
    """[B, C, F, H, W] -> [B * F, C, H, W]."""
    batch, channels, frames, height, width = g.shape_of(x)
    layer = network.add_shuffle(x)
    layer.first_transpose = (0, 2, 1, 3, 4)
    layer.reshape_dims = (batch * frames, channels, height, width)
    return layer.get_output(0)


def spatio_temporal_resnet(network, x, weights, prefix: str, *, batch: int, frames: int,
                           eps: float, temporal_eps: float, spatial_is_sigmoid: bool, work,
                           temb_silu=None):
    """diffusers SpatioTemporalResBlock: spatial resnet, temporal resnet, AlphaBlender."""
    spatial = resnet2d(network, x, weights, f"{prefix}.spatial_res_block", eps, work, temb_silu)
    spatial = to_clip_layout(network, spatial, batch, frames)
    temporal = _temporal_resnet(network, spatial, weights, f"{prefix}.temporal_res_block",
                                temporal_eps, work, temb_silu)
    mixing = g.blend_constants(weights[f"{prefix}.time_mixer.mix_factor"], work,
                               spatial_is_sigmoid=spatial_is_sigmoid)
    return to_frame_layout(network, g.blend(network, spatial, temporal, mixing, work))


def spatial_tokens(network, x):
    """[N, C, H, W] -> [N, H * W, C]."""
    batch, channels, height, width = g.shape_of(x)
    layer = network.add_shuffle(x)
    layer.first_transpose = (0, 2, 3, 1)
    layer.reshape_dims = (batch, height * width, channels)
    return layer.get_output(0)


def spatial_map(network, tokens, height: int, width: int):
    """[N, H * W, C] -> [N, C, H, W]."""
    batch, _, channels = g.shape_of(tokens)
    layer = network.add_shuffle(tokens)
    layer.reshape_dims = (batch, height, width, channels)
    layer.second_transpose = (0, 3, 1, 2)
    return layer.get_output(0)


def map_attention(network, x, weights, prefix: str, eps: float, work):
    """diffusers Attention(norm_num_groups=32, bias=True, residual_connection=True), one head."""
    _, _, height, width = g.shape_of(x)
    normed = g.group_norm(network, x, weights[f"{prefix}.group_norm.weight"],
                          weights[f"{prefix}.group_norm.bias"], GROUPS, eps, work)
    attended = g.self_attention(network, spatial_tokens(network, normed), weights, prefix, 1, work,
                                bias=True)
    return g.add(network, spatial_map(network, attended, height, width), x)


def upsample_nearest(network, x):
    """F.interpolate(scale_factor=2, mode="nearest") on [N, C, H, W]."""
    batch, channels, height, width = g.shape_of(x)
    layer = network.add_resize(x)
    layer.resize_mode = trt.InterpolationMode.NEAREST
    layer.coordinate_transformation = trt.ResizeCoordinateTransformation.ASYMMETRIC
    layer.nearest_rounding = trt.ResizeRoundMode.FLOOR
    layer.shape = (batch, channels, 2 * height, 2 * width)
    return layer.get_output(0)


def downsample_asymmetric(network, x, weights, prefix: str, work):
    """Downsample2D(padding=0): zero-pad right and bottom by one, then a stride-2 3x3 conv."""
    return g.conv(network, x, weights[f"{prefix}.weight"], weights[f"{prefix}.bias"], work,
                  stride=(2, 2), padding=(0, 0), post_padding=(1, 1))
