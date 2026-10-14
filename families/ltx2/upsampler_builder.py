# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LTX-2.5 latent spatial upsampler (``LTX2LatentUpsamplerModel``) as a TensorRT plan.

The two-stage pipeline denoises at half resolution, doubles the latent grid with this model and
refines at full resolution (diffusers ``LTX2LatentUpsamplePipeline`` with ``latents_normalized=False``
on the stage 1 ``output_type="latent"`` latents, then stage 2's ``prepare_latents``).

Engine I/O:
    Inputs:
        latents     [1, F*H*W, 128]     fp32  packed, normalized stage 1 video latents (DiT layout)
    Outputs:
        upsampled   [1, F*2H*2W, 128]   fp32  packed, normalized latents on the 2x spatial grid

The engine denormalizes with the VAE's ``latents_mean`` / ``latents_std`` (``scaling_factor``) in
fp32, unpacks to ``[1, C, F, H, W]`` and runs the upsampler in bf16 (the precision diffusers runs it
in): zero-padded 3x3x3 convolutions, GroupNorm(32) with fp32 statistics, SiLU, residual blocks,
the per-frame 3x3 convolution + 2x pixel shuffle, residual blocks and the final convolution. It
then normalizes again (in fp32; diffusers renormalizes the bf16 tensor in bf16) and packs the
tokens. Temporal upsampling and rational scales other than 2 are not used by LTX-2.5 and are
rejected.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorrt as trt

from .checkpoint import Checkpoint
from .graph import Graph, build_plan, make_logger, new_network

GROUP_NORM_EPS = 1e-5


def _conv(g: Graph, x, weight: np.ndarray, bias: np.ndarray):
    """Zero-padded ``nn.Conv3d`` (or a per-frame ``nn.Conv2d`` given a ``[O, I, 1, k, k]`` weight)."""
    out_c, _, kt, kh, kw = (int(s) for s in weight.shape)
    layer = g.net.add_convolution_nd(x, out_c, (kt, kh, kw), g.weights(weight, x.dtype), g.weights(bias, x.dtype))
    layer.stride_nd = (1, 1, 1)
    layer.padding_nd = (kt // 2, kh // 2, kw // 2)
    return layer.get_output(0)


def _group_norm(g: Graph, x, gamma: np.ndarray, beta: np.ndarray, groups: int = 32):
    """``nn.GroupNorm`` with fp32 statistics and affine, rounded once to the input dtype."""
    b, c, f, h, w = (int(s) for s in x.shape)
    xf = g.reshape(g.cast(x, trt.float32), (b, groups, (c // groups) * f * h * w))
    mean = g.reduce(xf, trt.ReduceOperation.AVG, 2)
    centered = g.sub(xf, mean)
    var = g.reduce(g.mul(centered, centered), trt.ReduceOperation.AVG, 2)
    inv = g.unary(g.unary(g.add(var, g.scalar(GROUP_NORM_EPS, trt.float32, 3)), trt.UnaryOperation.SQRT),
                  trt.UnaryOperation.RECIP)
    y = g.reshape(g.mul(centered, inv), (b, c, f, h, w))
    y = g.add(g.mul(y, g.const(gamma.reshape(1, c, 1, 1, 1), trt.float32)),
              g.const(beta.reshape(1, c, 1, 1, 1), trt.float32))
    return g.cast(y, x.dtype)


def _res_block(g: Graph, ck: Checkpoint, p: str, x):
    h = _conv(g, x, ck.get(f"{p}.conv1.weight"), ck.get(f"{p}.conv1.bias"))
    h = g.silu(_group_norm(g, h, ck.get(f"{p}.norm1.weight", np.float32), ck.get(f"{p}.norm1.bias", np.float32)))
    h = _conv(g, h, ck.get(f"{p}.conv2.weight"), ck.get(f"{p}.conv2.bias"))
    h = _group_norm(g, h, ck.get(f"{p}.norm2.weight", np.float32), ck.get(f"{p}.norm2.bias", np.float32))
    return g.silu(g.add(h, x))


def _pixel_shuffle_2d(g: Graph, x):
    """``PixelShuffleND(2)`` per frame: ``[B, C*4, F, H, W]`` -> ``[B, C, F, 2H, 2W]``."""
    b, cc, f, h, w = (int(s) for s in x.shape)
    c = cc // 4
    y = g.reshape(x, (b, c, 2, 2, f, h, w))
    return g.reshape(y, (b, c, f, 2 * h, 2 * w), first=(0, 1, 4, 5, 2, 6, 3))


def _upsampler_weights(ck: Checkpoint, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """The 2x spatial upsampler convolution as a per-frame ``[O, I, 1, 3, 3]`` weight."""
    if cfg.get("use_rational_resampler", True):
        if float(cfg.get("rational_spatial_scale", 2.0)) != 2.0:
            raise NotImplementedError("only the 2x rational resampler (no blur downsample) is implemented")
        prefix = "upsampler.conv"
    else:
        prefix = "upsampler.0"
    weight = ck.get(f"{prefix}.weight")
    return weight.reshape(weight.shape[0], weight.shape[1], 1, *weight.shape[2:]), ck.get(f"{prefix}.bias")


def build_latent_upsampler_engine(upsampler_dir: str | Path, vae_dir: str | Path, *, latent_frames: int,
                                  latent_height: int, latent_width: int, verbose: bool = False) -> bytes:
    ck = Checkpoint(upsampler_dir)
    cfg = ck.config()
    if int(cfg.get("dims", 3)) != 3:
        raise NotImplementedError("the LTX-2.5 latent upsampler uses 3D convolutions (dims=3)")
    if not cfg.get("spatial_upsample", True) or cfg.get("temporal_upsample", False):
        raise NotImplementedError("only the spatial 2x latent upsampler is implemented")
    vae = Checkpoint(vae_dir)
    latent_c = int(cfg.get("in_channels", 128))
    blocks = int(cfg.get("num_blocks_per_stage", 4))
    scaling = float(vae.config().get("scaling_factor", 1.0))
    mean = vae.get("latents_mean", np.float32).reshape(1, 1, latent_c)
    std = vae.get("latents_std", np.float32).reshape(1, 1, latent_c)

    builder, network = new_network(make_logger(verbose))
    g = Graph(network)
    f, h, w = latent_frames, latent_height, latent_width
    z = network.add_input("latents", trt.float32, (1, f * h * w, latent_c))
    x = g.add(g.mul(z, g.const(std / scaling, trt.float32)), g.const(mean, trt.float32))
    x = g.cast(g.reshape(g.transpose(x, (0, 2, 1)), (1, latent_c, f, h, w)), trt.bfloat16)

    x = _conv(g, x, ck.get("initial_conv.weight"), ck.get("initial_conv.bias"))
    x = g.silu(_group_norm(g, x, ck.get("initial_norm.weight", np.float32), ck.get("initial_norm.bias", np.float32)))
    for i in range(blocks):
        x = _res_block(g, ck, f"res_blocks.{i}", x)
    x = _pixel_shuffle_2d(g, _conv(g, x, *_upsampler_weights(ck, cfg)))
    for i in range(blocks):
        x = _res_block(g, ck, f"post_upsample_res_blocks.{i}", x)
    x = _conv(g, x, ck.get("final_conv.weight"), ck.get("final_conv.bias"))

    tokens = f * 2 * h * 2 * w
    y = g.reshape(g.cast(x, trt.float32), (1, latent_c, tokens), second=(0, 2, 1))
    y = g.mul(g.sub(y, g.const(mean, trt.float32)), g.const(scaling / std, trt.float32))
    g.mark_output(y, "upsampled", trt.float32)
    print(f"[ltx2] Building latent upsampler engine (latent {f}x{h}x{w} -> {f}x{2 * h}x{2 * w}) ...",
          file=sys.stderr)
    return build_plan(builder, network, label="latent upsampler")
