# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LTX-2.5 video VAE decoder (``AutoencoderKLLTX2Video.decoder``) as a TensorRT plan.

Engine I/O:
    Inputs:
        latents   [1, S, 128]  fp32  packed, normalized video latents (DiT layout, S = F*H*W tokens)
    Outputs:
        frames    [T, H*32, W*32, 3]  fp16  RGB in [0, 1] (``(x + 1) / 2`` clamped, the pipeline's
                                       ``postprocess_video``). Tile plans (``clamp_output=False``)
                                       leave ``(x + 1) / 2`` unclamped: the runtime clamps after
                                       blending the tiles (``vae_tiling.py``).

The engine denormalizes with the VAE's ``latents_mean`` / ``latents_std`` (``scaling_factor``),
unpacks the tokens to ``[1, C, F, H, W]`` and runs the non-causal ``LTX2VideoDecoder3d``:
temporal replicate padding + spatial zero padding for every 3x3x3 convolution, per-pixel RMS
norms (eps 1e-8) and SiLU in fp32 islands, bf16 convolutions, per-block upsamplers with their own
stride (spatiotemporal / temporal / spatial) and depth-to-space shuffles, then the 4x4 spatial
unpatchify. ``timestep_conditioning`` and decoder noise injection are not used by LTX-2.5 and
are rejected.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tensorrt as trt

from .checkpoint import Checkpoint
from .graph import Graph, build_plan, make_logger, new_network

_UPSAMPLE_STRIDES = {"spatiotemporal": (2, 2, 2), "temporal": (2, 1, 1), "spatial": (1, 2, 2)}


def _conv3d(g: Graph, x, weight: np.ndarray, bias: np.ndarray | None, *, causal: bool = False):
    """``LTX2VideoCausalConv3d`` (non-causal mode) or a plain 1x1x1 ``nn.Conv3d``."""
    b, c, t, h, w = (int(s) for s in x.shape)
    out_c, _, kt, kh, kw = (int(s) for s in weight.shape)
    if kt > 1:
        pad = kt - 1 if causal else (kt - 1) // 2
        first = g.slice(x, (0, 0, 0, 0, 0), (b, c, 1, h, w))
        parts = [first] * pad + [x]
        if not causal:
            last = g.slice(x, (0, 0, t - 1, 0, 0), (b, c, 1, h, w))
            parts += [last] * pad
        x = g.concat(parts, axis=2)
    layer = g.net.add_convolution_nd(x, out_c, (kt, kh, kw), g.weights(weight, x.dtype),
                                     g.weights(bias, x.dtype) if bias is not None else trt.Weights())
    layer.stride_nd = (1, 1, 1)
    layer.padding_nd = (0, kh // 2, kw // 2)
    return layer.get_output(0)


def _norm_silu(g: Graph, x, eps: float = 1e-8):
    """``PerChannelRMSNorm`` (``x / sqrt(mean_c(x^2) + eps)``) then SiLU, one fp32 island."""
    out = x.dtype
    xf = g.cast(x, trt.float32)
    ms = g.reduce(g.mul(xf, xf), trt.ReduceOperation.AVG, 1)
    inv = g.unary(g.unary(g.add(ms, g.scalar(eps, trt.float32, 5)), trt.UnaryOperation.SQRT),
                  trt.UnaryOperation.RECIP)
    return g.cast(g.silu(g.mul(xf, inv)), out)


def _channel_layer_norm(g: Graph, x, gamma: np.ndarray, beta: np.ndarray, eps: float):
    out = x.dtype
    c = int(x.shape[1])
    xf = g.cast(x, trt.float32)
    mean = g.reduce(xf, trt.ReduceOperation.AVG, 1)
    cen = g.sub(xf, mean)
    var = g.reduce(g.mul(cen, cen), trt.ReduceOperation.AVG, 1)
    inv = g.unary(g.unary(g.add(var, g.scalar(eps, trt.float32, 5)), trt.UnaryOperation.SQRT),
                  trt.UnaryOperation.RECIP)
    y = g.mul(cen, inv)
    y = g.add(g.mul(y, g.const(gamma.reshape(1, c, 1, 1, 1), trt.float32)),
              g.const(beta.reshape(1, c, 1, 1, 1), trt.float32))
    return g.cast(y, out)


def _resnet(g: Graph, ck: Checkpoint, p: str, x, eps: float):
    h = _norm_silu(g, x)
    h = _conv3d(g, h, ck.get(f"{p}.conv1.conv.weight"), ck.get(f"{p}.conv1.conv.bias"))
    h = _norm_silu(g, h)
    h = _conv3d(g, h, ck.get(f"{p}.conv2.conv.weight"), ck.get(f"{p}.conv2.conv.bias"))
    shortcut = x
    if ck.has(f"{p}.norm3.weight"):
        shortcut = _channel_layer_norm(g, shortcut, ck.get(f"{p}.norm3.weight", np.float32),
                                       ck.get(f"{p}.norm3.bias", np.float32), eps)
    for key in (f"{p}.conv_shortcut", f"{p}.conv_shortcut.conv"):
        if ck.has(f"{key}.weight"):
            shortcut = _conv3d(g, shortcut, ck.get(f"{key}.weight"), ck.maybe(f"{key}.bias"))
            break
    return g.add(h, shortcut)


def _depth_to_space(g: Graph, x, stride: tuple[int, int, int]):
    """``LTX2VideoUpsampler3d`` shuffle: ``[B, C*s0*s1*s2, F, H, W]`` -> ``[B, C, F*s0-(s0-1), H*s1, W*s2]``."""
    b, cc, f, h, w = (int(s) for s in x.shape)
    s0, s1, s2 = stride
    c = cc // (s0 * s1 * s2)
    y = g.reshape(x, (b, c, s0, s1, s2, f, h, w))
    y = g.reshape(y, (b, c, f * s0, h * s1, w * s2), first=(0, 1, 5, 2, 6, 3, 7, 4))
    if s0 > 1:
        y = g.slice(y, (0, 0, s0 - 1, 0, 0), (b, c, f * s0 - (s0 - 1), h * s1, w * s2))
    return y


def _count(ck: Checkpoint, fmt: str) -> int:
    n = 0
    while ck.has(fmt.format(n)):
        n += 1
    return n


def build_vae_decoder_engine(vae_dir: str | Path, *, latent_frames: int, latent_height: int, latent_width: int,
                             verbose: bool = False, precision: str = "bf16", clamp_output: bool = True) -> bytes:
    ck = Checkpoint(vae_dir)
    cfg = ck.config()
    if cfg.get("timestep_conditioning"):
        raise NotImplementedError("timestep-conditioned LTX-2 VAE decoders are not supported")
    if any(bool(v) for v in (cfg.get("decoder_inject_noise") or ())):
        raise NotImplementedError("LTX-2 VAE decoder noise injection is not supported")
    if cfg.get("decoder_causal", False):
        raise NotImplementedError("the LTX-2.5 decoder is non-causal; causal decoding is not implemented")
    if cfg.get("decoder_spatial_padding_mode", "zeros") != "zeros":
        raise NotImplementedError("only zero spatial padding is implemented")
    if any(bool(v) for v in (cfg.get("upsample_residual") or ())):
        raise NotImplementedError("residual upsamplers are not used by LTX-2.5 and are not implemented")
    patch, patch_t = int(cfg.get("patch_size", 4)), int(cfg.get("patch_size_t", 1))
    if patch_t != 1:
        raise NotImplementedError("temporal patching is not implemented")
    eps = float(cfg.get("resnet_norm_eps", 1e-6))
    channels = list(reversed(cfg["decoder_block_out_channels"]))
    factors = list(reversed(cfg["upsample_factor"]))
    scaling = list(reversed(cfg["decoder_spatio_temporal_scaling"]))
    up_types = list(cfg["upsample_type"])
    latent_c = int(cfg.get("latent_channels", 128))
    dt = {"bf16": trt.bfloat16, "fp16": trt.float16, "fp32": trt.float32}[precision]

    builder, network = new_network(make_logger(verbose))
    g = Graph(network)
    f, h, w = latent_frames, latent_height, latent_width
    s = f * h * w
    z = network.add_input("latents", trt.float32, (1, s, latent_c))
    mean = ck.get("latents_mean", np.float32).reshape(1, 1, latent_c)
    std = ck.get("latents_std", np.float32).reshape(1, 1, latent_c)
    sf = float(cfg.get("scaling_factor", 1.0))
    x = g.add(g.mul(z, g.const(std / sf, trt.float32)), g.const(mean, trt.float32))
    x = g.reshape(g.transpose(x, (0, 2, 1)), (1, latent_c, f, h, w))
    x = g.cast(x, dt)

    x = _conv3d(g, x, ck.get("decoder.conv_in.conv.weight"), ck.get("decoder.conv_in.conv.bias"))
    for i in range(_count(ck, "decoder.mid_block.resnets.{}.conv1.conv.weight")):
        x = _resnet(g, ck, f"decoder.mid_block.resnets.{i}", x, eps)
    for bi in range(len(channels)):
        p = f"decoder.up_blocks.{bi}"
        if ck.has(f"{p}.conv_in.conv1.conv.weight"):
            x = _resnet(g, ck, f"{p}.conv_in", x, eps)
        if scaling[bi]:
            stride = _UPSAMPLE_STRIDES[up_types[bi]]
            x = _conv3d(g, x, ck.get(f"{p}.upsamplers.0.conv.conv.weight"),
                        ck.get(f"{p}.upsamplers.0.conv.conv.bias"))
            x = _depth_to_space(g, x, stride)
        for ri in range(_count(ck, p + ".resnets.{}.conv1.conv.weight")):
            x = _resnet(g, ck, f"{p}.resnets.{ri}", x, eps)
        expected = channels[bi] // factors[bi]
        if int(x.shape[1]) != expected:
            raise ValueError(f"VAE up block {bi}: {int(x.shape[1])} channels, config expects {expected}")
    x = _norm_silu(g, x)
    x = _conv3d(g, x, ck.get("decoder.conv_out.conv.weight"), ck.get("decoder.conv_out.conv.bias"))
    b, cc, t, hh, ww = (int(v) for v in x.shape)
    c = cc // (patch * patch)
    # reshape(B, C, p_t, p, p, F, H, W).permute(0, 1, 5, 2, 6, 4, 7, 3) -> [B, C, F, H*p, W*p]
    x = g.reshape(x, (b, c, 1, patch, patch, t, hh, ww))
    x = g.reshape(x, (c, t, hh * patch, ww * patch), first=(0, 1, 5, 2, 6, 4, 7, 3))
    x = g.cast(x, trt.float32)
    x = g.mul(g.add(x, g.scalar(1.0, trt.float32, 4)), g.scalar(0.5, trt.float32, 4))
    if clamp_output:
        x = g.maximum(g.minimum(x, g.scalar(1.0, trt.float32, 4)), g.scalar(0.0, trt.float32, 4))
    frames = g.transpose(x, (1, 2, 3, 0))  # [T, H, W, 3]
    g.mark_output(frames, "frames", trt.float16)
    print(f"[ltx2] Building video VAE decoder engine (latent {f}x{h}x{w} -> {t}x{hh * patch}x{ww * patch}, "
          f"{precision}{'' if clamp_output else ', tile'}) ...", file=sys.stderr)
    return build_plan(builder, network, label="video VAE decoder")


def vae_config(vae_dir: str | Path) -> dict:
    return json.loads((Path(vae_dir) / "config.json").read_text(encoding="utf-8"))
