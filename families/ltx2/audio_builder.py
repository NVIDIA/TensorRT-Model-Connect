# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LTX-2.5 audio decoder: ``AutoencoderKLLTX2Audio.decoder`` + ``LTX2VocoderWithBWE`` as one TensorRT plan.

Engine I/O:
    Inputs:
        audio_latents  [1, Sa, 128]  fp32  packed, normalized audio latents (DiT layout)
    Outputs:
        waveform       [1, 2, N]     fp32  48 kHz stereo in [-1, 1], N = Sa*4-3 mel frames * 160 * 3
        mel            [1, 2, T, 64] fp32  (debug builds only) the audio VAE log-mel output

Everything runs in fp32 (the vocoder's SnakeBeta / log-mel path is precision sensitive and cheap):
- latent denormalization with the audio VAE ``latents_mean`` / ``latents_std``, unpack to ``[1, 8, Sa, 16]``;
- the causal (time axis) Conv2d decoder with pixel norms, nearest x2 upsampling that drops the first
  time row, and the crop to ``Sa*4-3`` frames x 64 mel bins;
- stage-1 vocoder (BigVGAN-style: transposed-conv upsamplers, three parallel dilated ResBlocks per
  stage averaged, anti-aliased SnakeBeta activations, Kaiser up/down filters with replicate padding);
- causal STFT (conv1d with the checkpoint's windowed DFT basis), magnitude, mel matmul, log clamp;
- BWE generator on that mel plus the Hann sinc x3 resampler of the stage-1 waveform, clamp, crop.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import tensorrt as trt

from .checkpoint import Checkpoint
from .graph import Graph, build_plan, make_logger, new_network

F32 = np.float32


# ---------------------------------------------------------------------- 1D / 2D primitives (fp32)


def _pad_replicate_last(g: Graph, x, left: int, right: int):
    """Replicate padding on the last axis of ``[B, C, L]``."""
    b, c, n = (int(s) for s in x.shape)
    parts = []
    if left:
        parts += [g.slice(x, (0, 0, 0), (b, c, 1))] * left
    parts.append(x)
    if right:
        parts += [g.slice(x, (0, 0, n - 1), (b, c, 1))] * right
    return g.concat(parts, axis=2) if len(parts) > 1 else x


def _conv1d(g: Graph, x, weight: np.ndarray, bias: np.ndarray | None, *, stride: int = 1, dilation: int = 1,
            padding: int = 0, groups: int = 1):
    """``F.conv1d`` on ``[B, C, L]`` (as a 1xK Conv2d)."""
    b, c, n = (int(s) for s in x.shape)
    out_c, _, k = (int(s) for s in weight.shape)
    x4 = g.reshape(x, (b, c, 1, n))
    layer = g.net.add_convolution_nd(x4, out_c, (1, k), g.weights(weight.reshape(out_c, -1, 1, k), trt.float32),
                                     g.weights(bias, trt.float32) if bias is not None else trt.Weights())
    layer.stride_nd = (1, stride)
    layer.dilation_nd = (1, dilation)
    layer.padding_nd = (0, padding)
    layer.num_groups = groups
    y = layer.get_output(0)
    return g.reshape(y, (b, out_c, int(y.shape[3])))


def _conv_transpose1d(g: Graph, x, weight: np.ndarray, bias: np.ndarray | None, *, stride: int, padding: int = 0,
                      groups: int = 1):
    """``F.conv_transpose1d`` on ``[B, C, L]``; ``weight`` is ``[C_in, C_out / groups, K]``."""
    b, c, n = (int(s) for s in x.shape)
    cin, cout_g, k = (int(s) for s in weight.shape)
    out_c = cout_g * groups
    x4 = g.reshape(x, (b, c, 1, n))
    layer = g.net.add_deconvolution_nd(x4, out_c, (1, k), g.weights(weight.reshape(cin, cout_g, 1, k), trt.float32),
                                       g.weights(bias, trt.float32) if bias is not None else trt.Weights())
    layer.stride_nd = (1, stride)
    layer.padding_nd = (0, padding)
    layer.num_groups = groups
    y = layer.get_output(0)
    return g.reshape(y, (b, out_c, int(y.shape[3])))


def _snake_beta(g: Graph, x, alpha: np.ndarray, beta: np.ndarray, eps: float = 1e-9):
    """``x + 1 / (exp(beta) + eps) * sin(x * exp(alpha))^2`` (logscale SnakeBeta)."""
    c = int(x.shape[1])
    a = g.const(np.exp(alpha.astype(np.float64)).astype(F32).reshape(1, c, 1), trt.float32)
    inv_b = g.const((1.0 / (np.exp(beta.astype(np.float64)) + eps)).astype(F32).reshape(1, c, 1), trt.float32)
    s = g.unary(g.mul(x, a), trt.UnaryOperation.SIN)
    return g.add(x, g.mul(inv_b, g.mul(s, s)))


def _kaiser_up(g: Graph, x, filt: np.ndarray, ratio: int):
    k = filt.size
    pad = k // ratio - 1
    pad_left = pad * ratio + (k - ratio) // 2
    pad_right = pad * ratio + (k - ratio + 1) // 2
    c = int(x.shape[1])
    xp = _pad_replicate_last(g, x, pad, pad)
    w = np.tile(filt.reshape(1, 1, k), (c, 1, 1)).astype(F32) * F32(ratio)
    y = _conv_transpose1d(g, xp, w, None, stride=ratio, groups=c)
    n = int(y.shape[2])
    return g.slice(y, (0, 0, pad_left), (int(y.shape[0]), c, n - pad_left - pad_right))


def _kaiser_down(g: Graph, x, filt: np.ndarray, ratio: int):
    k = filt.size
    pad_left = k // 2 + (k % 2) - 1
    pad_right = k // 2
    c = int(x.shape[1])
    xp = _pad_replicate_last(g, x, pad_left, pad_right)
    w = np.tile(filt.reshape(1, 1, k), (c, 1, 1)).astype(F32)
    return _conv1d(g, xp, w, None, stride=ratio, groups=c)


def _aa_act(g: Graph, ck: Checkpoint, p: str, x, ratio: int):
    """``AntiAliasAct1d(SnakeBeta)``: Kaiser x2 up, SnakeBeta, Kaiser x2 down."""
    x = _kaiser_up(g, x, ck.get(f"{p}.upsample.filter", F32).reshape(-1), ratio)
    x = _snake_beta(g, x, ck.get(f"{p}.act.alpha", F32), ck.get(f"{p}.act.beta", F32))
    return _kaiser_down(g, x, ck.get(f"{p}.downsample.filter", F32).reshape(-1), ratio)


def _resblock(g: Graph, ck: Checkpoint, p: str, x, kernel: int, dilations, ratio: int):
    for j, d in enumerate(dilations):
        xt = _aa_act(g, ck, f"{p}.acts1.{j}", x, ratio)
        xt = _conv1d(g, xt, ck.get(f"{p}.convs1.{j}.weight", F32), ck.maybe(f"{p}.convs1.{j}.bias", F32),
                     dilation=d, padding=d * (kernel - 1) // 2)
        xt = _aa_act(g, ck, f"{p}.acts2.{j}", xt, ratio)
        xt = _conv1d(g, xt, ck.get(f"{p}.convs2.{j}.weight", F32), ck.maybe(f"{p}.convs2.{j}.bias", F32),
                     padding=(kernel - 1) // 2)
        x = g.add(x, xt)
    return x


def _vocoder_stage(g: Graph, ck: Checkpoint, p: str, mel, cfg: dict, *, prefix_cfg: str = ""):
    """``LTX2Vocoder.forward`` (snakebeta + antialias variant) on ``mel`` ``[B, C, T, M]``."""
    def c(key):
        return cfg[prefix_cfg + key]

    if c("act_fn") not in ("snakebeta", "snake") or not c("antialias"):
        raise NotImplementedError("only the anti-aliased SnakeBeta vocoder is implemented")
    b, ch, t, m = (int(s) for s in mel.shape)
    x = g.reshape(g.transpose(mel, (0, 1, 3, 2)), (b, ch * m, t))
    x = _conv1d(g, x, ck.get(f"{p}.conv_in.weight", F32), ck.maybe(f"{p}.conv_in.bias", F32), padding=3)
    ratio = int(c("antialias_ratio"))
    kernels = c("resnet_kernel_sizes")
    dils = c("resnet_dilations")
    n_res = len(kernels)
    for i, (stride, k) in enumerate(zip(c("upsample_factors"), c("upsample_kernel_sizes"))):
        x = _conv_transpose1d(g, x, ck.get(f"{p}.upsamplers.{i}.weight", F32), ck.maybe(f"{p}.upsamplers.{i}.bias", F32),
                              stride=int(stride), padding=(int(k) - int(stride)) // 2)
        outs = [_resblock(g, ck, f"{p}.resnets.{i * n_res + j}", x, int(kernels[j]), dils[j], ratio)
                for j in range(n_res)]
        acc = outs[0]
        for o in outs[1:]:
            acc = g.add(acc, o)
        x = g.mul(acc, g.scalar(1.0 / n_res, trt.float32, 3))
    x = _aa_act(g, ck, f"{p}.act_out", x, ratio)
    x = _conv1d(g, x, ck.get(f"{p}.conv_out.weight", F32), ck.maybe(f"{p}.conv_out.bias", F32), padding=3)
    final = c("final_act_fn")
    if final == "tanh":
        x = g.net.add_activation(x, trt.ActivationType.TANH).get_output(0)
    elif final == "clamp":
        x = g.maximum(g.minimum(x, g.scalar(1.0, trt.float32, 3)), g.scalar(-1.0, trt.float32, 3))
    return x


def hann_resampler_filter(ratio: int) -> tuple[np.ndarray, int, int, int]:
    """diffusers ``UpSample1d(window_type="hann")``: (filter, pad, pad_left, pad_right)."""
    rolloff = 0.99
    lowpass_filter_width = 6
    width = math.ceil(lowpass_filter_width / rolloff)
    k = 2 * width * ratio + 1
    t = (np.arange(k, dtype=F32) / F32(ratio) - F32(width)) * F32(rolloff)
    tc = np.clip(t, -lowpass_filter_width, lowpass_filter_width)
    window = np.cos(tc * F32(math.pi) / F32(lowpass_filter_width) / F32(2)) ** 2
    filt = (np.sinc(t) * window * F32(rolloff) / F32(ratio)).astype(F32)
    return filt, width, 2 * width * ratio, k - ratio


def _bwe_vocoder(g: Graph, ck: Checkpoint, cfg: dict, mel, *, debug: bool = False):
    x = _vocoder_stage(g, ck, "vocoder", mel, cfg)  # [1, 2, n] at 16 kHz
    if debug:
        g.mark_output(g.cast(x, trt.float32), "stage1", trt.float32)
    b, ch, n = (int(s) for s in x.shape)
    hop = int(cfg["hop_length"])
    if n % hop:
        x = g.concat([x, g.const(np.zeros((b, ch, hop - n % hop), F32), trt.float32)], axis=2)
    n_pad = int(x.shape[2])
    # MelSTFT on every channel: causal left zero pad, windowed DFT conv, magnitude, mel, log clamp.
    win = int(cfg["window_length"])
    left = max(0, win - hop)
    w = g.reshape(x, (b * ch, 1, n_pad))
    w = g.concat([g.const(np.zeros((b * ch, 1, left), F32), trt.float32), w], axis=2)
    basis = ck.get("mel_stft.stft_fn.forward_basis", F32)
    spec = _conv1d(g, w, basis, None, stride=hop)  # [B*C, 2*nf, frames]
    nf = basis.shape[0] // 2
    frames = int(spec.shape[2])
    re = g.slice(spec, (0, 0, 0), (b * ch, nf, frames))
    im = g.slice(spec, (0, nf, 0), (b * ch, nf, frames))
    mag = g.unary(g.add(g.mul(re, re), g.mul(im, im)), trt.UnaryOperation.SQRT)
    mel_basis = ck.get("mel_stft.mel_basis", F32)  # [n_mels, nf]
    mel2 = g.net.add_matrix_multiply(g.const(mel_basis.reshape(1, *mel_basis.shape), trt.float32),
                                     trt.MatrixOperation.NONE, mag, trt.MatrixOperation.NONE).get_output(0)
    log_mel = g.unary(g.maximum(mel2, g.scalar(1e-5, trt.float32, 3)), trt.UnaryOperation.LOG)
    n_mels = int(mel_basis.shape[0])
    mel_bwe = g.transpose(g.reshape(log_mel, (b, ch, n_mels, frames)), (0, 1, 3, 2))  # [B, C, frames, mels]
    residual = _vocoder_stage(g, ck, "bwe_generator", mel_bwe, cfg, prefix_cfg="bwe_")
    ratio = int(cfg["output_sampling_rate"]) // int(cfg["input_sampling_rate"])
    filt, pad, pad_left, pad_right = hann_resampler_filter(ratio)
    xp = _pad_replicate_last(g, x, pad, pad)
    k = filt.size
    skip = _conv_transpose1d(g, xp, np.tile(filt.reshape(1, 1, k), (ch, 1, 1)) * F32(ratio), None, stride=ratio,
                             groups=ch)
    skip = g.slice(skip, (0, 0, pad_left), (b, ch, int(skip.shape[2]) - pad_left - pad_right))
    out = g.add(residual, skip)
    out = g.maximum(g.minimum(out, g.scalar(1.0, trt.float32, 3)), g.scalar(-1.0, trt.float32, 3))
    n_out = n * int(cfg["output_sampling_rate"]) // int(cfg["input_sampling_rate"])
    return g.slice(out, (0, 0, 0), (b, ch, n_out))


# ---------------------------------------------------------------------- audio VAE decoder


def _pixel_norm(g: Graph, x, eps: float):
    ms = g.reduce(g.mul(x, x), trt.ReduceOperation.AVG, 1)
    inv = g.unary(g.unary(g.add(ms, g.scalar(eps, trt.float32, 4)), trt.UnaryOperation.SQRT),
                  trt.UnaryOperation.RECIP)
    return g.mul(x, inv)


def _causal_conv2d(g: Graph, x, weight: np.ndarray, bias: np.ndarray | None, axis: str):
    out_c, _, kh, kw = (int(s) for s in weight.shape)
    ph, pw = kh - 1, kw - 1
    if axis == "height":
        pre, post = (ph, pw // 2), (0, pw - pw // 2)
    elif axis == "none":
        pre, post = (ph // 2, pw // 2), (ph - ph // 2, pw - pw // 2)
    else:
        raise NotImplementedError(f"audio VAE causality_axis={axis!r}")
    layer = g.net.add_convolution_nd(x, out_c, (kh, kw), g.weights(weight, trt.float32),
                                     g.weights(bias, trt.float32) if bias is not None else trt.Weights())
    layer.pre_padding = pre
    layer.post_padding = post
    return layer.get_output(0)


def _audio_resnet(g: Graph, ck: Checkpoint, p: str, x, axis: str, eps: float):
    h = g.silu(_pixel_norm(g, x, eps))
    h = _causal_conv2d(g, h, ck.get(f"{p}.conv1.conv.weight", F32), ck.maybe(f"{p}.conv1.conv.bias", F32), axis)
    h = g.silu(_pixel_norm(g, h, eps))
    h = _causal_conv2d(g, h, ck.get(f"{p}.conv2.conv.weight", F32), ck.maybe(f"{p}.conv2.conv.bias", F32), axis)
    if ck.has(f"{p}.nin_shortcut.conv.weight"):
        x = _causal_conv2d(g, x, ck.get(f"{p}.nin_shortcut.conv.weight", F32),
                           ck.maybe(f"{p}.nin_shortcut.conv.bias", F32), axis)
    elif ck.has(f"{p}.conv_shortcut.conv.weight"):
        x = _causal_conv2d(g, x, ck.get(f"{p}.conv_shortcut.conv.weight", F32),
                           ck.maybe(f"{p}.conv_shortcut.conv.bias", F32), axis)
    return g.add(x, h)


def _nearest2x(g: Graph, x):
    b, c, h, w = (int(s) for s in x.shape)
    x = g.gather(x, g.const(np.arange(2 * h, dtype=np.int32) // 2, trt.int32), 2)
    return g.gather(x, g.const(np.arange(2 * w, dtype=np.int32) // 2, trt.int32), 3)


def add_audio_vae_decoder(g: Graph, ck: Checkpoint, cfg: dict, z):
    """``LTX2AudioDecoder.forward`` on ``z`` ``[1, 8, frames, mel/4]`` -> mel ``[1, out_ch, T, mel_bins]``."""
    if cfg.get("norm_type", "pixel") != "pixel" or cfg.get("mid_block_add_attention") or cfg.get("attn_resolutions"):
        raise NotImplementedError("only the pixel-norm, attention-free LTX-2 audio decoder is implemented")
    axis = cfg.get("causality_axis", "height")
    eps = 1e-6
    ch_mult = list(cfg["ch_mult"])
    frames = int(z.shape[2])
    target_t = frames * 4
    if axis is not None:
        target_t = max(target_t - 3, 1)
    target_m = int(cfg.get("mel_bins") or z.shape[3])
    out_ch = int(cfg["output_channels"])
    x = _causal_conv2d(g, z, ck.get("decoder.conv_in.conv.weight", F32), ck.maybe("decoder.conv_in.conv.bias", F32),
                       axis)
    x = _audio_resnet(g, ck, "decoder.mid.block_1", x, axis, eps)
    x = _audio_resnet(g, ck, "decoder.mid.block_2", x, axis, eps)
    for level in reversed(range(len(ch_mult))):
        for bi in range(int(cfg["num_res_blocks"]) + 1):
            x = _audio_resnet(g, ck, f"decoder.up.{level}.block.{bi}", x, axis, eps)
        if level != 0:
            x = _nearest2x(g, x)
            x = _causal_conv2d(g, x, ck.get(f"decoder.up.{level}.upsample.conv.conv.weight", F32),
                               ck.maybe(f"decoder.up.{level}.upsample.conv.conv.bias", F32), axis)
            if axis == "height":
                b, c, h, w = (int(s) for s in x.shape)
                x = g.slice(x, (0, 0, 1, 0), (b, c, h - 1, w))
    x = g.silu(_pixel_norm(g, x, eps))
    x = _causal_conv2d(g, x, ck.get("decoder.conv_out.conv.weight", F32), ck.maybe("decoder.conv_out.conv.bias", F32),
                       axis)
    b, c, t, m = (int(s) for s in x.shape)
    x = g.slice(x, (0, 0, 0, 0), (b, min(c, out_ch), min(t, target_t), min(m, target_m)))
    if t < target_t or m < target_m:
        x = g.net.add_padding_nd(x, (0, 0), (max(target_t - t, 0), max(target_m - m, 0))).get_output(0)
    return x


# ---------------------------------------------------------------------- engine


def build_audio_decoder_engine(model_dir: str | Path, *, audio_frames: int, debug_mel: bool = False,
                               verbose: bool = False, tf32: bool = True, shapes: dict | None = None):
    """Serialized plan; ``shapes`` (optional) receives the ``waveform`` output shape."""
    model_dir = Path(model_dir)
    vae = Checkpoint(model_dir / "audio_vae")
    voc = Checkpoint(model_dir / "vocoder")
    vcfg, ocfg = vae.config(), voc.config()
    latent_c = int(vcfg["latent_channels"])
    mel_bins = int(vcfg["mel_bins"])
    latent_m = mel_bins // 4
    packed = latent_c * latent_m
    builder, network = new_network(make_logger(verbose))
    g = Graph(network)
    z = network.add_input("audio_latents", trt.float32, (1, audio_frames, packed))
    mean = vae.get("latents_mean", F32).reshape(1, 1, -1)
    std = vae.get("latents_std", F32).reshape(1, 1, -1)
    if mean.shape[-1] != packed:
        raise ValueError(f"audio latent statistics have {mean.shape[-1]} channels, packed latents {packed}")
    x = g.add(g.mul(z, g.const(std, trt.float32)), g.const(mean, trt.float32))
    # [1, L, C*M] -> unflatten(2, (C, M)) -> transpose(1, 2) -> [1, C, L, M]
    x = g.reshape(x, (1, audio_frames, latent_c, latent_m))
    x = g.transpose(x, (0, 2, 1, 3))
    mel = add_audio_vae_decoder(g, vae, vcfg, x)
    if debug_mel:
        g.mark_output(mel, "mel", trt.float32)
    wave = _bwe_vocoder(g, voc, ocfg, mel, debug=debug_mel)
    g.mark_output(wave, "waveform", trt.float32)
    if shapes is not None:
        shapes["waveform"] = [int(v) for v in wave.shape]
    print(f"[ltx2] Building audio decoder engine (audio latents {audio_frames} -> mel {int(mel.shape[2])} frames -> "
          f"{int(wave.shape[2])} samples @ {ocfg['output_sampling_rate']} Hz) ...", file=sys.stderr)
    return build_plan(builder, network, label="audio decoder", tf32=tf32)
