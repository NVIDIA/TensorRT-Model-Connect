# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Speech tokenizer engines: the per-frame feedback step and the voice-prompt encoder.

Both tokenizers are stacks of causal convolutions over ``[1, C, 1, T]``. In the
feedback engine they run one latent frame at a time: every convolution keeps
the input frames its kernel still needs from the previous step, and all those
frames travel in one flat ``state_in`` / ``state_out`` pair the runtime swaps
between calls. A transposed convolution with kernel ``2 * stride`` needs only
the previous input frame. Starting from zero states reproduces the reference's
zero-padded causal convolutions over the whole sequence, so frame-by-frame
output equals decoding the full latent sequence at once.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from . import graph as g

ACOUSTIC_ENCODER = "model.acoustic_tokenizer.encoder."
ACOUSTIC_DECODER = "model.acoustic_tokenizer.decoder."
SEMANTIC_ENCODER = "model.semantic_tokenizer.encoder."
CONNECTOR_EPS = 1e-6


def stream_state_layout(cfg) -> list[tuple[int, int]]:
    """``(channels, frames)`` of every streaming state, in graph order."""
    layout: list[tuple[int, int]] = []
    acoustic = cfg.acoustic
    tail = acoustic.kernel_size - 1
    channels = list(reversed(acoustic.stage_channels()))
    depths = list(reversed(acoustic.depths))
    layout.append((acoustic.vae_dim, tail))
    layout += [(channels[0], tail)] * depths[0]
    for index, _ in enumerate(reversed(acoustic.ratios)):
        layout.append((channels[index], 1))
        layout += [(channels[index + 1], tail)] * depths[index + 1]
    layout.append((channels[-1], tail))
    semantic = cfg.semantic
    channels = list(semantic.stage_channels())
    layout.append((semantic.channels, tail))
    layout += [(channels[0], tail)] * semantic.depths[0]
    for index, ratio in enumerate(semantic.ratios):
        layout.append((channels[index], ratio))
        layout += [(channels[index + 1], tail)] * semantic.depths[index + 1]
    layout.append((channels[-1], tail))
    return layout


def stream_state_size(cfg) -> int:
    return sum(channels * frames for channels, frames in stream_state_layout(cfg))


class _States:
    """Hands out the slices of ``state_in`` and collects ``state_out``; inert without a source."""

    def __init__(self, net: g.Net, source: trt.ITensor | None, layout=None):
        self.net = net
        self.source = source
        self.layout = list(layout or [])
        self.offset = 0
        self.taken = 0
        self.outputs: list[trt.ITensor] = []

    def take(self, channels: int, frames: int) -> trt.ITensor:
        expected = self.layout[self.taken]
        if expected != (channels, frames):
            raise RuntimeError(
                f"stream state {self.taken} is {(channels, frames)}, layout says {expected}"
            )
        part = self.net.slice(self.source, (self.offset,), (channels * frames,))
        self.offset += channels * frames
        self.taken += 1
        return self.net.reshape(part, (1, channels, 1, frames))

    def give(self, tensor: trt.ITensor, channels: int, frames: int) -> None:
        self.outputs.append(self.net.reshape(tensor, (channels * frames,)))


def _causal_conv(net, states, x, weight, bias, *, stride=1, groups=1):
    weight = np.asarray(weight)
    channels = int(weight.shape[1]) * groups
    left = (int(weight.shape[2]) - 1) - (stride - 1)
    if states.source is None:
        return net.conv(x, weight, bias, stride=stride, groups=groups, left_pad=left)
    frames = int(x.shape[3])
    full = net.concat([states.take(channels, left), x], 3)
    states.give(net.slice(full, (0, 0, 0, frames), (1, channels, 1, left)), channels, left)
    return net.conv(full, weight, bias, stride=stride, groups=groups)


def _causal_deconv(net, states, x, weight, bias, *, stride):
    weight = np.asarray(weight)
    in_channels, out_channels, kernel = (int(d) for d in weight.shape)
    if kernel != 2 * stride:
        raise ValueError("vibevoice transposed convolutions must have kernel == 2 * stride")
    frames = int(x.shape[3])
    full = net.concat([states.take(in_channels, 1), x], 3)
    states.give(net.slice(x, (0, 0, 0, frames - 1), (1, in_channels, 1, 1)), in_channels, 1)
    out = net.deconv(full, weight, bias, stride=stride)
    return net.slice(out, (0, 0, 0, stride), (1, out_channels, 1, frames * stride))


def _block(net, states, x, weights, base, eps, channels):
    """ConvNeXt-style block: channel RMSNorm, depthwise causal conv, gated FFN."""
    residual = x
    y = net.rms_norm(x, weights.f32(base + "norm.weight"), eps, 1)
    y = _causal_conv(
        net,
        states,
        y,
        weights.f32(base + "mixer.conv.conv.conv.weight"),
        weights.f32(base + "mixer.conv.conv.conv.bias"),
        groups=channels,
    )
    y = net.mul(y, net.const(weights.f32(base + "gamma"), (1, channels, 1, 1)))
    x = net.add(residual, y)
    residual = x
    y = net.rms_norm(x, weights.f32(base + "ffn_norm.weight"), eps, 1)
    y = net.conv(
        y,
        weights.f32(base + "ffn.linear1.weight")[:, :, None],
        weights.f32(base + "ffn.linear1.bias"),
    )
    y = net.gelu(y)
    y = net.conv(
        y,
        weights.f32(base + "ffn.linear2.weight")[:, :, None],
        weights.f32(base + "ffn.linear2.bias"),
    )
    y = net.mul(y, net.const(weights.f32(base + "ffn_gamma"), (1, channels, 1, 1)))
    return net.add(residual, y)


def _encoder(net, states, x, weights, prefix, tokenizer):
    """``[1, 1, 1, T]`` audio -> ``[1, vae_dim, 1, T / hop]`` latents."""
    channels = tokenizer.stage_channels()
    x = _causal_conv(
        net,
        states,
        x,
        weights.f32(prefix + "downsample_layers.0.0.conv.conv.weight"),
        weights.f32(prefix + "downsample_layers.0.0.conv.conv.bias"),
    )
    for block in range(tokenizer.depths[0]):
        x = _block(
            net, states, x, weights, f"{prefix}stages.0.{block}.", tokenizer.eps, channels[0]
        )
    for index, ratio in enumerate(tokenizer.ratios):
        layer = f"{prefix}downsample_layers.{index + 1}.0.conv.conv."
        x = _causal_conv(
            net, states, x, weights.f32(layer + "weight"), weights.f32(layer + "bias"), stride=ratio
        )
        for block in range(tokenizer.depths[index + 1]):
            x = _block(
                net,
                states,
                x,
                weights,
                f"{prefix}stages.{index + 1}.{block}.",
                tokenizer.eps,
                channels[index + 1],
            )
    return _causal_conv(
        net,
        states,
        x,
        weights.f32(prefix + "head.conv.conv.weight"),
        weights.f32(prefix + "head.conv.conv.bias"),
    )


def _decoder(net, states, x, weights, prefix, tokenizer):
    """``[1, vae_dim, 1, F]`` latents -> ``[1, 1, 1, F * hop]`` audio."""
    channels = list(reversed(tokenizer.stage_channels()))
    depths = list(reversed(tokenizer.depths))
    x = _causal_conv(
        net,
        states,
        x,
        weights.f32(prefix + "upsample_layers.0.0.conv.conv.weight"),
        weights.f32(prefix + "upsample_layers.0.0.conv.conv.bias"),
    )
    for block in range(depths[0]):
        x = _block(
            net, states, x, weights, f"{prefix}stages.0.{block}.", tokenizer.eps, channels[0]
        )
    for index, ratio in enumerate(reversed(tokenizer.ratios)):
        layer = f"{prefix}upsample_layers.{index + 1}.0.convtr.convtr."
        x = _causal_deconv(
            net, states, x, weights.f32(layer + "weight"), weights.f32(layer + "bias"), stride=ratio
        )
        for block in range(depths[index + 1]):
            x = _block(
                net,
                states,
                x,
                weights,
                f"{prefix}stages.{index + 1}.{block}.",
                tokenizer.eps,
                channels[index + 1],
            )
    return _causal_conv(
        net,
        states,
        x,
        weights.f32(prefix + "head.conv.conv.weight"),
        weights.f32(prefix + "head.conv.conv.bias"),
    )


def _connector(net, x, weights, prefix):
    y = net.linear(x, weights.f32(prefix + "fc1.weight"), weights.f32(prefix + "fc1.bias"))
    y = net.rms_norm(y, weights.f32(prefix + "norm.weight"), CONNECTOR_EPS, -1)
    return net.linear(y, weights.f32(prefix + "fc2.weight"), weights.f32(prefix + "fc2.bias"))


def _latent_factors(net, weights, rank):
    scaling = net.const(weights.f32("model.speech_scaling_factor").reshape(()), (1,) * rank)
    bias = net.const(weights.f32("model.speech_bias_factor").reshape(()), (1,) * rank)
    return scaling, bias


def build_feedback_engine(weights, cfg, *, dtype, verbose: bool = False) -> bytes:
    """One generated latent -> its audio chunk and the next LM input embedding."""
    builder, network, config = g.builder_and_network(verbose)
    net = g.Net(network, dtype)
    layout = stream_state_layout(cfg)
    size = stream_state_size(cfg)
    latent_in = network.add_input("latent", trt.float32, (1, cfg.latent_size))
    state_in = network.add_input("state_in", dtype, (size,))
    states = _States(net, state_in, layout)

    latent = net.cast(latent_in, dtype)
    scaling, bias = _latent_factors(net, weights, 2)
    unscaled = net.sub(net.div(latent, scaling), bias)
    audio = _decoder(
        net,
        states,
        net.reshape(unscaled, (1, cfg.latent_size, 1, 1)),
        weights,
        ACOUSTIC_DECODER,
        cfg.acoustic,
    )
    semantic = _encoder(net, states, audio, weights, SEMANTIC_ENCODER, cfg.semantic)
    if states.taken != len(layout):
        raise RuntimeError("feedback engine did not consume its whole stream state layout")

    pcm = net.cast(net.reshape(audio, (1, cfg.acoustic.hop_length)), trt.float32)
    pcm.name = "audio"
    network.mark_output(pcm)
    embed = net.add(
        _connector(net, latent, weights, "model.acoustic_connector."),
        _connector(
            net,
            net.reshape(semantic, (1, cfg.semantic.vae_dim)),
            weights,
            "model.semantic_connector.",
        ),
    )
    embed.name = "next_embed"
    network.mark_output(embed)
    state_out = net.concat(states.outputs, 0)
    state_out.name = "state_out"
    network.mark_output(state_out)
    return g.serialize(builder, network, config, "feedback")


def build_voice_encoder_engine(
    weights, cfg, *, dtype, max_samples: int, verbose: bool = False
) -> bytes:
    """Voice-prompt audio -> LM input embeddings, with the sigma-VAE sampling noise."""
    builder, network, config = g.builder_and_network(verbose)
    net = g.Net(network, dtype)
    hop = cfg.acoustic.hop_length
    audio = network.add_input("audio", trt.float32, (1, 1, 1, -1))
    noise = network.add_input("noise", trt.float32, (1, -1, cfg.acoustic.vae_dim))
    noise_scale = network.add_input("noise_scale", trt.float32, (1,))

    latents = _encoder(
        net, _States(net, None), net.cast(audio, dtype), weights, ACOUSTIC_ENCODER, cfg.acoustic
    )
    latents = net.reshape(latents, (1, cfg.acoustic.vae_dim, -1), second_transpose=(0, 2, 1))
    # sigma-VAE sampling: latents + (vae_std * z) * eps, every product in the model dtype.
    std = net.mul(
        net.scalar(cfg.acoustic_vae_std, 3), net.cast(net.reshape(noise_scale, (1, 1, 1)), dtype)
    )
    latents = net.add(latents, net.mul(std, net.cast(noise, dtype)))
    scaling, bias = _latent_factors(net, weights, 3)
    features = net.mul(net.add(latents, bias), scaling)
    embeds = _connector(net, features, weights, "model.acoustic_connector.")
    embeds.name = "embeds"
    network.mark_output(embeds)

    frames = max_samples // hop
    opt_frames = min(frames, 75)
    profile = builder.create_optimization_profile()
    profile.set_shape("audio", (1, 1, 1, hop), (1, 1, 1, opt_frames * hop), (1, 1, 1, frames * hop))
    profile.set_shape(
        "noise",
        (1, 1, cfg.acoustic.vae_dim),
        (1, opt_frames, cfg.acoustic.vae_dim),
        (1, frames, cfg.acoustic.vae_dim),
    )
    config.add_optimization_profile(profile)
    return g.serialize(builder, network, config, "voice encoder")
