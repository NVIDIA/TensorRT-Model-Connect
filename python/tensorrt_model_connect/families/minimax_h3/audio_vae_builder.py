# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT builders for the official MiniMax-H3 audio VAE boundaries.

The deployed engines are self-contained: reference waveforms are encoded and
normalized by the float32 DAC/causal-attention path, while generated diffusion
latents are denormalized and decoded by the float32 DAC/BigVGAN path. PyTorch
is used only while exporting model-owned graphs to ONNX; the native runtime
neither imports Python nor invokes a callback.
"""

from __future__ import annotations

import gc
import io
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tensorrt_model_connect import trt_compat

from .checkpoint import load_selected_component_state_dict
from .config import AUDIO_VAE_DECODER_DEFAULT_WORKSPACE_BYTES, resolve_workspace_bytes


AUDIO_BATCH = 2
AUDIO_LATENT_CHANNELS = 32
AUDIO_LATENT_FRAMES = 207
AUDIO_SAMPLE_RATE = 32_000
AUDIO_HOP_LENGTH = 800
AUDIO_OUTPUT_SAMPLES = AUDIO_LATENT_FRAMES * AUDIO_HOP_LENGTH
AUDIO_REFERENCE_MIN_SAMPLES = 2 * AUDIO_SAMPLE_RATE
AUDIO_REFERENCE_OPT_SAMPLES = AUDIO_OUTPUT_SAMPLES
AUDIO_REFERENCE_MAX_SAMPLES = 15 * AUDIO_SAMPLE_RATE
AUDIO_VAE_ENCODER_DEFAULT_WORKSPACE_BYTES = 32 << 30


@dataclass(frozen=True)
class AudioVaeDecoderConfig:
    latent_dim: int
    latent_channels: int
    decoder_dim: int
    decoder_rates: tuple[int, ...]
    decoder_kernel_sizes: tuple[int, ...]
    resblock_kernel_sizes: tuple[int, ...]
    resblock_dilation_sizes: tuple[tuple[int, ...], ...]
    sampling_rate: int
    latents_mean: tuple[float, ...]
    latents_std: tuple[float, ...]
    encoder_dim: int = 64
    encoder_rates: tuple[int, ...] = (2, 4, 4, 5, 5)
    num_attention_heads: int = 8

    @property
    def hop_length(self) -> int:
        return math.prod(self.decoder_rates)

    @property
    def encoder_hop_length(self) -> int:
        return math.prod(self.encoder_rates)


_EXPECTED_ARCHITECTURE = {
    "encoder_dim": 64,
    "encoder_rates": (2, 4, 4, 5, 5),
    "latent_dim": 2048,
    "latent_channels": AUDIO_LATENT_CHANNELS,
    "decoder_dim": 1024,
    "decoder_rates": (5, 5, 2, 2, 2, 2, 2),
    "decoder_kernel_sizes": (9, 9, 4, 4, 4, 4, 4),
    "num_attention_heads": 8,
    "resblock_kernel_sizes": (3, 7, 11),
    "resblock_dilation_sizes": ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    "sampling_rate": AUDIO_SAMPLE_RATE,
}


def _as_int_tuple(value: object, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in value
    ):
        raise ValueError(f"MiniMax-H3 audio VAE {name} must be an integer array")
    return tuple(value)


def _as_nested_int_tuple(value: object, *, name: str) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"MiniMax-H3 audio VAE {name} must be an array of integer arrays")
    return tuple(_as_int_tuple(item, name=name) for item in value)


def _finite_channel_values(value: object, *, name: str, positive: bool) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != AUDIO_LATENT_CHANNELS:
        raise ValueError(f"MiniMax-H3 audio VAE {name} must contain {AUDIO_LATENT_CHANNELS} values")
    result: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise ValueError(f"MiniMax-H3 audio VAE {name} must contain finite numbers")
        try:
            converted = float(item)
        except (TypeError, ValueError) as error:
            raise ValueError(f"MiniMax-H3 audio VAE {name} must contain finite numbers") from error
        if not math.isfinite(converted) or (positive and converted <= 0.0):
            qualifier = "finite positive" if positive else "finite"
            raise ValueError(f"MiniMax-H3 audio VAE {name} must contain {qualifier} numbers")
        result.append(converted)
    return tuple(result)


def validate_audio_vae_decoder_config(raw: object) -> AudioVaeDecoderConfig:
    """Validate the exact released Diffusers audio-decoder architecture."""

    if not isinstance(raw, dict):
        raise ValueError("MiniMax-H3 audio VAE config must be a JSON object")

    observed: dict[str, object] = {}
    for name, expected in _EXPECTED_ARCHITECTURE.items():
        value = raw.get(name)
        if isinstance(expected, tuple):
            if expected and isinstance(expected[0], tuple):
                value = _as_nested_int_tuple(value, name=name)
            else:
                value = _as_int_tuple(value, name=name)
        elif not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"MiniMax-H3 audio VAE {name} must be an integer")
        observed[name] = value

    mismatches = {
        name: (observed[name], expected)
        for name, expected in _EXPECTED_ARCHITECTURE.items()
        if observed[name] != expected
    }
    if mismatches:
        raise ValueError(f"Unsupported MiniMax-H3 audio VAE architecture: {mismatches}")

    means = _finite_channel_values(raw.get("latents_mean"), name="latents_mean", positive=False)
    stds = _finite_channel_values(raw.get("latents_std"), name="latents_std", positive=True)
    config = AudioVaeDecoderConfig(
        latent_dim=int(observed["latent_dim"]),
        latent_channels=int(observed["latent_channels"]),
        decoder_dim=int(observed["decoder_dim"]),
        decoder_rates=tuple(observed["decoder_rates"]),
        decoder_kernel_sizes=tuple(observed["decoder_kernel_sizes"]),
        resblock_kernel_sizes=tuple(observed["resblock_kernel_sizes"]),
        resblock_dilation_sizes=tuple(observed["resblock_dilation_sizes"]),
        sampling_rate=int(observed["sampling_rate"]),
        latents_mean=means,
        latents_std=stds,
        encoder_dim=int(observed["encoder_dim"]),
        encoder_rates=tuple(observed["encoder_rates"]),
        num_attention_heads=int(observed["num_attention_heads"]),
    )
    if config.hop_length != AUDIO_HOP_LENGTH or config.encoder_hop_length != AUDIO_HOP_LENGTH:
        raise ValueError("MiniMax-H3 audio VAE rates do not match the 800-sample hop length")
    return config


def load_audio_vae_config(audio_vae_dir: str | Path) -> AudioVaeDecoderConfig:
    path = Path(audio_vae_dir) / "config.json"
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read MiniMax-H3 audio VAE config: {path}") from error
    return validate_audio_vae_decoder_config(raw)


def load_audio_vae_decoder_config(audio_vae_dir: str | Path) -> AudioVaeDecoderConfig:
    """Backward-compatible name for callers building only the decoder."""

    return load_audio_vae_config(audio_vae_dir)


def validate_audio_reference_samples(samples: object, *, sample_rate: int) -> int:
    """Validate one upmixed stereo reference before native encoder enqueue.

    TensorRT profiles can bound a dynamic dimension, but cannot express the
    800-sample modulus or validate waveform values.  Callers must enforce this
    contract before submitting the input binding.

    Returns:
        The exact number of audio-latent frames produced by the encoder.
    """

    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool):
        raise ValueError("MiniMax-H3 reference audio sample_rate must be an integer")
    if sample_rate != AUDIO_SAMPLE_RATE:
        raise ValueError(
            f"MiniMax-H3 reference audio must be {AUDIO_SAMPLE_RATE} Hz, got {sample_rate}"
        )
    if not isinstance(samples, np.ndarray) or samples.dtype != np.float32:
        raise ValueError("MiniMax-H3 reference audio must be a float32 NumPy array")
    if samples.ndim != 3 or samples.shape[:2] != (AUDIO_BATCH, 1):
        raise ValueError(
            "MiniMax-H3 reference audio must have stereo-as-batch shape [2, 1, samples]"
        )
    num_samples = samples.shape[2]
    if not AUDIO_REFERENCE_MIN_SAMPLES <= num_samples <= AUDIO_REFERENCE_MAX_SAMPLES:
        raise ValueError("MiniMax-H3 reference audio must contain between 2 and 15 seconds")
    if num_samples % AUDIO_HOP_LENGTH != 0:
        raise ValueError(
            f"MiniMax-H3 reference audio must be aligned to {AUDIO_HOP_LENGTH} samples"
        )
    if not np.isfinite(samples).all():
        raise ValueError("MiniMax-H3 reference audio must contain only finite samples")
    if np.any(samples < -1.0) or np.any(samples > 1.0):
        raise ValueError("MiniMax-H3 reference audio samples must be in [-1, 1]")
    return num_samples // AUDIO_HOP_LENGTH


def _make_encoder_module(torch: Any, config: AudioVaeDecoderConfig):
    """Reconstruct the official encoder, attention projection, and mean head."""

    if config.latent_dim % config.latent_channels != 0:
        raise ValueError("MiniMax-H3 audio latent_dim must be divisible by latent_channels")
    if config.latent_dim % config.num_attention_heads != 0:
        raise ValueError("MiniMax-H3 audio latent_dim must be divisible by num_attention_heads")

    nn = torch.nn
    functional = torch.nn.functional
    weight_norm = torch.nn.utils.weight_norm

    class Snake1d(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.alpha = nn.Parameter(torch.ones(1, channels, 1))

        def forward(self, hidden_states):
            return hidden_states + (self.alpha + 1e-9).reciprocal() * torch.sin(
                self.alpha * hidden_states
            ).pow(2)

    class ResidualUnit(nn.Module):
        def __init__(self, dim: int, dilation: int):
            super().__init__()
            self.block = nn.Sequential(
                Snake1d(dim),
                weight_norm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size=7,
                        dilation=dilation,
                        padding=((7 - 1) * dilation) // 2,
                    )
                ),
                Snake1d(dim),
                weight_norm(nn.Conv1d(dim, dim, kernel_size=1)),
            )

        def forward(self, hidden_states):
            residual = self.block(hidden_states)
            # H3's odd kernel and symmetric dilation padding preserve length
            # exactly. The generic upstream defensive crop is unreachable and
            # would freeze a symbolic ONNX length through a Python condition.
            return hidden_states + residual

    class EncoderBlock(nn.Module):
        def __init__(self, dim: int, stride: int):
            super().__init__()
            self.block = nn.Sequential(
                ResidualUnit(dim // 2, dilation=1),
                ResidualUnit(dim // 2, dilation=3),
                ResidualUnit(dim // 2, dilation=9),
                Snake1d(dim // 2),
                weight_norm(
                    nn.Conv1d(
                        dim // 2,
                        dim,
                        kernel_size=2 * stride,
                        stride=stride,
                        padding=math.ceil(stride / 2),
                    )
                ),
            )

        def forward(self, hidden_states):
            return self.block(hidden_states)

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            dim = config.encoder_dim
            blocks: list[Any] = [weight_norm(nn.Conv1d(1, dim, kernel_size=7, padding=3))]
            for stride in config.encoder_rates:
                dim *= 2
                blocks.append(EncoderBlock(dim, stride=stride))
            blocks.extend(
                [
                    Snake1d(dim),
                    weight_norm(nn.Conv1d(dim, config.latent_dim, kernel_size=3, padding=1)),
                ]
            )
            self.block = nn.Sequential(*blocks)

        def forward(self, hidden_states):
            return self.block(hidden_states)

    class GeGluMlp(nn.Module):
        def __init__(self, in_features: int, hidden_features: int):
            super().__init__()
            self.norm = nn.LayerNorm(in_features)
            self.act = nn.GELU(approximate="tanh")
            self.w0 = nn.Linear(in_features, hidden_features)
            self.w1 = nn.Linear(in_features, hidden_features)
            self.w2 = nn.Linear(hidden_features, in_features)

        def forward(self, hidden_states):
            hidden_states = self.norm(hidden_states)
            return self.w2(self.act(self.w0(hidden_states)) * self.w1(hidden_states))

    class CausalAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.out_dim = config.latent_channels
            self.num_heads = config.num_attention_heads
            self.head_dim = config.latent_dim // config.num_attention_heads
            self.qkv = nn.Linear(config.latent_dim, config.latent_dim * 3, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(config.latent_dim))
            self.v_bias = nn.Parameter(torch.zeros(config.latent_dim))
            self.register_buffer("zero_k_bias", torch.zeros(config.latent_dim))
            self.proj = nn.Linear(config.latent_channels, config.latent_channels)

        def forward(self, hidden_states):
            batch_size, sequence_length, _ = hidden_states.shape
            qkv = functional.linear(
                hidden_states,
                self.qkv.weight,
                torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)),
            )
            query, key, value = (
                qkv.reshape(
                    batch_size,
                    sequence_length,
                    3,
                    self.num_heads,
                    self.head_dim,
                )
                .permute(2, 0, 1, 3, 4)
                .unbind(0)
            )
            query = query.permute(0, 2, 1, 3)
            key = key.permute(0, 2, 1, 3)
            value = value.permute(0, 2, 1, 3)
            hidden_states = functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
            ).permute(0, 2, 1, 3)
            hidden_states = torch.mean(hidden_states, dim=2)
            hidden_states = functional.adaptive_avg_pool1d(hidden_states, self.out_dim)
            return self.proj(hidden_states)

    class AttentionProjection(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(config.latent_dim)
            self.attn = CausalAttention()
            self.proj = nn.Linear(config.latent_dim, config.latent_channels)
            self.norm3 = nn.LayerNorm(config.latent_dim)
            self.norm2 = nn.LayerNorm(config.latent_channels)
            self.mlp = GeGluMlp(config.latent_channels, config.latent_channels * 2)

        def forward(self, hidden_states):
            hidden_states = self.proj(self.norm3(hidden_states)) + self.attn(
                self.norm1(hidden_states)
            )
            return hidden_states + self.mlp(self.norm2(hidden_states))

    class StaticAudioVaeEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "_latents_mean",
                torch.tensor(config.latents_mean, dtype=torch.float32).view(1, 1, -1),
                persistent=False,
            )
            self.register_buffer(
                "_latents_std",
                torch.tensor(config.latents_std, dtype=torch.float32).view(1, 1, -1),
                persistent=False,
            )
            self.encoder = Encoder()
            self.pre_block = AttentionProjection()
            self.mean_proj = nn.Conv1d(config.latent_channels, config.latent_channels, 1)

        def forward(self, audio_samples):
            hidden_states = self.encoder(audio_samples.to(torch.float32))
            hidden_states = self.pre_block(hidden_states.transpose(1, 2)).transpose(1, 2)
            mean = self.mean_proj(hidden_states).transpose(1, 2)
            return ((mean - self._latents_mean) / self._latents_std).to(torch.float32)

    return StaticAudioVaeEncoder().float().eval()


def _make_decoder_module(
    torch: Any,
    config: AudioVaeDecoderConfig,
    *,
    batch: int = AUDIO_BATCH,
    latent_frames: int = AUDIO_LATENT_FRAMES,
):
    """Reconstruct only the official Diffusers decoder under checkpoint names."""

    nn = torch.nn
    functional = torch.nn.functional
    weight_norm = torch.nn.utils.weight_norm

    def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int):
        half_size = kernel_size // 2
        attenuation = 2.285 * (half_size - 1) * math.pi * (4 * half_width) + 7.95
        if attenuation > 50.0:
            beta = 0.1102 * (attenuation - 8.7)
        elif attenuation >= 21.0:
            beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
        else:
            beta = 0.0
        window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
        if kernel_size % 2 == 0:
            positions = torch.arange(-half_size, half_size) + 0.5
        else:
            positions = torch.arange(kernel_size) - half_size
        filter_ = 2 * cutoff * window * torch.sinc(2 * cutoff * positions)
        return (filter_ / filter_.sum()).view(1, 1, kernel_size)

    class SnakeBeta(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.alpha = nn.Parameter(torch.zeros(channels))
            self.beta = nn.Parameter(torch.zeros(channels))

        def forward(self, hidden_states):
            alpha = torch.exp(self.alpha.unsqueeze(0).unsqueeze(-1))
            beta = torch.exp(self.beta.unsqueeze(0).unsqueeze(-1))
            return hidden_states + (beta + 1e-9).reciprocal() * torch.sin(
                alpha * hidden_states
            ).pow(2)

    class LowPassFilter1d(nn.Module):
        def __init__(
            self,
            cutoff: float,
            half_width: float,
            stride: int,
            kernel_size: int,
            channels: int,
        ):
            super().__init__()
            even = kernel_size % 2 == 0
            self.pad_left = kernel_size // 2 - int(even)
            self.pad_right = kernel_size // 2
            self.stride = stride
            self.channels = channels
            self.register_buffer("filter", kaiser_sinc_filter1d(cutoff, half_width, kernel_size))

        def forward(self, hidden_states):
            hidden_states = functional.pad(
                hidden_states, (self.pad_left, self.pad_right), mode="replicate"
            )
            return functional.conv1d(
                hidden_states,
                self.filter.expand(self.channels, -1, -1),
                stride=self.stride,
                groups=self.channels,
            )

    class UpSample1d(nn.Module):
        def __init__(self, ratio: int, kernel_size: int, channels: int):
            super().__init__()
            self.ratio = ratio
            self.stride = ratio
            self.channels = channels
            self.pad = kernel_size // ratio - 1
            self.pad_left = self.pad * self.stride + (kernel_size - self.stride) // 2
            self.pad_right = self.pad * self.stride + (kernel_size - self.stride + 1) // 2
            self.register_buffer(
                "filter",
                kaiser_sinc_filter1d(
                    cutoff=0.5 / ratio,
                    half_width=0.6 / ratio,
                    kernel_size=kernel_size,
                ),
            )

        def forward(self, hidden_states):
            hidden_states = functional.pad(hidden_states, (self.pad, self.pad), mode="replicate")
            hidden_states = self.ratio * functional.conv_transpose1d(
                hidden_states,
                self.filter.expand(self.channels, -1, -1),
                stride=self.stride,
                groups=self.channels,
            )
            return hidden_states[..., self.pad_left : -self.pad_right]

    class DownSample1d(nn.Module):
        def __init__(self, ratio: int, kernel_size: int, channels: int):
            super().__init__()
            self.lowpass = LowPassFilter1d(
                cutoff=0.5 / ratio,
                half_width=0.6 / ratio,
                stride=ratio,
                kernel_size=kernel_size,
                channels=channels,
            )

        def forward(self, hidden_states):
            return self.lowpass(hidden_states)

    class Activation1d(nn.Module):
        def __init__(self, activation, channels: int, ratio: int = 2, kernel_size: int = 12):
            super().__init__()
            self.act = activation
            self.upsample = UpSample1d(ratio, kernel_size, channels)
            self.downsample = DownSample1d(ratio, kernel_size, channels)

        def forward(self, hidden_states):
            hidden_states = self.upsample(hidden_states)
            hidden_states = self.act(hidden_states)
            return self.downsample(hidden_states)

    class AMPBlock(nn.Module):
        def __init__(self, channels: int, kernel_size: int, dilation: tuple[int, ...]):
            super().__init__()
            self.convs1 = nn.ModuleList(
                [
                    weight_norm(
                        nn.Conv1d(
                            channels,
                            channels,
                            kernel_size,
                            dilation=value,
                            padding=(kernel_size * value - value) // 2,
                        )
                    )
                    for value in dilation
                ]
            )
            self.convs2 = nn.ModuleList(
                [
                    weight_norm(
                        nn.Conv1d(
                            channels,
                            channels,
                            kernel_size,
                            dilation=1,
                            padding=(kernel_size - 1) // 2,
                        )
                    )
                    for _ in dilation
                ]
            )
            self.activations = nn.ModuleList(
                [Activation1d(SnakeBeta(channels), channels) for _ in range(2 * len(dilation))]
            )

        def forward(self, hidden_states):
            acts1, acts2 = self.activations[::2], self.activations[1::2]
            for conv1, conv2, act1, act2 in zip(self.convs1, self.convs2, acts1, acts2):
                residual = conv1(act1(hidden_states))
                residual = conv2(act2(residual))
                hidden_states = residual + hidden_states
            return hidden_states

    class BigVGANDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_kernels = len(config.resblock_kernel_sizes)
            self.num_upsamples = len(config.decoder_rates)
            self.conv_pre = weight_norm(
                nn.Conv1d(config.latent_dim, config.decoder_dim, 7, 1, padding=3)
            )
            self.ups = nn.ModuleList()
            for index, (rate, kernel) in enumerate(
                zip(config.decoder_rates, config.decoder_kernel_sizes)
            ):
                self.ups.append(
                    nn.ModuleList(
                        [
                            weight_norm(
                                nn.ConvTranspose1d(
                                    config.decoder_dim // (2**index),
                                    config.decoder_dim // (2 ** (index + 1)),
                                    kernel,
                                    rate,
                                    padding=(kernel - rate) // 2,
                                )
                            )
                        ]
                    )
                )
            self.resblocks = nn.ModuleList()
            channels = config.decoder_dim
            for index in range(self.num_upsamples):
                channels = config.decoder_dim // (2 ** (index + 1))
                for kernel, dilation in zip(
                    config.resblock_kernel_sizes, config.resblock_dilation_sizes
                ):
                    self.resblocks.append(AMPBlock(channels, kernel, dilation))
            self.activation_post = Activation1d(SnakeBeta(channels), channels)
            self.conv_post = weight_norm(nn.Conv1d(channels, 1, 7, 1, padding=3, bias=False))

        def forward(self, hidden_states):
            hidden_states = self.conv_pre(hidden_states)
            for index in range(self.num_upsamples):
                hidden_states = self.ups[index][0](hidden_states)
                residual = None
                for kernel_index in range(self.num_kernels):
                    block = self.resblocks[index * self.num_kernels + kernel_index](hidden_states)
                    residual = block if residual is None else residual + block
                hidden_states = residual / self.num_kernels
            hidden_states = self.activation_post(hidden_states)
            hidden_states = self.conv_post(hidden_states)
            return torch.clamp(hidden_states, min=-1.0, max=1.0)

    class StaticAudioVaeDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "_latents_mean",
                torch.tensor(config.latents_mean, dtype=torch.float32).view(1, -1, 1),
                persistent=False,
            )
            self.register_buffer(
                "_latents_std",
                torch.tensor(config.latents_std, dtype=torch.float32).view(1, -1, 1),
                persistent=False,
            )
            self.dec_in_proj = nn.Conv1d(config.latent_channels, config.latent_dim, 1)
            self.decoder = BigVGANDecoder()

        def forward(self, audio_latents):
            # The diffusion output is normalized.  Diffusers performs this
            # exact per-channel inverse transform immediately before decode.
            denormalized = audio_latents.to(torch.float32) * self._latents_std
            denormalized = denormalized + self._latents_mean
            waveform = self.decoder(self.dec_in_proj(denormalized))
            # ONNX cannot infer the static extent through its replicate-pad /
            # slice lowering.  This no-op reshape records the actual fixed ABI.
            waveform = waveform.reshape(batch, 1, latent_frames * config.hop_length)
            return waveform.to(torch.float32)

    return StaticAudioVaeDecoder().float().eval()


def _load_fp32_weights(torch: Any, module: Any, audio_vae_dir: Path, *, component: str) -> None:
    expected = tuple(module.state_dict())
    state = load_selected_component_state_dict(audio_vae_dir, expected)
    wrong_dtype = sorted(name for name, value in state.items() if value.dtype != torch.float32)
    if wrong_dtype:
        raise ValueError(
            f"MiniMax-H3 audio VAE {component} checkpoint tensors must be float32: {wrong_dtype}"
        )
    module.load_state_dict(state, strict=True)


def _load_decoder_weights(torch: Any, module: Any, audio_vae_dir: Path) -> None:
    _load_fp32_weights(torch, module, audio_vae_dir, component="decoder")


def _load_encoder_weights(torch: Any, module: Any, audio_vae_dir: Path) -> None:
    _load_fp32_weights(torch, module, audio_vae_dir, component="encoder")


def _remove_weight_normalization(torch: Any, module: Any) -> None:
    """Freeze official weight-normalized convolutions before ONNX tracing."""

    for child in module.modules():
        try:
            torch.nn.utils.remove_weight_norm(child)
        except ValueError:
            pass


def _export_decoder_onnx(
    audio_vae_dir: Path, config: AudioVaeDecoderConfig, verbose: bool
) -> bytes:
    import torch

    module = _make_decoder_module(torch, config)
    _load_decoder_weights(torch, module, audio_vae_dir)
    _remove_weight_normalization(torch, module)
    dummy = torch.zeros(
        (AUDIO_BATCH, AUDIO_LATENT_CHANNELS, AUDIO_LATENT_FRAMES), dtype=torch.float32
    )
    onnx_buffer = io.BytesIO()
    if verbose:
        print(
            "[trtmc build]   Exporting official MiniMax-H3 float32 audio VAE "
            f"decoder ({AUDIO_LATENT_FRAMES} latents -> {AUDIO_OUTPUT_SAMPLES} samples) ...",
            file=sys.stderr,
        )
    # PyTorch 2.12's aarch64 oneDNN JIT can reject the decoder's longest
    # depthwise Conv1d during tracing.  Eager convolution and the exported ONNX
    # graph are unchanged; disable only that CPU implementation while tracing.
    with (
        torch.inference_mode(),
        torch.backends.mkldnn.flags(
            enabled=False,
            deterministic=None,
            allow_tf32=None,
            fp32_precision=None,
        ),
    ):
        torch.onnx.export(
            module,
            dummy,
            onnx_buffer,
            opset_version=17,
            input_names=["audio_latents"],
            output_names=["waveform"],
            dynamo=False,
        )
    payload = onnx_buffer.getvalue()
    del dummy, module
    gc.collect()
    return payload


def _export_encoder_onnx(
    audio_vae_dir: Path, config: AudioVaeDecoderConfig, verbose: bool
) -> bytes:
    import torch

    module = _make_encoder_module(torch, config)
    _load_encoder_weights(torch, module, audio_vae_dir)
    _remove_weight_normalization(torch, module)
    dummy = torch.zeros((AUDIO_BATCH, 1, AUDIO_REFERENCE_OPT_SAMPLES), dtype=torch.float32)
    onnx_buffer = io.BytesIO()
    if verbose:
        print(
            "[trtmc build]   Exporting official MiniMax-H3 float32 audio VAE "
            "reference encoder with a dynamic 2-15 second input ...",
            file=sys.stderr,
        )
    with torch.inference_mode():
        torch.onnx.export(
            module,
            dummy,
            onnx_buffer,
            opset_version=17,
            input_names=["audio_samples"],
            output_names=["audio_condition_rows"],
            dynamic_axes={
                "audio_samples": {2: "num_samples"},
                "audio_condition_rows": {1: "num_audio_latents"},
            },
            dynamo=False,
        )
    payload = onnx_buffer.getvalue()
    del dummy, module
    gc.collect()
    return payload


def _build_serialized_engine(
    onnx_bytes: bytes,
    *,
    verbose: bool,
    workspace_bytes: int | None,
) -> bytes:
    trt = trt_compat.get_trt()
    logger = trt.Logger(trt.Logger.INFO if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        trt_compat.network_creation_flags(explicit_batch=True, strongly_typed=True)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_bytes):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError(
            "MiniMax-H3 audio VAE decoder ONNX parsing failed:\n" + "\n".join(errors)
        )

    if network.num_inputs != 1 or network.num_outputs != 1:
        raise RuntimeError(
            "MiniMax-H3 audio VAE decoder ONNX must expose exactly one input and one output"
        )
    input_tensor = network.get_input(0)
    output_tensor = network.get_output(0)
    input_contract = (input_tensor.name, tuple(input_tensor.shape), input_tensor.dtype)
    output_contract = (output_tensor.name, tuple(output_tensor.shape), output_tensor.dtype)
    expected_input = (
        "audio_latents",
        (AUDIO_BATCH, AUDIO_LATENT_CHANNELS, AUDIO_LATENT_FRAMES),
        trt.float32,
    )
    expected_output = ("waveform", (AUDIO_BATCH, 1, AUDIO_OUTPUT_SAMPLES), trt.float32)
    if input_contract != expected_input or output_contract != expected_output:
        raise RuntimeError(
            "MiniMax-H3 audio VAE decoder ONNX contract mismatch: "
            f"input={input_contract}, output={output_contract}"
        )

    build_config = builder.create_builder_config()
    resolved_workspace = resolve_workspace_bytes(
        workspace_bytes, default_bytes=AUDIO_VAE_DECODER_DEFAULT_WORKSPACE_BYTES
    )
    pool = trt.MemoryPoolType.WORKSPACE
    build_config.set_memory_pool_limit(pool, resolved_workspace)
    if int(build_config.get_memory_pool_limit(pool)) != resolved_workspace:
        raise RuntimeError("TensorRT did not apply the requested MiniMax-H3 audio workspace limit")
    if verbose:
        print(
            "[trtmc build]   Building complete MiniMax-H3 float32 audio VAE decoder "
            f"TensorRT engine ({AUDIO_SAMPLE_RATE} Hz) ...",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, build_config)
    if plan is None:
        raise RuntimeError("TensorRT MiniMax-H3 audio VAE decoder build failed")
    return bytes(plan)


def _build_serialized_encoder_engine(
    onnx_bytes: bytes,
    *,
    verbose: bool,
    workspace_bytes: int | None,
) -> bytes:
    trt = trt_compat.get_trt()
    logger = trt.Logger(trt.Logger.INFO if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        trt_compat.network_creation_flags(explicit_batch=True, strongly_typed=True)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_bytes):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError(
            "MiniMax-H3 audio VAE encoder ONNX parsing failed:\n" + "\n".join(errors)
        )

    if network.num_inputs != 1 or network.num_outputs != 1:
        raise RuntimeError(
            "MiniMax-H3 audio VAE encoder ONNX must expose exactly one input and one output"
        )
    input_tensor = network.get_input(0)
    output_tensor = network.get_output(0)
    input_contract = (input_tensor.name, tuple(input_tensor.shape), input_tensor.dtype)
    output_contract = (output_tensor.name, tuple(output_tensor.shape), output_tensor.dtype)
    expected_input = ("audio_samples", (AUDIO_BATCH, 1, -1), trt.float32)
    expected_output = (
        "audio_condition_rows",
        (AUDIO_BATCH, -1, AUDIO_LATENT_CHANNELS),
        trt.float32,
    )
    if input_contract != expected_input or output_contract != expected_output:
        raise RuntimeError(
            "MiniMax-H3 audio VAE encoder ONNX contract mismatch: "
            f"input={input_contract}, output={output_contract}"
        )

    build_config = builder.create_builder_config()
    resolved_workspace = resolve_workspace_bytes(
        workspace_bytes, default_bytes=AUDIO_VAE_ENCODER_DEFAULT_WORKSPACE_BYTES
    )
    pool = trt.MemoryPoolType.WORKSPACE
    build_config.set_memory_pool_limit(pool, resolved_workspace)
    if int(build_config.get_memory_pool_limit(pool)) != resolved_workspace:
        raise RuntimeError("TensorRT did not apply the requested MiniMax-H3 audio workspace limit")
    # Hugging Face keeps the Ref2VA audio encoder on the full-FP32 path.
    build_config.clear_flag(trt.BuilderFlag.TF32)

    profile = builder.create_optimization_profile()
    profile_shapes = (
        (AUDIO_BATCH, 1, AUDIO_REFERENCE_MIN_SAMPLES),
        (AUDIO_BATCH, 1, AUDIO_REFERENCE_OPT_SAMPLES),
        (AUDIO_BATCH, 1, AUDIO_REFERENCE_MAX_SAMPLES),
    )
    if profile.set_shape("audio_samples", *profile_shapes) is False:
        raise RuntimeError("TensorRT rejected the MiniMax-H3 audio encoder dynamic profile")
    if build_config.add_optimization_profile(profile) < 0:
        raise RuntimeError("TensorRT failed to add the MiniMax-H3 audio encoder dynamic profile")

    if verbose:
        print(
            "[trtmc build]   Building MiniMax-H3 float32 audio VAE reference encoder "
            "TensorRT engine (2-15 seconds at 32000 Hz) ...",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, build_config)
    if plan is None:
        raise RuntimeError("TensorRT MiniMax-H3 audio VAE encoder build failed")
    return bytes(plan)


def build_audio_vae_decoder_engine(
    audio_vae_dir: str | Path,
    *,
    verbose: bool = False,
    workspace_bytes: int | None = None,
) -> bytes:
    """Build normalized stereo audio latents -> clamped mono-batch waveform."""

    root = Path(audio_vae_dir)
    config = load_audio_vae_config(root)
    onnx_bytes = _export_decoder_onnx(root, config, verbose)
    try:
        return _build_serialized_engine(
            onnx_bytes,
            verbose=verbose,
            workspace_bytes=workspace_bytes,
        )
    finally:
        del onnx_bytes
        gc.collect()


def build_audio_vae_encoder_engine(
    audio_vae_dir: str | Path,
    *,
    verbose: bool = False,
    workspace_bytes: int | None = None,
) -> bytes:
    """Build the dynamic Ref2VA waveform -> normalized condition-row engine."""

    root = Path(audio_vae_dir)
    config = load_audio_vae_config(root)
    onnx_bytes = _export_encoder_onnx(root, config, verbose)
    try:
        return _build_serialized_encoder_engine(
            onnx_bytes,
            verbose=verbose,
            workspace_bytes=workspace_bytes,
        )
    finally:
        del onnx_bytes
        gc.collect()
