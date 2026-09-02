# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometry and weight preparation for the vocoder.

A DAC-style decoder: latents in, stereo waveform out. From the reference
(diffusers v0.40.0, ``models/autoencoders/minimax_music3_vocoder.py``,
Apache-2.0, (c) The MiniMax Team and The HuggingFace Team)::

    latents (b, 128, L) -> reshape (b*2, 64, L)      the two channels fold here
      dec_in_proj   Conv1d 64 -> 1024, k=1           not weight-normed
      conv_in       Conv1d 1024 -> 1536, k=7, pad=3
      four blocks, each: Snake -> ConvTranspose -> three residual units
      snake_out, conv_out Conv1d 96 -> 1, k=7, pad=3
      tanh
    -> reshape (b, 2, samples)

Stereo is two mono streams through one set of weights, which is why
``dec_in_proj`` takes 64 channels while the config declares 128, and why
``conv_out`` emits one.

Every convolution but ``dec_in_proj`` is weight-normalised, stored as a
``weight_g`` magnitude and a ``weight_v`` direction. :func:`fuse_weight_norm`
collapses them, because TensorRT has no weight-norm layer and the product is
constant once the checkpoint is loaded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

LATENT_CHANNELS = 128
DECODER_INPUT_DIM = 1024
DECODER_HIDDEN_DIM = 1536
UPSAMPLING_RATIOS = (8, 8, 4, 2)
SAMPLING_RATE = 44100
OUTPUT_CHANNELS = 1

#: The two audio channels are decoded as two folded streams.
STREAMS = 2
STREAM_CHANNELS = LATENT_CHANNELS // STREAMS

#: Residual units inside every block, by dilation.
RESIDUAL_DILATIONS = (1, 3, 9)
RESIDUAL_KERNEL = 7

CONV_IN_KERNEL = 7
CONV_OUT_KERNEL = 7


@dataclass(frozen=True)
class Block:
    """One upsampling block."""

    index: int
    input_dim: int
    output_dim: int
    stride: int

    @property
    def kernel_size(self) -> int:
        return 2 * self.stride

    @property
    def padding(self) -> int:
        return math.ceil(self.stride / 2)


def blocks() -> tuple[Block, ...]:
    """Return the four blocks, with the widths the reference derives."""

    return tuple(
        Block(
            index=index,
            input_dim=DECODER_HIDDEN_DIM // (2**index),
            output_dim=DECODER_HIDDEN_DIM // (2 ** (index + 1)),
            stride=stride,
        )
        for index, stride in enumerate(UPSAMPLING_RATIOS)
    )


def upsample_factor() -> int:
    """Waveform samples produced per latent frame."""

    factor = 1
    for stride in UPSAMPLING_RATIOS:
        factor *= stride
    return factor


def block_output_length(length: int, stride: int) -> int:
    """Length after one transposed convolution.

    ``(L - 1) * stride - 2 * ceil(stride / 2) + 2 * stride`` reduces to
    ``L * stride`` for every stride the checkpoint uses, so the block chain is
    an exact integer upsample with no trimming.
    """

    padding = math.ceil(stride / 2)
    return (length - 1) * stride - 2 * padding + 2 * stride


def waveform_samples(latent_length: int) -> int:
    """Samples one window of ``latent_length`` latent frames decodes to."""

    length = latent_length
    for stride in UPSAMPLING_RATIOS:
        length = block_output_length(length, stride)
    return length


def residual_padding(dilation: int) -> int:
    """Padding that keeps a dilated residual convolution length-preserving."""

    return (RESIDUAL_KERNEL - 1) * dilation // 2


def fuse_weight_norm(weight_g, weight_v):
    """Return the effective convolution weight of a weight-normalised layer.

    ``weight = g * v / ||v||`` with the norm taken over every dimension but
    the first, which is how ``torch.nn.utils.weight_norm`` parameterises a
    convolution.
    """

    import numpy as np

    magnitude = np.asarray(weight_g, dtype=np.float32)
    direction = np.asarray(weight_v, dtype=np.float32)
    axes = tuple(range(1, direction.ndim))
    norm = np.sqrt(np.sum(direction * direction, axis=axes, keepdims=True))
    magnitude = magnitude.reshape(-1, *([1] * (direction.ndim - 1)))
    return np.ascontiguousarray(magnitude * direction / norm, dtype=np.float32)


def snake(hidden, alpha):
    """The Snake activation: ``x + sin(alpha * x) ** 2 / (alpha + 1e-9)``."""

    import numpy as np

    x = np.asarray(hidden, dtype=np.float32)
    a = np.asarray(alpha, dtype=np.float32)
    return x + np.sin(a * x) ** 2 / (a + 1e-9)
