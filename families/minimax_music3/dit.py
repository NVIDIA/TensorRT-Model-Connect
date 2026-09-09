# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometry of the flow-matching diffusion transformer.

From the reference (diffusers v0.40.0,
``models/transformers/transformer_minimax_music3.py``, Apache-2.0,
(c) The MiniMax Team and The HuggingFace Team)::

    h = cat((latents, zeros_like(latents), condition.transpose(1, 2)), dim=1)
    h = preprocess_conv(h) + h                     residual, kernel 1, no bias
    h = proj_in(h.transpose(1, 2))
    h = cat((time_embed(fourier(t)).unsqueeze(1), h), dim=1)   time as a prefix
    thirty-six times:
        h = h + attn(layer_norm(h), partial_rope)
        first, second = ff_in(layer_norm(h)).chunk(2, -1)
        h = h + ff_out(first * silu(second))
    h = proj_out(h[:, 1:])                          the prefix is dropped
    h = postprocess_conv(h.transpose(1, 2)) + h     residual again

Four details do not survive a reading of the config alone, and each changes
the graph:

* The input width is ``2 * in_channels + condition_dim`` = 2304, because a
  zero block the size of the latents is concatenated between them. Text-to-
  music never fills it.
* Both stem convolutions are **residual**: ``conv(x) + x``, not ``conv(x)``.
* The timestep is a prefix **token**, not an added bias, and it is sliced off
  before ``proj_out``. Attention therefore runs over ``length + 1``.
* ``ff_in`` emits twice the inner width and the halves are used in the order
  ``first * silu(second)`` -- the opposite of the usual gate convention.

Only the first ``rotary_dim`` of each head rotates; the rest passes through.
"""

from __future__ import annotations

import math

IN_CHANNELS = 128
CONDITION_DIM = 2048
NUM_LAYERS = 36
NUM_ATTENTION_HEADS = 32
ATTENTION_HEAD_DIM = 64
FF_INNER_DIM = 8192
ROTARY_DIM = 32
FOURIER_EMBEDDING_DIM = 256
ROPE_THETA = 10000.0

INNER_DIM = NUM_ATTENTION_HEADS * ATTENTION_HEAD_DIM
CONCAT_CHANNELS = 2 * IN_CHANNELS + CONDITION_DIM


def attention_scale() -> float:
    return ATTENTION_HEAD_DIM ** -0.5


def sequence_length(latent_length: int) -> int:
    """Attention length: the latents plus the timestep prefix token."""

    if latent_length < 1:
        raise ValueError(f"latent_length must be positive, got {latent_length}")
    return latent_length + 1


def rotary_tables(seq_len: int):
    """Return ``(cos, sin)`` of shape ``(seq_len, rotary_dim)``.

    ``freqs`` is built over half the rotary width and then duplicated, which is
    why the tables are as wide as the rotated slice rather than half of it.
    """

    import numpy as np

    if seq_len < 1:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    exponents = np.arange(0, ROTARY_DIM, 2, dtype=np.float32) / ROTARY_DIM
    inv_freq = 1.0 / (ROPE_THETA ** exponents)
    steps = np.arange(seq_len, dtype=np.float32)
    freqs = np.outer(steps, inv_freq)
    freqs = np.concatenate((freqs, freqs), axis=-1)
    return (
        np.ascontiguousarray(np.cos(freqs), dtype=np.float32),
        np.ascontiguousarray(np.sin(freqs), dtype=np.float32),
    )


def apply_partial_rope(hidden, cos, sin):
    """Rotate the leading ``rotary_dim`` of each head and pass the rest."""

    import numpy as np

    x = np.asarray(hidden, dtype=np.float32)
    rotary_dim = np.shape(cos)[-1]
    rotated = x[..., :rotary_dim]
    first, second = np.split(rotated, 2, axis=-1)
    rotate_half = np.concatenate((-second, first), axis=-1)
    cos = np.asarray(cos, dtype=np.float32)[:, None, :]
    sin = np.asarray(sin, dtype=np.float32)[:, None, :]
    rotated = rotated * cos + rotate_half * sin
    return np.concatenate((rotated, x[..., rotary_dim:]), axis=-1)


def fourier_features(timestep, weight):
    """``cat(cos(2 pi t w), sin(2 pi t w))`` -- the trained Fourier projection."""

    import numpy as np

    t = np.asarray(timestep, dtype=np.float32).reshape(-1, 1)
    w = np.asarray(weight, dtype=np.float32).reshape(-1, 1)
    angles = 2.0 * math.pi * (t @ w.T)
    return np.concatenate((np.cos(angles), np.sin(angles)), axis=-1)


def fourier_weight_rows() -> int:
    """Rows of ``time_proj.weight``: half the embedding, cos and sin doubling it."""

    return FOURIER_EMBEDDING_DIM // 2
