# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-computed constants of the Granite Speech graphs (NumPy only).

Front end: the reference computes ``torchaudio.transforms.MelSpectrogram`` with its defaults (a
periodic Hann window zero-padded to ``n_fft``, centred frames with reflect padding, power 2, HTK
mel filters without normalisation). The graph evaluates the short-time transform as a strided
convolution with the windowed DFT basis of :func:`dft_basis` over the waveform padded by
:func:`reflect_pad`, then projects with :func:`mel_filterbank`.

Encoder: Shaw relative positions read ``E[c - r + max_pos_emb]``; :func:`relative_table` holds the
``2C - 1`` rows a block of ``C`` frames can reach. Decoder: :func:`rope_tables` are Granite's
default rotary ``cos``/``sin`` in FP32.
"""

from __future__ import annotations

import math

import numpy as np

from .config import FrontEnd, Text


def hann_window(win_length: int) -> np.ndarray:
    """Periodic Hann window (``torch.hann_window(win_length)``)."""
    n = np.arange(win_length, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * math.pi * n / win_length)


def padded_window(fe: FrontEnd) -> np.ndarray:
    """The window centred in ``n_fft`` samples, as ``torch.stft`` pads a shorter window."""
    left = (fe.n_fft - fe.win_length) // 2
    window = np.zeros(fe.n_fft, dtype=np.float64)
    window[left:left + fe.win_length] = hann_window(fe.win_length)
    return window


def dft_basis(fe: FrontEnd) -> np.ndarray:
    """``[2 * bins, n_fft]`` FP32: windowed cosine rows, then windowed sine rows (one-sided bins)."""
    bins = fe.n_fft // 2 + 1
    k = np.arange(bins, dtype=np.float64)[:, None]
    n = np.arange(fe.n_fft, dtype=np.float64)[None, :]
    angle = 2.0 * math.pi * ((k * n) % fe.n_fft) / fe.n_fft
    window = padded_window(fe)[None, :]
    return np.concatenate([np.cos(angle) * window, np.sin(angle) * window]).astype(np.float32)


def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(fe: FrontEnd) -> np.ndarray:
    """``[bins, n_mels]`` FP32 triangular HTK filters from 0 Hz to Nyquist, not area-normalised."""
    bins = fe.n_fft // 2 + 1
    freqs = np.linspace(0.0, fe.sample_rate // 2, bins, dtype=np.float64)
    points = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(fe.sample_rate / 2.0), fe.n_mels + 2))
    spacing = points[1:] - points[:-1]
    slopes = points[None, :] - freqs[:, None]
    down = -slopes[:, :-2] / spacing[:-1]
    up = slopes[:, 2:] / spacing[1:]
    return np.maximum(0.0, np.minimum(down, up)).astype(np.float32)


def reflect_pad(samples: np.ndarray, fe: FrontEnd) -> np.ndarray:
    """Centred-frame padding (``numpy`` "reflect", as ``torch.stft`` pads); the runtime does the same."""
    return np.pad(np.asarray(samples, dtype=np.float32), fe.n_fft // 2, mode="reflect")


def relative_table(embedding: np.ndarray, context: int, max_pos: int) -> np.ndarray:
    """``[D, 2C - 1]``: column ``j`` is ``E[max_pos + C - 1 - j]``, the row for offset ``c - r = C - 1 - j``."""
    rows = np.asarray(embedding)[max_pos + context - 1 - np.arange(2 * context - 1)]
    return np.ascontiguousarray(rows.T)


def rope_tables(text: Text, length: int) -> tuple[np.ndarray, np.ndarray]:
    """``cos``/``sin`` ``[length, head_dim]`` as Granite's default rotary embedding computes them in FP32."""
    dim = text.head_dim
    exponent = np.arange(0, dim, 2, dtype=np.int64).astype(np.float32) / np.float32(dim)
    inv_freq = (np.float32(1.0) / (np.float32(text.rope_theta) ** exponent)).astype(np.float32)
    freqs = np.arange(length, dtype=np.float32)[:, None] * inv_freq[None, :]
    angles = np.concatenate([freqs, freqs], axis=1).astype(np.float32)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)
