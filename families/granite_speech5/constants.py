# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-computed constants of the Granite Speech 5 graph (NumPy only).

Front end: the reference computes ``torchaudio.transforms.MelSpectrogram`` with its defaults (a
periodic Hann window zero-padded to ``n_fft``, centred frames with reflect padding, power 2, HTK
mel filters from 0 Hz to ``sample_rate // 2`` without normalisation). The graph evaluates the
short-time transform as a strided convolution with the windowed DFT basis of :func:`dft_basis`
over the waveform the runtime pads (:func:`pad_waveform`), then projects with
:func:`mel_filterbank`.

Encoder: Shaw relative positions add ``q_i · E[i - j + max_position_embeddings]`` inside a block;
:func:`relative_table` holds the ``2C - 1`` rows a block of ``C`` frames reaches.
"""

from __future__ import annotations

import math

import numpy as np

from .config import FrontEnd


def hann_window(win_length: int) -> np.ndarray:
    """Periodic Hann window (``torch.hann_window(win_length)``, which is ``[1]`` for one sample)."""
    if win_length == 1:
        return np.ones(1, dtype=np.float64)
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
    k = np.arange(fe.bins, dtype=np.float64)[:, None]
    n = np.arange(fe.n_fft, dtype=np.float64)[None, :]
    angle = 2.0 * math.pi * ((k * n) % fe.n_fft) / fe.n_fft
    window = padded_window(fe)[None, :]
    return np.concatenate([np.cos(angle) * window, np.sin(angle) * window]).astype(np.float32)


def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(fe: FrontEnd) -> np.ndarray:
    """``[bins, n_mels]`` FP32 triangular HTK filters from 0 Hz to ``sample_rate // 2``, not area-normalised."""
    nyquist = float(fe.sample_rate // 2)
    freqs = np.linspace(0.0, nyquist, fe.bins, dtype=np.float64)
    points = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(nyquist), fe.n_mels + 2))
    spacing = points[1:] - points[:-1]
    slopes = points[None, :] - freqs[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        down = -slopes[:, :-2] / spacing[:-1]
        up = slopes[:, 2:] / spacing[1:]
        bank = np.maximum(0.0, np.minimum(down, up)).astype(np.float32)
    if not np.all(np.isfinite(bank)):  # coinciding mel points (a zero-width filter) divide by zero
        raise ValueError(f"granite_speech5 mel filters for {fe.n_mels} bins up to {nyquist} Hz are not finite")
    return bank


def delta_weights(fe: FrontEnd) -> tuple[list[int], float]:
    """``torchaudio.functional.compute_deltas`` taps ``1..n`` and its denominator ``n(n + 1)(2n + 1) / 3``."""
    n = (fe.delta_win_length - 1) // 2
    return list(range(1, n + 1)), n * (n + 1) * (2 * n + 1) / 3


def pad_waveform(samples: np.ndarray, fe: FrontEnd) -> np.ndarray:
    """The engine's input: the clip, the reference's zero tail sample if it needs one, then reflect padding.

    The runtime pads the same way (``ctc.cpp``).
    """
    clip = np.asarray(samples, dtype=np.float32)
    if fe.needs_tail_sample(int(clip.shape[0])):
        clip = np.concatenate([clip, np.zeros(1, dtype=np.float32)])
    return np.pad(clip, fe.n_fft // 2, mode="reflect")


def relative_table(embedding: np.ndarray, context: int, max_positions: int) -> np.ndarray:
    """``[D, 2C - 1]``: column ``m`` is ``E[max_positions + C - 1 - m]``, the row of offset ``i - j = C - 1 - m``."""
    rows = np.asarray(embedding)[max_positions + context - 1 - np.arange(2 * context - 1)]
    return np.ascontiguousarray(rows.T)
