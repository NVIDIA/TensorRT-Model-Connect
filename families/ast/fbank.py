# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The constants AST's Kaldi filter-bank frontend needs, built once at bundle time.

The runtime does framing, a windowed real FFT, one matrix multiply and a log.
Everything that is pure arithmetic over the configuration - the analysis window
and the triangular mel filters - is computed here and shipped in the bundle, so
the mel scale never has to be spelled out twice.

``tests/test_model.py`` checks both against ``torchaudio.compliance.kaldi``,
which is what the reference feature extractor calls.
"""

from __future__ import annotations

import numpy as np


def mel_scale(frequency):
    """Kaldi's mel scale. HTK's constants, not Slaney's."""
    return 1127.0 * np.log1p(np.asarray(frequency, dtype=np.float64) / 700.0)


def hann_window(frame_length: int) -> np.ndarray:
    """The symmetric Hann window, which is what Kaldi calls ``hanning``.

    Not ``numpy.hanning``'s periodic sibling and not the periodic window a
    spectrogram would normally use: the endpoint is included. The periodic
    form moves the normalised features by 0.086, which is small enough to look
    plausible and large enough to be wrong.
    """
    index = np.arange(frame_length, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * index / (frame_length - 1))).astype(np.float32)


def mel_filters(num_mel_bins: int, n_fft: int, sample_rate: float,
                low_freq: float = 20.0, high_freq: float = 0.0) -> np.ndarray:
    """Kaldi's triangular mel filter bank, shaped ``[num_mel_bins, n_fft // 2 + 1]``.

    Kaldi builds the triangles over ``n_fft // 2`` bins and never gives the
    Nyquist bin any weight, so the last column is zero by construction rather
    than by truncation. A non-positive ``high_freq`` is an offset from Nyquist,
    which is how the reference spells "up to Nyquist".
    """
    num_fft_bins = n_fft // 2
    nyquist = 0.5 * sample_rate
    if high_freq <= 0.0:
        high_freq += nyquist
    if not 0.0 <= low_freq < high_freq <= nyquist:
        raise ValueError(f"ast mel band [{low_freq}, {high_freq}] does not fit under {nyquist} Hz")

    fft_bin_width = sample_rate / n_fft
    mel_low, mel_high = mel_scale(low_freq), mel_scale(high_freq)
    delta = (mel_high - mel_low) / (num_mel_bins + 1)

    left = (mel_low + np.arange(num_mel_bins, dtype=np.float64) * delta)[:, None]
    centre, right = left + delta, left + 2.0 * delta
    bins = mel_scale(fft_bin_width * np.arange(num_fft_bins, dtype=np.float64))[None, :]

    up = (bins - left) / (centre - left)
    down = (right - bins) / (right - centre)
    triangles = np.maximum(0.0, np.minimum(up, down))
    return np.pad(triangles, ((0, 0), (0, 1))).astype(np.float32)
