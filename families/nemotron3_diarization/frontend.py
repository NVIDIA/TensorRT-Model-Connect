# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The constants of the log-mel front end, shipped with the bundle.

The reference feature extractor takes its filters from ``librosa.filters.mel``
with the Slaney scale and Slaney area normalisation, and its analysis window
from ``torch.hann_window(win_length, periodic=False)``. Both are computed here
with the same arithmetic, so the build needs neither library and the runtime
never re-derives a mel scale; ``tests/test_model.py`` holds them against the
originals.
"""

from __future__ import annotations

import numpy as np

_SLANEY_SP = 200.0 / 3.0
_SLANEY_MIN_LOG_HZ = 1000.0
_SLANEY_MIN_LOG_MEL = _SLANEY_MIN_LOG_HZ / _SLANEY_SP
_SLANEY_LOG_STEP = np.log(6.4) / 27.0


def _hz_to_mel(hz: float) -> float:
    if hz >= _SLANEY_MIN_LOG_HZ:
        return _SLANEY_MIN_LOG_MEL + np.log(hz / _SLANEY_MIN_LOG_HZ) / _SLANEY_LOG_STEP
    return hz / _SLANEY_SP


def _mel_to_hz(mels: np.ndarray) -> np.ndarray:
    hz = _SLANEY_SP * mels
    logarithmic = mels >= _SLANEY_MIN_LOG_MEL
    hz[logarithmic] = _SLANEY_MIN_LOG_HZ * np.exp(
        _SLANEY_LOG_STEP * (mels[logarithmic] - _SLANEY_MIN_LOG_MEL))
    return hz


def mel_filters(num_mel_bins: int, n_fft: int, sample_rate: int) -> np.ndarray:
    """``[num_mel_bins, n_fft // 2 + 1]`` float32 triangles over 0 Hz to Nyquist.

    The rounding follows librosa: the triangles are evaluated in float64 and
    stored as float32, then scaled by the float64 area normalisation.
    """
    bins = n_fft // 2 + 1
    frequencies = np.linspace(0.0, sample_rate / 2.0, bins)
    edges = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(sample_rate / 2.0),
                                   num_mel_bins + 2))
    widths = np.diff(edges)
    ramps = np.subtract.outer(edges, frequencies)
    weights = np.zeros((num_mel_bins, bins), dtype=np.float32)
    for index in range(num_mel_bins):
        lower = -ramps[index] / widths[index]
        upper = ramps[index + 2] / widths[index + 1]
        weights[index] = np.maximum(0.0, np.minimum(lower, upper))
    weights *= (2.0 / (edges[2:num_mel_bins + 2] - edges[:num_mel_bins]))[:, np.newaxis]
    return np.ascontiguousarray(weights)


def symmetric_hann(win_length: int) -> np.ndarray:
    """``torch.hann_window(win_length, periodic=False)`` in float32.

    Symmetric, not periodic: the window touches zero at both ends. The products
    and the cosine are taken in float32, the order torch evaluates them in.
    """
    positions = np.arange(win_length, dtype=np.float32)
    angles = positions * np.float32(2.0 * np.pi / (win_length - 1))
    return np.ascontiguousarray(np.float32(0.5) - np.float32(0.5) * np.cos(angles))
