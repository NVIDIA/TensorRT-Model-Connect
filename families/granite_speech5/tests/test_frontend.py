# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Front-end constants and the graph's front-end arithmetic against GraniteSpeech5FeatureExtractor.

``numpy_features`` computes, step by step, what ``encoder.add_front_end`` builds into the engine,
from the same constants and the same padded waveform; ``test_graph.py`` checks the engine itself.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from families.granite_speech5.config import FrontEnd
from families.granite_speech5.constants import (
    delta_weights,
    dft_basis,
    hann_window,
    mel_filterbank,
    pad_waveform,
    padded_window,
)

DATA = Path(__file__).resolve().parent / "data"
FRONT_END = FrontEnd(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=80, delta_win_length=3,
                     floor_db=8.0)
# Largest difference to the extractor over the fixtures and noise clips below: 1.4e-5 (CPU, float64
# against torchaudio's float32 FFT). Without the tail sample the last frames move by 0.18 or more.
FEATURE_TOLERANCE = 1e-3


def numpy_features(samples: np.ndarray, fe: FrontEnd, *, tail: bool = True) -> np.ndarray:
    """The graph's front end in float64: DFT basis, power, mel, log10 floor, deltas, pair stacking."""
    padded = pad_waveform(samples, fe) if tail else np.pad(np.asarray(samples, np.float32), fe.n_fft // 2,
                                                             mode="reflect")
    frames = 1 + (padded.shape[0] - fe.n_fft) // fe.hop_length
    windows = np.lib.stride_tricks.sliding_window_view(padded.astype(np.float64), fe.n_fft)[::fe.hop_length][:frames]
    spectrum = windows @ dft_basis(fe).astype(np.float64).T
    power = spectrum[:, :fe.bins] ** 2 + spectrum[:, fe.bins:] ** 2
    power = power[: frames // 2 * 2]
    mel = power @ mel_filterbank(fe).astype(np.float64)
    log10 = np.log(np.maximum(mel, 1.0e-10)) / math.log(10.0)
    logmel = np.maximum(log10, log10.max() - fe.floor_db) * 0.25 + 1.0
    taps, denominator = delta_weights(fe)
    last = logmel.shape[0] - 1
    index = np.arange(logmel.shape[0])
    delta = sum(tap * (logmel[np.minimum(index + tap, last)] - logmel[np.maximum(index - tap, 0)]) for tap in taps)
    frame = np.concatenate([logmel, delta / denominator], axis=1)
    return frame.reshape(-1, 4 * fe.n_mels)


def _extractor():
    from transformers import GraniteSpeech5FeatureExtractor

    return GraniteSpeech5FeatureExtractor(num_mel_bins=80, sampling_rate=16000, n_fft=512, win_length=400,
                                          hop_length=160, delta_win_length=3, logmel_floor_db=8.0)


def _reference(samples: np.ndarray) -> np.ndarray:
    return _extractor()(samples, sampling_rate=16000, return_tensors="np")["input_features"][0]


def _clips() -> dict[str, np.ndarray]:
    import soundfile

    clips = {}
    for path in sorted(DATA.glob("*.wav")):
        audio, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
        assert rate == 16000 and audio.shape[1] == 1
        clips[path.stem] = np.ascontiguousarray(audio[:, 0])
    rng = np.random.default_rng(7)
    # An odd number of whole hops (the tail sample), an even one, and lengths between hops.
    for length in (112160, 7 * 160, 8 * 160, 16000 + 37, 33 * 160 + 159):
        clips[f"noise-{length}"] = (0.1 * rng.standard_normal(length)).astype(np.float32)
    return clips


def test_window_and_filterbank_match_torchaudio():
    import torch
    import torchaudio

    window = torch.hann_window(400, dtype=torch.float64).numpy()
    assert np.allclose(padded_window(FRONT_END)[56:456], window, atol=1e-12)
    assert not padded_window(FRONT_END)[:56].any() and not padded_window(FRONT_END)[456:].any()
    reference = torchaudio.functional.melscale_fbanks(257, 0.0, 8000.0, 80, 16000, norm=None, mel_scale="htk")
    # torchaudio evaluates the filters in float32 (5.3e-6 from the float64 ones, rounded once).
    assert np.abs(mel_filterbank(FRONT_END) - reference.numpy()).max() < 1e-5


def test_numpy_front_end_matches_the_extractor():
    for name, samples in _clips().items():
        reference = _reference(samples)
        features = numpy_features(samples, FRONT_END)
        assert features.shape == reference.shape, name
        assert np.abs(features - reference).max() < FEATURE_TOLERANCE, name


@pytest.mark.parametrize("length", [7 * 160, 112160])
def test_the_tail_sample_is_needed(length):
    """Without the reference's zero tail sample the last frames differ beyond the tolerance."""
    samples = (0.1 * np.random.default_rng(length).standard_normal(length)).astype(np.float32)
    assert FRONT_END.needs_tail_sample(length)
    reference = _reference(samples)
    assert np.abs(numpy_features(samples, FRONT_END) - reference).max() < FEATURE_TOLERANCE
    assert np.abs(numpy_features(samples, FRONT_END, tail=False) - reference).max() > 10 * FEATURE_TOLERANCE


def test_waveform_padding_matches_numpy_reflect():
    samples = np.arange(7 * 160, dtype=np.float32)
    padded = pad_waveform(samples, FRONT_END)
    expected = np.pad(np.concatenate([samples, np.zeros(1, np.float32)]), 256, mode="reflect")
    assert np.array_equal(padded, expected)
    assert np.array_equal(pad_waveform(samples[:-1], FRONT_END), np.pad(samples[:-1], 256, mode="reflect"))


@pytest.mark.parametrize("length", [1, 2, 3, 400, 512])
def test_hann_window_matches_torch_at_every_length(length):
    import torch

    assert np.allclose(hann_window(length), torch.hann_window(length, dtype=torch.float64).numpy(), atol=1e-12)


def test_degenerate_mel_filters_are_refused():
    # A 1 Hz clock has no frequency range: every mel point coincides and the filters divide by zero.
    degenerate = FrontEnd(sample_rate=1, n_fft=512, win_length=400, hop_length=160, n_mels=80,
                          delta_win_length=3, floor_db=8.0)
    with pytest.raises(ValueError, match="mel filters"):
        mel_filterbank(degenerate)
