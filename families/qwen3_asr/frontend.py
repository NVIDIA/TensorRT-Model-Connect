# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audio front-end constants and encoder geometry of Qwen3-ASR (NumPy only).

The values follow ``Qwen3ASRFeatureExtractor``, ``Qwen3ASRProcessor`` and ``Qwen3ASREncoder`` in
Hugging Face Transformers 5.22.2:

* log-mel features with 128 Slaney mel bins over 0-8 kHz, a 400-point periodic Hann window and a
  160-sample hop; ``samples // 160`` frames per clip;
* clips shorter than ``min_length`` samples are zero-padded first, and the padding counts as audio;
* the mel axis is cut into chunks of ``2 * n_window`` frames; three stride-2 convolutions turn a
  chunk into ``post_cnn_length(frames)`` audio tokens (13 for a full chunk of 100 frames);
* encoder attention runs inside windows of ``n_window_infer / (2 * n_window)`` chunks.

The native runtime computes the features itself; the bundle carries the filterbank built here.
"""

from __future__ import annotations

import numpy as np

# The processor's (code, name) table: the forced-language prefill names the language in full.
LANGUAGES: tuple[tuple[str, str], ...] = (
    ("ar", "Arabic"),
    ("yue", "Cantonese"),
    ("zh", "Chinese"),
    ("cs", "Czech"),
    ("da", "Danish"),
    ("nl", "Dutch"),
    ("en", "English"),
    ("fil", "Filipino"),
    ("fi", "Finnish"),
    ("fr", "French"),
    ("de", "German"),
    ("el", "Greek"),
    ("hi", "Hindi"),
    ("hu", "Hungarian"),
    ("id", "Indonesian"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("mk", "Macedonian"),
    ("ms", "Malay"),
    ("fa", "Persian"),
    ("pl", "Polish"),
    ("pt", "Portuguese"),
    ("ro", "Romanian"),
    ("ru", "Russian"),
    ("es", "Spanish"),
    ("sv", "Swedish"),
    ("th", "Thai"),
    ("tr", "Turkish"),
    ("vi", "Vietnamese"),
)


def post_cnn_length(frames: int) -> int:
    """Tokens three ``kernel 3, stride 2, padding 1`` convolutions keep from ``frames`` frames."""
    if frames < 0:
        raise ValueError("frame count must be non-negative")
    for _ in range(3):
        frames = (frames - 1) // 2 + 1 if frames > 0 else 0
    return frames


def mel_frame_count(samples: int, *, hop_length: int, min_samples: int) -> int:
    """Mel frames of a clip of ``samples`` samples after the short-clip padding."""
    if samples <= 0:
        raise ValueError("a clip needs at least one sample")
    return max(samples, min_samples) // hop_length


def audio_token_count(mel_frames: int, *, chunk_frames: int) -> int:
    """Audio tokens (``<|audio_pad|>`` placeholders) for ``mel_frames`` valid mel frames."""
    full, rest = divmod(mel_frames, chunk_frames)
    return full * post_cnn_length(chunk_frames) + post_cnn_length(rest)


def chunk_token_counts(mel_frames: int, *, chunk_frames: int) -> list[int]:
    """Valid tokens of every chunk; only the last chunk can be partial."""
    if mel_frames <= 0:
        raise ValueError("a clip needs at least one mel frame")
    chunks = -(-mel_frames // chunk_frames)
    counts = [post_cnn_length(chunk_frames)] * (chunks - 1)
    counts.append(post_cnn_length(mel_frames - (chunks - 1) * chunk_frames))
    return counts


def _hertz_to_mel_slaney(frequency: np.ndarray | float) -> np.ndarray | float:
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = 27.0 / np.log(6.4)
    mels = 3.0 * frequency / 200.0
    if isinstance(frequency, np.ndarray):
        log_region = frequency >= min_log_hertz
        mels[log_region] = min_log_mel + np.log(frequency[log_region] / min_log_hertz) * logstep
    elif frequency >= min_log_hertz:
        mels = min_log_mel + np.log(frequency / min_log_hertz) * logstep
    return mels


def _mel_to_hertz_slaney(mels: np.ndarray) -> np.ndarray:
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = np.log(6.4) / 27.0
    frequency = 200.0 * mels / 3.0
    log_region = mels >= min_log_mel
    frequency[log_region] = min_log_hertz * np.exp(logstep * (mels[log_region] - min_log_mel))
    return frequency


def slaney_mel_filters(
    *, num_frequency_bins: int, num_mel_filters: int, max_frequency: float, sampling_rate: int
) -> np.ndarray:
    """Slaney-scale, Slaney-normalised triangular filters ``[num_frequency_bins, num_mel_filters]``.

    A transcription of ``transformers.audio_utils.mel_filter_bank(..., norm="slaney",
    mel_scale="slaney")`` with ``min_frequency = 0``; computed in float64 and returned as float32,
    which is how the feature extractor applies it.
    """
    if num_frequency_bins < 2 or num_mel_filters < 1 or max_frequency <= 0.0:
        raise ValueError("invalid mel filterbank geometry")
    mel_min = _hertz_to_mel_slaney(0.0)
    mel_max = _hertz_to_mel_slaney(float(max_frequency))
    mel_frequencies = np.linspace(mel_min, mel_max, num_mel_filters + 2)
    filter_frequencies = _mel_to_hertz_slaney(mel_frequencies)
    fft_frequencies = np.linspace(0, sampling_rate // 2, num_frequency_bins)
    filter_diff = np.diff(filter_frequencies)
    slopes = np.expand_dims(filter_frequencies, 0) - np.expand_dims(fft_frequencies, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    filters = np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))
    enorm = 2.0 / (filter_frequencies[2 : num_mel_filters + 2] - filter_frequencies[:num_mel_filters])
    filters *= np.expand_dims(enorm, 0)
    return np.ascontiguousarray(filters.astype(np.float32))


def sinusoid_positions(length: int, channels: int, max_timescale: float = 10000.0) -> np.ndarray:
    """``SinusoidsPositionEmbedding`` table ``[length, channels]``: sines, then cosines, in float32."""
    if channels % 2 or channels < 4 or length < 1:
        raise ValueError("sinusoid table needs an even channel count of at least 4")
    increment = np.float32(-np.log(max_timescale) / (channels // 2 - 1))
    inverse = np.exp(increment * np.arange(channels // 2, dtype=np.float32)).astype(np.float32)
    scaled = np.arange(length, dtype=np.float32)[:, None] * inverse[None, :]
    return np.ascontiguousarray(
        np.concatenate([np.sin(scaled), np.cos(scaled)], axis=1).astype(np.float32)
    )
