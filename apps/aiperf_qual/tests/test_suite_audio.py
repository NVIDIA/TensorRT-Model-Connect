# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A suite resamples dataset audio once, so the native and TRTMC sides read the same samples."""

import base64
import io

import numpy as np
import pytest

from trtmc_aiperf_qual.config import ConfigError
from trtmc_aiperf_qual.suites import _hf_rows, _wav_media

soundfile = pytest.importorskip("soundfile")
pytest.importorskip("scipy")


def _encoded(audio, rate):
    buffer = io.BytesIO()
    soundfile.write(buffer, audio, rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _decoded(media):
    assert media["$file"]["suffix"] == ".wav"
    return soundfile.read(io.BytesIO(base64.b64decode(media["$file"]["b64"])), dtype="float32")


def test_audio_is_resampled_once_to_the_suite_rate():
    tone = 0.5 * np.sin(2 * np.pi * 440 * np.arange(16_000) / 16_000)
    audio, rate = _decoded(_wav_media(_encoded(tone, 16_000), 24_000))
    assert rate == 24_000
    assert audio.shape == (24_000,)
    expected = 0.5 * np.sin(2 * np.pi * 440 * np.arange(24_000) / 24_000)
    np.testing.assert_allclose(audio[200:-200], expected[200:-200], atol=2e-3)


@pytest.mark.parametrize("sample_rate", [None, 16_000])
def test_audio_keeps_its_samples_without_a_different_suite_rate(sample_rate):
    source = (np.arange(-800, 800) * 16).astype(np.float32) / 32768  # exact PCM16 values
    audio, rate = _decoded(_wav_media(_encoded(source, 16_000), sample_rate))
    assert rate == 16_000
    np.testing.assert_array_equal(audio, source)


@pytest.mark.parametrize("sample_rate", [0, -24_000, 24_000.0, "24000", True])
def test_invalid_suite_rates_are_rejected_before_the_dataset_loads(sample_rate):
    with pytest.raises(ConfigError, match="audio_sample_rate"):
        _hf_rows("dataset", object(), "revision", {"audio_sample_rate": sample_rate}, {})
