# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent acoustic input checks before complete-checkpoint qualification."""

from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from families.nemotron_omni.preprocessing import mel_filters


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("Owning C++ frontend requires a C++ compiler")
    family = Path(__file__).parents[1]
    executable = tmp_path_factory.mktemp("omni-audio") / "probe"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(family / "runtime"),
            str(family / "runtime/audio_frontend.cpp"),
            str(family / "tests/cpp/audio_frontend_probe.cpp"),
            "-o",
            str(executable),
        ],
        check=True,
    )
    return executable


def execute(probe, tmp_path, samples, maximum):
    wave, filters = tmp_path / "wave.f32", tmp_path / "filters.f32"
    np.asarray(samples, np.float32).tofile(wave)
    mel_filters().tofile(filters)
    return subprocess.run([str(probe), str(wave), str(filters), str(maximum)], capture_output=True)


def test_filterbank_matches_independent_librosa():
    librosa = pytest.importorskip("librosa")
    reference = librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, norm="slaney")
    np.testing.assert_array_equal(mel_filters(), reference)


@pytest.mark.parametrize("length", [320, 1280, 16017, 32411])
def test_audio_features_match_hf(probe, tmp_path, length):
    pytest.importorskip("torch")
    pytest.importorskip("librosa")
    from transformers.models.parakeet.feature_extraction_parakeet import ParakeetFeatureExtractor

    waveform = np.random.default_rng(17).normal(0, 0.2, length).astype(np.float32)
    reference = ParakeetFeatureExtractor(feature_size=128)(
        waveform, sampling_rate=16000, return_tensors="np"
    )
    expected = reference["input_features"][0, reference["attention_mask"][0].astype(bool)]
    result = execute(probe, tmp_path, waveform, len(expected))
    assert result.returncode == 0, result.stderr.decode()
    actual = np.frombuffer(result.stdout, np.float32).reshape(-1, 128)
    assert actual.shape == (length // 160, 128)
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)


def test_silence_is_finite_zero_without_padding(probe, tmp_path):
    result = execute(probe, tmp_path, np.zeros(16017), 100)
    assert result.returncode == 0, result.stderr.decode()
    np.testing.assert_array_equal(np.frombuffer(result.stdout, np.float32), np.zeros(12800))


@pytest.mark.parametrize(
    "samples,maximum", [(np.zeros(319), 100), (np.zeros(16000), 99), (np.full(1600, np.nan), 100)]
)
def test_invalid_audio_is_rejected_not_truncated(probe, tmp_path, samples, maximum):
    result = execute(probe, tmp_path, samples, maximum)
    assert result.returncode != 0
    assert not result.stdout
    assert b"Omni audio requires" in result.stderr
