# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU integration checks for the Native loader's soxr-HQ input contract."""

from __future__ import annotations

import ctypes
from pathlib import Path
import shutil
import struct
import subprocess

import pytest


REPO = Path(__file__).resolve().parents[3]
CLI = r"""
#include "families/personaplex/runtime/resampler.h"
#include <fstream>
#include <iostream>
#include <vector>
int main(int argc, char** argv) {
    if (argc != 5) return 2;
    try {
        std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
        const auto bytes = input.tellg();
        std::vector<float> samples(static_cast<std::size_t>(bytes) / sizeof(float));
        input.seekg(0);
        input.read(reinterpret_cast<char*>(samples.data()), bytes);
        auto output = trtmc::resample_audio(samples.data(), samples.size(),
                                            std::stoi(argv[3]), std::stoi(argv[4]));
        std::ofstream file(argv[2], std::ios::binary);
        file.write(reinterpret_cast<const char*>(output.data()), output.size() * sizeof(float));
    } catch (const std::exception& error) {
        std::cerr << error.what();
        return 1;
    }
}
"""


@pytest.fixture(scope="module")
def resampler_cli(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("the CPU integration check requires a C++ compiler")
    work = tmp_path_factory.mktemp("personaplex-resampler")
    source, executable = work / "main.cpp", work / "resampler"
    source.write_text(CLI)
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Werror",
            "-I",
            str(REPO),
            str(source),
            str(REPO / "families/personaplex/runtime/resampler.cpp"),
            "-ldl",
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return executable


def run_resampler(executable, work, samples, source_rate, target_rate=24000):
    source, output = work / "input.f32", work / "output.f32"
    source.write_bytes(struct.pack(f"<{len(samples)}f", *samples))
    result = subprocess.run(
        [str(executable), str(source), str(output), str(source_rate), str(target_rate)],
        capture_output=True,
        text=True,
    )
    return result, output


def have_soxr():
    try:
        ctypes.CDLL("libsoxr.so.0")
    except OSError:
        return False
    return True


def require_soxr():
    if not have_soxr():
        pytest.skip("HQ waveform checks require the family runtime dependency libsoxr0")


def test_identity_and_invalid_rate(resampler_cli, tmp_path):
    samples = (0.25, -0.5, 0.75)
    result, output = run_resampler(resampler_cli, tmp_path, samples, 24000)
    assert result.returncode == 0, result.stderr
    assert output.read_bytes() == struct.pack("<3f", *samples)
    result, _ = run_resampler(resampler_cli, tmp_path, samples, 0)
    assert result.returncode != 0
    assert "positive rates" in result.stderr
    if not have_soxr():
        result, _ = run_resampler(resampler_cli, tmp_path, samples, 16000)
        assert result.returncode != 0
        assert "install libsoxr0" in result.stderr


def test_matches_native_soxr_hq(resampler_cli, tmp_path):
    require_soxr()
    samples = [0.0] * 64
    for index, value in ((0, 0.25), (8, -0.5), (31, 1.0), (63, 0.125)):
        samples[index] = value
    # librosa 0.11.0 / python-soxr 1.1.0, float32, res_type="soxr_hq".
    # Include start/end transients, interpolation phases, and interior samples.
    expected = {
        0: 0.25145572423934937,
        1: 0.1089545264840126,
        2: -0.061442479491233826,
        7: 0.0388762466609478,
        12: -0.486540824174881,
        13: -0.2189978063106537,
        46: 0.8075547218322754,
        47: 0.8009353876113892,
        48: 0.043039966374635696,
        80: -0.011325065046548843,
        94: 0.10438264906406403,
        95: 0.10148961842060089,
    }
    result, output = run_resampler(resampler_cli, tmp_path, samples, 16000)
    assert result.returncode == 0, result.stderr
    data = output.read_bytes()
    assert len(data) == 96 * 4
    values = struct.unpack("<96f", data)
    for index, value in expected.items():
        assert values[index] == pytest.approx(value, abs=1e-6, rel=0), index


@pytest.mark.parametrize(
    "rate,count,expected,padded",
    [
        (22050, 44100, 48001, True),
        (44101, 4411, 2401, True),
        (192001, 1, 1, True),
        (16000, 64, 96, False),
    ],
)
def test_librosa_length_and_zero_padding(resampler_cli, tmp_path, rate, count, expected, padded):
    require_soxr()
    result, output = run_resampler(resampler_cli, tmp_path, [0.25] * count, rate)
    assert result.returncode == 0, result.stderr
    data = output.read_bytes()
    assert len(data) == expected * 4
    assert (struct.unpack("<f", data[-4:])[0] == 0.0) == padded
