/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <complex>
#include <cstdint>
#include <memory>
#include <vector>

namespace trtmc::moss_transcribe_diarize {

// The checkpoint feature extractor's filter bank, [freq_bins, mel_bins] row-major.
struct MelFilterBank {
    std::int32_t freq_bins{0};
    std::int32_t mel_bins{0};
    std::vector<float> weights;
};

MelFilterBank parse_mel_filter_bank(const std::vector<char>& bytes);

// Discrete Fourier transform of any length through Bluestein's algorithm on a
// power-of-two FFT, in double precision.
class Dft {
  public:
    explicit Dft(std::size_t length);
    std::size_t length() const { return length_; }
    // In place; `values` must hold length() elements.
    void forward(std::vector<std::complex<double>>& values) const;

  private:
    void fft(std::vector<std::complex<double>>& data, bool inverse) const;

    std::size_t length_;
    std::size_t padded_;
    std::vector<std::complex<double>> chirp_;
    std::vector<std::complex<double>> kernel_spectrum_;
    std::vector<std::complex<double>> twiddles_;
};

// Whisper log-mel features of one fixed-length chunk, as WhisperFeatureExtractor
// computes them for audio zero-padded to the chunk length: reflect-padded
// centred STFT with a periodic Hann window, power spectrum, mel projection,
// log10 with a 1e-10 floor, the last frame dropped, values clamped to the
// chunk maximum minus 8, then (x + 4) / 4. Arithmetic is double precision.
class LogMelExtractor {
  public:
    LogMelExtractor(MelFilterBank bank, std::int32_t n_fft, std::int32_t hop_length,
                    std::int32_t chunk_samples, std::int32_t mel_frames);

    // `count` samples (at most one chunk) -> [mel_bins, mel_frames] row-major.
    std::vector<float> chunk(const float* samples, std::int64_t count) const;

    std::int32_t mel_bins() const { return bank_.mel_bins; }
    std::int32_t mel_frames() const { return mel_frames_; }

  private:
    void frame_pair_power(const double* first, const double* second,
                          std::vector<std::complex<double>>& work, double* power_first,
                          double* power_second) const;

    MelFilterBank bank_;
    std::int32_t n_fft_;
    std::int32_t hop_length_;
    std::int32_t chunk_samples_;
    std::int32_t mel_frames_;
    Dft dft_;
    std::vector<double> window_;
    std::vector<std::int32_t> first_bin_;
    std::vector<std::int32_t> end_bin_;
};

} // namespace trtmc::moss_transcribe_diarize
