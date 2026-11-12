/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace trtmc::voxtral {

// The checkpoint's WhisperFeatureExtractor contract, read from runtime.json.
struct FrontendConfig {
    std::int32_t sample_rate{16000};
    std::int64_t chunk_samples{480000};
    std::int32_t n_fft{400};
    std::int32_t hop_length{160};
    std::int32_t num_mel_bins{128};
    std::int32_t mel_frames_per_chunk{3000};
};

// Complex DFT of a fixed length whose prime factors are 2, 3 and 5.
class Fft {
  public:
    explicit Fft(std::int32_t length);
    // out[k] = sum_j in[j] * exp(-2 pi i j k / length) for k in [0, length).
    void forward(const std::complex<double>* input, std::complex<double>* output);
    std::int32_t length() const noexcept { return length_; }

  private:
    void transform(const std::complex<double>* input, std::size_t stride, std::size_t size,
                   std::complex<double>* output);
    std::int32_t length_;
    std::vector<std::complex<double>> twiddles_;
    std::vector<std::complex<double>> scratch_;
};

// Interleaved PCM to one channel by averaging the channels of every frame.
std::vector<float> downmix(const float* samples, std::size_t count, std::uint32_t channels);

// Whole 30 s chunks needed to cover `samples` (at least one).
std::int64_t chunk_count(std::size_t samples, std::int64_t chunk_samples);

// Log-mel features of the zero-padded clip, laid out [chunks, mels, frames per chunk]:
// center (reflect) padded STFT with a periodic Hann window, power spectrum,
// `filters` ([n_fft / 2 + 1, mels]), log10 clamped at 1e-10, floored at the
// clip-wide maximum minus 8, then (x + 4) / 4. The STFT's last frame is dropped.
std::vector<float> log_mel_features(const std::vector<float>& mono,
                                    const std::vector<float>& filters,
                                    const FrontendConfig& config);

} // namespace trtmc::voxtral
