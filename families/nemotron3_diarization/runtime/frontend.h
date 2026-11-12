/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace trtmc::nemotron3_diarization {

// The log-mel front end of the reference feature extractor, with its window
// and mel filters computed by the family's Python side and carried in the
// bundle. Nothing here normalises the features: the reference does not.
struct FrontEndConfig {
    std::int32_t sample_rate{16000};
    std::int32_t n_fft{512};
    std::int32_t win_length{400};
    std::int32_t hop_length{160};
    std::int32_t num_mel_bins{128};
    float preemphasis{0.97F};
    float log_guard{5.9604644775390625e-08F}; // 2**-24
    std::vector<float> window;                // [win_length], symmetric Hann
    std::vector<float> mel_filters;           // [num_mel_bins, n_fft / 2 + 1], row-major
};

// Row-major [frames, num_mel_bins]. The analysis windows are centred and the
// signal zero-padded by n_fft / 2 at both ends, so there are 1 + L / hop frames
// for L samples, of which the first L / hop are valid; the rest are zero.
struct LogMel {
    std::vector<float> values;
    std::int64_t frames{0};
    std::int64_t valid_frames{0};
};

class LogMelFrontEnd {
  public:
    explicit LogMelFrontEnd(FrontEndConfig config);

    LogMel compute(const float* mono, std::int64_t count) const;
    const FrontEndConfig& config() const noexcept { return config_; }

  private:
    void power_spectrum(const std::vector<float>& padded, std::int64_t start,
                        std::vector<std::complex<double>>& buffer, std::vector<float>& power) const;

    FrontEndConfig config_;
    std::vector<float> padded_window_;          // [n_fft], the window centred in the transform
    std::vector<std::size_t> bit_reversed_;     // [n_fft]
    std::vector<std::complex<double>> twiddle_; // [n_fft / 2]
    std::vector<std::size_t> filter_begin_, filter_end_; // nonzero bins of each mel filter
};

// Interleaved channels averaged to mono, as the reference feature extractor
// does for multichannel input. One channel is copied through unchanged.
std::vector<float> mix_to_mono(const float* samples, std::size_t count, std::uint32_t channels);

} // namespace trtmc::nemotron3_diarization
