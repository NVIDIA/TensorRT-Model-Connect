/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace trtmc {
namespace rnnt {

struct MelResult {
    std::vector<float> data; // [n_mels, n_frames] row-major
    int32_t n_mels{0};
    int32_t n_frames{0};
};

enum class MelLogScale {
    kLog10Normalized,
    kNaturalLog,
};

struct MelSpectrogramOptions {
    int32_t n_fft{400};
    int32_t win_length{0};
    int32_t hop_length{160};
    int32_t chunk_length_s{30};
    int32_t sample_rate{16000};
    bool symmetric_window{false};
    bool center_window_in_fft{false};
    float preemphasis{0.0F};
    MelLogScale log_scale{MelLogScale::kLog10Normalized};
    bool normalize_per_feature{false};
};

MelResult extract_configured_mel_spectrogram(const float* samples, int32_t n_samples,
                                             const float* mel_filters, int32_t n_freq_bins,
                                             int32_t n_mel_bins,
                                             const MelSpectrogramOptions& options);

MelResult extract_rnnt_mel_spectrogram(const float* samples, int32_t n_samples,
                                       const float* mel_filters, int32_t n_freq_bins,
                                       int32_t n_mel_bins, int32_t n_fft, int32_t win_length,
                                       int32_t hop_length, int32_t chunk_length_s,
                                       int32_t sample_rate, float preemphasis);

} // namespace rnnt
} // namespace trtmc
