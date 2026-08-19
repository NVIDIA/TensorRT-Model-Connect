/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <memory>
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

namespace detail {

std::vector<float> rfft_power(const std::vector<float>& input);

} // namespace detail

struct IncrementalMelStats {
    int64_t accepted_source_samples{0};
    int64_t generated_resampled_samples{0};
    int64_t computed_mel_frames{0};
};

class IncrementalMelSpectrogram {
  public:
    IncrementalMelSpectrogram(const float* mel_filters, int32_t n_freq_bins, int32_t n_mel_bins,
                              MelSpectrogramOptions options, int32_t input_sample_rate);
    ~IncrementalMelSpectrogram();

    IncrementalMelSpectrogram(IncrementalMelSpectrogram&&) noexcept;
    IncrementalMelSpectrogram& operator=(IncrementalMelSpectrogram&&) noexcept;
    IncrementalMelSpectrogram(const IncrementalMelSpectrogram&) = delete;
    IncrementalMelSpectrogram& operator=(const IncrementalMelSpectrogram&) = delete;

    void accept_audio(const float* samples, int32_t n_samples);
    void ensure_frames(int32_t end_frame, bool final);
    void reset();

    int32_t available_frames() const;
    int32_t frame_count() const;
    int32_t n_mels() const;
    float value(int32_t mel_bin, int32_t frame) const;
    IncrementalMelStats stats() const;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
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
