/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace trtmc {
namespace voicechat_audio {

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

int32_t reflect_index(int32_t index, int32_t length);

} // namespace detail

class IncrementalMelSpectrogram {
  public:
    IncrementalMelSpectrogram(const float* mel_filters, int32_t n_freq_bins, int32_t n_mel_bins,
                              MelSpectrogramOptions options, int32_t input_sample_rate,
                              const float* exact_window, int32_t exact_window_length);
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

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace voicechat_audio
} // namespace trtmc
