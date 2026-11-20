/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace trtmc {
namespace canary {

namespace detail {

// Internal seam used by the focused frontend tests. Production callers should
// use extract_mel_spectrogram().
std::vector<float> rfft_power(const std::vector<float>& input);

} // namespace detail

struct MelResult {
    std::vector<float> data; // [n_mels, n_frames] row-major
    int32_t n_mels{0};
    int32_t n_frames{0};     // total frames (audio is chunk-padded, so this is the full length)
    int32_t valid_frames{0}; // frames covering the real (pre-chunk-padding) audio
};

class RangeRunner;

// A range of a shared frame pass holds at least this many frames, and a range
// of a shared log and normalization pass at least this many row values, so a
// short clip stays on the calling thread.
constexpr int32_t kCanaryMelMinFramesPerRange = 8;
constexpr int32_t kCanaryMelMinRowValuesPerRange = 2 * 1024;

// With a runner, the frames and then the mel rows are computed in ranges that
// the runner's threads take (canary_parallel.h). Every value is computed by
// the same statements however a pass is cut and whichever thread takes a
// range, so the result does not depend on `runner` or on the two shares.
MelResult
extract_mel_spectrogram(const float* samples, int32_t n_samples, const float* mel_filters,
                        int32_t n_freq_bins, int32_t n_mel_bins, int32_t n_fft, int32_t win_length,
                        int32_t hop_length, int32_t chunk_length_s, int32_t sample_rate,
                        float preemph, bool normalize_per_feature, RangeRunner* runner = nullptr,
                        int32_t min_frames_per_range = kCanaryMelMinFramesPerRange,
                        int32_t min_row_values_per_range = kCanaryMelMinRowValuesPerRange);

} // namespace canary
} // namespace trtmc
