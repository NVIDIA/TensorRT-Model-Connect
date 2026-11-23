/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace trtmc::vibevoice_asr {

// Average interleaved channels into one mono track (finite samples only).
std::vector<float> downmix(const float* interleaved, std::size_t frames, std::uint32_t channels);

// Largest exact polyphase table the resampler precomputes, in coefficients.
inline constexpr std::size_t kMaxResampleTable = std::size_t{1} << 20;

// Length of resample()'s output for `samples` input samples,
// ceil(samples * to_rate / from_rate), computed without resampling so a request
// can be sized first. Throws std::invalid_argument for a rate that is not
// positive or a length beyond the int64 range.
std::size_t resampled_length(std::size_t samples, std::int32_t from_rate, std::int32_t to_rate);

// The taps [begin, end) of an output sample's tap list, whose first tap reads input
// sample `first`, that fall on the input samples [0, count). The resampler visits only
// these, so its work per output stays within the input even when the kernel is wider.
struct TapSpan {
    std::int64_t begin{0};
    std::int64_t end{0};
};
TapSpan input_taps(std::int64_t first, std::int64_t taps, std::int64_t count);

// Kaiser-windowed sinc polyphase resampler. The output holds
// ceil(input.size() * to_rate / from_rate) samples, the length librosa.resample
// returns, so the speech-token count and stated duration follow the reference.
// Rate pairs whose exact phase table (one row per output phase) would exceed
// `max_table_coefficients` interpolate a kernel table instead, so memory and
// setup stay bounded for any pair of rates (for example 192001 Hz). A resampled
// sample beyond the float range throws std::invalid_argument.
std::vector<float> resample(const std::vector<float>& input, std::int32_t from_rate,
                            std::int32_t to_rate,
                            std::size_t max_table_coefficients = kMaxResampleTable);

// VibeVoice AudioNormalizer: scale to target_dbfs RMS, then divide by the peak
// (plus eps) when it exceeds 1. Arithmetic follows the reference's float32 steps.
void normalize_loudness(std::vector<float>& samples, double target_dbfs, double eps);

} // namespace trtmc::vibevoice_asr
