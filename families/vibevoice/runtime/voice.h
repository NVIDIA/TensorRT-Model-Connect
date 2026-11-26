/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc::vibevoice {

struct WavAudio {
    std::vector<float> samples; // interleaved
    std::uint32_t sample_rate{0};
    std::uint32_t channels{0};
};

// PCM 16/24/32-bit integer or 32-bit float RIFF/WAVE, including WAVE_FORMAT_EXTENSIBLE.
// Reads only the chunk headers, the format and the samples: a file with more than
// ``max_samples`` samples (all channels) is refused before they are allocated, and a
// NaN or infinite float sample is refused.
WavAudio read_wav(const std::string& path, std::int64_t max_samples);

// Whether every one of `count` samples is a finite number.
bool all_finite(const float* samples, std::size_t count);

// The reference feature extractor's voice-prompt preparation: scale to the
// target RMS level, limit the peak to 1, then zero-pad to whole frames. Nonfinite
// samples are refused.
std::vector<float> prepare_voice(std::vector<float> samples, float target_dbfs, float eps,
                                 std::int64_t hop_length);

} // namespace trtmc::vibevoice
