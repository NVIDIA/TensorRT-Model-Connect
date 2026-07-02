/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace trtmc {

struct SpeechWaveformTrimResult {
    bool trimmed{false};
    std::size_t expected_samples{0};
};

inline SpeechWaveformTrimResult
trim_speech_waveform_to_generated_frames(int32_t sample_rate, float frame_rate,
                                         int32_t generated_frames, std::vector<float>& waveform) {
    SpeechWaveformTrimResult result;
    if (waveform.empty() || generated_frames <= 0 || frame_rate <= 0.0F) {
        return result;
    }

    const int32_t samples_per_frame =
        static_cast<int32_t>(std::lround(static_cast<float>(sample_rate) / frame_rate));
    if (samples_per_frame <= 0) {
        return result;
    }

    result.expected_samples =
        static_cast<std::size_t>(generated_frames) * static_cast<std::size_t>(samples_per_frame);
    if (result.expected_samples == 0 || result.expected_samples >= waveform.size()) {
        return result;
    }

    waveform.resize(result.expected_samples);
    result.trimmed = true;
    return result;
}

struct SpeechPeakNormalizeResult {
    bool normalized{false};
    float peak{0.0F};
    float scale{1.0F};
};

inline SpeechPeakNormalizeResult peak_normalize_speech_waveform(std::vector<float>& waveform,
                                                                float target_peak = 0.95F) {
    SpeechPeakNormalizeResult result;
    if (waveform.empty()) {
        return result;
    }

    for (float sample : waveform) {
        result.peak = std::max(result.peak, std::abs(sample));
    }
    if (result.peak <= 1.0F) {
        return result;
    }

    result.scale = target_peak / result.peak;
    for (float& sample : waveform) {
        sample *= result.scale;
    }
    result.normalized = true;
    return result;
}

} // namespace trtmc
