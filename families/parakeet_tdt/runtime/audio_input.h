/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/parakeet_tdt/runtime/resampler.h"
#include "trtmc/internal/audio.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace trtmc::tdt {

inline std::vector<float> prepare_audio(const internal::SpeechTranscriptionRequest& request) {
    const auto& audio = request.audio;
    if (request.source_language.has_value())
        throw std::invalid_argument("Parakeet TDT does not support source_language control");
    if (audio.channels == 0 || audio.samples.empty() || audio.samples.data() == nullptr ||
        audio.samples.size() % audio.channels != 0)
        throw std::invalid_argument(
            "Parakeet TDT requires complete nonempty interleaved PCM frames");
    const auto rate = audio.sample_rate.value_or(16000);
    if (rate == 0 || rate > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()))
        throw std::invalid_argument("Parakeet TDT input sample rate is out of range");
    const auto frames = audio.samples.size() / audio.channels;
    if (frames > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
        frames > static_cast<std::uint64_t>(rate) * 30)
        throw std::invalid_argument("Parakeet TDT supports at most 30 seconds per request");

    std::vector<float> mono(frames);
    for (std::size_t frame = 0; frame < frames; ++frame) {
        double sum = 0;
        for (std::uint32_t channel = 0; channel < audio.channels; ++channel) {
            const auto value = audio.samples[frame * audio.channels + channel];
            if (!std::isfinite(value))
                throw std::invalid_argument("Parakeet TDT PCM must contain only finite samples");
            sum += value;
        }
        mono[frame] = static_cast<float>(sum / audio.channels);
    }
    if (rate == 16000)
        return mono;
    auto result = resample_linear(mono.data(), static_cast<std::int32_t>(frames),
                                  static_cast<std::int32_t>(rate), 16000);
    if (result.empty())
        throw std::invalid_argument("Parakeet TDT input is shorter than one model-rate sample");
    return result;
}

} // namespace trtmc::tdt
