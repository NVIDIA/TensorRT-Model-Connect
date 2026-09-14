/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "trtmc/task.h"

#include <nlohmann/json.hpp>
#include <stdexcept>

inline nlohmann::json audio_observation(const trtmc::AudioResult& result) {
    const auto count = result.samples.size();
    if (result.num_samples < 0 ||
        (result.num_samples != 0 && static_cast<std::size_t>(result.num_samples) != count))
        throw std::runtime_error("audio benchmark: sample count does not match buffer");
    if (result.sample_rate <= 0 || result.num_channels <= 0)
        throw std::runtime_error("audio benchmark: sample rate and channel count must be positive");
    if (count % result.num_channels != 0)
        throw std::runtime_error("audio benchmark: incomplete interleaved audio frame");
    return {{"output_samples", count},
            {"num_samples", count},
            {"output_audio_seconds",
             static_cast<double>(count) / result.num_channels / result.sample_rate},
            {"sample_rate", result.sample_rate},
            {"num_channels", result.num_channels}};
}
