/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace trtmc::flux3 {

struct TextConfig {
    std::string prompt_prefix;
    std::string prompt_suffix;
    std::int32_t pad_token_id{0};
    std::int32_t pad_multiple{0};
    std::int32_t max_tokens{0};
    std::int32_t embed_dim{0};
};

struct ScheduleConfig {
    std::int32_t default_steps{0};
    double shift{0.0};
};

struct TimestepEmbeddingConfig {
    std::int32_t dim{0};
    std::int32_t max_period{0};
};

// The bundle's runtime.json (format flux3-runtime-v1) after validation.
struct RuntimeConfig {
    std::int32_t height{0};
    std::int32_t width{0};
    std::int32_t latent_height{0};
    std::int32_t latent_width{0};
    std::int32_t latent_channels{0};
    TextConfig text;
    ScheduleConfig schedule;
    TimestepEmbeddingConfig timestep_embedding;
};

inline constexpr std::int32_t kMaxSteps = 1000;

// Throws std::runtime_error naming the first missing, mistyped or
// inconsistent field. Keys the runtime does not use are ignored.
RuntimeConfig parse_runtime_config(std::string_view text);

} // namespace trtmc::flux3
