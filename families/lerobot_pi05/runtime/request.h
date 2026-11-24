/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "trtmc/internal/model.h"

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace trtmc::lerobot_pi05 {
struct NoiseOptions {
    bool has_seed{false};
    std::uint64_t seed{0};
    std::vector<float> noise; // Empty unless supplied explicitly.
};
inline const internal::ConfigField kFields[] = {
    {"seed", internal::ConfigKind::I64, std::nullopt,
     "Seed for the initial flow noise. Omit for a fresh random draw on every call."},
    {"noise", internal::ConfigKind::F64List, std::nullopt,
     "Explicit initial noise of chunk_size x max_action_dim values, row-major. It overrides "
     "seed; use it to reproduce a reference run."},
};

inline NoiseOptions parse_options(internal::ConfigView config, std::size_t expected) {
    NoiseOptions options;
    for (const auto& entry : config) {
        if (entry.name == "seed") {
            const auto seed = std::get<std::int64_t>(entry.value);
            if (seed < 0)
                throw std::invalid_argument("lerobot_pi05 seed must not be negative");
            options.has_seed = true;
            options.seed = static_cast<std::uint64_t>(seed);
        } else if (entry.name == "noise") {
            const auto values = std::get<Span<const double>>(entry.value);
            if (values.size() != expected)
                throw std::invalid_argument("lerobot_pi05 noise must have chunk_size x "
                                            "max_action_dim values");
            options.noise.reserve(expected);
            for (const double value : values) {
                if (!std::isfinite(value))
                    throw std::invalid_argument("lerobot_pi05 noise must be finite");
                options.noise.push_back(static_cast<float>(value));
            }
        }
    }
    return options;
}

inline std::vector<float> initial_noise(const NoiseOptions& options, std::size_t size,
                                        std::mt19937_64& rng) {
    if (!options.noise.empty())
        return options.noise;
    std::vector<float> noise(size);
    std::mt19937_64 seeded(options.seed);
    auto& engine = options.has_seed ? seeded : rng;
    std::normal_distribution<float> normal(0.0F, 1.0F);
    for (float& value : noise)
        value = normal(engine);
    return noise;
}

} // namespace trtmc::lerobot_pi05
