/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/lerobot_groot/runtime/tokenizer.h"
#include "trtmc/internal/action.h"

#include <random>
#include <string>
#include <vector>

namespace trtmc::lerobot_groot {

struct NoiseOptions {
    bool has_seed{false};
    std::uint64_t seed{0};
    std::vector<float> noise;
};
struct RgbFrame {
    const std::uint8_t* borrowed{nullptr};
    std::vector<std::uint8_t> converted;
    const std::uint8_t* data() const { return borrowed ? borrowed : converted.data(); }
};

Span<const internal::ConfigField> action_fields();
NoiseOptions parse_noise(internal::ConfigView config, std::size_t count);
std::vector<float> initial_noise(const NoiseOptions& options, std::size_t count,
                                 std::mt19937_64& rng);
std::vector<RgbFrame> ordered_rgb(const internal::ImagesStateTextObservation& observation,
                                  const std::vector<std::string>& cameras, std::uint32_t size);
void validate_state(Span<const float> state, std::size_t width);
std::vector<std::int32_t> instruction_tokens(std::string_view text, bool formalize,
                                             const ITokenizer& tokenizer, std::int32_t limit);

} // namespace trtmc::lerobot_groot
