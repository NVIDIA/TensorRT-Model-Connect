/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace trtmc {

struct MimiEncodePlan {
    bool input_fits{false};
    int32_t valid_frames{0};
};

constexpr int32_t kMimiFrontendTokensPerFrame = 2;
constexpr int32_t kMimiAttentionContext = 250;
constexpr float kMimiAttentionMaskPenalty = -1.0e10F;

struct MimiRingAttentionInputs {
    std::array<int32_t, kMimiFrontendTokensPerFrame> position_ids{};
    std::array<int32_t, kMimiFrontendTokensPerFrame> cache_indices{};
    std::array<float, kMimiFrontendTokensPerFrame * kMimiAttentionContext> mask{};
};

inline MimiEncodePlan build_mimi_encode_plan(int32_t input_samples, int32_t engine_input_samples,
                                             int32_t engine_frames) {
    MimiEncodePlan plan;
    if (input_samples <= 0 || engine_input_samples <= 0 || engine_frames <= 0 ||
        input_samples > engine_input_samples) {
        return plan;
    }

    plan.input_fits = true;
    const auto numerator = static_cast<int64_t>(input_samples) * engine_frames;
    plan.valid_frames = static_cast<int32_t>(
        (numerator + static_cast<int64_t>(engine_input_samples) - 1) / engine_input_samples);
    plan.valid_frames = std::clamp(plan.valid_frames, 1, engine_frames);
    return plan;
}

inline MimiRingAttentionInputs build_mimi_ring_attention_inputs(int32_t frame) {
    MimiRingAttentionInputs inputs;
    std::fill(inputs.mask.begin(), inputs.mask.end(), kMimiAttentionMaskPenalty);

    const int32_t position = frame * kMimiFrontendTokensPerFrame;
    const int32_t end_offset = position + kMimiFrontendTokensPerFrame;
    const int32_t end_index = end_offset % kMimiAttentionContext;
    for (int32_t query = 0; query < kMimiFrontendTokensPerFrame; ++query) {
        const int32_t query_position = position + query;
        inputs.position_ids[static_cast<std::size_t>(query)] = query_position;
        inputs.cache_indices[static_cast<std::size_t>(query)] =
            query_position % kMimiAttentionContext;
        for (int32_t column = 0; column < kMimiAttentionContext; ++column) {
            const bool invalid = column >= end_offset;
            const int32_t ring_delta = column - end_index;
            const int32_t key_position =
                invalid ? -1
                        : end_offset + ring_delta - (ring_delta > 0 ? kMimiAttentionContext : 0);
            const int32_t delta = query_position - key_position;
            if (key_position >= 0 && delta >= 0 && delta < kMimiAttentionContext) {
                inputs.mask[static_cast<std::size_t>(query) * kMimiAttentionContext + column] =
                    0.0F;
            }
        }
    }
    return inputs;
}

} // namespace trtmc
