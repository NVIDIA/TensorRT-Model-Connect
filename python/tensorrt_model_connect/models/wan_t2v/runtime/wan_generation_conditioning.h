/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "wan_generation_plan.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace trtmc {
namespace diffusion {

struct WanConditioningInputs {
    std::vector<float> encoder_attn_mask;
    std::vector<int32_t> null_ids;
};

struct WanTextConditioning {
    std::vector<float> text_projected;
    std::vector<float> null_text;
};

inline bool wan_conditioning_values_are_finite(const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
}

inline std::vector<int32_t> normalize_wan_t5_token_ids(std::vector<int32_t> ids, int32_t max_len,
                                                       bool native_special_frame = true) {
    constexpr int32_t kT5NativePrefixTokenId = 2;
    constexpr int32_t kT5EosTokenId = 1;
    if (native_special_frame && !ids.empty() && ids.front() == kT5NativePrefixTokenId)
        ids.erase(ids.begin());
    if (ids.empty() || ids.back() != kT5EosTokenId)
        ids.push_back(kT5EosTokenId);
    if (max_len <= 0)
        return {};
    if (static_cast<int32_t>(ids.size()) > max_len) {
        ids.resize(static_cast<std::size_t>(max_len));
        ids.back() = kT5EosTokenId;
    }
    return ids;
}

inline WanConditioningInputs make_wan_conditioning_inputs(const WanDiffusionConfig& config,
                                                          const WanLayout& layout,
                                                          const std::vector<int32_t>& input_ids) {
    WanConditioningInputs inputs;
    inputs.null_ids.assign(static_cast<std::size_t>(layout.seq_len), 0);
    if (!inputs.null_ids.empty())
        inputs.null_ids[0] = 1;

    if (!config.use_rope) {
        inputs.encoder_attn_mask.assign(static_cast<std::size_t>(layout.seq_len), -10000.0F);
        for (std::size_t index = 0;
             index < input_ids.size() && index < static_cast<std::size_t>(layout.seq_len);
             ++index) {
            if (input_ids[index] != 0)
                inputs.encoder_attn_mask[index] = 0.0F;
        }
        if (!input_ids.empty() && !inputs.encoder_attn_mask.empty())
            inputs.encoder_attn_mask[0] = 0.0F;
    }

    return inputs;
}

template <typename RunT5EncoderFn, typename ProjectTextFn>
bool build_wan_text_conditioning(const std::vector<int32_t>& input_ids,
                                 const WanConditioningInputs& inputs, int32_t seq_len,
                                 std::string& error, RunT5EncoderFn&& run_t5_encoder,
                                 ProjectTextFn&& project_text, WanTextConditioning& conditioning) {
    std::vector<float> text_embeddings;
    if (!run_t5_encoder(input_ids, text_embeddings, error))
        return false;
    if (!wan_conditioning_values_are_finite(text_embeddings)) {
        error = "Wan prompt T5 embeddings contain non-finite values";
        return false;
    }
    project_text(text_embeddings, seq_len, conditioning.text_projected);
    if (!wan_conditioning_values_are_finite(conditioning.text_projected)) {
        error = "Wan projected prompt conditioning contains non-finite values";
        return false;
    }

    std::vector<float> null_embeddings;
    if (!run_t5_encoder(inputs.null_ids, null_embeddings, error))
        return false;
    if (!wan_conditioning_values_are_finite(null_embeddings)) {
        error = "Wan null-prompt T5 embeddings contain non-finite values";
        return false;
    }
    project_text(null_embeddings, seq_len, conditioning.null_text);
    if (!wan_conditioning_values_are_finite(conditioning.null_text)) {
        error = "Wan projected null-prompt conditioning contains non-finite values";
        return false;
    }
    return true;
}

inline std::vector<float> make_wan_initial_latents(std::size_t latent_count, uint32_t seed = 42U) {
    constexpr double kPi = 3.14159265358979323846;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<float> latents(latent_count, 0.0F);
    for (std::size_t index = 0; index < latents.size(); index += 2) {
        double u1 = dist(rng);
        double u2 = dist(rng);
        if (u1 < 1e-12)
            u1 = 1e-12;
        const double radius = std::sqrt(-2.0 * std::log(u1));
        const double theta = 2.0 * kPi * u2;
        latents[index] = static_cast<float>(radius * std::cos(theta));
        if (index + 1 < latents.size())
            latents[index + 1] = static_cast<float>(radius * std::sin(theta));
    }
    return latents;
}

inline bool resolve_wan_initial_latents(std::size_t latent_count,
                                        const std::vector<float>& supplied_latents,
                                        int32_t requested_seed, std::vector<float>& latents,
                                        std::string& error) {
    if (!supplied_latents.empty()) {
        if (supplied_latents.size() != latent_count) {
            error = "Wan initial latent count mismatch: expected " + std::to_string(latent_count) +
                    ", got " + std::to_string(supplied_latents.size());
            return false;
        }
        latents = supplied_latents;
        return true;
    }

    const uint32_t seed = requested_seed >= 0 ? static_cast<uint32_t>(requested_seed) : 42U;
    latents = make_wan_initial_latents(latent_count, seed);
    return true;
}

} // namespace diffusion
} // namespace trtmc
