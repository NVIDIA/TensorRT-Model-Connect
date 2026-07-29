/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/pixart/pixart_generation_plan.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace trtmc {
namespace diffusion {

struct PixArtConditioningInputs {
    std::vector<float> encoder_attn_mask;
    std::vector<int32_t> null_ids;
};

struct PixArtTextConditioning {
    std::vector<float> text_projected;
    std::vector<float> null_text;
};

// Diffusers normalizes every PixArt prompt with lower().strip() even when the optional
// clean-caption dependencies are unavailable. Keep that baseline preprocessing in native C++.
inline std::string preprocess_pixart_prompt(const std::string& prompt) {
    const auto is_space = [](unsigned char value) { return std::isspace(value) != 0; };
    const auto first = std::find_if_not(prompt.begin(), prompt.end(), is_space);
    const auto last = std::find_if_not(prompt.rbegin(), prompt.rend(), is_space).base();
    if (first >= last)
        return {};

    std::string normalized(first, last);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return normalized;
}

// HF truncates prompt content while retaining the tokenizer's special suffix. The native
// tokenizer frames the full prompt first, so the PixArt runtime must apply the same ordering.
inline std::vector<int32_t>
normalize_pixart_t5_input_ids(const std::vector<int32_t>& input_ids, std::size_t sequence_length,
                              const std::vector<int32_t>& special_suffix_ids) {
    if (input_ids.size() <= sequence_length)
        return input_ids;

    const std::size_t suffix_length = std::min(sequence_length, special_suffix_ids.size());
    const std::size_t content_length = sequence_length - suffix_length;

    std::vector<int32_t> normalized;
    normalized.reserve(sequence_length);
    normalized.insert(normalized.end(), input_ids.begin(), input_ids.begin() + content_length);
    normalized.insert(normalized.end(), special_suffix_ids.end() - suffix_length,
                      special_suffix_ids.end());
    return normalized;
}

inline std::vector<float> make_pixart_t5_attention_mask(const std::vector<int32_t>& input_ids,
                                                        std::size_t sequence_length) {
    std::vector<float> mask(sequence_length, -1e9F);
    const std::size_t token_count = std::min(sequence_length, input_ids.size());
    for (std::size_t index = 0; index < token_count; ++index) {
        if (input_ids[index] != 0)
            mask[index] = 0.0F;
    }
    return mask;
}

inline std::vector<float> make_pixart_null_attention_mask(std::size_t sequence_length) {
    std::vector<float> mask(sequence_length, -10000.0F);
    if (!mask.empty()) {
        mask[0] = 0.0F;
    }
    return mask;
}

inline PixArtConditioningInputs
make_pixart_conditioning_inputs(const PixArtDiffusionConfig& config, const PixArtLayout& layout,
                                const std::vector<int32_t>& input_ids) {
    PixArtConditioningInputs inputs;
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
bool build_pixart_text_conditioning(const std::vector<int32_t>& input_ids,
                                    const PixArtConditioningInputs& inputs, int32_t seq_len,
                                    std::string& error, RunT5EncoderFn&& run_t5_encoder,
                                    ProjectTextFn&& project_text,
                                    PixArtTextConditioning& conditioning) {
    std::vector<float> text_embeddings;
    if (!run_t5_encoder(input_ids, text_embeddings, error))
        return false;
    project_text(text_embeddings, seq_len, conditioning.text_projected);

    std::vector<float> null_embeddings;
    if (!run_t5_encoder(inputs.null_ids, null_embeddings, error))
        return false;
    project_text(null_embeddings, seq_len, conditioning.null_text);
    return true;
}

inline std::vector<float> make_pixart_initial_latents(std::size_t latent_count,
                                                      uint32_t seed = 42U) {
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

inline bool resolve_pixart_initial_latents(std::size_t latent_count,
                                           const std::vector<float>& supplied_latents,
                                           int32_t requested_seed, std::vector<float>& latents,
                                           std::string& error) {
    if (!supplied_latents.empty()) {
        if (supplied_latents.size() != latent_count) {
            error = "PixArt initial latents contain " + std::to_string(supplied_latents.size()) +
                    " floats; expected " + std::to_string(latent_count);
            return false;
        }
        latents = supplied_latents;
        return true;
    }

    const auto seed = requested_seed >= 0 ? static_cast<uint32_t>(requested_seed) : 42U;
    latents = make_pixart_initial_latents(latent_count, seed);
    return true;
}

} // namespace diffusion
} // namespace trtmc
