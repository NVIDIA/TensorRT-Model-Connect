/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen_image2_1/runtime/sampling.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>

namespace trtmc::qwen_image2_1 {

std::vector<float> flow_match_sigmas(const RuntimeConfig& config, std::int32_t num_steps) {
    if (num_steps < 1)
        throw std::invalid_argument("Qwen-Image 2.1 needs at least one denoising step");
    const auto count = static_cast<std::size_t>(num_steps);

    // mu grows linearly with the image token count between the two configured anchors.
    const double slope = (config.max_shift - config.base_shift) /
                         static_cast<double>(config.max_image_seq_len - config.base_image_seq_len);
    const double mu = config.base_shift + slope * static_cast<double>(config.image_tokens() -
                                                                      config.base_image_seq_len);
    const double shift = std::exp(mu);

    std::vector<double> sigmas(count);
    const double last = 1.0 / static_cast<double>(num_steps);
    for (std::size_t index = 0; index < count; ++index) {
        const double fraction =
            count == 1 ? 0.0 : static_cast<double>(index) / static_cast<double>(count - 1);
        const double sigma = 1.0 + fraction * (last - 1.0);
        sigmas[index] = shift / (shift + (1.0 / sigma - 1.0));
    }
    if (config.shift_terminal != 0.0) {
        // A single step ends at sigma 1, which leaves the stretch undefined (0 / 0).
        const double scale = (1.0 - sigmas.back()) / (1.0 - config.shift_terminal);
        if (scale <= 0.0)
            throw std::invalid_argument("Qwen-Image 2.1 needs at least two denoising steps");
        for (auto& sigma : sigmas)
            sigma = 1.0 - (1.0 - sigma) / scale;
    }

    std::vector<float> result(count + 1, 0.0F);
    std::transform(sigmas.begin(), sigmas.end(), result.begin(),
                   [](double sigma) { return static_cast<float>(sigma); });
    return result;
}

float round_to_bfloat16(float value) {
    if (!std::isfinite(value))
        return value;
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t dropped = bits & 0xFFFFU;
    bits &= 0xFFFF0000U;
    // Sign-magnitude storage: rounding the magnitude up is adding one unit in the last kept place.
    if (dropped > 0x8000U || (dropped == 0x8000U && (bits & 0x10000U) != 0))
        bits += 0x10000U;
    std::memcpy(&value, &bits, sizeof(bits));
    return value;
}

float reference_timestep(float sigma, std::int32_t num_train_timesteps) {
    const auto scale = static_cast<float>(num_train_timesteps);
    return round_to_bfloat16(round_to_bfloat16(sigma * scale) / scale);
}

void euler_step(std::vector<float>& latents, const std::vector<float>& velocity, float sigma,
                float sigma_next) {
    if (latents.size() != velocity.size())
        throw std::invalid_argument("Qwen-Image 2.1 velocity does not match the latents");
    const float delta = sigma_next - sigma;
    for (std::size_t index = 0; index < latents.size(); ++index)
        latents[index] += delta * velocity[index];
}

std::vector<float> gaussian_latents(std::size_t count, std::uint64_t seed) {
    std::mt19937_64 generator(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<float> latents(count);
    for (auto& value : latents)
        value = normal(generator);
    return latents;
}

std::vector<float> pack_latents(const float* chw, std::int32_t channels, std::int32_t height,
                                std::int32_t width) {
    const auto plane = static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
    std::vector<float> tokens(plane * static_cast<std::size_t>(channels));
    for (std::size_t channel = 0; channel < static_cast<std::size_t>(channels); ++channel) {
        for (std::size_t token = 0; token < plane; ++token)
            tokens[token * static_cast<std::size_t>(channels) + channel] =
                chw[channel * plane + token];
    }
    return tokens;
}

std::vector<float> unpack_latents(const std::vector<float>& tokens, std::int32_t channels,
                                  std::int32_t height, std::int32_t width) {
    const auto plane = static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
    if (tokens.size() != plane * static_cast<std::size_t>(channels))
        throw std::invalid_argument("Qwen-Image 2.1 latent tokens have an unexpected size");
    std::vector<float> chw(tokens.size());
    for (std::size_t channel = 0; channel < static_cast<std::size_t>(channels); ++channel) {
        for (std::size_t token = 0; token < plane; ++token)
            chw[channel * plane + token] =
                tokens[token * static_cast<std::size_t>(channels) + channel];
    }
    return chw;
}

std::vector<float> image_to_hwc_unit(const float* chw, std::int32_t channels, std::int32_t height,
                                     std::int32_t width) {
    const auto plane = static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
    std::vector<float> hwc(plane * static_cast<std::size_t>(channels));
    for (std::size_t channel = 0; channel < static_cast<std::size_t>(channels); ++channel) {
        for (std::size_t pixel = 0; pixel < plane; ++pixel) {
            const float unit = chw[channel * plane + pixel] * 0.5F + 0.5F;
            hwc[pixel * static_cast<std::size_t>(channels) + channel] =
                std::clamp(unit, 0.0F, 1.0F);
        }
    }
    return hwc;
}

} // namespace trtmc::qwen_image2_1
