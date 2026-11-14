/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Every operation here mirrors one rounded float32 (or float64) operation of
// the reference; the file is compiled without floating-point contraction.

#include "families/chroma/runtime/sampling.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>

namespace trtmc::chroma {

float round_to_bfloat16(float value) {
    if (std::isnan(value))
        return value;
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t rounding = 0x7FFFU + ((bits >> 16) & 1U);
    bits = (bits + rounding) & 0xFFFF0000U;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::vector<float> flow_match_sigmas(std::int32_t steps, float shift) {
    if (steps < 1)
        throw std::invalid_argument("Chroma needs at least one denoising step");
    const double start = 1.0;
    const double stop = 1.0 / static_cast<double>(steps);
    std::vector<float> sigmas(static_cast<std::size_t>(steps) + 1, 0.0F);
    const double step = steps > 1 ? (stop - start) / static_cast<double>(steps - 1) : 0.0;
    for (std::int32_t index = 0; index < steps; ++index) {
        double value = static_cast<double>(index) * step;
        value = value + start;
        if (index == steps - 1 && steps > 1)
            value = stop;
        const auto sigma = static_cast<float>(value);
        const float numerator = shift * sigma;
        const float scaled = (shift - 1.0F) * sigma;
        const float denominator = 1.0F + scaled;
        sigmas[static_cast<std::size_t>(index)] = numerator / denominator;
    }
    return sigmas;
}

float conditioned_timestep(float timestep, bool bfloat16) {
    if (!bfloat16) {
        const float divided = timestep / 1000.0F;
        return divided * 1000.0F;
    }
    const float stored = round_to_bfloat16(timestep);
    const float divided = round_to_bfloat16(stored / 1000.0F);
    return round_to_bfloat16(divided * 1000.0F);
}

std::array<float, 16> timestep_features(float timestep) {
    std::array<float, 16> features{};
    constexpr int kHalf = 8;
    const float log_period = static_cast<float>(std::log(10000.0));
    for (int index = 0; index < kHalf; ++index) {
        const float exponent =
            (-log_period * static_cast<float>(index)) / static_cast<float>(kHalf);
        const float frequency = std::exp(exponent);
        const float argument = timestep * frequency;
        features[static_cast<std::size_t>(index)] = std::cos(argument);
        features[static_cast<std::size_t>(index + kHalf)] = std::sin(argument);
    }
    return features;
}

std::vector<float> denoiser_text_mask(const std::vector<std::int32_t>& tokenizer_mask) {
    std::size_t valid = 0;
    for (const auto value : tokenizer_mask)
        valid += value != 0 ? 1U : 0U;
    std::vector<float> mask(tokenizer_mask.size(), 0.0F);
    for (std::size_t index = 0; index < mask.size() && index <= valid; ++index)
        mask[index] = 1.0F;
    return mask;
}

std::vector<float> pack_latents(const std::vector<float>& latents, const LatentShape& shape) {
    if (latents.size() != shape.count() || shape.height % 2 != 0 || shape.width % 2 != 0)
        throw std::invalid_argument("Chroma latents do not match the bundle latent shape");
    const auto half_h = shape.height / 2;
    const auto half_w = shape.width / 2;
    std::vector<float> packed(latents.size());
    std::size_t out = 0;
    for (std::int32_t row = 0; row < half_h; ++row)
        for (std::int32_t column = 0; column < half_w; ++column)
            for (std::int32_t channel = 0; channel < shape.channels; ++channel)
                for (std::int32_t dy = 0; dy < 2; ++dy)
                    for (std::int32_t dx = 0; dx < 2; ++dx) {
                        const auto y = static_cast<std::size_t>(2 * row + dy);
                        const auto x = static_cast<std::size_t>(2 * column + dx);
                        packed[out++] = latents[(static_cast<std::size_t>(channel) *
                                                     static_cast<std::size_t>(shape.height) +
                                                 y) *
                                                    static_cast<std::size_t>(shape.width) +
                                                x];
                    }
    return packed;
}

std::vector<float> unpack_latents(const std::vector<float>& packed, const LatentShape& shape) {
    if (packed.size() != shape.count() || shape.height % 2 != 0 || shape.width % 2 != 0)
        throw std::invalid_argument("Chroma packed latents do not match the bundle latent shape");
    const auto half_h = shape.height / 2;
    const auto half_w = shape.width / 2;
    std::vector<float> latents(packed.size());
    std::size_t in = 0;
    for (std::int32_t row = 0; row < half_h; ++row)
        for (std::int32_t column = 0; column < half_w; ++column)
            for (std::int32_t channel = 0; channel < shape.channels; ++channel)
                for (std::int32_t dy = 0; dy < 2; ++dy)
                    for (std::int32_t dx = 0; dx < 2; ++dx) {
                        const auto y = static_cast<std::size_t>(2 * row + dy);
                        const auto x = static_cast<std::size_t>(2 * column + dx);
                        latents[(static_cast<std::size_t>(channel) *
                                     static_cast<std::size_t>(shape.height) +
                                 y) *
                                    static_cast<std::size_t>(shape.width) +
                                x] = packed[in++];
                    }
    return latents;
}

std::vector<float> initial_noise(std::uint32_t seed, const LatentShape& shape) {
    std::mt19937 generator(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<float> latents(shape.count());
    for (auto& value : latents)
        value = normal(generator);
    return pack_latents(latents, shape);
}

namespace {

float model_type(float value, bool bfloat16) {
    return bfloat16 ? round_to_bfloat16(value) : value;
}

} // namespace

void to_model_type(std::vector<float>& values, bool bfloat16) {
    if (!bfloat16)
        return;
    for (auto& value : values)
        value = round_to_bfloat16(value);
}

void apply_guidance(std::vector<float>& positive, const std::vector<float>& negative, float scale,
                    bool bfloat16) {
    if (positive.size() != negative.size())
        throw std::invalid_argument("Chroma guidance predictions differ in size");
    for (std::size_t index = 0; index < positive.size(); ++index) {
        const float difference = model_type(positive[index] - negative[index], bfloat16);
        const float scaled = model_type(scale * difference, bfloat16);
        positive[index] = model_type(negative[index] + scaled, bfloat16);
    }
}

void euler_step(std::vector<float>& latents, const std::vector<float>& velocity, float sigma,
                float sigma_next, bool bfloat16) {
    if (latents.size() != velocity.size())
        throw std::invalid_argument("Chroma velocity does not match the latents");
    // The scheduler upcasts the sample to float32; delta * velocity takes the velocity's type,
    // and the new sample is cast back to it.
    const float delta = model_type(sigma_next - sigma, bfloat16);
    for (std::size_t index = 0; index < latents.size(); ++index) {
        const float step = model_type(delta * velocity[index], bfloat16);
        latents[index] = model_type(latents[index] + step, bfloat16);
    }
}

void to_vae_input(std::vector<float>& latents, float scaling, float shift, bool bfloat16) {
    for (auto& value : latents) {
        const float scaled = model_type(value / scaling, bfloat16);
        value = model_type(scaled + shift, bfloat16);
    }
}

std::vector<float> to_rgb(const std::vector<float>& image, std::int32_t height, std::int32_t width,
                          bool bfloat16) {
    const auto plane = static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
    if (image.size() != 3 * plane)
        throw std::invalid_argument("Chroma decoder output has an unexpected size");
    std::vector<float> pixels(3 * plane);
    for (std::size_t pixel = 0; pixel < plane; ++pixel)
        for (std::size_t channel = 0; channel < 3; ++channel) {
            const float half = model_type(image[channel * plane + pixel] / 2.0F, bfloat16);
            const float shifted = model_type(half + 0.5F, bfloat16);
            pixels[pixel * 3 + channel] = std::clamp(shifted, 0.0F, 1.0F);
        }
    return pixels;
}

} // namespace trtmc::chroma
