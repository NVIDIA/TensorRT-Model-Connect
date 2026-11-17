/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/flux3/runtime/sampling.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>

namespace trtmc::flux3 {
namespace {

constexpr float kTimestepScale = 1000.0F;
constexpr std::size_t kImageChannels = 3;

// numpy.linspace in float64: start + i * step, with the last value set to stop.
std::vector<double> linspace(double start, double stop, std::int32_t count) {
    std::vector<double> values(static_cast<std::size_t>(count), start);
    if (count < 2)
        return values;
    const double step = (stop - start) / static_cast<double>(count - 1);
    for (std::size_t index = 0; index < values.size(); ++index)
        values[index] = static_cast<double>(index) * step + start;
    values.back() = stop;
    return values;
}

} // namespace

float round_bf16(float value) noexcept {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7FFFFFFFU) > 0x7F800000U) {
        bits = 0x7FC00000U;
    } else {
        bits += 0x7FFFU + ((bits >> 16U) & 1U);
        bits &= 0xFFFF0000U;
    }
    float rounded = 0.0F;
    std::memcpy(&rounded, &bits, sizeof(rounded));
    return rounded;
}

Schedule make_schedule(std::int32_t num_steps, double shift) {
    if (num_steps < 1)
        throw std::invalid_argument("flux3 schedule needs at least one step");
    // NumPy casts the Python shift and shift - 1 to float32 before the
    // float32 array arithmetic shift * s / (1 + (shift - 1) * s).
    const float scale = static_cast<float>(shift);
    const float scale_minus_one = static_cast<float>(shift - 1.0);
    Schedule schedule;
    for (const double value : linspace(1.0, 1.0 / static_cast<double>(num_steps), num_steps)) {
        const float unshifted = static_cast<float>(value);
        const float sigma = (scale * unshifted) / (1.0F + scale_minus_one * unshifted);
        schedule.sigmas.push_back(sigma);
        schedule.timesteps.push_back(sigma * kTimestepScale);
    }
    schedule.sigmas.push_back(0.0F);
    schedule.timesteps.push_back(0.0F);
    return schedule;
}

float timestep_feed(float timestep) noexcept {
    return round_bf16(round_bf16(timestep / kTimestepScale) * kTimestepScale);
}

std::vector<float> timestep_projection(float feed, std::int32_t dim, std::int32_t max_period) {
    if (dim < 2 || dim % 2 != 0 || max_period < 2)
        throw std::invalid_argument("flux3 timestep projection settings are invalid");
    const auto half = static_cast<std::size_t>(dim / 2);
    const float log_period = static_cast<float>(-std::log(static_cast<double>(max_period)));
    std::vector<float> projection(2 * half);
    for (std::size_t index = 0; index < half; ++index) {
        const float exponent = log_period * static_cast<float>(index) / static_cast<float>(half);
        const float argument = feed * std::exp(exponent);
        projection[index] = std::cos(argument);
        projection[half + index] = std::sin(argument);
    }
    return projection;
}

std::string frame_prompt(std::string_view prefix, std::string_view prompt,
                         std::string_view suffix) {
    std::string text;
    text.reserve(prefix.size() + prompt.size() + suffix.size());
    text.append(prefix).append(prompt).append(suffix);
    return text;
}

std::int32_t text_window(std::size_t real_tokens, std::int32_t pad_multiple,
                         std::int32_t max_tokens) {
    if (pad_multiple < 1 || max_tokens < pad_multiple || max_tokens % pad_multiple != 0)
        throw std::invalid_argument("flux3 text window settings are invalid");
    if (real_tokens == 0)
        throw std::invalid_argument("flux3 prompt framing produced no tokens");
    if (real_tokens > static_cast<std::size_t>(max_tokens))
        throw std::invalid_argument("flux3 prompt has " + std::to_string(real_tokens) +
                                    " tokens after chat framing; this bundle accepts at most " +
                                    std::to_string(max_tokens) + " and never truncates");
    const auto multiple = static_cast<std::size_t>(pad_multiple);
    return static_cast<std::int32_t>((real_tokens + multiple - 1) / multiple * multiple);
}

std::vector<std::int32_t> pad_tokens(std::vector<std::int32_t> ids, std::int32_t window,
                                     std::int32_t pad_token_id) {
    if (window < 0 || ids.size() > static_cast<std::size_t>(window))
        throw std::invalid_argument("flux3 token ids do not fit the text window");
    ids.resize(static_cast<std::size_t>(window), pad_token_id);
    return ids;
}

std::vector<float> pack_latents(Span<const float> channel_major, const LatentShape& shape) {
    if (channel_major.size() != shape.count())
        throw std::invalid_argument("flux3 latent count does not match its shape");
    const auto channels = static_cast<std::size_t>(shape.channels);
    const auto plane =
        static_cast<std::size_t>(shape.height) * static_cast<std::size_t>(shape.width);
    std::vector<float> tokens(channel_major.size());
    for (std::size_t channel = 0; channel < channels; ++channel) {
        for (std::size_t position = 0; position < plane; ++position)
            tokens[position * channels + channel] =
                round_bf16(channel_major[channel * plane + position]);
    }
    return tokens;
}

std::vector<float> gaussian_latents(std::uint32_t seed, std::size_t count) {
    std::mt19937 generator(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<float> values(count);
    for (auto& value : values)
        value = normal(generator);
    return values;
}

void euler_update(std::vector<float>& latents, Span<const float> velocity, float sigma,
                  float next_sigma) {
    if (velocity.size() != latents.size())
        throw std::invalid_argument("flux3 velocity does not match the latents");
    const float delta = round_bf16(next_sigma - sigma);
    for (std::size_t index = 0; index < latents.size(); ++index) {
        const float product = round_bf16(delta * velocity[index]);
        latents[index] = round_bf16(latents[index] + product);
    }
}

std::vector<float> image_to_pixels(Span<const float> planar, std::int32_t height,
                                   std::int32_t width) {
    const auto plane = static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
    if (height < 1 || width < 1 || planar.size() != kImageChannels * plane)
        throw std::invalid_argument("flux3 image does not match its shape");
    std::vector<float> pixels(planar.size());
    for (std::size_t channel = 0; channel < kImageChannels; ++channel) {
        for (std::size_t position = 0; position < plane; ++position) {
            const float value = planar[channel * plane + position];
            if (!std::isfinite(value))
                throw std::runtime_error("flux3 VAE decoder produced a non-finite value");
            pixels[position * kImageChannels + channel] =
                std::clamp(value / 2.0F + 0.5F, 0.0F, 1.0F);
        }
    }
    return pixels;
}

} // namespace trtmc::flux3
