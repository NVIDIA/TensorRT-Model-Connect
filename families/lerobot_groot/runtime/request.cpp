/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/lerobot_groot/runtime/request.h"

#include "families/lerobot_groot/runtime/prompt.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace trtmc::lerobot_groot {
namespace {
const internal::ConfigField kFields[] = {
    {"seed", internal::ConfigKind::I64, std::nullopt,
     "Seed for the initial flow noise. Omit for a fresh random draw on every call."},
    {"noise", internal::ConfigKind::F64List, std::nullopt,
     "Explicit initial noise of action_horizon x max_action_dim values (40 x 132), row-major. It "
     "overrides seed; use it to reproduce a reference run."},
};

} // namespace

Span<const internal::ConfigField> action_fields() {
    return {kFields, std::size(kFields)};
}

NoiseOptions parse_noise(internal::ConfigView config, std::size_t count) {
    NoiseOptions options;
    for (const auto& entry : config) {
        if (entry.name == "seed") {
            const auto seed = std::get<std::int64_t>(entry.value);
            if (seed < 0)
                throw std::invalid_argument("lerobot_groot seed must not be negative");
            options.has_seed = true;
            options.seed = static_cast<std::uint64_t>(seed);
        } else if (entry.name == "noise") {
            const auto values = std::get<Span<const double>>(entry.value);
            if (values.size() != count)
                throw std::invalid_argument(
                    "lerobot_groot noise must have action_horizon x max_action_dim values");
            options.noise.reserve(count);
            for (const double value : values) {
                if (!std::isfinite(value) || !std::isfinite(static_cast<float>(value)))
                    throw std::invalid_argument("lerobot_groot noise must be finite");
                options.noise.push_back(static_cast<float>(value));
            }
        }
    }
    return options;
}

std::vector<float> initial_noise(const NoiseOptions& options, std::size_t count,
                                 std::mt19937_64& rng) {
    if (!options.noise.empty())
        return options.noise;
    std::vector<float> noise(count);
    std::mt19937_64 seeded(options.seed);
    auto& engine = options.has_seed ? seeded : rng;
    std::normal_distribution<float> normal(0.0F, 1.0F);
    for (float& value : noise)
        value = normal(engine);
    return noise;
}

std::vector<RgbFrame> ordered_rgb(const internal::ImagesStateTextObservation& observation,
                                  const std::vector<std::string>& cameras, std::uint32_t size) {
    const std::size_t values = static_cast<std::size_t>(size) * size * 3;
    std::vector<const internal::ImageView*> views(cameras.size(), nullptr);
    for (const auto& named : observation.images) {
        const auto found = std::find(cameras.begin(), cameras.end(), std::string(named.name));
        if (found == cameras.end())
            throw std::invalid_argument("lerobot_groot does not accept a camera named '" +
                                        std::string(named.name) + "'");
        const auto& image = named.image;
        const std::size_t element =
            image.format == internal::ImageFormat::Float32 ? sizeof(float) : 1;
        if (image.height != size || image.width != size || image.channels != 3 ||
            image.data == nullptr || image.byte_size != values * element)
            throw std::invalid_argument("lerobot_groot camera '" + std::string(named.name) +
                                        "' must be " + std::to_string(size) + "x" +
                                        std::to_string(size) + "x3");
        views[static_cast<std::size_t>(found - cameras.begin())] = &image;
    }

    std::vector<RgbFrame> frames(views.size());
    for (std::size_t slot = 0; slot < views.size(); ++slot) {
        const internal::ImageView* image = views[slot];
        if (image == nullptr)
            throw std::invalid_argument("lerobot_groot requires camera '" + cameras[slot] + "'");
        const std::uint8_t* rgb = static_cast<const std::uint8_t*>(image->data);
        if (image->format == internal::ImageFormat::Float32) {
            // LeRobot's float image path: clamp to [0, 1], scale by 255 and truncate to uint8.
            auto& bytes = frames[slot].converted;
            bytes.resize(values);
            const auto* source = static_cast<const float*>(image->data);
            for (std::size_t index = 0; index < values; ++index) {
                if (!std::isfinite(source[index]))
                    throw std::invalid_argument("lerobot_groot camera '" + cameras[slot] +
                                                "' contains a non-finite pixel");
                bytes[index] =
                    static_cast<std::uint8_t>(std::clamp(source[index], 0.0F, 1.0F) * 255.0F);
            }
        } else {
            frames[slot].borrowed = rgb;
        }
    }
    return frames;
}

void validate_state(Span<const float> state, std::size_t width) {
    if (state.size() != width)
        throw std::invalid_argument("lerobot_groot state must have " + std::to_string(width) +
                                    " values");
    for (const float value : state)
        if (!std::isfinite(value))
            throw std::invalid_argument("lerobot_groot state must be finite");
}

std::vector<std::int32_t> instruction_tokens(std::string_view text, bool formalize,
                                             const ITokenizer& tokenizer, std::int32_t limit) {
    const auto max_bytes = static_cast<std::size_t>(limit) * kInstructionBytesPerToken;
    if (text.size() > max_bytes)
        throw std::invalid_argument("lerobot_groot instruction exceeds " +
                                    std::to_string(max_bytes) + " bytes");
    const auto instruction = tokenizer.encode(formalize_instruction(text, formalize));
    if (instruction.size() > static_cast<std::size_t>(limit))
        throw std::invalid_argument("lerobot_groot instruction exceeds " + std::to_string(limit) +
                                    " tokens");
    return instruction;
}

} // namespace trtmc::lerobot_groot
