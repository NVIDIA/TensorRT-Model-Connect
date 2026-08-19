/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/cosmos3/conditioning.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::cosmos3 {
namespace {

constexpr int32_t kPadTokenId = 151643;
constexpr int32_t kEosTokenId = 151645;
constexpr int32_t kStartOfGenerationTokenId = 151652;
constexpr int32_t kTemporalModalityMargin = 15000;
constexpr float kRopeTheta = 5000000.0F;
constexpr float kMaskedAttentionValue = -3.38953139e38F;

std::string strip_trailing_periods(std::string text) {
    while (!text.empty() && text.back() == '.')
        text.pop_back();
    return text;
}

void fill_rotary_row(std::vector<float>& cos_values, std::vector<float>& sin_values,
                     std::size_t row, float temporal, float height, float width) {
    constexpr int32_t kHalfDimension = kHeadDimension / 2;
    for (int32_t frequency_index = 0; frequency_index < kHalfDimension; ++frequency_index) {
        float position = temporal;
        if (frequency_index < 60) {
            if (frequency_index % 3 == 1)
                position = height;
            else if (frequency_index % 3 == 2)
                position = width;
        }
        const float exponent =
            -2.0F * static_cast<float>(frequency_index) / static_cast<float>(kHeadDimension);
        const float inverse_frequency = std::pow(kRopeTheta, exponent);
        const float phase = position * inverse_frequency;
        const float cosine = std::cos(phase);
        const float sine = std::sin(phase);
        const auto base = row * kHeadDimension;
        cos_values[base + static_cast<std::size_t>(frequency_index)] = cosine;
        cos_values[base + static_cast<std::size_t>(frequency_index + kHalfDimension)] = cosine;
        sin_values[base + static_cast<std::size_t>(frequency_index)] = sine;
        sin_values[base + static_cast<std::size_t>(frequency_index + kHalfDimension)] = sine;
    }
}

std::size_t latent_offset(int32_t channel, int32_t frame, int32_t height, int32_t width) {
    return (((static_cast<std::size_t>(channel) * kLatentFrames + frame) * kLatentHeight + height) *
            kLatentWidth) +
           width;
}

std::size_t patch_offset(int32_t frame, int32_t patch_height, int32_t patch_width,
                         int32_t inner_height, int32_t inner_width, int32_t channel) {
    const auto row =
        (static_cast<std::size_t>(frame) * kPatchHeight + patch_height) * kPatchWidth + patch_width;
    const auto column = (static_cast<std::size_t>(inner_height) * kLatentPatchSize + inner_width) *
                            kLatentChannels +
                        channel;
    return row * kPatchDimension + column;
}

} // namespace

std::string format_chat_prompt(const std::string& text, bool negative) {
    std::string augmented = strip_trailing_periods(text);
    if (!augmented.empty())
        augmented += ". ";
    if (negative) {
        augmented += "The video is not 7.9 seconds long and is not of 24 FPS. "
                     "This video is not of 720x1280 resolution.";
    } else {
        augmented += "The video is 7.9 seconds long and is of 24 FPS. "
                     "This video is of 720x1280 resolution.";
    }
    return "<|im_start|>system\n"
           "You are a helpful assistant who will generate videos from a give prompt."
           "<|im_end|>\n<|im_start|>user\n" +
           augmented + "<|im_end|>\n<|im_start|>assistant\n";
}

PromptInputs prepare_prompt_inputs(const ITokenizer& tokenizer, const std::string& text,
                                   bool negative) {
    PromptInputs result;
    auto ids = tokenizer.encode(format_chat_prompt(text, negative));
    ids.push_back(kEosTokenId);
    ids.push_back(kStartOfGenerationTokenId);
    if (ids.empty() || ids.size() > static_cast<std::size_t>(kTextTokens)) {
        throw std::invalid_argument("Cosmos3 prompt exceeds the qualified 4096-token profile");
    }
    result.real_text_tokens = static_cast<int32_t>(ids.size());
    result.input_ids.assign(kTextTokens, kPadTokenId);
    std::copy(ids.begin(), ids.end(), result.input_ids.begin());

    result.text_rotary_cos.resize(static_cast<std::size_t>(kTextTokens) * kHeadDimension);
    result.text_rotary_sin.resize(static_cast<std::size_t>(kTextTokens) * kHeadDimension);
    for (int32_t token = 0; token < kTextTokens; ++token) {
        const float position = static_cast<float>(token);
        fill_rotary_row(result.text_rotary_cos, result.text_rotary_sin,
                        static_cast<std::size_t>(token), position, position, position);
    }

    result.vision_rotary_cos.resize(static_cast<std::size_t>(kVisionTokens) * kHeadDimension);
    result.vision_rotary_sin.resize(static_cast<std::size_t>(kVisionTokens) * kHeadDimension);
    const float temporal_offset =
        static_cast<float>(result.real_text_tokens + kTemporalModalityMargin);
    std::size_t row = 0;
    for (int32_t frame = 0; frame < kLatentFrames; ++frame) {
        for (int32_t height = 0; height < kPatchHeight; ++height) {
            for (int32_t width = 0; width < kPatchWidth; ++width, ++row) {
                fill_rotary_row(result.vision_rotary_cos, result.vision_rotary_sin, row,
                                temporal_offset + static_cast<float>(frame),
                                static_cast<float>(height), static_cast<float>(width));
            }
        }
    }

    result.generation_attention_mask.assign(kTextTokens + kVisionTokens, 0.0F);
    std::fill(result.generation_attention_mask.begin() + result.real_text_tokens,
              result.generation_attention_mask.begin() + kTextTokens, kMaskedAttentionValue);
    return result;
}

std::vector<float> patchify_latents(const std::vector<float>& latents) {
    const std::size_t expected =
        static_cast<std::size_t>(kLatentChannels) * kLatentFrames * kLatentHeight * kLatentWidth;
    if (latents.size() != expected)
        throw std::invalid_argument("Cosmos3 latent tensor has an invalid size");
    std::vector<float> patches(static_cast<std::size_t>(kVisionTokens) * kPatchDimension, 0.0F);
    for (int32_t frame = 0; frame < kLatentFrames; ++frame) {
        for (int32_t patch_height = 0; patch_height < kPatchHeight; ++patch_height) {
            for (int32_t patch_width = 0; patch_width < kPatchWidth; ++patch_width) {
                for (int32_t inner_height = 0; inner_height < kLatentPatchSize; ++inner_height) {
                    const int32_t height = patch_height * kLatentPatchSize + inner_height;
                    if (height >= kLatentHeight)
                        continue;
                    for (int32_t inner_width = 0; inner_width < kLatentPatchSize; ++inner_width) {
                        const int32_t width = patch_width * kLatentPatchSize + inner_width;
                        if (width >= kLatentWidth)
                            continue;
                        for (int32_t channel = 0; channel < kLatentChannels; ++channel) {
                            patches[patch_offset(frame, patch_height, patch_width, inner_height,
                                                 inner_width, channel)] =
                                latents[latent_offset(channel, frame, height, width)];
                        }
                    }
                }
            }
        }
    }
    return patches;
}

std::vector<float> unpatchify_latents(const std::vector<float>& patches) {
    if (patches.size() != static_cast<std::size_t>(kVisionTokens) * kPatchDimension)
        throw std::invalid_argument("Cosmos3 patch tensor has an invalid size");
    std::vector<float> latents(static_cast<std::size_t>(kLatentChannels) * kLatentFrames *
                               kLatentHeight * kLatentWidth);
    for (int32_t frame = 0; frame < kLatentFrames; ++frame) {
        for (int32_t patch_height = 0; patch_height < kPatchHeight; ++patch_height) {
            for (int32_t patch_width = 0; patch_width < kPatchWidth; ++patch_width) {
                for (int32_t inner_height = 0; inner_height < kLatentPatchSize; ++inner_height) {
                    const int32_t height = patch_height * kLatentPatchSize + inner_height;
                    if (height >= kLatentHeight)
                        continue;
                    for (int32_t inner_width = 0; inner_width < kLatentPatchSize; ++inner_width) {
                        const int32_t width = patch_width * kLatentPatchSize + inner_width;
                        if (width >= kLatentWidth)
                            continue;
                        for (int32_t channel = 0; channel < kLatentChannels; ++channel) {
                            latents[latent_offset(channel, frame, height, width)] =
                                patches[patch_offset(frame, patch_height, patch_width, inner_height,
                                                     inner_width, channel)];
                        }
                    }
                }
            }
        }
    }
    return latents;
}

} // namespace trtmc::cosmos3
