/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/cosmos3/runtime/tokenizer.h"

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc::cosmos3 {

inline constexpr int32_t kLatentChannels = 48;
inline constexpr int32_t kLatentFrames = 48;
inline constexpr int32_t kLatentHeight = 45;
inline constexpr int32_t kLatentWidth = 80;
inline constexpr int32_t kLatentPatchSize = 2;
inline constexpr int32_t kPatchHeight = 23;
inline constexpr int32_t kPatchWidth = 40;
inline constexpr int32_t kPatchDimension = 192;
inline constexpr int32_t kVisionTokens = 44160;
inline constexpr int32_t kTextTokens = 4096;
inline constexpr int32_t kHeadDimension = 128;

struct PromptInputs {
    std::vector<int32_t> input_ids;
    std::vector<float> text_rotary_cos;
    std::vector<float> text_rotary_sin;
    std::vector<float> vision_rotary_cos;
    std::vector<float> vision_rotary_sin;
    std::vector<float> generation_attention_mask;
    int32_t real_text_tokens{0};
};

std::string format_chat_prompt(const std::string& text, bool negative);
PromptInputs prepare_prompt_inputs(const ITokenizer& tokenizer, const std::string& text,
                                   bool negative);

std::vector<float> patchify_latents(const std::vector<float>& latents);
std::vector<float> unpatchify_latents(const std::vector<float>& patches);

} // namespace trtmc::cosmos3
