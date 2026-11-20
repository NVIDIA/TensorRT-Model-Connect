/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/qwen_image2_1/runtime/runtime_config.h"
#include "families/qwen_image2_1/runtime/tokenizer.h"

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc::qwen_image2_1 {

// One prompt in the frame the text encoder reads.
struct PromptTokens {
    // Token ids of the whole frame, padded to the text encoder's input length.
    std::vector<std::int32_t> ids;
    // Text tokens after the system turn.
    std::int32_t length{0};
};

// Frames and tokenizes one prompt. It reads only the tokenizer and the bundle's static
// configuration, so it needs no lock. Throws std::invalid_argument when the prompt does not fit
// the bundle's text budget.
PromptTokens tokenize_prompt(const ITokenizer& tokenizer, const RuntimeConfig& config,
                             const std::string& prompt);

} // namespace trtmc::qwen_image2_1
