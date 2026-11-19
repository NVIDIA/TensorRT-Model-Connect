/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen_image2_1/runtime/prompt_tokens.h"

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace trtmc::qwen_image2_1 {

PromptTokens tokenize_prompt(const ITokenizer& tokenizer, const RuntimeConfig& config,
                             const std::string& prompt) {
    // An empty prompt would leave the encoder nothing to read; the pipeline substitutes a space.
    const std::string text = prompt.empty() ? " " : prompt;
    auto ids = tokenizer.encode(config.prompt_prefix + text + config.prompt_suffix);

    const auto prefix = static_cast<std::size_t>(config.prefix_tokens);
    const auto capacity = prefix + static_cast<std::size_t>(config.text_tokens);
    if (ids.size() <= prefix)
        throw std::runtime_error("Qwen-Image 2.1 prompt frame lost its system turn");
    if (ids.size() > capacity) {
        throw std::invalid_argument(
            "Qwen-Image 2.1 prompt needs " + std::to_string(ids.size() - prefix) +
            " text tokens; this bundle was built for " + std::to_string(config.text_tokens));
    }
    const auto length = static_cast<std::int32_t>(ids.size() - prefix);
    ids.resize(capacity, config.pad_token_id);
    return {std::move(ids), length};
}

} // namespace trtmc::qwen_image2_1
