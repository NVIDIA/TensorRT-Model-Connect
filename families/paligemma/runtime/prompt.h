/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/paligemma/runtime/paligemma_bpe.h"
#include "trtmc/internal/language.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace trtmc::paligemma {

struct PromptParts {
    std::vector<internal::ImageView> images;
    std::string prefix;
};

// One user message whose image parts all precede its text parts: the only
// arrangement the PaliGemma prompt format expresses. At least one image is
// required, as in the reference processor.
PromptParts collect_prompt(Span<const internal::ImagesTextMessage> messages,
                           Span<const internal::ToolDefinitionView> tools);

struct PromptTokens {
    int32_t image_token_id{-1};
    int32_t image_tokens_per_image{0};
    int32_t bos_token_id{-1};
    // Text the reference tokenizer splits out as special tokens. A prefix that
    // contains one is refused rather than tokenized differently.
    std::vector<std::string> reserved_text;
};

// Text of the added tokens that the reference splits out of a prompt but that
// SentencePiece does not match as one piece (the image token included).
std::vector<std::string> reserved_text(const PaligemmaBpeTokenizer& tokenizer,
                                       const std::vector<int32_t>& token_ids,
                                       int32_t image_token_id);

// <image> x (images * tokens), <bos>, prefix and the newline separator,
// exactly as the reference processor builds the prompt. A prompt longer than
// max_tokens is rejected, before tokenization when its length alone proves it.
std::vector<int32_t> prompt_ids(const PromptParts& prompt, const PaligemmaBpeTokenizer& tokenizer,
                                const PromptTokens& tokens, std::size_t max_tokens);

} // namespace trtmc::paligemma
