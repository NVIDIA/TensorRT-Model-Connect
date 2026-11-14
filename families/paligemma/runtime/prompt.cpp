/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/paligemma/runtime/prompt.h"

#include <stdexcept>
#include <variant>

namespace trtmc::paligemma {

PromptParts collect_prompt(Span<const internal::ImagesTextMessage> messages,
                           Span<const internal::ToolDefinitionView> tools) {
    if (!tools.empty())
        throw std::invalid_argument("PaliGemma does not expose tools");
    if (messages.size() != 1 || messages[0].role != internal::MessageRole::User)
        throw std::invalid_argument(
            "PaliGemma takes exactly one user message; it has no chat template");
    PromptParts prompt;
    bool text_seen = false;
    for (const auto& part : messages[0].parts) {
        if (const auto* image = std::get_if<internal::ImageView>(&part)) {
            if (text_seen)
                throw std::invalid_argument("PaliGemma places every image before the prefix text");
            prompt.images.push_back(*image);
        } else if (const auto* text = std::get_if<internal::TextPartView>(&part)) {
            text_seen = true;
            prompt.prefix += text->text;
        } else {
            throw std::invalid_argument("PaliGemma prompts contain only images and text");
        }
    }
    if (prompt.images.empty())
        throw std::invalid_argument("PaliGemma requires at least one image");
    return prompt;
}

std::vector<std::string> reserved_text(const PaligemmaBpeTokenizer& tokenizer,
                                       const std::vector<int32_t>& token_ids,
                                       int32_t image_token_id) {
    const auto& pieces = tokenizer.asset().pieces;
    std::vector<std::string> result;
    for (const int32_t id : token_ids) {
        if (id == image_token_id)
            result.emplace_back("<image>");
        else if (id >= 0 && static_cast<std::size_t>(id) < pieces.size())
            result.push_back(pieces[static_cast<std::size_t>(id)].text);
        else
            throw std::invalid_argument("PaliGemma reserved token id is outside the vocabulary");
    }
    return result;
}

std::vector<int32_t> prompt_ids(const PromptParts& prompt, const PaligemmaBpeTokenizer& tokenizer,
                                const PromptTokens& tokens, std::size_t max_tokens) {
    if (tokens.image_token_id < 0 || tokens.image_tokens_per_image <= 0 || tokens.bos_token_id < 0)
        throw std::invalid_argument("PaliGemma prompt tokens are not configured");
    constexpr const char* kTooLong =
        "PaliGemma prompt and max_new_tokens exceed the built sequence capacity";
    const std::size_t fixed =
        prompt.images.size() * static_cast<std::size_t>(tokens.image_tokens_per_image) + 1;
    // A token covers at most longest_piece_bytes() of normalized text, so this
    // lower bound on the token count rejects an oversized prefix before it is
    // tokenized; a prefix whose tokens fit always passes it.
    const std::size_t longest = tokenizer.longest_piece_bytes();
    const std::size_t length = tokenizer.normalization_never_shortens()
                                   ? prompt.prefix.size() + 1
                                   : tokenizer.normalize(prompt.prefix + "\n").size();
    if (fixed > max_tokens || (length + longest - 1) / longest > max_tokens - fixed)
        throw std::invalid_argument(kTooLong);
    for (const auto& reserved : tokens.reserved_text)
        if (!reserved.empty() && prompt.prefix.find(reserved) != std::string::npos)
            throw std::invalid_argument("PaliGemma prefix text contains the special token " +
                                        reserved);
    const auto encoded = tokenizer.encode(prompt.prefix + "\n", false);
    if (encoded.size() > max_tokens - fixed)
        throw std::invalid_argument(kTooLong);
    std::vector<int32_t> ids(fixed - 1, tokens.image_token_id);
    ids.push_back(tokens.bos_token_id);
    ids.insert(ids.end(), encoded.begin(), encoded.end());
    return ids;
}

} // namespace trtmc::paligemma
