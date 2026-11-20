/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/gemma3n/runtime/prompt.h"

#include <stdexcept>
#include <variant>

namespace trtmc::gemma3n {
namespace {

using namespace internal;

constexpr std::string_view kImageToken = "<image_soft_token>";

// The characters Python's str.isspace() accepts.
bool python_space(char32_t c) {
    return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 ||
           c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 ||
           c == 0x202F || c == 0x205F || c == 0x3000;
}

std::size_t utf8_width(unsigned char lead) {
    return lead < 0x80 ? 1 : (lead >> 5) == 0x6 ? 2 : (lead >> 4) == 0xE ? 3 : 4;
}

char32_t decode_at(std::string_view text, std::size_t position, std::size_t width) {
    char32_t code = static_cast<unsigned char>(text[position]);
    if (width > 1) {
        code &= 0x7F >> width;
        for (std::size_t k = 1; k < width; ++k)
            code = (code << 6) | (static_cast<unsigned char>(text[position + k]) & 0x3F);
    }
    return code;
}

void reject_placeholder(std::string_view text) {
    // The reference processor expands every placeholder in the rendered text,
    // so a literal one would no longer match the supplied images.
    if (text.find(kImageToken) != std::string_view::npos)
        throw std::invalid_argument("gemma3n text must not contain <image_soft_token>");
}

} // namespace

std::string strip(std::string_view text) {
    std::size_t begin = 0;
    while (begin < text.size()) {
        const auto width = utf8_width(static_cast<unsigned char>(text[begin]));
        if (begin + width > text.size() || !python_space(decode_at(text, begin, width)))
            break;
        begin += width;
    }
    std::size_t end = text.size();
    while (end > begin) {
        std::size_t start = end - 1;
        while (start > begin && (static_cast<unsigned char>(text[start]) & 0xC0) == 0x80)
            --start;
        const auto width = end - start;
        if (width != utf8_width(static_cast<unsigned char>(text[start])) ||
            !python_space(decode_at(text, start, width)))
            break;
        end = start;
    }
    return std::string(text.substr(begin, end - begin));
}

std::string image_sequence(int soft_tokens) {
    std::string sequence = "\n\n<start_of_image>";
    for (int i = 0; i < soft_tokens; ++i)
        sequence += kImageToken;
    return sequence + "<end_of_image>\n\n";
}

Prompt render_prompt(Span<const ImagesTextMessage> messages, std::string_view image_sequence) {
    if (messages.empty())
        throw std::invalid_argument("gemma3n requires at least one message");
    Prompt prompt;
    prompt.text = "<bos>";
    std::size_t first = 0;
    std::string prefix;
    if (messages[0].role == MessageRole::System) {
        // The template prefixes the first user turn with the system text plus
        // a blank line; it reads only the first content item, untrimmed.
        const auto& parts = messages[0].parts;
        const auto* text = parts.size() == 1 ? std::get_if<TextPartView>(&parts[0]) : nullptr;
        if (text == nullptr)
            throw std::invalid_argument("gemma3n system messages carry exactly one text part");
        reject_placeholder(text->text);
        prefix = std::string(text->text) + "\n\n";
        first = 1;
    }
    if (first == messages.size())
        throw std::invalid_argument("gemma3n requires a user message");
    for (std::size_t i = first; i < messages.size(); ++i) {
        const auto& message = messages[i];
        const bool user_turn = (i - first) % 2 == 0;
        if (message.role != (user_turn ? MessageRole::User : MessageRole::Assistant))
            throw std::invalid_argument(
                "gemma3n conversation roles must alternate user/assistant/user/...");
        prompt.text += user_turn ? "<start_of_turn>user\n" : "<start_of_turn>model\n";
        if (i == first)
            prompt.text += prefix;
        for (const auto& part : message.parts) {
            if (const auto* text = std::get_if<TextPartView>(&part)) {
                reject_placeholder(text->text);
                prompt.text += strip(text->text);
            } else if (const auto* image = std::get_if<ImageView>(&part)) {
                prompt.images.push_back(*image);
                prompt.text += image_sequence;
            } else {
                throw std::invalid_argument("gemma3n messages carry only text and image parts");
            }
        }
        prompt.text += "<end_of_turn>\n";
    }
    prompt.text += "<start_of_turn>model\n";
    return prompt;
}

} // namespace trtmc::gemma3n
