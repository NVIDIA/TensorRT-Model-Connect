/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/glm_ocr/runtime/prompt.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string_view>

namespace trtmc::glm_ocr {
namespace {

constexpr std::string_view kImage = "<|begin_of_image|><|image|><|end_of_image|>";

// Code points Python's str.isspace() accepts (the template strips assistant text with it).
bool python_space(std::uint32_t cp) {
    return (cp >= 0x9 && cp <= 0xD) || (cp >= 0x1C && cp <= 0x20) || cp == 0x85 || cp == 0xA0 ||
           cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 ||
           cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

// Byte length and value of the code point starting at `pos` (lenient: one byte when invalid).
std::size_t next_code_point(std::string_view text, std::size_t pos, std::uint32_t& cp) {
    const auto lead = static_cast<std::uint8_t>(text[pos]);
    const std::size_t length = lead < 0x80           ? 1
                               : (lead >> 5) == 0x6  ? 2
                               : (lead >> 4) == 0xE  ? 3
                               : (lead >> 3) == 0x1E ? 4
                                                     : 1;
    if (length == 1 || pos + length > text.size()) {
        cp = lead;
        return 1;
    }
    cp = lead & (0xFF >> (length + 1));
    for (std::size_t i = 1; i < length; ++i)
        cp = (cp << 6) | (static_cast<std::uint8_t>(text[pos + i]) & 0x3F);
    return length;
}

std::string python_strip(std::string_view text) {
    std::size_t begin = 0, end = 0, pos = 0;
    bool seen = false;
    while (pos < text.size()) {
        std::uint32_t cp = 0;
        const auto length = next_code_point(text, pos, cp);
        if (!python_space(cp)) {
            if (!seen)
                begin = pos;
            seen = true;
            end = pos + length;
        }
        pos += length;
    }
    return seen ? std::string(text.substr(begin, end - begin)) : std::string();
}

std::string_view text_of(const internal::ImagesTextPart& part, const char* role) {
    if (const auto* text = std::get_if<internal::TextPartView>(&part))
        return text->text;
    throw std::invalid_argument(std::string("glm_ocr ") + role +
                                " messages accept text parts only");
}

} // namespace

RenderedPrompt render_prompt(const internal::ImagesTextToTextRequest& request) {
    using internal::MessageRole;
    if (!request.tools.empty())
        throw std::invalid_argument("glm_ocr does not accept tool definitions");
    if (request.messages.empty())
        throw std::invalid_argument("glm_ocr request has no messages");
    if (request.messages[request.messages.size() - 1].role != MessageRole::User)
        throw std::invalid_argument("glm_ocr conversation must end with a user message");
    RenderedPrompt prompt;
    prompt.text = "[gMASK]<sop>";
    for (const auto& message : request.messages) {
        if (message.parts.empty())
            throw std::invalid_argument("glm_ocr messages must not be empty");
        switch (message.role) {
        case MessageRole::System:
            prompt.text += "<|system|>\n";
            for (const auto& part : message.parts)
                prompt.text += text_of(part, "system");
            break;
        case MessageRole::User:
            prompt.text += "<|user|>\n";
            for (const auto& part : message.parts) {
                if (const auto* text = std::get_if<internal::TextPartView>(&part)) {
                    prompt.text += text->text;
                } else if (const auto* image = std::get_if<internal::ImageView>(&part)) {
                    prompt.text += kImage;
                    prompt.images.push_back(*image);
                } else {
                    throw std::invalid_argument(
                        "glm_ocr user messages accept text and image parts only");
                }
            }
            break;
        case MessageRole::Assistant: {
            std::string content;
            for (const auto& part : message.parts)
                content += text_of(part, "assistant");
            if (content.find("<think>") != std::string::npos ||
                content.find("</think>") != std::string::npos)
                throw std::invalid_argument("glm_ocr assistant history must not carry reasoning");
            prompt.text += "<|assistant|>\n<think></think>";
            const auto stripped = python_strip(content);
            if (!stripped.empty())
                prompt.text += "\n" + stripped;
            break;
        }
        default:
            throw std::invalid_argument("glm_ocr accepts system, user and assistant messages");
        }
    }
    prompt.text += "<|assistant|>\n";
    return prompt;
}

PromptLayout layout_prompt(const std::vector<std::int32_t>& ids, std::int32_t image_token_id,
                           const std::vector<ImageGrid>& images, int merge_size) {
    PromptLayout layout;
    std::size_t image = 0;
    std::int32_t current = 0;
    for (const auto id : ids) {
        if (id != image_token_id) {
            layout.ids.push_back(id);
            layout.positions.push_back({current, current, current});
            ++current;
            continue;
        }
        if (image >= images.size())
            throw std::invalid_argument("glm_ocr prompt has more image placeholders than images");
        const int rows = images[image].grid_h / merge_size;
        const int cols = images[image].grid_w / merge_size;
        ++image;
        layout.image_offsets.push_back(layout.ids.size());
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) {
                layout.ids.push_back(image_token_id);
                layout.positions.push_back({current, current + r, current + c});
            }
        current += std::max(rows, cols);
    }
    if (image != images.size())
        throw std::invalid_argument("glm_ocr prompt has fewer image placeholders than images");
    std::int32_t largest = -1;
    for (const auto& position : layout.positions)
        largest = std::max({largest, position[0], position[1], position[2]});
    layout.next_position = largest + 1;
    return layout;
}

void text_rotary(const std::vector<std::array<std::int32_t, 3>>& positions, int head_dim,
                 float theta, const std::array<int, 3>& sections, std::vector<float>& cos,
                 std::vector<float>& sin) {
    const int half = head_dim / 2;
    if (sections[0] + sections[1] + sections[2] != half)
        throw std::logic_error("glm_ocr mrope sections do not cover the rotary pairs");
    std::vector<float> inv_freq(half);
    std::vector<int> axis(half);
    for (int i = 0, owner = 0, bound = sections[0]; i < half; ++i) {
        while (i >= bound)
            bound += sections[++owner];
        axis[i] = owner;
        const float exponent = static_cast<float>(2 * i) / static_cast<float>(head_dim);
        inv_freq[i] = 1.0F / std::pow(theta, exponent);
    }
    cos.resize(positions.size() * half);
    sin.resize(positions.size() * half);
    for (std::size_t row = 0; row < positions.size(); ++row)
        for (int i = 0; i < half; ++i) {
            const float angle = static_cast<float>(positions[row][axis[i]]) * inv_freq[i];
            cos[row * half + i] = std::cos(angle);
            sin[row * half + i] = std::sin(angle);
        }
}

} // namespace trtmc::glm_ocr
