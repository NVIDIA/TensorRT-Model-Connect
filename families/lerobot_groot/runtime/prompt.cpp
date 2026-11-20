/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/lerobot_groot/runtime/prompt.h"

#include <algorithm>
#include <stdexcept>

namespace trtmc::lerobot_groot {
namespace {

// str.isspace() for ASCII: the C whitespace plus the four separators 0x1c-0x1f.
bool python_space(unsigned char c) {
    return c == ' ' || (c >= '\t' && c <= '\r') || (c >= 0x1c && c <= 0x1f);
}

bool ascii_alnum(unsigned char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

} // namespace

std::string formalize_instruction(std::string_view text, bool formalize) {
    const std::string_view source = text.empty() ? kDefaultInstruction : text;
    if (std::any_of(source.begin(), source.end(),
                    [](char c) { return static_cast<unsigned char>(c) >= 0x80; }))
        throw std::invalid_argument(
            "lerobot_groot accepts ASCII instructions only, as LeRobot's lower() and isalnum() "
            "and the tokenizer's NFC normalization are not reproduced for other text");
    if (!formalize)
        return std::string(source);
    std::string out;
    out.reserve(source.size());
    for (const char raw : source) {
        const auto c = static_cast<unsigned char>(raw);
        if (ascii_alnum(c))
            out.push_back(static_cast<char>(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c));
        else if (python_space(c) || c == '_')
            out.push_back(static_cast<char>(c));
    }
    // LeRobot runs the batch helper twice; the second, unformalising pass replaces text that
    // formalised to nothing with the default instruction verbatim.
    return out.empty() ? std::string(kDefaultInstruction) : out;
}

Prompt build_prompt(const std::vector<std::int32_t>& instruction, std::int32_t cameras,
                    std::int32_t grid, const TemplateTokens& tokens) {
    if (cameras <= 0 || grid <= 0)
        throw std::invalid_argument("lerobot_groot prompt needs cameras and an image grid");
    Prompt prompt;
    std::vector<std::int32_t> temporal, height, width;
    std::int32_t next = 0;
    std::int32_t image_tokens = 0;
    const auto text = [&](std::int32_t id) {
        prompt.token_ids.push_back(id);
        prompt.image_index.push_back(0);
        prompt.image_mask.push_back(0);
        temporal.push_back(next);
        height.push_back(next);
        width.push_back(next);
        ++next;
    };
    text(tokens.im_start);
    for (const auto id : tokens.user)
        text(id);
    for (std::int32_t camera = 0; camera < cameras; ++camera) {
        text(tokens.vision_start);
        for (std::int32_t row = 0; row < grid; ++row) {
            for (std::int32_t column = 0; column < grid; ++column) {
                prompt.token_ids.push_back(tokens.image_pad);
                prompt.image_index.push_back(image_tokens++);
                prompt.image_mask.push_back(1);
                temporal.push_back(next);
                height.push_back(next + row);
                width.push_back(next + column);
            }
        }
        next += grid;
        text(tokens.vision_end);
    }
    for (const auto id : instruction)
        text(id);
    text(tokens.im_end);
    for (const auto id : tokens.newline)
        text(id);

    prompt.positions.reserve(temporal.size() * 3);
    for (const auto* row : {&temporal, &height, &width})
        prompt.positions.insert(prompt.positions.end(), row->begin(), row->end());
    return prompt;
}

} // namespace trtmc::lerobot_groot
