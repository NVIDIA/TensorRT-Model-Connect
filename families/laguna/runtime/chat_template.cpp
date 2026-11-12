/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/laguna/runtime/chat_template.h"

namespace trtmc::laguna {

namespace {

// The code points Python's str.isspace() accepts; the template applies strip() and
// rstrip() to the system message.
bool is_space(char32_t cp) {
    return (cp >= 0x09 && cp <= 0x0D) || (cp >= 0x1C && cp <= 0x20) || cp == 0x85 || cp == 0xA0 ||
           cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 ||
           cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

// Decodes the UTF-8 sequence at `at`; an invalid or truncated sequence is one non-space unit.
char32_t decode(std::string_view text, std::size_t at, std::size_t& length) {
    const auto byte = [&](std::size_t k) { return static_cast<unsigned char>(text[at + k]); };
    const unsigned char lead = byte(0);
    length = 1;
    if (lead < 0x80)
        return lead;
    const std::size_t n = (lead & 0xE0) == 0xC0   ? 2
                          : (lead & 0xF0) == 0xE0 ? 3
                          : (lead & 0xF8) == 0xF0 ? 4
                                                  : 0;
    if (n == 0 || at + n > text.size())
        return 0xFFFD;
    char32_t cp = lead & (0xFFU >> (n + 1));
    for (std::size_t k = 1; k < n; ++k) {
        if ((byte(k) & 0xC0) != 0x80)
            return 0xFFFD;
        cp = (cp << 6) | (byte(k) & 0x3F);
    }
    length = n;
    return cp;
}

std::string_view rstrip(std::string_view text) {
    std::size_t keep = 0;
    for (std::size_t at = 0, length = 0; at < text.size(); at += length) {
        if (!is_space(decode(text, at, length)))
            keep = at + length;
    }
    return text.substr(0, keep);
}

} // namespace

std::string render_chat(std::string_view user, std::string_view system_prompt,
                        bool enable_thinking) {
    const std::string_view system = system_prompt.empty() ? kDefaultSystemMessage : system_prompt;
    std::string text = "\xE3\x80\x88|EOS|\xE3\x80\x89"; // 〈|EOS|〉
    const std::string_view stripped = rstrip(system);
    if (!stripped.empty()) {
        text += "<system>\n\n";
        text += stripped;
        text += "\n</system>\n";
    }
    text += "<user>\n";
    text += user;
    text += "\n</user>\n<assistant>\n";
    text += enable_thinking ? "<think>" : "</think>";
    return text;
}

} // namespace trtmc::laguna
