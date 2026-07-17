/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/wan2_2_ti2v/prompt_cleaner.h"

#include <charconv>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace trtmc::wan2_2 {
namespace {

bool is_continuation(unsigned char value) {
    return (value & 0xC0U) == 0x80U;
}

uint32_t decode_utf8(std::string_view text, std::size_t& offset) {
    const auto first = static_cast<unsigned char>(text[offset++]);
    if (first < 0x80U)
        return first;

    int continuation_count = 0;
    uint32_t codepoint = 0;
    uint32_t minimum = 0;
    if ((first & 0xE0U) == 0xC0U) {
        continuation_count = 1;
        codepoint = first & 0x1FU;
        minimum = 0x80U;
    } else if ((first & 0xF0U) == 0xE0U) {
        continuation_count = 2;
        codepoint = first & 0x0FU;
        minimum = 0x800U;
    } else if ((first & 0xF8U) == 0xF0U) {
        continuation_count = 3;
        codepoint = first & 0x07U;
        minimum = 0x10000U;
    } else {
        throw std::invalid_argument("Wan2.2 prompt contains invalid UTF-8");
    }

    if (offset + static_cast<std::size_t>(continuation_count) > text.size())
        throw std::invalid_argument("Wan2.2 prompt contains truncated UTF-8");
    for (int index = 0; index < continuation_count; ++index) {
        const auto next = static_cast<unsigned char>(text[offset++]);
        if (!is_continuation(next))
            throw std::invalid_argument("Wan2.2 prompt contains invalid UTF-8 continuation");
        codepoint = (codepoint << 6U) | (next & 0x3FU);
    }
    if (codepoint < minimum || codepoint > 0x10FFFFU ||
        (codepoint >= 0xD800U && codepoint <= 0xDFFFU))
        throw std::invalid_argument("Wan2.2 prompt contains invalid UTF-8 code point");
    return codepoint;
}

void append_utf8(std::string& output, uint32_t codepoint) {
    if (codepoint <= 0x7FU) {
        output.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7FFU) {
        output.push_back(static_cast<char>(0xC0U | (codepoint >> 6U)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else if (codepoint <= 0xFFFFU) {
        output.push_back(static_cast<char>(0xE0U | (codepoint >> 12U)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else {
        output.push_back(static_cast<char>(0xF0U | (codepoint >> 18U)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    }
}

bool is_unicode_whitespace(uint32_t codepoint) {
    switch (codepoint) {
    case 0x0009U:
    case 0x000AU:
    case 0x000BU:
    case 0x000CU:
    case 0x000DU:
    case 0x001CU:
    case 0x001DU:
    case 0x001EU:
    case 0x001FU:
    case 0x0020U:
    case 0x0085U:
    case 0x00A0U:
    case 0x1680U:
    case 0x2000U:
    case 0x2001U:
    case 0x2002U:
    case 0x2003U:
    case 0x2004U:
    case 0x2005U:
    case 0x2006U:
    case 0x2007U:
    case 0x2008U:
    case 0x2009U:
    case 0x200AU:
    case 0x2028U:
    case 0x2029U:
    case 0x202FU:
    case 0x205FU:
    case 0x3000U:
        return true;
    default:
        return false;
    }
}

uint32_t repair_character_width(uint32_t codepoint) {
    // ftfy.fix_text enables fix_character_width.  Unicode's full-width ASCII
    // block is a fixed offset from the corresponding ASCII characters.
    if (codepoint >= 0xFF01U && codepoint <= 0xFF5EU)
        return codepoint - 0xFEE0U;
    return codepoint == 0x3000U ? 0x0020U : codepoint;
}

bool parse_numeric_entity(std::string_view entity, uint32_t& value) {
    if (entity.size() < 2 || entity.front() != '#')
        return false;
    int base = 10;
    std::size_t begin = 1;
    if (begin < entity.size() && (entity[begin] == 'x' || entity[begin] == 'X')) {
        base = 16;
        ++begin;
    }
    if (begin == entity.size())
        return false;
    const char* first = entity.data() + begin;
    const char* last = entity.data() + entity.size();
    const auto result = std::from_chars(first, last, value, base);
    return result.ec == std::errc{} && result.ptr == last && value <= 0x10FFFFU &&
           !(value >= 0xD800U && value <= 0xDFFFU);
}

bool named_entity(std::string_view entity, std::string_view& replacement) {
    if (entity == "amp")
        replacement = "&";
    else if (entity == "quot")
        replacement = "\"";
    else if (entity == "apos" || entity == "#39")
        replacement = "'";
    else if (entity == "lt")
        replacement = "<";
    else if (entity == "gt")
        replacement = ">";
    else if (entity == "nbsp")
        replacement = "\xC2\xA0";
    else
        return false;
    return true;
}

std::string html_unescape_once(std::string_view text) {
    std::string output;
    output.reserve(text.size());
    for (std::size_t index = 0; index < text.size();) {
        if (text[index] != '&') {
            output.push_back(text[index++]);
            continue;
        }
        const auto semicolon = text.find(';', index + 1);
        if (semicolon == std::string_view::npos || semicolon - index > 32) {
            output.push_back(text[index++]);
            continue;
        }
        const auto entity = text.substr(index + 1, semicolon - index - 1);
        uint32_t numeric = 0;
        std::string_view replacement;
        if (parse_numeric_entity(entity, numeric)) {
            append_utf8(output, numeric == 0 ? 0xFFFDU : numeric);
        } else if (named_entity(entity, replacement)) {
            output.append(replacement);
        } else {
            output.append(text.substr(index, semicolon - index + 1));
        }
        index = semicolon + 1;
    }
    return output;
}

std::string width_repair(std::string_view text) {
    std::string output;
    output.reserve(text.size());
    for (std::size_t offset = 0; offset < text.size();)
        append_utf8(output, repair_character_width(decode_utf8(text, offset)));
    return output;
}

std::string collapse_whitespace(std::string_view text) {
    std::string output;
    output.reserve(text.size());
    bool pending_space = false;
    bool has_output = false;
    for (std::size_t offset = 0; offset < text.size();) {
        const uint32_t codepoint = decode_utf8(text, offset);
        if (is_unicode_whitespace(codepoint)) {
            pending_space = has_output;
            continue;
        }
        if (pending_space)
            output.push_back(' ');
        append_utf8(output, codepoint);
        pending_space = false;
        has_output = true;
    }
    return output;
}

} // namespace

std::string clean_t5_prompt(std::string_view text) {
    // ftfy's default HTML fixer resolves nested entities before applying its
    // character-width repair.  Four bounded passes cover the nesting that
    // ftfy accepts while preventing adversarial inputs from causing unbounded
    // work.  Wan then explicitly applies two additional unescape passes.
    std::string cleaned(text);
    for (int pass = 0; pass < 4; ++pass) {
        auto unescaped = html_unescape_once(cleaned);
        if (unescaped == cleaned)
            break;
        cleaned = std::move(unescaped);
    }
    cleaned = width_repair(cleaned);
    cleaned = html_unescape_once(cleaned);
    cleaned = html_unescape_once(cleaned);
    return collapse_whitespace(cleaned);
}

} // namespace trtmc::wan2_2
