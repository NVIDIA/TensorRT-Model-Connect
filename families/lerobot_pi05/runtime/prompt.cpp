/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/lerobot_pi05/runtime/prompt.h"

#include <algorithm>
#include <array>
#include <iterator>
#include <stdexcept>

namespace trtmc::lerobot_pi05 {
namespace {

struct Utf8Codepoint {
    char32_t value{0};
    std::size_t offset{0};
    std::size_t length{0};
};

struct Utf8Prefix {
    std::size_t length{0};
    char32_t value{0};
    char32_t minimum{0};
};

Utf8Prefix decode_utf8_prefix(uint8_t first) {
    if (first <= 0x7FU) {
        return Utf8Prefix{1, first, 0};
    }

    if ((first & 0xE0U) == 0xC0U) {
        return Utf8Prefix{2, first & 0x1FU, 0x80};
    }
    if ((first & 0xF0U) == 0xE0U) {
        return Utf8Prefix{3, first & 0x0FU, 0x800};
    }
    if ((first & 0xF8U) == 0xF0U) {
        return Utf8Prefix{4, first & 0x07U, 0x10000};
    }
    throw std::invalid_argument("Pi 0.5 prompt contains invalid UTF-8");
}

char32_t decode_utf8_continuations(std::string_view text, std::size_t offset,
                                   const Utf8Prefix& prefix) {
    char32_t value = prefix.value;
    for (std::size_t i = 1; i < prefix.length; ++i) {
        const auto byte = static_cast<uint8_t>(text[offset + i]);
        if ((byte & 0xC0U) != 0x80U) {
            throw std::invalid_argument("Pi 0.5 prompt contains invalid UTF-8 continuation byte");
        }
        value = static_cast<char32_t>((value << 6U) | (byte & 0x3FU));
    }
    return value;
}

bool is_invalid_utf8_scalar(char32_t value, char32_t minimum) {
    if (value < minimum || value > 0x10FFFFU) {
        return true;
    }
    return value >= 0xD800U && value <= 0xDFFFU;
}

Utf8Codepoint decode_utf8(std::string_view text, std::size_t offset) {
    if (offset >= text.size()) {
        throw std::invalid_argument("Pi 0.5 prompt UTF-8 offset is out of range");
    }

    const Utf8Prefix prefix = decode_utf8_prefix(static_cast<uint8_t>(text[offset]));
    if (offset + prefix.length > text.size()) {
        throw std::invalid_argument("Pi 0.5 prompt contains truncated UTF-8");
    }
    const char32_t value = decode_utf8_continuations(text, offset, prefix);
    if (is_invalid_utf8_scalar(value, prefix.minimum)) {
        throw std::invalid_argument("Pi 0.5 prompt contains a non-canonical UTF-8 codepoint");
    }
    return Utf8Codepoint{value, offset, prefix.length};
}

struct CodepointRange {
    char32_t first;
    char32_t last;
};

bool is_python_whitespace(char32_t value) {
    // CPython's Unicode whitespace set (Py_UNICODE_ISSPACE), used by str.strip().
    constexpr std::array<CodepointRange, 10> kWhitespaceRanges = {
        CodepointRange{0x0009U, 0x000DU}, CodepointRange{0x001CU, 0x0020U},
        CodepointRange{0x0085U, 0x0085U}, CodepointRange{0x00A0U, 0x00A0U},
        CodepointRange{0x1680U, 0x1680U}, CodepointRange{0x2000U, 0x200AU},
        CodepointRange{0x2028U, 0x2029U}, CodepointRange{0x202FU, 0x202FU},
        CodepointRange{0x205FU, 0x205FU}, CodepointRange{0x3000U, 0x3000U},
    };
    for (const auto& range : kWhitespaceRanges) {
        if (value >= range.first && value <= range.last) {
            return true;
        }
    }
    return false;
}

std::vector<Utf8Codepoint> decode_prompt(std::string_view prompt) {
    std::vector<Utf8Codepoint> result;
    for (std::size_t offset = 0; offset < prompt.size();) {
        auto codepoint = decode_utf8(prompt, offset);
        offset += codepoint.length;
        result.push_back(codepoint);
    }
    return result;
}

} // namespace

int32_t discretize_state_value(float value) {
    std::array<double, 256> boundaries{};
    for (std::size_t i = 0; i < boundaries.size(); ++i)
        boundaries[i] = -1.0 + static_cast<double>(i) / 128.0;
    const auto iterator =
        std::upper_bound(boundaries.begin(), boundaries.end(), static_cast<double>(value));
    return static_cast<int32_t>(std::distance(boundaries.begin(), iterator)) - 1;
}

std::vector<int32_t> discretize_state(const std::vector<float>& state) {
    std::vector<int32_t> output;
    output.reserve(state.size());
    std::transform(state.begin(), state.end(), std::back_inserter(output),
                   [](float value) { return discretize_state_value(value); });
    return output;
}

std::string clean_prompt(std::string_view prompt) {
    const auto codepoints = decode_prompt(prompt);
    std::size_t first = 0;
    while (first < codepoints.size() && is_python_whitespace(codepoints[first].value)) {
        ++first;
    }
    std::size_t last = codepoints.size();
    while (last > first && is_python_whitespace(codepoints[last - 1].value)) {
        --last;
    }
    if (first == last) {
        return {};
    }

    const std::size_t begin_offset = codepoints[first].offset;
    const std::size_t end_offset = codepoints[last - 1].offset + codepoints[last - 1].length;
    std::string output(prompt.substr(begin_offset, end_offset - begin_offset));
    std::replace(output.begin(), output.end(), '_', ' ');
    std::replace(output.begin(), output.end(), '\n', ' ');
    return output;
}

std::string format_prompt(std::string_view prompt, const std::vector<float>& state) {
    const auto bins = discretize_state(state);
    std::string output = "Task: " + clean_prompt(prompt) + ", State: ";
    for (std::size_t i = 0; i < bins.size(); ++i) {
        if (i != 0) {
            output.push_back(' ');
        }
        output.append(std::to_string(bins[i]));
    }
    output.append(";\nAction: ");
    return output;
}

} // namespace trtmc::lerobot_pi05
