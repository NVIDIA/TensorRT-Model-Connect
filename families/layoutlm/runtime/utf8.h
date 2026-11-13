/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc::layoutlm {

struct CodePoint {
    char32_t value{0};
    std::size_t begin{0}; // Byte offset of the first byte.
    std::size_t end{0};   // Byte offset one past the last byte.
};

// Strict UTF-8 decoding: overlong forms, surrogates, values above U+10FFFF and
// truncated sequences are rejected, because the reference only ever sees valid
// Unicode strings.
inline std::vector<CodePoint> decode_utf8(std::string_view text) {
    std::vector<CodePoint> result;
    result.reserve(text.size());
    std::size_t index = 0;
    while (index < text.size()) {
        const auto lead = static_cast<unsigned char>(text[index]);
        std::size_t length = 0;
        char32_t value = 0;
        if (lead < 0x80) {
            length = 1;
            value = lead;
        } else if ((lead & 0xE0) == 0xC0) {
            length = 2;
            value = lead & 0x1F;
        } else if ((lead & 0xF0) == 0xE0) {
            length = 3;
            value = lead & 0x0F;
        } else if ((lead & 0xF8) == 0xF0) {
            length = 4;
            value = lead & 0x07;
        } else {
            throw std::invalid_argument("text is not valid UTF-8");
        }
        if (index + length > text.size())
            throw std::invalid_argument("text is not valid UTF-8");
        for (std::size_t offset = 1; offset < length; ++offset) {
            const auto next = static_cast<unsigned char>(text[index + offset]);
            if ((next & 0xC0) != 0x80)
                throw std::invalid_argument("text is not valid UTF-8");
            value = (value << 6) | (next & 0x3F);
        }
        static constexpr char32_t kMinimum[] = {0, 0, 0x80, 0x800, 0x10000};
        if (value < kMinimum[length] || value > 0x10FFFF || (value >= 0xD800 && value <= 0xDFFF))
            throw std::invalid_argument("text is not valid UTF-8");
        result.push_back({value, index, index + length});
        index += length;
    }
    return result;
}

// Python's str.isspace() set (Unicode 15): what str.split() and str.strip()
// without arguments treat as whitespace. It is wider than Unicode White_Space:
// it also contains the information separators U+001C..U+001F.
inline bool is_python_space(char32_t value) noexcept {
    return (value >= 0x09 && value <= 0x0D) || (value >= 0x1C && value <= 0x20) || value == 0x85 ||
           value == 0xA0 || value == 0x1680 || (value >= 0x2000 && value <= 0x200A) ||
           value == 0x2028 || value == 0x2029 || value == 0x202F || value == 0x205F ||
           value == 0x3000;
}

} // namespace trtmc::layoutlm
