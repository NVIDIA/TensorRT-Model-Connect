/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <algorithm>
#include <codecvt>
#include <locale>
#include <string>

namespace trtmc::laya {
inline std::string trim_unicode(const std::string& text) {
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
    const auto points = converter.from_bytes(text);
    // Python str.strip() includes the C0 information separators as whitespace.
    auto whitespace = [](char32_t c) {
        return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 ||
               c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 ||
               c == 0x202F || c == 0x205F || c == 0x3000;
    };
    const auto begin = std::find_if_not(points.begin(), points.end(), whitespace);
    if (begin == points.end())
        return "";
    const auto end = std::find_if_not(points.rbegin(), points.rend(), whitespace).base();
    return converter.to_bytes(std::u32string(begin, end));
}
} // namespace trtmc::laya
