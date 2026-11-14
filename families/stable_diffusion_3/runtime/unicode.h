/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Unicode properties the tokenizers need, over the generated tables in
// unicode_data.h: simple lowercase, the \p{L} / \p{N} / mark categories,
// and NFC (canonical decomposition, canonical ordering, composition).

#include "families/stable_diffusion_3/runtime/unicode_data.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::stable_diffusion_3::unicode {

template <std::size_t N>
bool in_ranges(const unicode_data::Range (&ranges)[N], char32_t cp) {
    const auto* end = ranges + N;
    const auto* found =
        std::upper_bound(ranges, end, cp, [](char32_t value, const unicode_data::Range& range) {
            return value < range.lo;
        });
    return found != ranges && cp <= (found - 1)->hi;
}

inline bool is_letter(char32_t cp) {
    return in_ranges(unicode_data::kLetters, cp);
}
inline bool is_number(char32_t cp) {
    return in_ranges(unicode_data::kNumbers, cp);
}
inline bool is_mark(char32_t cp) {
    return in_ranges(unicode_data::kMarks, cp);
}
// Grapheme_Cluster_Break Control, CR or LF: a grapheme cluster ends before and
// after one (UAX #29 GB4, GB5), except CR LF (GB3).
inline bool is_grapheme_control(char32_t cp) {
    return in_ranges(unicode_data::kGraphemeControls, cp);
}
// Grapheme_Cluster_Break Extend, ZWJ or SpacingMark: joins the cluster before
// it (GB9, GB9a).
inline bool is_grapheme_extend(char32_t cp) {
    return in_ranges(unicode_data::kGraphemeExtend, cp);
}
// Grapheme_Cluster_Break Prepend: joins the code point after it (GB9b).
inline bool is_grapheme_prepend(char32_t cp) {
    return in_ranges(unicode_data::kGraphemePrepend, cp);
}

// Unicode White_Space (PropList.txt): the \s of the tokenizers library's
// regexes and what its added tokens' lstrip / rstrip absorb.
inline bool is_white_space(char32_t cp) {
    return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F ||
           cp == 0x205F || cp == 0x3000;
}

// Simple lowercase mapping; U+0130 (the one multi-code-point case) is the caller's.
inline char32_t lowercase(char32_t cp) {
    if (cp < 0x80)
        return cp >= 'A' && cp <= 'Z' ? cp + 32 : cp;
    for (const auto& range : unicode_data::kLowercase) {
        if (cp < range.lo)
            break;
        if (cp <= range.hi && (cp - range.lo) % static_cast<char32_t>(range.stride) == 0)
            return static_cast<char32_t>(static_cast<std::int64_t>(cp) + range.delta);
    }
    return cp;
}

inline std::uint8_t combining_class(char32_t cp) {
    const auto& table = unicode_data::kCombiningClasses;
    const auto* end = table + sizeof(table) / sizeof(table[0]);
    const auto* found = std::upper_bound(
        table, end, cp, [](char32_t value, const auto& range) { return value < range.lo; });
    if (found == table || cp > (found - 1)->hi)
        return 0;
    return (found - 1)->ccc;
}

namespace detail {

constexpr char32_t kSBase = 0xAC00, kLBase = 0x1100, kVBase = 0x1161, kTBase = 0x11A7;
constexpr char32_t kLCount = 19, kVCount = 21, kTCount = 28;
constexpr char32_t kNCount = kVCount * kTCount, kSCount = kLCount * kNCount;

inline void decompose(char32_t cp, std::u32string& out) {
    if (cp >= kSBase && cp < kSBase + kSCount) {
        const char32_t index = cp - kSBase;
        out.push_back(kLBase + index / kNCount);
        out.push_back(kVBase + (index % kNCount) / kTCount);
        if (index % kTCount != 0)
            out.push_back(kTBase + index % kTCount);
        return;
    }
    const auto& table = unicode_data::kDecompositions;
    const auto* end = table + sizeof(table) / sizeof(table[0]);
    const auto* found = std::lower_bound(
        table, end, cp, [](const auto& entry, char32_t value) { return entry.cp < value; });
    if (found == end || found->cp != cp) {
        out.push_back(cp);
        return;
    }
    decompose(found->first, out);
    if (found->second != 0)
        decompose(found->second, out);
}

inline char32_t compose(char32_t first, char32_t second) {
    if (first >= kLBase && first < kLBase + kLCount && second >= kVBase &&
        second < kVBase + kVCount)
        return kSBase + ((first - kLBase) * kVCount + (second - kVBase)) * kTCount;
    if (first >= kSBase && first < kSBase + kSCount && (first - kSBase) % kTCount == 0 &&
        second > kTBase && second < kTBase + kTCount)
        return first + (second - kTBase);
    const auto& table = unicode_data::kCompositions;
    const auto* end = table + sizeof(table) / sizeof(table[0]);
    const auto key = std::make_pair(first, second);
    const auto* found = std::lower_bound(table, end, key, [](const auto& entry, const auto& value) {
        return std::make_pair(entry.first, entry.second) < value;
    });
    if (found == end || found->first != first || found->second != second)
        return 0;
    return found->composite;
}

} // namespace detail

// NFC of a code point sequence (UAX #15: decompose, order, compose).
inline std::u32string nfc(const std::u32string& text) {
    bool trivial = true;
    for (const char32_t cp : text)
        trivial = trivial && cp < 0x300;
    if (trivial)
        return text; // nothing below U+0300 decomposes, combines or composes
    std::u32string buffer;
    buffer.reserve(text.size() + 8);
    for (const char32_t cp : text)
        detail::decompose(cp, buffer);
    // Canonical ordering: stable sort of every run of non-starters by class.
    for (std::size_t i = 1; i < buffer.size(); ++i) {
        const auto cls = combining_class(buffer[i]);
        if (cls == 0)
            continue;
        std::size_t j = i;
        while (j > 0) {
            const auto previous = combining_class(buffer[j - 1]);
            if (previous == 0 || previous <= cls)
                break;
            std::swap(buffer[j - 1], buffer[j]);
            --j;
        }
    }
    if (buffer.empty())
        return buffer;
    std::size_t starter = 0;
    int last_class = combining_class(buffer[0]) == 0 ? 0 : 256;
    std::size_t out = 1;
    for (std::size_t i = 1; i < buffer.size(); ++i) {
        const char32_t cp = buffer[i];
        const int cls = combining_class(cp);
        const char32_t composite = detail::compose(buffer[starter], cp);
        if (composite != 0 && (last_class < cls || last_class == 0)) {
            buffer[starter] = composite;
            continue;
        }
        if (cls == 0)
            starter = out;
        last_class = cls;
        buffer[out++] = cp;
    }
    buffer.resize(out);
    return buffer;
}

// UTF-8 <-> code points; a malformed byte decodes to U+FFFD.
inline std::u32string decode_utf8(const std::string& text) {
    std::u32string out;
    out.reserve(text.size());
    std::size_t pos = 0;
    while (pos < text.size()) {
        const auto c = static_cast<unsigned char>(text[pos]);
        std::size_t length = c < 0x80             ? 1
                             : (c & 0xE0) == 0xC0 ? 2
                             : (c & 0xF0) == 0xE0 ? 3
                             : (c & 0xF8) == 0xF0 ? 4
                                                  : 0;
        if (length == 0 || pos + length > text.size()) {
            out.push_back(0xFFFD);
            ++pos;
            continue;
        }
        char32_t cp = length == 1   ? c
                      : length == 2 ? (c & 0x1F)
                      : length == 3 ? (c & 0x0F)
                                    : (c & 0x07);
        bool valid = true;
        for (std::size_t k = 1; k < length; ++k) {
            const auto next = static_cast<unsigned char>(text[pos + k]);
            valid = valid && (next & 0xC0) == 0x80;
            cp = (cp << 6) | (next & 0x3F);
        }
        out.push_back(valid ? cp : 0xFFFD);
        pos += valid ? length : 1;
    }
    return out;
}

inline std::string encode_utf8(const std::u32string& text) {
    std::string out;
    out.reserve(text.size());
    for (const char32_t cp : text) {
        if (cp < 0x80) {
            out.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }
    return out;
}

inline std::string nfc_utf8(const std::string& text) {
    for (const char c : text) {
        // Only code points from U+0300 up can change, and they need two or
        // more UTF-8 bytes led by 0xCC or higher.
        if (static_cast<unsigned char>(c) >= 0xCC)
            return encode_utf8(nfc(decode_utf8(text)));
    }
    return text;
}

} // namespace trtmc::stable_diffusion_3::unicode
