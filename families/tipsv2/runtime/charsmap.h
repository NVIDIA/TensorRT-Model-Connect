/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace trtmc::tipsv2 {

// SentencePiece's serialized normalization map: a little-endian Darts double-array trie followed
// by a pool of NUL-terminated UTF-8 replacements. For this checkpoint the map *is* `nmt_nfkc`;
// substituting a general Unicode normalizer changes token ids, so the bytes travel in the bundle
// and are read here exactly as SentencePiece reads them.
//
// This class answers one question - "what does the map do to the start of this string?" - because
// that is the shape SentencePiece's normalizer needs: it walks the input prefix by prefix, so
// that whitespace collapsing and the dummy prefix can act between the replacements.
class PrecompiledCharsMap {
  public:
    void load(std::string_view raw) {
        units_.clear();
        replacements_.clear();
        if (raw.size() < 4) {
            throw std::invalid_argument("tipsv2 normalization map is truncated");
        }
        const auto word = [&raw](std::size_t index) -> std::uint32_t {
            return static_cast<std::uint32_t>(static_cast<unsigned char>(raw[index])) |
                   (static_cast<std::uint32_t>(static_cast<unsigned char>(raw[index + 1])) << 8) |
                   (static_cast<std::uint32_t>(static_cast<unsigned char>(raw[index + 2])) << 16) |
                   (static_cast<std::uint32_t>(static_cast<unsigned char>(raw[index + 3])) << 24);
        };
        const std::uint32_t trie_bytes = word(0);
        if (trie_bytes == 0 || trie_bytes % 4 != 0 || trie_bytes > raw.size() - 4) {
            throw std::invalid_argument("tipsv2 normalization map has an invalid trie size");
        }
        units_.reserve(trie_bytes / 4);
        for (std::size_t index = 4; index < std::size_t{trie_bytes} + 4U; index += 4) {
            units_.push_back(word(index));
        }
        replacements_.assign(raw.substr(std::size_t{trie_bytes} + 4U));
    }

    bool empty() const noexcept { return units_.empty(); }

    // The longest trie match at the start of `text`. `second == 0` means the map says nothing
    // about this position and the caller keeps the input as it is.
    std::pair<std::string_view, std::size_t> prefix(std::string_view text) const {
        if (units_.empty() || text.empty()) {
            return {{}, 0};
        }
        std::uint32_t node = offset(units_[0]);
        std::size_t matched = 0;
        std::uint32_t replacement = 0;
        for (std::size_t end = 0; end < text.size(); ++end) {
            const auto byte = static_cast<unsigned char>(text[end]);
            // The trie's keys are NUL-terminated, so an embedded NUL can never continue a key.
            if (byte == 0) {
                break;
            }
            node ^= byte;
            if (node >= units_.size()) {
                break;
            }
            const std::uint32_t unit = units_[node];
            // The label keeps bit 31 so that a leaf unit can never compare equal to a byte.
            if ((unit & 0x800000FFU) != byte) {
                break;
            }
            node ^= offset(unit);
            if ((unit & 0x100U) != 0U) {
                if (node >= units_.size()) {
                    throw std::runtime_error("tipsv2 normalization map has an invalid leaf");
                }
                // Keep walking: the *last* leaf on the path is the longest match.
                replacement = units_[node] & 0x7FFFFFFFU;
                matched = end + 1;
            }
        }
        if (matched == 0) {
            return {{}, 0};
        }
        if (replacement >= replacements_.size()) {
            throw std::runtime_error("tipsv2 normalization map has an out-of-range replacement");
        }
        const auto end = replacements_.find('\0', replacement);
        if (end == std::string::npos) {
            throw std::runtime_error("tipsv2 normalization map has an unterminated replacement");
        }
        return {std::string_view(replacements_).substr(replacement, end - replacement), matched};
    }

  private:
    static std::uint32_t offset(std::uint32_t unit) noexcept {
        return (unit >> 10) << ((unit & 0x200U) >> 6);
    }

    std::vector<std::uint32_t> units_;
    std::string replacements_;
};

} // namespace trtmc::tipsv2
