/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::mt5 {

// SentencePiece's serialized normalization map: a little-endian Darts trie
// followed by NUL-terminated UTF-8 replacements. It belongs to the checkpoint;
// substituting a generic Unicode normalization algorithm changes token IDs.
class CharsMap {
  public:
    void load(const std::string& encoded) {
        std::vector<unsigned char> bytes;
        unsigned value = 0;
        int bits = 0;
        const std::string alphabet =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (const unsigned char c : encoded) {
            if (c == '=') break;
            const auto position = alphabet.find(c);
            if (position == std::string::npos)
                throw std::runtime_error("mt5 invalid normalization map encoding");
            value = (value << 6) | static_cast<unsigned>(position);
            bits += 6;
            if (bits >= 8) {
                bits -= 8;
                bytes.push_back(static_cast<unsigned char>((value >> bits) & 255U));
            }
        }
        auto word = [&bytes](std::size_t index) -> std::uint32_t {
            if (index > bytes.size() || bytes.size() - index < 4)
                throw std::runtime_error("mt5 truncated normalization map");
            return std::uint32_t(bytes[index]) | (std::uint32_t(bytes[index+1]) << 8)
                | (std::uint32_t(bytes[index+2]) << 16) | (std::uint32_t(bytes[index+3]) << 24);
        };
        const auto size = word(0);
        if (size == 0 || size % 4 || size > bytes.size() - 4)
            throw std::runtime_error("mt5 invalid normalization trie size");
        units_.clear();
        for (std::size_t index = 4; index < size + 4U; index += 4)
            units_.push_back(word(index));
        replacements_.assign(bytes.begin() + size + 4U, bytes.end());
    }

    std::string normalize(const std::string& input) const {
        if (units_.empty()) return input;
        std::string result;
        for (std::size_t start = 0; start < input.size();) {
            std::uint32_t node = offset(units_[0]);
            std::size_t matched = 0;
            std::uint32_t replacement = 0;
            for (std::size_t end = start; end < input.size(); ++end) {
                const auto c = static_cast<unsigned char>(input[end]);
                if (c == 0) break;
                node ^= c;
                if (node >= units_.size()) break;
                const auto unit = units_[node];
                if ((unit & 0x800000FFU) != c) break;
                node ^= offset(unit);
                if (unit & 0x100U) {
                    if (node >= units_.size())
                        throw std::runtime_error("mt5 invalid normalization leaf");
                    replacement = units_[node] & 0x7FFFFFFFU;
                    matched = end - start + 1;
                }
            }
            if (matched) {
                if (replacement >= replacements_.size())
                    throw std::runtime_error("mt5 invalid normalization replacement");
                const auto end = replacements_.find('\0', replacement);
                if (end == std::string::npos)
                    throw std::runtime_error("mt5 unterminated normalization replacement");
                result.append(replacements_, replacement, end - replacement);
                start += matched;
            } else {
                // Preserve one UTF-8 codepoint when the trie has no replacement.
                const auto c = static_cast<unsigned char>(input[start]);
                const std::size_t size = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
                if (size > input.size() - start)
                    throw std::invalid_argument("mt5 input contains truncated UTF-8");
                result.append(input, start, size);
                start += size;
            }
        }
        return result;
    }

  private:
    static std::uint32_t offset(std::uint32_t unit) {
        return (unit >> 10) << ((unit & 0x200U) >> 6);
    }
    std::vector<std::uint32_t> units_;
    std::string replacements_;
};

} // namespace trtmc::mt5
