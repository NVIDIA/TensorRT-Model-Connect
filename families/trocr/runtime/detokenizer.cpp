/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/trocr/runtime/detokenizer.h"

#include <algorithm>
#include <array>
#include <optional>
#include <stdexcept>
#include <utility>

namespace trtmc::trocr {
namespace {

constexpr std::string_view kReplacement = "\xEF\xBF\xBD";

// GPT-2's bytes_to_unicode, inverted: printable Latin-1 bytes stand for
// themselves, the other 68 bytes for code points 256..323 in byte order.
std::array<std::int16_t, 324> byte_of_code_point() {
    std::array<std::int16_t, 324> table{};
    table.fill(-1);
    int next = 256;
    for (int byte = 0; byte < 256; ++byte) {
        const bool printable = (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) ||
                               (byte >= 174 && byte <= 255);
        table[static_cast<std::size_t>(printable ? byte : next++)] =
            static_cast<std::int16_t>(byte);
    }
    return table;
}

// Code points of a token string, or nothing when it is not well-formed UTF-8.
std::optional<std::vector<std::uint32_t>> code_points(std::string_view text) {
    std::vector<std::uint32_t> points;
    for (std::size_t index = 0; index < text.size();) {
        const auto lead = static_cast<unsigned char>(text[index]);
        std::size_t length = lead < 0x80           ? 1
                             : (lead >> 5) == 0x6  ? 2
                             : (lead >> 4) == 0xE  ? 3
                             : (lead >> 3) == 0x1E ? 4
                                                   : 0;
        if (length == 0 || index + length > text.size())
            return std::nullopt;
        std::uint32_t point = length == 1 ? lead : lead & (0x7F >> length);
        for (std::size_t k = 1; k < length; ++k) {
            const auto next = static_cast<unsigned char>(text[index + k]);
            if ((next & 0xC0) != 0x80)
                return std::nullopt;
            point = (point << 6) | (next & 0x3F);
        }
        points.push_back(point);
        index += length;
    }
    return points;
}

// The bytes a vocabulary entry stands for. An entry with a character outside
// the byte table is taken as its own UTF-8 bytes, as the reference decoder does.
std::string token_bytes(const std::string& token) {
    static const auto table = byte_of_code_point();
    const auto points = code_points(token);
    if (!points)
        return token;
    std::string bytes;
    bytes.reserve(points->size());
    for (const auto point : *points) {
        if (point >= table.size() || table[point] < 0)
            return token;
        bytes.push_back(static_cast<char>(table[point]));
    }
    return bytes;
}

} // namespace

std::string utf8_replace_invalid(std::string_view bytes) {
    std::string output;
    output.reserve(bytes.size());
    std::size_t index = 0;
    while (index < bytes.size()) {
        const auto lead = static_cast<unsigned char>(bytes[index]);
        if (lead < 0x80) {
            output.push_back(static_cast<char>(lead));
            ++index;
            continue;
        }
        // The second byte's range depends on the lead (no overlongs, no
        // surrogates, nothing above U+10FFFF); later bytes are 80..BF.
        std::size_t length = 0;
        unsigned char low = 0x80, high = 0xBF;
        if (lead >= 0xC2 && lead <= 0xDF) {
            length = 2;
        } else if (lead >= 0xE0 && lead <= 0xEF) {
            length = 3;
            low = lead == 0xE0 ? 0xA0 : 0x80;
            high = lead == 0xED ? 0x9F : 0xBF;
        } else if (lead >= 0xF0 && lead <= 0xF4) {
            length = 4;
            low = lead == 0xF0 ? 0x90 : 0x80;
            high = lead == 0xF4 ? 0x8F : 0xBF;
        }
        std::size_t valid = length == 0 ? 0 : 1;
        while (valid > 0 && valid < length && index + valid < bytes.size()) {
            const auto next = static_cast<unsigned char>(bytes[index + valid]);
            const auto lower = valid == 1 ? low : static_cast<unsigned char>(0x80);
            const auto upper = valid == 1 ? high : static_cast<unsigned char>(0xBF);
            if (next < lower || next > upper)
                break;
            ++valid;
        }
        if (length != 0 && valid == length) {
            output.append(bytes.substr(index, length));
            index += length;
        } else {
            // One replacement per maximal subpart: the lead and the
            // continuation bytes that could still have completed it.
            output.append(kReplacement);
            index += std::max<std::size_t>(valid, 1);
        }
    }
    return output;
}

ByteLevelDetokenizer::ByteLevelDetokenizer(std::vector<std::string> id_to_token,
                                           const std::vector<std::int32_t>& special_ids) {
    if (id_to_token.empty())
        throw std::invalid_argument("trocr vocabulary is empty");
    token_bytes_.reserve(id_to_token.size());
    for (const auto& token : id_to_token)
        token_bytes_.push_back(token_bytes(token));
    special_.assign(token_bytes_.size(), false);
    for (const auto id : special_ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= special_.size())
            throw std::invalid_argument("trocr special token id is outside the vocabulary");
        special_[static_cast<std::size_t>(id)] = true;
    }
}

std::string ByteLevelDetokenizer::decode(Span<const std::int32_t> ids) const {
    std::string bytes;
    for (const auto id : ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= token_bytes_.size())
            throw std::out_of_range("trocr generated an id outside the vocabulary");
        if (!special_[static_cast<std::size_t>(id)])
            bytes.append(token_bytes_[static_cast<std::size_t>(id)]);
    }
    return utf8_replace_invalid(bytes);
}

} // namespace trtmc::trocr
