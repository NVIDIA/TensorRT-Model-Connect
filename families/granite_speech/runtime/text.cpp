/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/granite_speech/runtime/text.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <utility>

namespace trtmc::granite_speech {
namespace {

constexpr std::string_view kReplacement = "\xEF\xBF\xBD";

int hex_value(char c) {
    if (c >= '0' && c <= '9')
        return c - '0';
    if (c >= 'a' && c <= 'f')
        return c - 'a' + 10;
    if (c >= 'A' && c <= 'F')
        return c - 'A' + 10;
    throw std::runtime_error("granite_speech vocabulary holds a non-hex token");
}

std::string from_hex(const std::string& hex) {
    if (hex.size() % 2 != 0)
        throw std::runtime_error("granite_speech vocabulary holds an odd-length hex token");
    std::string out(hex.size() / 2, '\0');
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = static_cast<char>(hex_value(hex[2 * i]) * 16 + hex_value(hex[2 * i + 1]));
    return out;
}

bool in_range(unsigned char value, unsigned char low, unsigned char high) {
    return value >= low && value <= high;
}

// Number of bytes of the well-formed sequence starting at `lead`, 0 for an invalid lead byte,
// and the accepted range of the second byte (Unicode Table 3-7).
std::size_t sequence_length(unsigned char lead, unsigned char& low, unsigned char& high) {
    low = 0x80;
    high = 0xBF;
    if (lead >= 0xC2 && lead <= 0xDF)
        return 2;
    if (lead >= 0xE0 && lead <= 0xEF) {
        if (lead == 0xE0)
            low = 0xA0;
        if (lead == 0xED)
            high = 0x9F;
        return 3;
    }
    if (lead >= 0xF0 && lead <= 0xF4) {
        if (lead == 0xF0)
            low = 0x90;
        if (lead == 0xF4)
            high = 0x8F;
        return 4;
    }
    return 0;
}

} // namespace

std::string utf8_lossy(std::string_view bytes) {
    std::string out;
    out.reserve(bytes.size());
    std::size_t index = 0;
    while (index < bytes.size()) {
        const auto lead = static_cast<unsigned char>(bytes[index]);
        if (lead < 0x80) {
            out.push_back(static_cast<char>(lead));
            ++index;
            continue;
        }
        unsigned char low = 0, high = 0;
        const std::size_t length = sequence_length(lead, low, high);
        if (length == 0) {
            out.append(kReplacement);
            ++index;
            continue;
        }
        std::size_t valid = 1;
        while (valid < length && index + valid < bytes.size()) {
            const auto next = static_cast<unsigned char>(bytes[index + valid]);
            const bool ok = valid == 1 ? in_range(next, low, high) : in_range(next, 0x80, 0xBF);
            if (!ok)
                break;
            ++valid;
        }
        if (valid == length)
            out.append(bytes.substr(index, length));
        else
            out.append(kReplacement);
        index += valid;
    }
    return out;
}

Vocabulary::Vocabulary(std::vector<std::string> token_bytes,
                       const std::vector<std::int32_t>& skipped)
    : bytes_(std::move(token_bytes)), skip_(bytes_.size(), false) {
    for (const auto id : skipped) {
        if (id < 0 || static_cast<std::size_t>(id) >= skip_.size())
            throw std::runtime_error("granite_speech vocabulary skips an id outside the table");
        skip_[static_cast<std::size_t>(id)] = true;
    }
}

Vocabulary Vocabulary::from_json(std::string_view json) {
    const auto parsed = nlohmann::json::parse(json.begin(), json.end());
    const auto& tokens = parsed.at("tokens");
    std::vector<std::string> bytes;
    bytes.reserve(tokens.size());
    for (const auto& token : tokens)
        bytes.push_back(from_hex(token.get<std::string>()));
    if (bytes.empty())
        throw std::runtime_error("granite_speech vocabulary is empty");
    return Vocabulary(std::move(bytes), parsed.at("skip").get<std::vector<std::int32_t>>());
}

std::string Vocabulary::decode(const std::vector<std::int32_t>& ids) const {
    std::string raw;
    for (const auto id : ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= bytes_.size())
            throw std::runtime_error("granite_speech generated an id outside the vocabulary: " +
                                     std::to_string(id));
        if (!skip_[static_cast<std::size_t>(id)])
            raw += bytes_[static_cast<std::size_t>(id)];
    }
    return utf8_lossy(raw);
}

} // namespace trtmc::granite_speech
