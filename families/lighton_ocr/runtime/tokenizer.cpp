/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/lighton_ocr/runtime/tokenizer.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <unordered_map>

namespace trtmc::lighton_ocr {
namespace {

std::unordered_map<std::uint32_t, unsigned char> byte_decoder() {
    std::unordered_map<std::uint32_t, unsigned char> result;
    std::uint32_t extra = 256;
    for (std::uint32_t byte = 0; byte < 256; ++byte) {
        const bool direct =
            (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || byte >= 174;
        result.emplace(direct ? byte : extra++, static_cast<unsigned char>(byte));
    }
    return result;
}

std::uint32_t read_codepoint(std::string_view text, std::size_t& position) {
    const auto first = static_cast<unsigned char>(text[position++]);
    if (first < 128)
        return first;
    const int continuation = first < 224 ? 1 : first < 240 ? 2 : 3;
    std::uint32_t value = first & ((1U << (6 - continuation)) - 1);
    for (int i = 0; i < continuation; ++i) {
        if (position >= text.size() || (static_cast<unsigned char>(text[position]) & 192) != 128)
            throw std::runtime_error("LightOnOCR tokenizer contains malformed UTF-8");
        value = (value << 6) | (static_cast<unsigned char>(text[position++]) & 63);
    }
    return value;
}

std::string token_bytes(std::string_view text,
                        const std::unordered_map<std::uint32_t, unsigned char>& decoder) {
    std::string result;
    for (std::size_t position = 0; position < text.size();) {
        const auto found = decoder.find(read_codepoint(text, position));
        if (found == decoder.end())
            throw std::runtime_error("LightOnOCR requires the checkpoint ByteLevel vocabulary");
        result.push_back(static_cast<char>(found->second));
    }
    return result;
}

bool continuation(unsigned char byte) {
    return (byte & 192) == 128;
}

int valid_prefix(std::string_view bytes, std::size_t position) {
    const auto first = static_cast<unsigned char>(bytes[position]);
    if (first < 128)
        return 1;
    const auto length = first >= 194 && first <= 223   ? 2
                        : first >= 224 && first <= 239 ? 3
                        : first >= 240 && first <= 244 ? 4
                                                       : 0;
    if (!length)
        return -1;
    for (int i = 1; i < length; ++i) {
        if (position + i >= bytes.size())
            return -static_cast<int>(bytes.size() - position);
        const auto byte = static_cast<unsigned char>(bytes[position + i]);
        if (!continuation(byte) || (i == 1 && first == 224 && byte < 160) ||
            (i == 1 && first == 237 && byte >= 160) || (i == 1 && first == 240 && byte < 144) ||
            (i == 1 && first == 244 && byte >= 144))
            return -i;
    }
    return length;
}

// Match the ByteLevel decoder's replacement policy for a truncated UTF-8 tail.
std::string replace_invalid_utf8(std::string_view bytes) {
    std::string result;
    for (std::size_t position = 0; position < bytes.size();) {
        const auto length = valid_prefix(bytes, position);
        if (length < 0) {
            result += "\xef\xbf\xbd";
            position += static_cast<std::size_t>(-length);
        } else {
            result.append(bytes.substr(position, static_cast<std::size_t>(length)));
            position += static_cast<std::size_t>(length);
        }
    }
    return result;
}

} // namespace

Tokenizer::Tokenizer(std::string_view document, std::int32_t vocab_size) {
    if (vocab_size <= 0)
        throw std::invalid_argument("LightOnOCR vocabulary size must be positive");
    const auto json = nlohmann::json::parse(document);
    if (json.at("model").at("type") != "BPE" || json.at("decoder").at("type") != "ByteLevel")
        throw std::runtime_error("LightOnOCR requires its BPE ByteLevel tokenizer");
    bytes_.resize(static_cast<std::size_t>(vocab_size));
    special_.resize(bytes_.size(), false);
    const auto decoder = byte_decoder();
    for (const auto& item : json.at("model").at("vocab").items()) {
        const auto id = item.value().get<std::int32_t>();
        if (id < 0 || id >= vocab_size)
            throw std::runtime_error("LightOnOCR tokenizer vocabulary ID is out of range");
        bytes_[id] = token_bytes(item.key(), decoder);
    }
    for (const auto& item : json.at("added_tokens")) {
        const auto id = item.at("id").get<std::int32_t>();
        if (id < 0 || id >= vocab_size)
            throw std::runtime_error("LightOnOCR added token ID is out of range");
        bytes_[id] = item.at("content").get<std::string>();
        special_[id] = item.at("special").get<bool>();
    }
}

std::string Tokenizer::decode(const std::vector<std::int32_t>& ids) const {
    std::string bytes;
    for (const auto id : ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= bytes_.size())
            throw std::invalid_argument("LightOnOCR generated token ID is out of range");
        if (!special_[id])
            bytes += bytes_[id];
    }
    return replace_invalid_utf8(bytes);
}

} // namespace trtmc::lighton_ocr
