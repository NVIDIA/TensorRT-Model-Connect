/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/lfm2/runtime/byte_level_decoder.h"

#include <array>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace trtmc {
namespace {

struct Utf8Unit {
    char32_t codepoint{0};
    std::size_t size{1};
    bool valid{false};
};

bool is_continuation(unsigned char byte) {
    return (byte & 0xC0U) == 0x80U;
}

std::size_t utf8_sequence_size(unsigned char first) {
    if (first < 0x80U)
        return 1;
    if (first >= 0xC2U && first <= 0xDFU)
        return 2;
    if (first >= 0xE0U && first <= 0xEFU)
        return 3;
    if (first >= 0xF0U && first <= 0xF4U)
        return 4;
    return 0;
}

bool is_valid_three_byte_prefix(unsigned char first, unsigned char second) {
    return (first != 0xE0U || second >= 0xA0U) && (first != 0xEDU || second <= 0x9FU);
}

bool is_valid_four_byte_prefix(unsigned char first, unsigned char second) {
    return (first != 0xF0U || second >= 0x90U) && (first != 0xF4U || second <= 0x8FU);
}

bool is_valid_scalar_prefix(unsigned char first, unsigned char second, std::size_t size) {
    if (size == 3)
        return is_valid_three_byte_prefix(first, second);
    if (size == 4)
        return is_valid_four_byte_prefix(first, second);
    return true;
}

bool is_valid_utf8_sequence(std::string_view text, std::size_t offset, std::size_t size,
                            unsigned char first) {
    if (size == 0 || size > text.size() - offset)
        return false;
    if (size > 1 &&
        !is_valid_scalar_prefix(first, static_cast<unsigned char>(text[offset + 1]), size))
        return false;
    for (std::size_t index = 1; index < size; ++index) {
        if (!is_continuation(static_cast<unsigned char>(text[offset + index])))
            return false;
    }
    return true;
}

char32_t decode_utf8_codepoint(std::string_view text, std::size_t offset, std::size_t size) {
    static constexpr std::array<unsigned char, 5> masks{0, 0x7FU, 0x1FU, 0x0FU, 0x07U};
    char32_t codepoint = static_cast<unsigned char>(text[offset]) & masks[size];
    for (std::size_t index = 1; index < size; ++index) {
        codepoint = (codepoint << 6U) | (static_cast<unsigned char>(text[offset + index]) & 0x3FU);
    }
    return codepoint;
}

Utf8Unit decode_utf8_unit(std::string_view text, std::size_t offset) {
    const auto first = static_cast<unsigned char>(text[offset]);
    const std::size_t size = utf8_sequence_size(first);
    if (is_valid_utf8_sequence(text, offset, size, first))
        return {decode_utf8_codepoint(text, offset, size), size, true};

    // Preserve malformed input one byte at a time. This also makes the inverse
    // safe for arbitrary byte strings rather than only well-formed user text.
    return {first, 1, false};
}

const std::unordered_map<char32_t, unsigned char>& gpt2_unicode_to_byte() {
    static const auto table = [] {
        std::array<bool, 256> direct{};
        for (int byte = 33; byte <= 126; ++byte)
            direct[static_cast<std::size_t>(byte)] = true;
        for (int byte = 161; byte <= 172; ++byte)
            direct[static_cast<std::size_t>(byte)] = true;
        for (int byte = 174; byte <= 255; ++byte)
            direct[static_cast<std::size_t>(byte)] = true;

        std::unordered_map<char32_t, unsigned char> inverse;
        inverse.reserve(256);
        int displaced = 0;
        for (int byte = 0; byte < 256; ++byte) {
            const char32_t codepoint = direct[static_cast<std::size_t>(byte)]
                                           ? static_cast<char32_t>(byte)
                                           : static_cast<char32_t>(256 + displaced++);
            inverse.emplace(codepoint, static_cast<unsigned char>(byte));
        }
        return inverse;
    }();
    return table;
}

class Lfm2ByteLevelDecoderTokenizer final : public ITokenizer {
  public:
    explicit Lfm2ByteLevelDecoderTokenizer(std::unique_ptr<ITokenizer> inner)
        : inner_(std::move(inner)) {
        if (!inner_)
            throw std::invalid_argument("LFM2 ByteLevel decoder requires an inner tokenizer");
    }

    std::vector<int32_t> encode(const std::string& text) const override {
        return inner_->encode(text);
    }
    std::string decode(const std::vector<int32_t>& ids) const override {
        return lfm2_decode_gpt2_byte_level(inner_->decode(ids));
    }
    int32_t id_for_token(std::string_view token) const override {
        return inner_->id_for_token(token);
    }
    std::string token_for_id(int32_t id) const override { return inner_->token_for_id(id); }

  private:
    std::unique_ptr<ITokenizer> inner_;
};

bool json_has_type(const nlohmann::json& value, std::string_view type) {
    return value.is_object() && value.value("type", std::string{}) == type;
}

bool is_sequence_byte_level_decoder(const nlohmann::json& decoder) {
    if (!json_has_type(decoder, "Sequence"))
        return false;
    const auto children = decoder.find("decoders");
    if (children == decoder.end() || !children->is_array() || children->size() != 1)
        return false;
    return json_has_type((*children)[0], "ByteLevel");
}

} // namespace

bool lfm2_uses_sequence_byte_level_decoder(const char* tokenizer_json, std::size_t size) {
    if (tokenizer_json == nullptr || size == 0)
        return false;
    try {
        const auto root = nlohmann::json::parse(tokenizer_json, tokenizer_json + size);
        const auto decoder = root.find("decoder");
        return decoder != root.end() && is_sequence_byte_level_decoder(*decoder);
    } catch (const nlohmann::json::exception&) {
        return false;
    }
}

std::string lfm2_decode_gpt2_byte_level(std::string_view encoded) {
    const auto& inverse = gpt2_unicode_to_byte();
    std::string decoded;
    decoded.reserve(encoded.size());
    for (std::size_t offset = 0; offset < encoded.size();) {
        const Utf8Unit unit = decode_utf8_unit(encoded, offset);
        if (unit.valid) {
            const auto found = inverse.find(unit.codepoint);
            if (found != inverse.end()) {
                decoded.push_back(static_cast<char>(found->second));
                offset += unit.size;
                continue;
            }
        }
        decoded.append(encoded.data() + offset, unit.size);
        offset += unit.size;
    }
    return decoded;
}

std::unique_ptr<ITokenizer> lfm2_wrap_byte_level_decoder(std::unique_ptr<ITokenizer> tokenizer) {
    return std::make_unique<Lfm2ByteLevelDecoderTokenizer>(std::move(tokenizer));
}

} // namespace trtmc
