/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/voxtral/runtime/text.h"

#include <cstring>
#include <stdexcept>

namespace trtmc::voxtral {
namespace {

constexpr const char* kReplacement = "\xEF\xBF\xBD";

std::uint32_t read_u32(const std::vector<char>& table, std::size_t offset) {
    if (offset + 4 > table.size())
        throw std::invalid_argument("voxtral vocabulary section is truncated");
    std::uint32_t value = 0;
    for (std::size_t index = 0; index < 4; ++index)
        value |= static_cast<std::uint32_t>(static_cast<unsigned char>(table[offset + index]))
                 << (8 * index);
    return value;
}

bool in_range(unsigned char value, unsigned char low, unsigned char high) {
    return value >= low && value <= high;
}

// Valid range of the second byte for a lead byte (Unicode Table 3-7).
bool second_byte_valid(unsigned char lead, unsigned char second) {
    if (lead == 0xE0)
        return in_range(second, 0xA0, 0xBF);
    if (lead == 0xED)
        return in_range(second, 0x80, 0x9F);
    if (lead == 0xF0)
        return in_range(second, 0x90, 0xBF);
    if (lead == 0xF4)
        return in_range(second, 0x80, 0x8F);
    return in_range(second, 0x80, 0xBF);
}

std::size_t sequence_length(unsigned char lead) {
    if (lead < 0x80)
        return 1;
    if (in_range(lead, 0xC2, 0xDF))
        return 2;
    if (in_range(lead, 0xE0, 0xEF))
        return 3;
    if (in_range(lead, 0xF0, 0xF4))
        return 4;
    return 0;
}

bool is_lowercase_ascii(char value) {
    return value >= 'a' && value <= 'z';
}

} // namespace

std::string decode_utf8_replace(std::string_view bytes) {
    std::string text;
    text.reserve(bytes.size());
    std::size_t index = 0;
    while (index < bytes.size()) {
        const auto lead = static_cast<unsigned char>(bytes[index]);
        const std::size_t length = sequence_length(lead);
        if (length == 1) {
            text.push_back(static_cast<char>(lead));
            ++index;
            continue;
        }
        if (length == 0) {
            text += kReplacement;
            ++index;
            continue;
        }
        // Consume the longest valid prefix; a broken sequence becomes one U+FFFD
        // and decoding resumes at the first byte that did not belong to it.
        std::size_t valid = 1;
        while (valid < length && index + valid < bytes.size()) {
            const auto next = static_cast<unsigned char>(bytes[index + valid]);
            const bool ok = valid == 1 ? second_byte_valid(lead, next) : in_range(next, 0x80, 0xBF);
            if (!ok)
                break;
            ++valid;
        }
        if (valid == length)
            text.append(bytes.data() + index, length);
        else
            text += kReplacement;
        index += valid;
    }
    return text;
}

Vocabulary::Vocabulary(const std::vector<char>& table, std::int32_t num_special_tokens)
    : num_special_tokens_(num_special_tokens) {
    const std::uint32_t count = read_u32(table, 0);
    if (count == 0 || num_special_tokens < 0 ||
        static_cast<std::uint32_t>(num_special_tokens) > count)
        throw std::invalid_argument("voxtral vocabulary section has an invalid token count");
    offsets_.resize(static_cast<std::size_t>(count) + 1);
    for (std::size_t index = 0; index < offsets_.size(); ++index)
        offsets_[index] = read_u32(table, 4 + 4 * index);
    const std::size_t base = 4 + 4 * offsets_.size();
    for (std::size_t index = 1; index < offsets_.size(); ++index) {
        if (offsets_[index] < offsets_[index - 1])
            throw std::invalid_argument("voxtral vocabulary offsets are not monotonic");
    }
    if (offsets_.front() != 0 || base + offsets_.back() != table.size())
        throw std::invalid_argument("voxtral vocabulary section size does not match its offsets");
    bytes_.assign(table.data() + base, offsets_.back());
}

std::string Vocabulary::decode(const std::vector<std::int32_t>& ids) const {
    std::string text;
    std::string run;
    for (const auto id : ids) {
        if (id < 0 || id >= size())
            throw std::out_of_range("voxtral token id is outside the vocabulary");
        if (id < num_special_tokens_) {
            text += decode_utf8_replace(run);
            run.clear();
            continue;
        }
        const auto index = static_cast<std::size_t>(id);
        run.append(bytes_, offsets_[index], offsets_[index + 1] - offsets_[index]);
    }
    text += decode_utf8_replace(run);
    if (text.size() >= 7 && text.compare(0, 5, "lang:") == 0 && is_lowercase_ascii(text[5]) &&
        is_lowercase_ascii(text[6]))
        text.erase(0, 7);
    return text;
}

std::vector<std::int32_t> transcription_prompt(const PromptContract& contract, std::int64_t chunks,
                                               std::optional<std::string_view> language) {
    if (chunks <= 0 || contract.audio_tokens_per_chunk <= 0)
        throw std::invalid_argument("voxtral prompt needs at least one audio chunk");
    const std::vector<std::int32_t>* suffix = &contract.suffix;
    if (language) {
        const auto found = contract.language_suffix.find(*language);
        if (found == contract.language_suffix.end()) {
            std::string supported;
            for (const auto& item : contract.language_suffix)
                supported += (supported.empty() ? "" : ", ") + item.first;
            throw std::invalid_argument("voxtral does not support source language '" +
                                        std::string(*language) + "'; supported: " + supported);
        }
        suffix = &found->second;
    }
    std::vector<std::int32_t> prompt(contract.prefix);
    prompt.insert(prompt.end(), static_cast<std::size_t>(chunks * contract.audio_tokens_per_chunk),
                  contract.audio_token_id);
    prompt.insert(prompt.end(), suffix->begin(), suffix->end());
    return prompt;
}

} // namespace trtmc::voxtral
