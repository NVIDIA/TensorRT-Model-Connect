/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc::voxtral {

// Bytes of every token id from the bundle's vocabulary.bin section.
class Vocabulary {
  public:
    Vocabulary(const std::vector<char>& table, std::int32_t num_special_tokens);
    std::int32_t size() const noexcept { return static_cast<std::int32_t>(offsets_.size()) - 1; }
    // The checkpoint tokenizer's decode with special tokens ignored: each run of
    // regular tokens is joined and decoded as UTF-8 (invalid bytes become
    // U+FFFD), and a leading "lang:xx" language tag is removed.
    std::string decode(const std::vector<std::int32_t>& ids) const;

  private:
    std::vector<std::uint32_t> offsets_;
    std::string bytes_;
    std::int32_t num_special_tokens_;
};

// UTF-8 decoding with U+FFFD for every maximal invalid subsequence.
std::string decode_utf8_replace(std::string_view bytes);

// The transcription prompt: prefix, one placeholder per audio embedding, suffix.
struct PromptContract {
    std::vector<std::int32_t> prefix;
    std::vector<std::int32_t> suffix; // automatic language
    std::map<std::string, std::vector<std::int32_t>, std::less<>> language_suffix; // by code
    std::int32_t audio_token_id{-1};
    std::int32_t audio_tokens_per_chunk{0};
};

std::vector<std::int32_t> transcription_prompt(const PromptContract& contract, std::int64_t chunks,
                                               std::optional<std::string_view> language);

} // namespace trtmc::voxtral
