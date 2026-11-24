/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc::vibevoice {

class ITokenizer {
  public:
    virtual ~ITokenizer() = default;
    virtual std::vector<std::int32_t> encode(const std::string& text) const = 0;
    // encode(), but throws std::length_error as soon as the ids exceed `max_tokens`,
    // before the rest of the text is merged.
    virtual std::vector<std::int32_t> encode_within(const std::string& text,
                                                    std::size_t max_tokens) const = 0;
    // The longest UTF-8 text whose ids can fit `max_tokens`: NFC shrinks UTF-8 at most
    // threefold and no token covers more than the widest token's bytes. Longer text can be
    // refused from its size alone, before it is copied or split.
    virtual std::size_t max_text_bytes(std::size_t max_tokens) const = 0;
    virtual std::string decode(const std::vector<std::int32_t>& ids) const = 0;
    virtual std::int32_t id_for_token(std::string_view token) const = 0;
    virtual std::string token_for_id(std::int32_t id) const = 0;
};

// `vocab_limit` is the language model's vocabulary size: every token id must lie
// below it, so a malformed tokenizer.json is rejected before it can index the
// embedding table.
std::unique_ptr<ITokenizer> CreateBpeTokenizer(const char* tokenizer_json_data,
                                               std::size_t tokenizer_json_size,
                                               bool add_special_tokens, std::int32_t vocab_limit);

} // namespace trtmc::vibevoice
