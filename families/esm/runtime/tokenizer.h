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

namespace trtmc {

class IEsmTokenizer {
  public:
    virtual ~IEsmTokenizer() = default;
    // Encodes the body only; <cls> and <eos> are added by the pipeline.
    virtual std::vector<std::int32_t> encode(const std::string& text) const = 0;
    virtual std::int32_t id_for_token(std::string_view token) const = 0;
    virtual std::string token_for_id(std::int32_t id) const = 0;
    virtual std::size_t size() const = 0;
};

// Reads ESM's vocab.txt: one token per line, the line number is the id.
//
// The matching rule is longest-token-first at each position, not a split on
// whitespace. The reference registers *every* vocabulary entry as a
// no-split token and runs its trie over the text first; only what the trie
// leaves behind reaches its `text.split()`. The visible effect is that
// "MK<mask>TV" becomes M, K, <mask>, T, V and that spaces disappear, which a
// whitespace split would get wrong in both directions.
std::unique_ptr<IEsmTokenizer> CreateEsmTokenizer(const char* vocab_data, std::size_t vocab_size);

} // namespace trtmc
