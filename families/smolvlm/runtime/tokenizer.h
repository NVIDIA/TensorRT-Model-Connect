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

namespace trtmc::smolvlm {

// Byte-level BPE for the SmolVLM tokenizer.json pipeline: literal added-token
// split (leftmost-longest), Digits(individual_digits) and ByteLevel(regex)
// pre-tokenization, rank-ordered merges, no normalizer and no post-processor.
class Tokenizer {
  public:
    explicit Tokenizer(std::string_view serialized);
    ~Tokenizer();
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    std::vector<int32_t> encode(std::string_view text) const;
    // Skips special tokens and replaces malformed UTF-8 with U+FFFD.
    std::string decode(const std::vector<int32_t>& ids) const;
    // The id of an added token's literal text, or -1.
    int32_t added_token_id(std::string_view content) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace trtmc::smolvlm
