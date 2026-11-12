/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace trtmc::openai_privacy_filter {

// Splits one UTF-8 segment (without added tokens) with the o200k pre-tokenizer
// rule, returning the byte ranges [begin, end) of the pieces in order.
// Throws std::invalid_argument for malformed UTF-8.
std::vector<std::pair<std::size_t, std::size_t>> o200k_split(std::string_view text);

// Byte-level BPE tokenizer of the privacy-filter checkpoints: added tokens are
// matched first (leftmost-longest), every other segment is split with the o200k
// rule, mapped to byte-level symbols and merged by rank. No special tokens are
// inserted. The tokenizer.json must declare exactly this configuration.
class O200kTokenizer {
  public:
    static O200kTokenizer from_json(std::string_view tokenizer_json);

    std::vector<std::int32_t> encode(std::string_view text) const;
    std::size_t vocabulary_size() const noexcept { return mVocabularySize; }

  private:
    void encode_piece(std::string_view piece, std::vector<std::int32_t>& ids) const;
    void encode_segment(std::string_view segment, std::vector<std::int32_t>& ids) const;

    std::unordered_map<std::string, std::int32_t> mVocab;
    std::unordered_map<std::string, std::int32_t> mMergeRanks; // key: left + '\0' + right
    std::vector<std::pair<std::string, std::int32_t>> mAddedTokens;
    std::size_t mVocabularySize{0};
};

} // namespace trtmc::openai_privacy_filter
