/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace trtmc::ministral3 {

struct Token {
    std::int32_t id{0};
    // End-exclusive byte range of the token in the encoded UTF-8 text.
    std::uint64_t byte_begin{0};
    std::uint64_t byte_end{0};
};

// Splits UTF-8 text into the byte spans the Tekken Split regex matches
// (behavior "Isolated"). Throws std::invalid_argument on invalid UTF-8.
std::vector<std::pair<std::size_t, std::size_t>> tekken_pre_tokenize(std::string_view text);

// The byte-level BPE tokenizer of one exact tokenizer.json contract: no
// normalizer; added tokens split first (leftmost-longest); a Sequence of the
// Tekken Split regex and ByteLevel without regex; BPE with ignore_merges and
// without unknown token, byte fallback or dropout; a template that adds no
// tokens. Loading any other contract throws.
class TekkenTokenizer {
  public:
    static std::shared_ptr<const TekkenTokenizer> load(const char* json, std::size_t size);

    std::vector<Token> encode(std::string_view text) const;
    std::int64_t vocab_size() const noexcept { return vocab_size_; }
    // The most bytes one token can cover. Tokens tile the text, so a text of
    // more than n * max_token_bytes() bytes has more than n tokens.
    std::size_t max_token_bytes() const noexcept { return max_token_bytes_; }

  private:
    struct Merge {
        std::uint32_t rank;
        std::int32_t id;
    };
    struct TrieNode {
        std::unordered_map<unsigned char, std::int32_t> next;
        std::int32_t token{-1};
    };

    TekkenTokenizer() = default;
    void encode_piece(std::string_view text, std::size_t begin, std::size_t end,
                      std::vector<Token>& out) const;
    void encode_segment(std::string_view text, std::size_t begin, std::size_t end,
                        std::vector<Token>& out) const;

    std::unordered_map<std::string, std::int32_t> vocab_;
    std::unordered_map<std::uint64_t, Merge> merges_;
    std::int32_t byte_ids_[256]{};
    std::vector<TrieNode> added_trie_;
    std::int64_t vocab_size_{0};
    std::size_t max_token_bytes_{1};
};

} // namespace trtmc::ministral3
