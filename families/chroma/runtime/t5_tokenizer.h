/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace trtmc::chroma {

// A prompt padded to the bundle's fixed text length.
struct TokenizedPrompt {
    std::vector<std::int32_t> ids;  // tokens, then </s>, then <pad>
    std::vector<std::int32_t> mask; // 1 for the tokens and </s>, 0 for padding
    std::size_t valid{0};           // count of mask ones
};

// The T5 Unigram tokenizer exactly as tokenizers JSON describes it for this
// family: special tokens matched in the raw text, whitespace split, a
// prepended and splitting U+2581 Metaspace, Unigram Viterbi segmentation with
// fused unknown runs, and an appended </s>. Construction rejects any other
// tokenizer configuration.
class T5Tokenizer {
  public:
    T5Tokenizer(const char* json, std::size_t size);

    // Token ids with </s>, without truncation or padding. Rejects invalid UTF-8.
    std::vector<std::int32_t> encode(std::string_view text) const;
    // Truncates to length - 1 tokens before </s>, then pads to length.
    TokenizedPrompt encode_padded(std::string_view text, std::size_t length) const;

  private:
    struct Node {
        std::unordered_map<unsigned char, std::int32_t> children;
        std::int32_t token{-1};
    };
    struct AddedToken {
        std::string content;
        std::int32_t id;
    };

    void insert(const std::string& piece, std::int32_t id);
    void encode_text(std::string_view text, std::vector<std::int32_t>& ids) const;
    void encode_piece(std::string_view piece, std::vector<std::int32_t>& ids) const;

    std::vector<Node> trie_;
    std::vector<double> scores_;
    std::unordered_map<std::string, std::int32_t> ids_;
    std::vector<AddedToken> added_;
    std::int32_t unk_id_{0};
    std::int32_t eos_id_{1};
    std::int32_t pad_id_{0};
    double unk_score_{0.0};
};

} // namespace trtmc::chroma
