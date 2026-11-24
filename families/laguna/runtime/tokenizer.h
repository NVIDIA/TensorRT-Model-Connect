/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Laguna byte-level BPE tokenizer.
//
// Implements exactly the tokenizer.json contract Laguna ships and rejects any
// other: added tokens matched first (longest match); then Split on newline runs
// `(?:\r?\n)+(?!\r?\n)` (MergedWithNext), Split on the Qwen-style word pattern
// (Isolated) and ByteLevel without its own regex; BPE merges by rank; a
// TemplateProcessing post-processor that prefixes special tokens; ByteLevel
// decoding with special tokens skipped.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc::laguna {

class Tokenizer {
  public:
    static std::unique_ptr<Tokenizer> create(std::string_view tokenizer_json);
    ~Tokenizer();
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    // add_special_tokens inserts the post-processor's leading special tokens. Throws
    // std::invalid_argument as soon as the ids exceed max_tokens, before tokenizing the rest of the
    // text.
    std::vector<std::int32_t> encode(std::string_view text, bool add_special_tokens,
                                     std::size_t max_tokens = SIZE_MAX) const;
    // The longest text that can encode to at most max_tokens ids: no token covers more input bytes
    // than the longest vocabulary or added token.
    std::size_t max_text_bytes(std::size_t max_tokens) const;
    // The largest id of a vocabulary, added or post-processor token.
    std::int32_t max_id() const;
    // Skips special tokens, as HF decode(skip_special_tokens=True).
    std::string decode(const std::vector<std::int32_t>& ids) const;
    // -1 when the token is not in the vocabulary.
    std::int32_t id_for_token(std::string_view token) const;

  private:
    struct State;
    explicit Tokenizer(std::unique_ptr<State> state);
    std::unique_ptr<State> state_;
};

// Pre-tokenizer pieces of one segment without added tokens (exposed for tests).
std::vector<std::string> pre_tokenize(std::string_view text);

} // namespace trtmc::laguna
