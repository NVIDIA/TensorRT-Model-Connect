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
#include <vector>

namespace trtmc::openai_gpt {

// The checkpoint's tokenizer.json, as the tokenizers library runs it: added tokens are
// matched in the raw text; the rest is normalised by the uncased BertNormalizer, split by
// BertPreTokenizer and encoded by BPE, which appends the end-of-word suffix to a word's
// last character and maps each unknown character to the unknown token. No special tokens
// are added. decode() is the BPEDecoder of the same suffix over the ids that are not
// special tokens.
class Tokenizer {
  public:
    explicit Tokenizer(std::string_view tokenizer_json);
    ~Tokenizer();
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    // The ids of `text`. Throws std::invalid_argument when the text is not well-formed
    // UTF-8 or has more than `max_tokens` ids; the text is read only until that is known.
    std::vector<std::int32_t> encode(std::string_view text, std::size_t max_tokens) const;
    std::string decode(const std::vector<std::int32_t>& ids) const;
    // Number of ids: the vocabulary, which holds every added token.
    std::int32_t vocab_size() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace trtmc::openai_gpt
