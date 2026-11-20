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

namespace trtmc::gemma3n {

// The Gemma 3n tokenizer.json pipeline: added tokens matched verbatim on the
// raw text (leftmost, then longest), spaces replaced by U+2581, and
// byte-fallback BPE over each remaining segment. The builder refuses
// tokenizers that configure anything else.
class Tokenizer {
  public:
    explicit Tokenizer(std::string_view serialized);
    ~Tokenizer();
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    std::vector<int32_t> encode(std::string_view text) const;
    // Decoding skips special added tokens, as skip_special_tokens=True does.
    std::string decode(const std::vector<int32_t>& ids) const;
    int32_t token_id(std::string_view token) const;
    int32_t vocab_size() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace trtmc::gemma3n
