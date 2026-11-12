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

namespace trtmc::apertus {

// Byte-level BPE of the Apertus tokenizer.json: added tokens are matched first,
// the remaining text is split by the serialized Split regex (ICU) and merged.
// encode() adds no special tokens; decode() skips special tokens.
class Tokenizer {
  public:
    explicit Tokenizer(std::string_view serialized);
    ~Tokenizer();
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    std::vector<std::int32_t> encode(std::string_view text) const;
    std::string decode(const std::vector<std::int32_t>& ids) const;
    std::int32_t vocab_size() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace trtmc::apertus
