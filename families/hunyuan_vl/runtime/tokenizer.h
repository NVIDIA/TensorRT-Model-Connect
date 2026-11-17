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

namespace trtmc::hunyuan_vl {
// Byte-level BPE over a sequence of isolating regex splits, as serialized in tokenizer.json.
class Tokenizer {
  public:
    explicit Tokenizer(std::string_view serialized);
    ~Tokenizer();
    // Added tokens in the text are matched first (leftmost, then longest), like the reference.
    std::vector<int32_t> encode(std::string_view text) const;
    // Skips special tokens; byte-level BPE output is not space-normalized.
    std::string decode(const std::vector<int32_t>& ids) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace trtmc::hunyuan_vl
