/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc::granite_speech {

// Byte-level detokenisation: the bytes of every token id, and the ids decoding skips (special
// tokens). Decoding concatenates the bytes of the kept ids and repairs invalid UTF-8 the way the
// reference tokenizer does: each maximal ill-formed subsequence becomes one U+FFFD.
class Vocabulary {
  public:
    Vocabulary() = default;
    Vocabulary(std::vector<std::string> token_bytes, const std::vector<std::int32_t>& skipped);

    static Vocabulary from_json(std::string_view json);

    std::size_t size() const noexcept { return bytes_.size(); }
    std::string decode(const std::vector<std::int32_t>& ids) const;

  private:
    std::vector<std::string> bytes_;
    std::vector<bool> skip_;
};

// Valid UTF-8 copied unchanged; every maximal ill-formed subsequence replaced by U+FFFD.
std::string utf8_lossy(std::string_view bytes);

} // namespace trtmc::granite_speech
