/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc::minicpmv4_6 {

// Byte-level BPE of the checkpoint's tokenizer.json: added tokens split first,
// NFC normalization and the declared Unicode Split pattern on the remaining
// text, with the Unicode versions of tokenizers' tables (9.0 for NFC, 16.0 for
// the pattern's classes), then BPE merges. Decoding skips special added tokens.
class Tokenizer {
  public:
    virtual ~Tokenizer() = default;
    // The ids of the text, or nothing when it needs more than `limit` ids. The text is read only as
    // far as that takes: one with more code points than max_code_points(limit) is refused unread,
    // and encoding stops at the first id beyond `limit`.
    virtual std::optional<std::vector<std::int32_t>> encode(std::string_view text,
                                                            std::size_t limit) const = 0;
    // The most code points that `tokens` ids can stand for: NFC composes at most four code points
    // into one (the longest canonical decomposition), and an id covers at most as many code points
    // as its longest token has.
    virtual std::size_t max_code_points(std::size_t tokens) const = 0;
    virtual std::string decode(const std::vector<std::int32_t>& ids) const = 0;
    virtual std::int32_t id_for_token(std::string_view token) const = 0;
    virtual std::int32_t vocabulary_size() const = 0;
};

std::unique_ptr<Tokenizer> create_tokenizer(const char* tokenizer_json, std::size_t size);

// The number of code points of UTF-8 text as ICU decodes it (an ill-formed sequence is one U+FFFD).
std::size_t code_points(std::string_view text);

// Exposed for focused tests.
std::vector<std::string> pre_tokenize(std::string_view text);
std::string utf8_lossy(std::string_view bytes);

} // namespace trtmc::minicpmv4_6
