/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/runtime/span.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace trtmc::glm_ocr {

// The byte-level BPE tokenizer of GLM-OCR's tokenizer.json: added tokens split the text
// first, the remaining spans pass the checkpoint's pre-tokenizer regex, byte-level mapping
// and BPE with ignore_merges. Construction rejects a tokenizer.json of any other shape.
class Tokenizer {
  public:
    explicit Tokenizer(std::string_view tokenizer_json);

    std::vector<std::int32_t> encode(std::string_view text) const;
    // Byte-level decoding; special added tokens are dropped when skip_special is set,
    // other added tokens contribute their text. Invalid UTF-8 becomes U+FFFD.
    std::string decode(Span<const std::int32_t> ids, bool skip_special) const;
    // Id of an added token by its exact content; throws when absent.
    std::int32_t added_token_id(std::string_view content) const;
    std::size_t vocab_size() const noexcept { return id_to_token_.size(); }

  private:
    struct Added {
        std::string content;
        std::int32_t id{0};
        bool special{false};
    };
    void encode_span(std::string_view text, std::vector<std::int32_t>& ids) const;
    void encode_piece(std::string_view piece, std::vector<std::int32_t>& ids) const;

    std::unordered_map<std::string, std::int32_t> vocab_;
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, std::int32_t> merge_rank_;
    std::vector<Added> added_;
    std::unordered_map<std::int32_t, std::size_t> added_by_id_;
    bool ignore_merges_{false};
    std::string byte_to_unicode_[256];
    std::unordered_map<std::uint32_t, std::uint8_t> unicode_to_byte_;
};

// Pre-tokenizer pieces of one span (exposed for tests). Each piece is a byte range.
std::vector<std::string_view> pre_tokenize(std::string_view text);

} // namespace trtmc::glm_ocr
