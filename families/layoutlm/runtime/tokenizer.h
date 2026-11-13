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
#include <vector>

namespace trtmc::layoutlm {

// Byte-level BPE over the checkpoint's tokenizer.json, applied to words that are
// already split (OCR words and whitespace-split question words). Each word is
// encoded the way the reference encodes `is_split_into_words` input: added tokens
// are matched first and keep their own ids, and every remaining piece is given one
// leading space (ByteLevel `add_prefix_space`) before the GPT-2 pre-tokenizer
// regex, the byte-to-Unicode map and the BPE merges run.
class WordTokenizer {
  public:
    static std::unique_ptr<WordTokenizer> create(std::string_view tokenizer_json);

    std::vector<std::int32_t> encode_word(std::string_view word) const;

    std::int32_t cls_id() const noexcept { return cls_id_; }
    std::int32_t sep_id() const noexcept { return sep_id_; }
    std::size_t vocab_size() const noexcept { return vocab_size_; }

  private:
    struct AddedToken {
        std::string content;
        std::int32_t id{0};
        bool lstrip{false};
        bool rstrip{false};
    };
    struct Merge {
        std::int32_t rank{0};
        std::int32_t new_id{0};
    };

    WordTokenizer() = default;
    void encode_piece(std::string_view piece, std::vector<std::int32_t>& ids) const;
    void encode_pre_token(std::string_view pre_token, std::vector<std::int32_t>& ids) const;
    const Merge* find_merge(std::int32_t left, std::int32_t right) const;

    std::unordered_map<std::string, std::int32_t> vocab_;
    std::vector<std::string> tokens_;
    std::unordered_map<std::uint64_t, Merge> merges_;
    std::vector<AddedToken> added_;
    std::int32_t byte_ids_[256]{};
    std::int32_t cls_id_{-1};
    std::int32_t sep_id_{-1};
    std::size_t vocab_size_{0};
};

} // namespace trtmc::layoutlm
