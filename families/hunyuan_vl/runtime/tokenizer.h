/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
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
    // Throws std::invalid_argument once the text needs more than `budget` tokens; a text whose
    // length alone needs more fails before it is tokenized.
    std::vector<int32_t> encode(std::string_view text,
                                size_t budget = std::numeric_limits<size_t>::max()) const;
    // Skips special tokens; byte-level BPE output is not space-normalized.
    std::string decode(const std::vector<int32_t>& ids) const;
    // The most bytes a text needing at most `budget` tokens may hold: a token spans at most the
    // longest vocabulary entry or added token. A longer text is refused even when the vocabulary
    // would drop some of its bytes (carriage returns, for example).
    size_t max_bytes(size_t budget) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Gathers consecutive text parts that are encoded together. A part that would make the gathered
// text longer than a text of `budget` tokens can be fails before it is copied.
class TextRun {
  public:
    TextRun(const Tokenizer& tokenizer, size_t budget) : tokenizer_(&tokenizer), budget_(budget) {}
    void append(std::string_view part) {
        if (part.size() > tokenizer_->max_bytes(budget_) - text_.size())
            throw std::invalid_argument("HunYuanVL text needs more tokens than its budget");
        text_ += part;
    }
    // Starts an empty run with a new budget.
    void restart(size_t budget) {
        budget_ = budget;
        text_.clear();
    }
    const std::string& text() const { return text_; }

  private:
    const Tokenizer* tokenizer_;
    size_t budget_;
    std::string text_;
};
} // namespace trtmc::hunyuan_vl
