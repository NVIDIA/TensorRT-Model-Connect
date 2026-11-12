/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

namespace trtmc {

// The checkpoint's uncased WordPiece tokenizer (BertNormalizer, BertPreTokenizer,
// WordPiece), read from its tokenizer.json. encode() returns the ids of the text
// alone: framing with [CLS] and [SEP] belongs to the pipeline, because the
// single-text and the text-pair Tasks frame differently. Special tokens written
// in the text, such as "[MASK]", are matched before normalisation and kept as
// their own ids, as the reference tokenizer does.
class IMobileBertTokenizer {
  public:
    virtual ~IMobileBertTokenizer() = default;
    virtual std::vector<std::int32_t> encode(std::string_view text) const = 0;
    // Number of ids, base vocabulary and added tokens together.
    virtual std::size_t size() const = 0;
    // Id of an exact token string, or -1.
    virtual std::int32_t id_for_token(std::string_view token) const = 0;
};

std::unique_ptr<IMobileBertTokenizer> CreateMobileBertTokenizer(const char* tokenizer_json,
                                                                std::size_t size);

} // namespace trtmc
