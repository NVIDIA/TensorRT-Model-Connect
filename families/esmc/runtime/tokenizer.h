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

namespace trtmc {

class IEsmcTokenizer {
  public:
    virtual ~IEsmcTokenizer() = default;
    // Encodes the body only; <cls> and <eos> are added by the pipeline.
    virtual std::vector<std::int32_t> encode(const std::string& text) const = 0;
    virtual std::int32_t id_for_token(std::string_view token) const = 0;
    virtual std::string token_for_id(std::int32_t id) const = 0;
    virtual std::size_t size() const = 0;
};

// Reads the bundle's vocab.txt and reproduces the reference tokenizer's rules.
//
// ESMC's tokenizer.json is a BPE model with no merges, no normaliser and no
// pre-tokenizer, and six added special tokens: the five <...> markers and the
// chain break "|". Concatenating those rules gives three rules that are
// visible in the ids, and the last two are where this differs from ESM-2's
// tokenizer in families/esm:
//
//   1. an added token matches whole and longest first, so "MK<mask>TV" is
//      M K <mask> T V rather than six characters;
//   2. everything else is split into *characters* of the vocabulary's
//      alphabet, so a residue string is one id per residue;
//   3. a character outside that alphabet becomes one <unk> of its own - so
//      "mktv" is four <unk>s, and a space is one <unk> rather than being
//      dropped. The reference never merges an unknown run.
//
// A character is a Unicode scalar value: a multi-byte UTF-8 character that is
// not in the alphabet is one <unk>, not one per byte.
std::unique_ptr<IEsmcTokenizer> CreateEsmcTokenizer(const char* vocab_data, std::size_t vocab_size);

} // namespace trtmc
