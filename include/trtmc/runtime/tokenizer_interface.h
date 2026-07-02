/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// ITokenizer: shared tokenizer interface.
// HF equivalent: AutoTokenizer / PreTrainedTokenizer.
//
// Native C++ tokenizers (BpeTokenizer, WordPieceTokenizer, UnigramTokenizer,
// VocabTokenizer, IpaTokenizer) implement this interface. Pipelines use
// ITokenizer without knowing the concrete type.

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

class ITokenizer {
public:
    virtual ~ITokenizer() = default;

    // Encode text → token IDs.
    virtual std::vector<int32_t> encode(const std::string& text) = 0;

    // Decode token IDs → text.
    virtual std::string decode(const std::vector<int32_t>& ids) = 0;
};

} // namespace trtmc
