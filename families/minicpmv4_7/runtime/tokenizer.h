/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <nlohmann/json_fwd.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc::minicpmv4_7 {

class ITokenizer {
  public:
    virtual ~ITokenizer() = default;
    // Prompt text to ids; added tokens are matched verbatim, other text is NFC-normalized,
    // split with the checkpoint's pattern and byte-level BPE encoded. No ids are added.
    virtual std::vector<std::int32_t> encode(const std::string& text) const = 0;
    // Ids to text without special tokens (Hugging Face skip_special_tokens=True).
    virtual std::string decode(const std::vector<std::int32_t>& ids) const = 0;
    virtual std::int32_t id_for_token(std::string_view token) const = 0;
};

// Accepts exactly the MiniCPM-V 4.7 tokenizer.json contract and fails on anything else.
std::unique_ptr<ITokenizer> CreateBpeTokenizer(const char* tokenizer_json_data,
                                               std::size_t tokenizer_json_size);

// The tokenizer's NFC with the Unicode 9.0 data of the tokenizers library: a character assigned
// later stays in place and bounds the normalization around it.
std::string tokenizer_nfc(std::string_view text);

// A tokenizer.json token id: the value when it is a JSON integer in [0, end), checked before it
// is narrowed to 32 bits, so a fractional or wider value never stands for a valid id.
std::optional<std::int32_t> tokenizer_json_id(const nlohmann::json& value, std::int32_t end);

} // namespace trtmc::minicpmv4_7
