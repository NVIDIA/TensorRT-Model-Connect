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

// Unicode White_Space used by the tokenizer's regex \s alternatives.
inline bool hunyuan_is_whitespace(char32_t cp) {
    return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F ||
           cp == 0x205F || cp == 0x3000;
}

class ITokenizer {
  public:
    virtual ~ITokenizer() = default;
    virtual std::vector<std::int32_t> encode(const std::string& text) const = 0;
    virtual std::string decode(const std::vector<std::int32_t>& ids) const = 0;
    virtual std::int32_t id_for_token(std::string_view token) const = 0;
    virtual std::string token_for_id(std::int32_t id) const = 0;
};

std::unique_ptr<ITokenizer> CreateBpeTokenizer(const char* tokenizer_json_data,
                                               std::size_t tokenizer_json_size,
                                               bool add_special_tokens = false);

} // namespace trtmc
