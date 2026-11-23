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
#include <vector>

namespace trtmc {

class ITokenizer {
  public:
    virtual ~ITokenizer() = default;
    virtual std::vector<std::int32_t> encode(const std::string& text) const = 0;
    virtual std::string decode(const std::vector<std::int32_t>& ids) const = 0;
    virtual std::int32_t id_for_token(std::string_view token) const = 0;
    virtual std::string token_for_id(std::int32_t id) const = 0;
};

// token_id_limit is the number of ids the model that reads the encoded text takes, the rows of
// its token embedding. A tokenizer.json that gives a token an id outside [0, token_id_limit)
// fails to load, so encode() never returns one.
std::unique_ptr<ITokenizer> CreateBpeTokenizer(const char* tokenizer_json_data,
                                               std::size_t tokenizer_json_size,
                                               bool add_special_tokens,
                                               std::int32_t token_id_limit);

} // namespace trtmc
