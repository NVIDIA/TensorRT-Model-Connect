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

std::unique_ptr<ITokenizer>
CreateIpaTokenizer(const char* phoneme_dict_data, std::size_t phoneme_dict_size,
                   const char* heteronyms_data, std::size_t heteronyms_size, const char* vocab_data,
                   std::size_t vocab_size, const char* config_data, std::size_t config_size);

} // namespace trtmc
