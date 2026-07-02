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

class ITokenizer {
public:
    virtual ~ITokenizer() = default;

    virtual std::vector<int32_t> encode(const std::string& text) const = 0;
    virtual std::string decode(const std::vector<int32_t>& ids) const = 0;

    virtual int32_t id_for_token(std::string_view token) const = 0;
    virtual std::string token_for_id(int32_t id) const = 0;
};

std::unique_ptr<ITokenizer> CreateVocabTokenizer(std::vector<std::string> vocab);
std::unique_ptr<ITokenizer> CreateIpaTokenizer(
    const char* phoneme_dict_data, std::size_t phoneme_dict_size,
    const char* heteronyms_data, std::size_t heteronyms_size,
    const char* vocab_data, std::size_t vocab_size,
    const char* config_data, std::size_t config_size);

std::unique_ptr<ITokenizer> CreateBpeTokenizer(
    const char* tokenizer_json_data,
    std::size_t tokenizer_json_size,
    bool add_special_tokens = false);

std::unique_ptr<ITokenizer> CreateWordPieceTokenizer(
    const char* tokenizer_json_data,
    std::size_t tokenizer_json_size,
    bool add_special_tokens = true);

std::unique_ptr<ITokenizer> CreateUnigramTokenizer(
    const char* tokenizer_json_data,
    std::size_t tokenizer_json_size,
    bool add_special_tokens = true);

} // namespace trtmc
