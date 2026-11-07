/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/m2m_100/runtime/request_tokens.h"
#include "families/m2m_100/runtime/runtime_config.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
    constexpr std::int32_t eos = 2;
    constexpr std::int32_t source = 7;
    std::vector<std::int32_t> ids{10, 11, eos};
    trtmc::m2m_100_apply_source_language_token(ids, eos, source);
    if (ids != std::vector<std::int32_t>{source, 10, 11, eos})
        return 2;
    ids = {10, 11};
    trtmc::m2m_100_apply_source_language_token(ids, eos, source);
    if (ids != std::vector<std::int32_t>{source, 10, 11, eos})
        return 3;
    ids = {source, 10, 11, eos};
    trtmc::m2m_100_apply_source_language_token(ids, eos, source);
    if (ids != std::vector<std::int32_t>{source, 10, 11, eos})
        return 4;
    ids = {10, 11, eos, source};
    trtmc::m2m_100_apply_source_language_token(ids, eos, source);
    if (ids != std::vector<std::int32_t>{source, 10, 11, eos})
        return 5;
    ids = {10};
    trtmc::m2m_100_apply_source_language_token(ids, eos, -1);
    if (ids != std::vector<std::int32_t>{10})
        return 6;
    if (trtmc::m2m_100_apply_forced_bos_token(10, 0, 7) != 7 ||
        trtmc::m2m_100_apply_forced_bos_token(10, 1, 7) != 10 ||
        trtmc::m2m_100_apply_forced_bos_token(10, 0, -1) != 10)
        return 7;

    const std::string valid =
        R"json({"vocab_size":100,"hidden_size":64,"bos_token_id":0,"eos_token_id":2,"pad_token_id":1,"num_attention_heads":4,"num_key_value_heads":4,"encoder_layers":2,"decoder_layers":2,"max_source_length":128,"decoder_start_token_id":2,"scale_embedding":true,"has_vision_engine":true,"is_encoder_decoder":true,"precision":"fp32","max_cache_length":128,"decoder_engine_layout":"single"})json";
    try {
        trtmc::m2m_100::validate_runtime_config_json(valid);
    } catch (const std::exception& error) {
        std::cerr << "valid runtime config rejected: " << error.what() << '\n';
        return 1;
    }

    const std::string invalid = valid.substr(0, valid.size() - 1) + ",\"extra\":1}";
    try {
        trtmc::m2m_100::validate_runtime_config_json(invalid);
    } catch (const std::exception&) {
        return 0;
    }
    std::cerr << "unexpected runtime field accepted\n";
    return 1;
}
