/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/bart/runtime/runtime_config.h"

#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    const std::string valid =
        R"json({"vocab_size":100,"hidden_size":64,"bos_token_id":0,"eos_token_id":2,"pad_token_id":1,"encoder_layers":2,"decoder_layers":2,"encoder_attention_heads":4,"decoder_attention_heads":4,"encoder_ffn_dim":128,"decoder_ffn_dim":128,"max_position_embeddings":256,"has_vision_engine":true,"is_encoder_decoder":true,"decoder_start_token_id":2,"forced_bos_token_id":0,"position_embedding_offset":2,"precision":"fp32","max_cache_length":128,"decoder_engine_layout":"single","tensor_parallel_size":1,"tensor_parallel_mode":"single"})json";
    try {
        trtmc::bart::validate_runtime_config_json(valid);
    } catch (const std::exception& error) {
        std::cerr << "valid runtime config rejected: " << error.what() << '\n';
        return 1;
    }

    const std::string invalid = valid.substr(0, valid.size() - 1) + ",\"extra\":1}";
    try {
        trtmc::bart::validate_runtime_config_json(invalid);
    } catch (const std::exception&) {
        return 0;
    }
    std::cerr << "unexpected runtime field accepted\n";
    return 1;
}
