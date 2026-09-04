/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/t5/runtime/runtime_config.h"

#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    const std::string valid =
        R"json({"vocab_size":100,"hidden_size":64,"bos_token_id":-1,"eos_token_id":1,"pad_token_id":0,"encoder_layers":2,"decoder_layers":2,"max_source_positions":128,"max_target_positions":128,"has_vision_engine":true,"is_encoder_decoder":true,"d_kv":16,"d_ff":128,"relative_attention_num_buckets":32,"precision":"fp32","max_cache_length":128,"decoder_engine_layout":"single","tensor_parallel_size":1,"tensor_parallel_mode":"single"})json";
    try {
        trtmc::t5::validate_runtime_config_json(valid);
    } catch (const std::exception& error) {
        std::cerr << "valid runtime config rejected: " << error.what() << '\n';
        return 1;
    }

    const std::string invalid = valid.substr(0, valid.size() - 1) + ",\"extra\":1}";
    try {
        trtmc::t5::validate_runtime_config_json(invalid);
    } catch (const std::exception&) {
        return 0;
    }
    std::cerr << "unexpected runtime field accepted\n";
    return 1;
}
