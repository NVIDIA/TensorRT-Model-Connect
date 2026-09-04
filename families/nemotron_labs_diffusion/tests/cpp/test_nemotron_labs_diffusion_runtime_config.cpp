/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/nemotron_labs_diffusion/runtime/runtime_config.h"

#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    const std::string valid =
        R"json({"vocab_size":100,"hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,"bos_token_id":0,"eos_token_id":2,"pad_token_id":1,"max_cache_length":128,"precision":"fp16"})json";
    try {
        trtmc::nemotron_labs_diffusion::validate_runtime_config_json(valid);
    } catch (const std::exception& error) {
        std::cerr << "valid runtime config rejected: " << error.what() << '\n';
        return 1;
    }

    const std::string invalid = valid.substr(0, valid.size() - 1) + ",\"extra\":1}";
    try {
        trtmc::nemotron_labs_diffusion::validate_runtime_config_json(invalid);
    } catch (const std::exception&) {
        return 0;
    }
    std::cerr << "unexpected runtime field accepted\n";
    return 1;
}
