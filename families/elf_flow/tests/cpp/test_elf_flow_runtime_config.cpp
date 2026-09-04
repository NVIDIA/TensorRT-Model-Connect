/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/elf_flow/runtime/runtime_config.h"

#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    const std::string valid =
        R"json({"max_length":32,"max_input_length":0,"input_dim":64,"text_encoder_dim":64,"vocab_size":100,"denoiser_noise_scale":1.0,"denoiser_p_mean":-1.5,"denoiser_p_std":0.8,"timestep_epsilon":0.05,"latent_mean":0.0,"latent_std":0.2,"encoder_pad_token_id":0})json";
    try {
        trtmc::elf_flow::validate_runtime_config_json(valid);
    } catch (const std::exception& error) {
        std::cerr << "valid runtime config rejected: " << error.what() << '\n';
        return 1;
    }

    const std::string invalid = valid.substr(0, valid.size() - 1) + ",\"extra\":1}";
    try {
        trtmc::elf_flow::validate_runtime_config_json(invalid);
    } catch (const std::exception&) {
        return 0;
    }
    std::cerr << "unexpected runtime field accepted\n";
    return 1;
}
