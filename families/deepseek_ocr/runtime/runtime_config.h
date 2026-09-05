/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>

namespace trtmc {

struct DeepseekOcrConfig {
    std::int32_t vocab_size{0};
    std::int32_t id_bos{0};
    std::int32_t id_eos{0};
    std::int32_t image_token_id{-1};
    std::int32_t vision_output_dim{0};
    bool has_position_input{true};
    std::int32_t num_layers{0};
    std::int32_t prefill_max_length{0};
    std::string present_k_pattern{"present_k_{i}"};
    std::string present_v_pattern{"present_v_{i}"};
};

struct DeepseekOcrRuntimeConfig {
    std::int32_t tensor_parallel_size{1};
    std::int32_t max_cache_length{0};
    DeepseekOcrConfig model;
    std::string cache_k_pattern;
    std::string cache_v_pattern;
};

DeepseekOcrRuntimeConfig deepseek_ocr_parse_runtime_config(const std::string& text);

} // namespace trtmc
