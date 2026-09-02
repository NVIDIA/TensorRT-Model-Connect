/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/qwen3_embedding/embedding_pipeline.h"
#include "runtime/models/qwen3_embedding/plugin_helpers.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require_close(float actual, float expected) {
    if (std::abs(actual - expected) > 1.0e-6F)
        throw std::runtime_error("unexpected normalized embedding value");
}

void test_last_token_pool_handles_right_padding() {
    const std::vector<float> hidden{
        1.0F, 0.0F,  0.0F, 2.0F,  3.0F, 4.0F,  9.0F, 9.0F,
        5.0F, 12.0F, 8.0F, 15.0F, 7.0F, 24.0F, 9.0F, 40.0F,
    };
    const std::vector<int32_t> mask{1, 1, 1, 0, 1, 1, 1, 0};

    const auto pooled = trtmc::qwen_last_token_pool_and_normalize(hidden, mask, 2, 4, 2);

    require_close(pooled[0], 0.6F);
    require_close(pooled[1], 0.8F);
    require_close(pooled[2], 0.28F);
    require_close(pooled[3], 0.96F);
}

void test_last_token_pool_handles_left_and_mixed_padding() {
    const std::vector<float> hidden{
        99.0F, 99.0F, 3.0F, 4.0F, 5.0F, 12.0F, 8.0F, 15.0F, 7.0F, 24.0F, 99.0F, 99.0F,
    };
    const std::vector<int32_t> mask{0, 1, 1, 1, 1, 0};

    const auto pooled = trtmc::qwen_last_token_pool_and_normalize(hidden, mask, 2, 3, 2);

    require_close(pooled[0], 5.0F / 13.0F);
    require_close(pooled[1], 12.0F / 13.0F);
    require_close(pooled[2], 7.0F / 25.0F);
    require_close(pooled[3], 24.0F / 25.0F);
}

void test_last_token_pool_rejects_empty_rows() {
    bool threw = false;
    try {
        (void)trtmc::qwen_last_token_pool_and_normalize(std::vector<float>(4, 0.0F),
                                                        std::vector<int32_t>{0, 0}, 1, 2, 2);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    if (!threw)
        throw std::runtime_error("empty attention-mask row was accepted");
}

void test_kernel_filename_component_cannot_escape_temp_directory() {
    const auto value = trtmc::sanitize_kernel_filename_component("../flashinfer/decode.so");
    if (value.find('/') != std::string::npos || value.find("..") != std::string::npos)
        throw std::runtime_error("kernel filename sanitizer preserved traversal syntax");
}

} // namespace

int main() {
    try {
        test_last_token_pool_handles_right_padding();
        test_last_token_pool_handles_left_and_mixed_padding();
        test_last_token_pool_rejects_empty_rows();
        test_kernel_filename_component_cannot_escape_temp_directory();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
