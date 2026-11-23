/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace trtmc::molmo2 {
namespace detail {
constexpr std::size_t rgb_index(std::size_t row, std::size_t column, std::size_t width,
                                std::size_t channel) {
    return (row * width + column) * 3 + channel;
}
} // namespace detail
struct ImageInputs {
    std::vector<float> pixels;
    std::vector<std::int32_t> pooling;
    int crops{0};
    int pooled_height{0};
    int pooled_width{0};
};
ImageInputs preprocess(const float* pixels, int height, int width);
std::string image_prompt(const ImageInputs& inputs);
bool is_image_token(std::int32_t token);
std::vector<float> attention_mask(const std::vector<std::int32_t>& tokens, int past);
} // namespace trtmc::molmo2
