/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {
struct Idefics3Image {
    int32_t rows{0}, columns{0};
    std::vector<std::vector<float>> views;
};
Idefics3Image preprocess_idefics3_image(const float* rgb, int32_t height, int32_t width,
                                      int32_t longest_edge, int32_t view_size);
std::string idefics3_image_prompt(int32_t rows, int32_t columns, int32_t sequence_length);
} // namespace trtmc
