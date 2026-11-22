/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "trtmc/runtime/tensor.h"
#include "trtmc/task.h"

#include <cstdint>
#include <vector>
namespace trtmc::d3 {
struct VisualInputs {
    std::vector<uint16_t> patches;
    std::vector<float> factors;
    std::vector<int32_t> indices;
    std::vector<float> angles;
    int64_t count;
    TensorMap tensors();
};
uint16_t bfloat16(float);
VisualInputs prepare_image(const ImageResult&, int32_t height, int32_t width, int32_t positions,
                           int32_t head_dim);
std::vector<float> rope_angles(const std::vector<int32_t>& ids, int32_t image_token, int32_t height,
                               int32_t width, std::size_t image_count);
} // namespace trtmc::d3
