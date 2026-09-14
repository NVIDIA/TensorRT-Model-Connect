/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace trtmc {

struct Yolo11PreprocessConfig {
    std::int32_t input_image_h{640};
    std::int32_t input_image_w{640};
    float pad_value{114.0F / 255.0F};
};

// How the source image was fitted into the square network input. The pipeline
// needs it to map boxes back, so it is returned rather than recomputed.
struct Yolo11Letterbox {
    float scale{1.0F};
    float pad_x{0.0F};
    float pad_y{0.0F};
};

// `pixels` is an interleaved RGB image in [0, 1], the layout the CLI's
// image reader produces. The result is planar CHW for the engine.
std::vector<float> preprocess_yolo11_image(const float* pixels, std::int32_t height,
                                           std::int32_t width, const Yolo11PreprocessConfig& config,
                                           Yolo11Letterbox& letterbox);

} // namespace trtmc
