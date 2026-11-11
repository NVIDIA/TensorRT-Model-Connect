/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "trtmc/internal/image.h"

#include <array>
#include <cstdint>
#include <vector>

namespace trtmc::lighton_ocr {

struct ImageConfig {
    std::int32_t patch_size{14};
    std::int32_t spatial_merge_size{2};
    std::int32_t max_image_size{1540};
    double rescale_factor{1.0 / 255.0};
    std::array<float, 3> mean{};
    std::array<float, 3> standard_deviation{};
};

struct PreparedImage {
    std::int32_t height{0};
    std::int32_t width{0};
    std::vector<float> pixels;
    std::vector<std::int32_t> position_ids;
};

PreparedImage preprocess(const internal::ImageView&, const ImageConfig&);

} // namespace trtmc::lighton_ocr
