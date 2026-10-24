/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "trtmc/task.h"

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc::lfm2_vl {
struct ImageOptions {
    int min_tokens{64};
    int max_tokens{256};
    int max_tiles{10};
    bool split{true};
    bool integer_pixels{true};
};
struct ImageTile {
    int patch_rows{0};
    int patch_cols{0};
    std::vector<float> pixels;
    std::vector<std::int32_t> unshuffle;
};
struct ProcessedImage {
    std::vector<ImageTile> tiles;
    std::string placeholders;
};
std::vector<float> resize_bilinear(const std::vector<float>& pixels, int height, int width,
                                   int channels, int out_height, int out_width, bool quantize);
ProcessedImage process_image(const ImageResult& image, const ImageOptions& options);
struct VisionInputs {
    std::int64_t patch_count{0};
    std::int64_t valid_patches{0};
    std::vector<float> pixels;
    std::vector<float> positions;
    std::vector<std::int32_t> mask;
};
VisionInputs prepare_vision_inputs(const ImageTile& tile, const ImageOptions& options,
                                   const std::vector<float>& position_table, int hidden,
                                   bool bfloat16);

std::vector<float> position_embeddings(const std::vector<float>& table, int grid, int hidden,
                                       int patch_rows, int patch_cols, bool bfloat16 = true);
} // namespace trtmc::lfm2_vl
