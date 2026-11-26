/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace trtmc::minicpmv4_7 {

// MiniCPMV4_6ImageProcessor settings that the bundle records (runtime.json "vision").
struct ProcessorConfig {
    int32_t patch_size{14};
    int32_t patches_per_side{70};
    int32_t scale_resolution{448};
    int32_t max_slice_nums{9};
    int32_t merge_factor{4}; // "16x": two 2x2 merges per LLM token side
    int32_t max_patches{4096};
    std::array<float, 3> mean{0.5F, 0.5F, 0.5F};
    std::array<float, 3> stddev{0.5F, 0.5F, 0.5F};
};

// One vision-engine input: a resized crop on a (grid_h, grid_w) patch grid.
struct Crop {
    int32_t grid_h{0};
    int32_t grid_w{0};
    std::vector<float> patches;        // [grid_h * grid_w, 3 * p * p], (channel, y, x) per patch
    std::vector<int32_t> position_ids; // [grid_h * grid_w]
    std::vector<int32_t> window_index; // [grid_h * grid_w]
    std::vector<int32_t> merge_index;  // [grid_h * grid_w / 4]
    int32_t tokens(int32_t merge_factor) const {
        return grid_h / merge_factor * (grid_w / merge_factor);
    }
};

// A preprocessed picture: the thumbnail first, then `rows x cols` detail slices in row-major order.
struct ImageCrops {
    int32_t rows{0};
    int32_t cols{0};
    std::vector<Crop> crops;
};

// Slice grid of the source processor; {0, 0} when the picture is not sliced.
std::array<int32_t, 2> sliced_grid(int32_t height, int32_t width, int32_t max_slice_nums,
                                   int32_t scale_resolution);
// Resized (height, width) in pixels for a source size given as (possibly fractional) numbers.
std::array<int32_t, 2> best_resize(double height, double width, int32_t scale_resolution,
                                   int32_t patch_size, int32_t merge_factor, bool allow_upscale);
std::array<int32_t, 2> refine_size(int32_t height, int32_t width, std::array<int32_t, 2> grid,
                                   int32_t scale_resolution, int32_t patch_size,
                                   int32_t merge_factor);

// torchvision uint8 bicubic resize with antialiasing (PyTorch's CPU fixed-point path), RGB HWC.
std::vector<uint8_t> resize_bicubic_u8(const uint8_t* rgb, int32_t width, int32_t height,
                                       int32_t target_width, int32_t target_height);

// Nearest-bucket indices into the learned patches_per_side^2 position grid.
std::vector<int32_t> nearest_position_ids(int32_t grid_h, int32_t grid_w, int32_t per_side);
// Row-major 2x2 blocks of a row-major (grid_h, grid_w) grid, row-major inside each block.
std::vector<int32_t> block_order(int32_t grid_h, int32_t grid_w);

// Full preprocessing of one RGB uint8 picture (HWC).
ImageCrops preprocess_image(const uint8_t* rgb, int32_t height, int32_t width,
                            const ProcessorConfig& config);

} // namespace trtmc::minicpmv4_7
