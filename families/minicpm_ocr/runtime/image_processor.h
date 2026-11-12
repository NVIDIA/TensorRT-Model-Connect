/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/internal/image.h"

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace trtmc::minicpm_ocr {

// MiniCPM-OCR image preprocessing ("4x" token mode, slice mode, upscaling of
// small images to the slice resolution). All learned computation stays in the
// vision engine; this file only resizes, slices, normalizes and lays out pixels.
struct ImageProcessorConfig {
    int32_t patch_size{14};
    int32_t merge_size{2};
    int32_t scale_resolution{448};
    int32_t max_slice_nums{9};
    bool slice_mode{true};
    int32_t position_grid{70};
    int32_t max_piece_patches{0};
    std::array<float, 3> mean{0.5F, 0.5F, 0.5F};
    std::array<float, 3> stddev{0.5F, 0.5F, 0.5F};

    void validate() const;
};

struct SliceGrid {
    int32_t rows{0};
    int32_t cols{0};
};

// One overview image or one slice, ready for the vision engine.
struct ImagePiece {
    int32_t height{0}; // patch rows
    int32_t width{0};  // patch columns
    // [height * width, 3 * patch * patch] normalized pixels. Patches are grouped
    // by merge window (window-major, then row and column inside the window), so
    // each run of merge * merge rows forms one visual token.
    std::vector<float> patches;
    std::vector<int32_t> position_ids; // same order as patches

    int32_t tokens(int32_t merge_size) const { return height / merge_size * (width / merge_size); }
};

struct ProcessedImage {
    std::vector<ImagePiece> pieces; // overview first, then slices in row-major order
    SliceGrid grid{};               // {0, 0} when the image is not sliced
};

// Reference-equivalent building blocks, exposed for unit tests.
std::optional<SliceGrid> sliced_grid(int32_t height, int32_t width, int32_t max_slice_nums,
                                     int32_t scale_resolution);
std::array<int32_t, 2> best_resize(double height, double width, int32_t scale_resolution,
                                   int32_t multiple, bool allow_upscale);
std::array<int32_t, 2> refine_size(int32_t height, int32_t width, SliceGrid grid,
                                   int32_t scale_resolution, int32_t multiple);
// Antialiased bicubic resize of a packed HWC RGB8 image with the integer
// arithmetic of PyTorch's CPU uint8 kernel (int16 weights, horizontal pass
// first, uint8 rounding between passes, unchanged axes skipped).
std::vector<uint8_t> resize_rgb8(const std::vector<uint8_t>& image, int32_t height, int32_t width,
                                 int32_t out_height, int32_t out_width);
// Row-major nearest-bucket indices into a grid x grid position table.
std::vector<int32_t> nearest_position_ids(int32_t height, int32_t width, int32_t grid);

ProcessedImage preprocess_image(const ImageProcessorConfig& config,
                                const internal::ImageView& image);

} // namespace trtmc::minicpm_ocr
