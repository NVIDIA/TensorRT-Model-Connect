/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/internal/image.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace trtmc::minicpmv4_6 {

// The largest max_slice_nums the runtime accepts (the builder applies the same bound). Slice grids
// are planned before any pixel work, in time linear in this bound; 4096 slices of the nominal
// 448-pixel size (64 rows each) already fill the decoder's 262144 positions.
inline constexpr std::int32_t kMaxSliceNums = 4096;

// Image-processor settings recorded in the bundle (vision.json).
struct ImageSettings {
    std::int32_t patch_size{14};
    std::int32_t positions_per_side{70};
    std::int32_t scale_resolution{448};
    std::int32_t max_slice_nums{9};
    std::int32_t max_patches{4096};
    std::array<float, 3> mean{0.5F, 0.5F, 0.5F};
    std::array<float, 3> stddev{0.5F, 0.5F, 0.5F};
    void validate() const;
};

// Source thumbnail size and optional slice grid of one image (pixels).
struct SlicePlan {
    std::int32_t source_height{0}, source_width{0};
    std::int32_t rows{0}, cols{0}; // 0 x 0 when the image is not sliced
    std::int32_t refine_height{0}, refine_width{0};
};

// One vision-engine launch: a patch grid with its pixel patches and grid indices.
struct VisionSlice {
    std::int32_t height{0}, width{0};       // patch grid, both multiples of four
    std::vector<float> patches;             // [h * w, 3 * patch * patch], channel-major per patch
    std::vector<std::int32_t> position_ids; // [h * w] learned-position rows
    std::vector<std::int32_t> window_index; // [h * w] patches in 2 x 2 window order
    std::vector<std::int32_t> merge_index;  // [h * w / 4] merged tokens in projector block order
    std::int32_t tokens() const { return height * width / 16; }
};

struct PreparedImage {
    std::vector<VisionSlice> slices; // source thumbnail first, then the grid slices row-major
    std::int32_t rows{0}, cols{0};
};

// Reference slicing arithmetic (16x token compression).
SlicePlan plan_slices(const ImageSettings& settings, std::int32_t height, std::int32_t width);

// The slice plan of an image view (its dimensions checked), and the vision rows the plan produces.
SlicePlan plan_image(const ImageSettings& settings, const internal::ImageView& image);
std::int64_t planned_rows(const ImageSettings& settings, const SlicePlan& plan);

// Separable antialiased Keys bicubic (a = -0.5) on RGB8 HWC data with int16
// fixed-point weights, horizontal pass first and rounding to RGB8 after each
// pass, as the reference resamples uint8 images.
std::vector<std::uint8_t> resize_rgb8(const std::uint8_t* pixels, std::int32_t height,
                                      std::int32_t width, std::int32_t out_height,
                                      std::int32_t out_width);

// Learned-position rows of a patch grid (reference bucketing, GPU arithmetic).
std::vector<std::int32_t> bucket_position_ids(std::int32_t height, std::int32_t width,
                                              std::int32_t side);
// Indices of a grid in 2 x 2 block order: blocks row-major, (dy, dx) inside.
std::vector<std::int32_t> block_order(std::int32_t height, std::int32_t width);

PreparedImage prepare_image(const ImageSettings& settings, const internal::ImageView& image);

// Provider placeholder text of one image (local index `image_index` in the request).
std::string image_placeholder(const PreparedImage& image, std::int32_t image_index);

} // namespace trtmc::minicpmv4_6
