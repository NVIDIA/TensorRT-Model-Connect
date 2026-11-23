/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "trtmc/internal/image.h"

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::smolvlm {

// 8-bit RGB, row-major HWC; the processor resamples and crops in this domain.
struct Rgb8Image {
    int height{0};
    int width{0};
    std::vector<uint8_t> pixels;
};

struct ImageProcessing {
    int longest_edge{0}; // Resize target of the longer side before tiling.
    int tile{0};         // Square tile edge, equal to the vision tower's image size.
    double rescale_factor{0};
    std::array<float, 3> mean{};
    std::array<float, 3> std{};
    int image_seq_len{0}; // Image tokens per tile.
};

struct TiledImage {
    int rows{0}; // Zero rows and columns: a single unsplit tile.
    int cols{0};
    int frames{0};             // rows * cols tiles, then the global view.
    std::vector<float> pixels; // [frames, 3, tile, tile], normalized.
};

// SmolVLMImageProcessor geometry. Sizes are (height, width).
std::pair<int, int> resize_to_longest_edge(int height, int width, int longest_edge);
std::pair<int, int> resize_for_tiles(int height, int width, int tile);

// Pillow's two-pass fixed-point Lanczos (support 3) on 8-bit channels,
// including its pass-order rule and its same-size copy.
Rgb8Image resize_lanczos(const Rgb8Image& input, int height, int width);

// Resize, tile, add the global view, rescale and normalize one image.
TiledImage tile_image(const Rgb8Image& image, const ImageProcessing& processing);

// Accept 3-channel UInt8 [0, 255] or Float32 [0, 1] host HWC pixels.
Rgb8Image to_rgb8(const internal::ImageView& view);

} // namespace trtmc::smolvlm
