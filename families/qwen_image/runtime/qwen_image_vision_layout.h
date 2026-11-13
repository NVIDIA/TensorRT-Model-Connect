/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Pixel layout the Qwen2.5-VL vision engine expects. Qwen2VLImageProcessor
// flattens patches in merge-group order: for each merge_size x merge_size block
// of patches (row-major over blocks), its patches row-major. The engine's patch
// embedding reads patches in raster order and then treats every merge_size^2
// consecutive patches as one merge group, so the pixels must be laid out so
// that raster patch k is the processor's patch k.

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace trtmc {
namespace qwen_image {

// Rearrange a [channels, height, width] image so its raster patch order is the
// processor's merge-group patch order.
inline std::vector<float> to_merge_group_patch_order(const std::vector<float>& chw, int channels,
                                                     int height, int width, int patch, int merge) {
    if (channels <= 0 || patch <= 0 || merge <= 0 || height % (patch * merge) != 0 ||
        width % (patch * merge) != 0 ||
        chw.size() != static_cast<std::size_t>(channels) * height * width) {
        throw std::invalid_argument("to_merge_group_patch_order: invalid image or patch geometry");
    }
    const int grid_w = width / patch;
    const int groups_h = height / patch / merge;
    const int groups_w = grid_w / merge;
    const std::size_t plane = static_cast<std::size_t>(height) * width;
    std::vector<float> out(chw.size());
    int destination = 0;
    for (int gh = 0; gh < groups_h; ++gh) {
        for (int gw = 0; gw < groups_w; ++gw) {
            for (int dh = 0; dh < merge; ++dh) {
                for (int dw = 0; dw < merge; ++dw, ++destination) {
                    const int source_row = (gh * merge + dh) * patch;
                    const int source_col = (gw * merge + dw) * patch;
                    const int target_row = destination / grid_w * patch;
                    const int target_col = destination % grid_w * patch;
                    for (int c = 0; c < channels; ++c) {
                        for (int y = 0; y < patch; ++y) {
                            const float* from = chw.data() + c * plane +
                                                static_cast<std::size_t>(source_row + y) * width +
                                                source_col;
                            float* to = out.data() + c * plane +
                                        static_cast<std::size_t>(target_row + y) * width +
                                        target_col;
                            std::memcpy(to, from, static_cast<std::size_t>(patch) * sizeof(float));
                        }
                    }
                }
            }
        }
    }
    return out;
}

} // namespace qwen_image
} // namespace trtmc
