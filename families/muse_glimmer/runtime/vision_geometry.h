/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace trtmc::muse_glimmer {
struct VisionGrid {
    std::vector<std::int32_t> position_indices;
    std::vector<float> position_weights;
    std::vector<float> rotary_cos;
    std::vector<float> rotary_sin;
    std::vector<std::int32_t> window_ids;
    std::vector<std::int32_t> merge_order;
};

// Return the patch-grid height and width, not the post-merge token grid.
std::pair<int, int> visual_patch_grid(int height, int width, int patch_size, int max_merged_tokens);
VisionGrid make_vision_grid(int rows, int cols, int table_side, int head_dim, float rope_theta,
                            int max_patches);
} // namespace trtmc::muse_glimmer
