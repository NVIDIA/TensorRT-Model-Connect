/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <vector>
namespace trtmc::nemotron_omni {
struct Grid {
    int rows, columns;
    int count() const { return rows * columns; }
};
// Shapes below are pixels on input, patches on output. Capacity includes all
// requested images; unlike a training processor, native never truncates text.
std::vector<Grid> image_grids(const std::vector<Grid>& shapes, int patch, int minimum_patches,
                              int maximum_patches, int available_tokens);
Grid video_grid(int height, int width, int patch, int target_patches, bool preserve_aspect);
} // namespace trtmc::nemotron_omni
