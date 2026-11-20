/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "trtmc/internal/image.h"

#include <array>
#include <vector>

namespace trtmc::cosmos3_edge {
struct Frame {
    internal::ImageView image;
    int height{}, width{}, patch{}, merge{};
    int rows() const { return height / patch; }
    int cols() const { return width / patch; }
    int tokens() const { return rows() * cols() / (merge * merge); }
};
Frame plan_frame(internal::ImageView image, int patch, int merge, int frames, int min_pixels,
                 int max_pixels, int max_patches);
std::vector<float> patchify(const Frame& frame);
void interpolate_positions(const Frame& frame, int table_side, std::vector<float>& weights);
} // namespace trtmc::cosmos3_edge
