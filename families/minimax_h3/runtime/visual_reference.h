/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace trtmc::minimax_h3 {

// Original VisualVAE geometry: 256-pixel tiles, at least 64-pixel overlaps,
// and 16-pixel alignment. This is not the Qwen vision patch layout.
struct VisualReferenceAxis {
    std::vector<int32_t> starts;
    std::vector<int32_t> lengths;
    std::vector<int32_t> overlaps;
};

VisualReferenceAxis make_visual_reference_axis(int32_t pixels);

// Each entry is a raw FP32 [48, latent_frames, H/16, W/16] tile returned by
// the native tile encoder. Stitch posterior moments BEFORE sampling, rather
// than independently sampling each tile or replacing the posterior by its mean.
std::vector<float>
stitch_visual_reference_moments(const std::vector<std::vector<std::vector<float>>>& tiles,
                                int32_t pixel_height, int32_t pixel_width, int32_t latent_frames);

} // namespace trtmc::minimax_h3
