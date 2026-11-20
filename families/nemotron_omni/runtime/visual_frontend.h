/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cstdint>
#include <vector>

namespace trtmc::nemotron_omni {
// Host-side RGB pixels are interleaved, finite float32 in [0, 1]. Resizing
// deliberately retains floating bicubic overshoot, as the checkpoint does.
std::vector<float> resize_rgb(const std::vector<float>& pixels, int height, int width,
                              int target_height, int target_width);
std::vector<float> patch_pixels(const std::vector<std::vector<float>>& frames, int height,
                                int width, int patch);
std::vector<int32_t> shuffle_indices(int rows, int columns);
std::vector<float> visual_positions(const std::vector<float>& table, int side, int hidden, int rows,
                                    int columns);
// Returned indices preserve original token order. Always keeps the first
// tubelet; later selection is stable for equal dissimilarity scores.
std::vector<int32_t> video_retained_indices(const std::vector<float>& features,
                                            int tokens_per_frame, int hidden, double pruning);
} // namespace trtmc::nemotron_omni
