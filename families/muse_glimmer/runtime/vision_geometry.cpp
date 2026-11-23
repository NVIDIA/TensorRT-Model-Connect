/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/muse_glimmer/runtime/vision_geometry.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace trtmc::muse_glimmer {
std::pair<int, int> visual_patch_grid(int height, int width, int patch_size,
                                      int max_merged_tokens) {
    if (std::min({height, width, patch_size, max_merged_tokens}) < 1)
        throw std::invalid_argument("Invalid Muse visual resize geometry");
    const double factor = 2.0 * patch_size;
    double h = height / factor, w = width / factor;
    if (h * w > max_merged_tokens) {
        const double aspect = w / h;
        h = std::sqrt(max_merged_tokens / aspect);
        w = h * aspect;
    }
    double error = std::numeric_limits<double>::infinity();
    std::pair<int, int> best{2, 2};
    for (const double y : {std::floor(h), std::ceil(h)}) {
        for (const double x : {std::floor(w), std::ceil(w)}) {
            if (y < 1 || x < 1 || y * x > max_merged_tokens)
                continue;
            const double candidate = std::abs(y / x - static_cast<double>(height) / width);
            if (candidate < error) {
                if (std::max(y, x) > std::numeric_limits<int>::max() / 2)
                    throw std::invalid_argument(
                        "Muse visual grid exceeds the supported integer range");
                error = candidate;
                best = {static_cast<int>(y) * 2, static_cast<int>(x) * 2};
            }
        }
    }
    return best;
}

VisionGrid make_vision_grid(int rows, int cols, int table_side, int head_dim, float rope_theta,
                            int max_patches) {
    if (rows < 2 || cols < 2 || rows % 2 || cols % 2 || table_side < 1 || head_dim < 4 ||
        head_dim % 4 || !(rope_theta > 0) || !std::isfinite(rope_theta) || max_patches < 4 ||
        rows > max_patches / cols || table_side > std::numeric_limits<int>::max() / table_side)
        throw std::invalid_argument("Invalid Muse visual grid or positional geometry");
    const int count = rows * cols;
    const auto n = static_cast<std::size_t>(count);
    VisionGrid grid;
    grid.position_indices.reserve(n * 4);
    grid.position_weights.reserve(n * 4);
    grid.rotary_cos.reserve(n * head_dim);
    grid.rotary_sin.reserve(n * head_dim);
    grid.merge_order.reserve(n);
    grid.window_mask.resize(n * n);
    std::vector<float> inv_freq(head_dim / 4);
    for (int i = 0; i < head_dim / 4; ++i)
        inv_freq[i] = 1.0F / std::pow(rope_theta, static_cast<float>(i) / (head_dim / 4));
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            // Align-corners-false interpolation into the learned table, with zero padding.
            const float y = (row + 0.5F) * (static_cast<float>(table_side) / rows) - 0.5F;
            const float x = (col + 0.5F) * (static_cast<float>(table_side) / cols) - 0.5F;
            const int y0 = static_cast<int>(std::floor(y)), x0 = static_cast<int>(std::floor(x));
            for (int a = 0; a < 2; ++a)
                for (int b = 0; b < 2; ++b) {
                    const int i = y0 + a, j = x0 + b;
                    grid.position_indices.push_back(std::clamp(i, 0, table_side - 1) * table_side +
                                                    std::clamp(j, 0, table_side - 1));
                    const float w = (a ? y - y0 : 1.0F - (y - y0)) * (b ? x - x0 : 1.0F - (x - x0));
                    grid.position_weights.push_back(
                        i >= 0 && i < table_side && j >= 0 && j < table_side ? w : 0.0F);
                }
            for (int d = 0; d < head_dim; ++d) {
                const int phase = d % (head_dim / 2);
                const float angle =
                    (phase < head_dim / 4 ? col + 1 : row + 1) * inv_freq[phase % (head_dim / 4)];
                grid.rotary_cos.push_back(std::cos(angle));
                grid.rotary_sin.push_back(std::sin(angle));
            }
            const auto offset = static_cast<std::size_t>(row * cols + col) * n;
            for (int t = 0; t < count; ++t)
                grid.window_mask[offset + t] = row / table_side == (t / cols) / table_side &&
                                                       col / table_side == (t % cols) / table_side
                                                   ? 0.0F
                                                   : -std::numeric_limits<float>::infinity();
        }
    }
    for (int y = 0; y < rows; y += 2)
        for (int x = 0; x < cols; x += 2)
            for (int a = 0; a < 2; ++a)
                for (int b = 0; b < 2; ++b)
                    grid.merge_order.push_back((y + a) * cols + x + b);
    return grid;
}
} // namespace trtmc::muse_glimmer
