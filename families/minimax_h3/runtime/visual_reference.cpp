/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minimax_h3/runtime/visual_reference.h"

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace trtmc::minimax_h3 {
namespace {

constexpr int32_t kRatio = 16;
constexpr int32_t kTile = 256;
constexpr int32_t kOverlap = 64;
constexpr int32_t kMoments = 48;

std::size_t checked_product(std::size_t lhs, std::size_t rhs) {
    if (rhs && lhs > std::numeric_limits<std::size_t>::max() / rhs)
        throw std::overflow_error("MiniMax-H3 reference moment dimensions overflow");
    return lhs * rhs;
}

} // namespace

VisualReferenceAxis make_visual_reference_axis(int32_t pixels) {
    if (pixels < 2 * kRatio || pixels % kRatio != 0)
        throw std::invalid_argument(
            "MiniMax-H3 reference axis must be latent-aligned and at least 32 pixels");
    if (pixels <= kTile)
        return {{0}, {pixels}, {}};
    // Use wide arithmetic before allocating: valid int32 image dimensions must
    // not overflow the overlap/coverage calculation.
    int64_t count = (static_cast<int64_t>(pixels) + kTile - 1) / kTile;
    while (kTile * count - kOverlap * (count - 1) < pixels)
        ++count;
    VisualReferenceAxis plan;
    plan.lengths.assign(static_cast<std::size_t>(count), kTile);
    plan.overlaps.assign(static_cast<std::size_t>(count - 1), kOverlap);
    const int64_t remaining = kTile * count - kOverlap * (count - 1) - pixels;
    for (int64_t index = 0; index < remaining / kRatio; ++index)
        plan.overlaps[static_cast<std::size_t>(index % (count - 1))] += kRatio;
    plan.starts.push_back(0);
    for (int64_t index = 0; index < count - 1; ++index)
        plan.starts.push_back(plan.starts.back() + kTile -
                              plan.overlaps[static_cast<std::size_t>(index)]);
    if (plan.starts.back() + kTile != pixels)
        throw std::logic_error("MiniMax-H3 reference tiles do not cover the exact input axis");
    return plan;
}

std::vector<float>
stitch_visual_reference_moments(const std::vector<std::vector<std::vector<float>>>& tiles,
                                int32_t pixel_height, int32_t pixel_width, int32_t latent_frames) {
    const auto vertical = make_visual_reference_axis(pixel_height);
    const auto horizontal = make_visual_reference_axis(pixel_width);
    if (latent_frames < 1 || latent_frames > 5)
        throw std::invalid_argument(
            "MiniMax-H3 tile moments require one image or one encoded clip");
    if (tiles.size() != vertical.starts.size())
        throw std::invalid_argument("MiniMax-H3 reference moment tile row count mismatch");
    const std::size_t planes = static_cast<std::size_t>(kMoments * latent_frames);
    const auto height = static_cast<std::size_t>(pixel_height / kRatio);
    const auto width = static_cast<std::size_t>(pixel_width / kRatio);
    // Validate every payload before allocation or indexing, including tiles
    // whose bottom/right overlap will be cropped from the final output.
    for (std::size_t row = 0; row < tiles.size(); ++row) {
        if (tiles[row].size() != horizontal.starts.size())
            throw std::invalid_argument("MiniMax-H3 reference moment tile column count mismatch");
        for (std::size_t col = 0; col < tiles[row].size(); ++col) {
            const auto tile_height = static_cast<std::size_t>(vertical.lengths[row] / kRatio);
            const auto tile_width = static_cast<std::size_t>(horizontal.lengths[col] / kRatio);
            if (tiles[row][col].size() !=
                checked_product(planes, checked_product(tile_height, tile_width)))
                throw std::invalid_argument(
                    "MiniMax-H3 reference moment tile payload size mismatch");
        }
    }
    std::vector<float> output(checked_product(planes, checked_product(height, width)));
    for (std::size_t row = 0; row < tiles.size(); ++row) {
        const auto tile_height = static_cast<std::size_t>(vertical.lengths[row] / kRatio);
        const auto y_begin = static_cast<std::size_t>(vertical.starts[row] / kRatio);
        const auto kept_height =
            tile_height - (row + 1 < tiles.size() ? vertical.overlaps[row] / kRatio : 0);
        const auto y_overlap =
            row ? static_cast<std::size_t>(vertical.overlaps[row - 1] / kRatio) : 0;
        for (std::size_t col = 0; col < tiles[row].size(); ++col) {
            const auto tile_width = static_cast<std::size_t>(horizontal.lengths[col] / kRatio);
            const auto x_begin = static_cast<std::size_t>(horizontal.starts[col] / kRatio);
            const auto kept_width =
                tile_width - (col + 1 < tiles[row].size() ? horizontal.overlaps[col] / kRatio : 0);
            const auto x_overlap =
                col ? static_cast<std::size_t>(horizontal.overlaps[col - 1] / kRatio) : 0;
            for (std::size_t plane = 0; plane < planes; ++plane) {
                for (std::size_t y = 0; y < kept_height; ++y) {
                    for (std::size_t x = 0; x < kept_width; ++x) {
                        float value = tiles[row][col][(plane * tile_height + y) * tile_width + x];
                        if (row && y < y_overlap) {
                            const auto upper_height =
                                static_cast<std::size_t>(vertical.lengths[row - 1] / kRatio);
                            const float upper =
                                tiles[row - 1][col]
                                     [(plane * upper_height + upper_height - y_overlap + y) *
                                          tile_width +
                                      x];
                            const float weight =
                                static_cast<float>(y) / static_cast<float>(y_overlap);
                            value = upper * (1.0F - weight) + value * weight;
                        }
                        // Upstream blends from the ORIGINAL left tile after the
                        // vertical blend. It does not reuse a previously blended
                        // tile or average the four-way corner symmetrically.
                        if (col && x < x_overlap) {
                            const auto left_width =
                                static_cast<std::size_t>(horizontal.lengths[col - 1] / kRatio);
                            const float left =
                                tiles[row][col - 1][(plane * tile_height + y) * left_width +
                                                    left_width - x_overlap + x];
                            const float weight =
                                static_cast<float>(x) / static_cast<float>(x_overlap);
                            value = left * (1.0F - weight) + value * weight;
                        }
                        output[(plane * height + y_begin + y) * width + x_begin + x] = value;
                    }
                }
            }
        }
    }
    return output;
}

} // namespace trtmc::minimax_h3
