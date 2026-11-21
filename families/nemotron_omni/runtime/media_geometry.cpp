/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "media_geometry.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace trtmc::nemotron_omni {
namespace {
int nearest_even(double value) {
    const double low = std::floor(value);
    if (low < 0 || low >= std::numeric_limits<int>::max())
        throw std::invalid_argument("Omni media dimension is not representable");
    const double fraction = value - low;
    return static_cast<int>(low) + (fraction > .5 || (fraction == .5 && static_cast<int>(low) % 2));
}
Grid image_grid(Grid input, int patch, int minimum, int available) {
    if (input.rows <= 0 || input.columns <= 0)
        throw std::invalid_argument("Omni image dimensions must be positive");
    const int h = nearest_even(double(input.rows) / patch + .5);
    const int w = nearest_even(double(input.columns) / patch + .5);
    if (!h || !w)
        throw std::invalid_argument("Omni image is too small for a complete patch");
    const double scale = std::min(1., std::sqrt(double(available) / (double(h) * w)));
    Grid result{std::max(1, int(std::floor(scale * h))), std::max(1, int(std::floor(scale * w)))};
    if (available > minimum && int64_t(result.rows) * result.columns < minimum) {
        const double up = std::sqrt(double(minimum) / (double(result.rows) * result.columns));
        result.rows = int(std::ceil(up * result.rows));
        result.columns = int(std::ceil(up * result.columns));
    }
    if (result.rows % 2)
        result.rows = int64_t(result.rows + 1) * result.columns <= available
                          ? result.rows + 1
                          : std::max(2, result.rows - 1);
    if (result.columns % 2)
        result.columns = int64_t(result.columns + 1) * result.rows <= available
                             ? result.columns + 1
                             : std::max(2, result.columns - 1);
    if (int64_t(result.rows) * result.columns > std::numeric_limits<int>::max())
        throw std::invalid_argument("Omni visual patch count is not representable");
    return result;
}
} // namespace
std::vector<Grid> image_grids(const std::vector<Grid>& shapes, int patch, int minimum, int maximum,
                              int available_tokens) {
    if (patch <= 0 || minimum < 4 || maximum < minimum || available_tokens <= 0 ||
        available_tokens > std::numeric_limits<int>::max() / 4)
        throw std::invalid_argument("Invalid Omni image token budget");
    if (shapes.empty())
        return {};
    const int budget = available_tokens * 4;
    if (int64_t(minimum) * shapes.size() > static_cast<uint64_t>(budget))
        throw std::invalid_argument(
            "Omni image minimum resolution exceeds remaining context capacity");
    std::vector<int> budgets(shapes.size(), std::max(minimum, std::min(maximum, budget)));
    for (int attempt = 0; attempt < 10; ++attempt) {
        std::vector<Grid> result;
        int64_t total = 0;
        for (size_t index = 0; index < shapes.size(); ++index) {
            result.push_back(image_grid(shapes[index], patch, minimum, budgets[index]));
            total += result.back().count();
        }
        if (total <= budget) {
            for (const auto& grid : result)
                if (grid.count() > maximum)
                    throw std::invalid_argument("Omni image grid exceeds built visual capacity");
            return result;
        }
        const double scale = double(budget) / total;
        bool decreased = false;
        for (size_t i = 0; i < shapes.size(); ++i) {
            const int next = std::max(minimum, int(result[i].count() * scale));
            decreased = decreased || next < budgets[i];
            budgets[i] = next;
        }
        if (!decreased)
            std::fill(budgets.begin(), budgets.end(), minimum);
    }
    throw std::invalid_argument("Omni image grids cannot fit the built context capacity");
}
Grid video_grid(int height, int width, int patch, int target_patches, bool preserve_aspect) {
    if (height <= 0 || width <= 0 || patch <= 0 || target_patches < 4)
        throw std::invalid_argument("Invalid Omni video geometry");
    if (!preserve_aspect) {
        const int side = std::max(2, int(std::sqrt(target_patches)) / 2 * 2);
        return {side, side};
    }
    const double aspect = double(width) / height;
    const int h = std::max(1, nearest_even(std::sqrt(target_patches / aspect)));
    const int w = std::max(1, nearest_even(std::sqrt(target_patches * aspect)));
    const int up_h = h + h % 2, up_w = w + w % 2;
    if (int64_t(up_h) * up_w <= target_patches)
        return {up_h, up_w};
    Grid result{std::max(2, h - h % 2), std::max(2, w - w % 2)};
    if (int64_t(result.rows) * result.columns > target_patches)
        throw std::invalid_argument("Omni video aspect ratio exceeds built visual capacity");
    return result;
}
} // namespace trtmc::nemotron_omni
