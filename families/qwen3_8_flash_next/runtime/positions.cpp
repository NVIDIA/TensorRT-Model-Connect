/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "positions.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace trtmc::qwen3_8_flash_next {

Positions prompt_positions(const std::vector<std::uint8_t>& types, const std::vector<Grid>& images,
                           const std::vector<Grid>& videos, std::int32_t merge,
                           std::int32_t capacity) {
    if (types.empty() || capacity <= 0 || types.size() > static_cast<std::size_t>(capacity) ||
        merge <= 0)
        throw std::invalid_argument("Invalid Flash-Next prompt length, capacity or spatial merge");
    Positions result{{}, 0};
    result.prompt.reserve(types.size());
    std::size_t image = 0, video = 0;
    std::int32_t tubelet = 0;
    for (std::size_t i = 0; i < types.size();) {
        if (types[i] == 0) {
            const auto pos = result.next++;
            result.prompt.push_back({pos, pos, pos});
            ++i;
            continue;
        }
        const auto kind = types[i];
        std::size_t end = i + 1;
        while (end < types.size() && types[end] == kind)
            ++end;
        const auto& grids = kind == 1 ? images : videos;
        const auto idx = kind == 1 ? image : video;
        if ((kind != 1 && kind != 2) || idx >= grids.size())
            throw std::invalid_argument("Flash-Next media tokens do not have a corresponding grid");
        const auto [t, h, w] = grids[idx];
        if (t <= 0 || h <= 0 || w <= 0 || h % merge || w % merge || (kind == 1 && t != 1))
            throw std::invalid_argument("Invalid Flash-Next media patch grid");
        const auto height = h / merge, width = w / merge;
        if (std::int64_t{height} * width != static_cast<std::int64_t>(end - i))
            throw std::invalid_argument(
                "Flash-Next media placeholder count does not match its grid");
        const auto start = result.next;
        for (std::int32_t y = 0; y < height; ++y)
            for (std::int32_t x = 0; x < width; ++x)
                result.prompt.push_back({start, start + y, start + x});
        result.next += std::max(height, width);
        if (kind == 1) {
            ++image;
        } else if (++tubelet == t) {
            ++video;
            tubelet = 0;
        }
        i = end;
    }
    if (image != images.size() || video != videos.size() || tubelet)
        throw std::invalid_argument("Unconsumed Flash-Next image or video grids");
    return result;
}

std::array<float, 64> rotary_row(Position position, float theta) {
    if (!std::isfinite(theta) || theta <= 0 ||
        *std::min_element(position.begin(), position.end()) < 0)
        throw std::invalid_argument("Invalid Flash-Next rotary parameters");
    std::array<float, 64> row;
    // mrope_section [11,11,10]: H at indices 1,4,...,31, W at 2,5,...,29.
    for (int k = 0; k < 32; ++k) {
        const float inverse = 1.0F / std::pow(theta, float(k) / 32.0F);
        const float phase = static_cast<float>(position[k % 3]) * inverse;
        row[k] = std::cos(phase);
        row[k + 32] = std::sin(phase);
    }
    return row;
}

} // namespace trtmc::qwen3_8_flash_next
