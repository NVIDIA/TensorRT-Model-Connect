/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace trtmc::qwen3_8_flash_next {

using Position = std::array<std::int32_t, 3>;
using Grid = std::array<std::int32_t, 3>; // temporal, height, width before spatial merge
struct Positions {
    std::vector<Position> prompt;
    // First position after the prompt; not the physical index when media are present.
    std::int32_t next;
};

// Consume the provider processor's token types in message order. A video is
// split into temporal tubelets separated by textual timestamps and vision delimiters.
Positions prompt_positions(const std::vector<std::uint8_t>& types, const std::vector<Grid>& images,
                           const std::vector<Grid>& videos, std::int32_t spatial_merge,
                           std::int32_t capacity);

// 32 interleaved T/H/W phase pairs for the trained partial rotary head.
std::array<float, 64> rotary_row(Position position, float theta);

} // namespace trtmc::qwen3_8_flash_next
