/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace trtmc {
namespace diffusion {
namespace flux_rope {

inline int32_t axis_position(std::size_t axis, int32_t temporal_pos, int32_t height_pos,
                             int32_t width_pos, int32_t sequence_pos) {
    switch (axis) {
    case 0:
        return temporal_pos;
    case 1:
        return height_pos;
    case 2:
        return width_pos;
    case 3:
        return sequence_pos;
    default:
        return 0;
    }
}

} // namespace flux_rope
} // namespace diffusion
} // namespace trtmc
