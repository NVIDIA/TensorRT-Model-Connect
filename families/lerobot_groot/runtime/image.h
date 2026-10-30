/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace trtmc::lerobot_groot {

constexpr std::int32_t kPatch = 16;
constexpr std::int32_t kTemporalPatch = 2;
constexpr std::int32_t kMerge = 2;
constexpr std::int32_t kPatchValues = 3 * kTemporalPatch * kPatch * kPatch;

// OpenCV cv::resize with INTER_AREA where it enlarges an 8-bit image: linear interpolation
// with INTER_AREA's coefficients in 11-bit fixed point, bit for bit. `src` is HWC with
// `channels` interleaved values; the result has the same layout.
std::vector<std::uint8_t> resize_area_up(const std::uint8_t* src, std::int32_t src_height,
                                         std::int32_t src_width, std::int32_t channels,
                                         std::int32_t height, std::int32_t width);

// One square RGB camera image (HWC uint8, `size` x `size`) through LeRobot's evaluation path:
// a centred `crop` x `crop` window (floored offset), INTER_AREA back to `size`, then the
// processor's rescale/normalise and patch flattening. Appends `(size / kPatch)^2` rows of
// kPatchValues floats, in merge-block order, to `out`.
void append_patches(const std::uint8_t* rgb, std::int32_t size, std::int32_t crop,
                    std::vector<float>& out);

} // namespace trtmc::lerobot_groot
