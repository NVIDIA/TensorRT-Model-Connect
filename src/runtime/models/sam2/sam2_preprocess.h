/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace trtmc::sam2 {

inline constexpr std::int32_t kPreprocessImageSize = 1024;
inline constexpr std::int32_t kPillowResizePrecisionBits = 22;
inline constexpr std::size_t kSam2RgbChannels = 3U;
inline constexpr std::size_t kSam2Rgb8ValueCount = 256U;
inline constexpr std::size_t kSam2Rgb8NormalizationTableElements =
    kSam2RgbChannels * kSam2Rgb8ValueCount;
using Sam2Rgb8NormalizationTable = std::array<float, kSam2Rgb8NormalizationTableElements>;

// Trivially-copyable description of one destination sample's contiguous
// source support. The host-generated plan is uploaded once and consumed by
// the CUDA RGB8 preprocessing kernels.
struct PillowResizeSpan {
    std::int32_t first{0};
    std::int32_t weight_offset{0};
    std::int32_t weight_count{0};
};

struct PillowResizeAxisPlan {
    std::vector<PillowResizeSpan> spans;
    std::vector<std::int32_t> weights;
};

// Generate Pillow 12.3's normalized bicubic coefficient table for one axis.
// Coefficients use signed 22-bit fixed-point quantization. This is public to
// the model-owned runtime so host and CUDA preprocessing cannot silently drift
// to independently generated tables.
PillowResizeAxisPlan makePillowBicubicAxisPlan(std::int32_t input_size, std::int32_t output_size);

// Final FP32 normalized values for every [channel][uint8] pair. Generation
// exactly follows the reference CPU expression: double division by 255,
// round-to-nearest conversion to float, then float subtraction and division.
// The device runtime uploads this 3 KiB table and performs a lookup instead of
// millions of slow FP64 divisions.
const Sam2Rgb8NormalizationTable& sam2Rgb8NormalizationTable();

} // namespace trtmc::sam2
