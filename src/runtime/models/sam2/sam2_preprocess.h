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
// source support. The same host-generated plan is consumed by the scalar
// qualification oracle and the CUDA RGB8 preprocessing kernels.
struct PillowResizeSpan {
    std::int32_t first{0};
    std::int32_t weight_offset{0};
    std::int32_t weight_count{0};
};

struct PillowResizeAxisPlan {
    std::int32_t input_size{0};
    std::int32_t output_size{0};
    std::vector<PillowResizeSpan> spans;
    std::vector<std::int32_t> weights;
};

struct PreprocessedFrame {
    // Exact Pillow-compatible uint8 RGB resize retained for qualification.
    std::vector<std::uint8_t> resized_rgb_hwc;
    // Float32 NCHW input, shape [1, 3, 1024, 1024].
    std::vector<float> pixel_values;
};

// Generate Pillow 12.3's normalized bicubic coefficient table for one axis.
// Coefficients use signed 22-bit fixed-point quantization. This is public to
// the model-owned runtime so host and CUDA preprocessing cannot silently drift
// to independently generated tables.
PillowResizeAxisPlan makePillowBicubicAxisPlan(std::int32_t input_size, std::int32_t output_size);

// Final FP32 normalized values for every [channel][uint8] pair. Generation
// exactly follows the qualification CPU expression: double division by 255,
// round-to-nearest conversion to float, then float subtraction and division.
// The device runtime uploads this 3 KiB table and performs a lookup instead of
// millions of slow FP64 divisions.
const Sam2Rgb8NormalizationTable& sam2Rgb8NormalizationTable();

// Pillow 12.3 Image.resize default for RGB is antialiased bicubic. This
// family-owned implementation mirrors Pillow's half-pixel coordinates,
// widened downsampling support, 22-bit coefficients, and uint8 rounding after
// each separable pass.
std::vector<std::uint8_t> resizePillowBicubicRgb(const std::uint8_t* input,
                                                 std::int32_t input_height,
                                                 std::int32_t input_width,
                                                 std::int32_t output_height,
                                                 std::int32_t output_width);

// The video ABI supplies tightly packed HWC RGB float32 values in [0, 1]. The
// reference decoded JPEGs are exact uint8/255 values; conversion deliberately
// returns to uint8 before resize, matching the source loader's operation order.
PreprocessedFrame preprocessFrame(const float* input, std::int32_t input_height,
                                  std::int32_t input_width);

// Exact uint8 source variant used by the JPEG decoder and CUDA candidate. It
// preserves the existing float ABI while removing the redundant uint8->float
// ->uint8 round trip for callers that already own decoded RGB8 pixels.
PreprocessedFrame preprocessRgb8Frame(const std::uint8_t* input, std::int32_t input_height,
                                      std::int32_t input_width);

} // namespace trtmc::sam2
