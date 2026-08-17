/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/sam2/sam2_preprocess.h"

#include <cstdint>
#include <cuda_runtime_api.h>

namespace trtmc::sam2 {

// Enqueue the exact fixed SAM2 RGB8 preprocessing pipeline on stream:
//   HWC RGB8 source -> Pillow horizontal uint8 pass ->
//   Pillow vertical uint8 pass + FP32 NCHW normalization.
//
// All pointers are CUDA-device pointers. The plans must be the device copies
// of makePillowBicubicAxisPlan(kOriginalImage{Width,Height}, 1024). The two
// resize passes retain Pillow's signed 22-bit coefficients, rounding bias,
// arithmetic shift, and uint8 clamp. The destination stays FP32 so the
// existing TensorRT image plan (including its graph-owned BF16 cast) is
// unchanged.
cudaError_t enqueueSam2PillowRgb8Preprocess(
    const std::uint8_t* source_rgb_hwc, std::uint8_t* horizontal_rgb_hwc, float* pixel_values_nchw,
    const PillowResizeSpan* horizontal_spans, const std::int32_t* horizontal_weights,
    const PillowResizeSpan* vertical_spans, const std::int32_t* vertical_weights,
    const float* normalization_table, cudaStream_t stream) noexcept;

// Qualification-only lookup fixture. values and normalized_values each hold
// exactly [3, 256] elements; callers populate every channel row with byte
// values 0..255, download the output, and compare all 768 FP32 bit patterns to
// sam2Rgb8NormalizationTable().
cudaError_t enqueueSam2Rgb8NormalizationQualification(const std::uint8_t* values,
                                                      float* normalized_values,
                                                      const float* normalization_table,
                                                      cudaStream_t stream) noexcept;

} // namespace trtmc::sam2
