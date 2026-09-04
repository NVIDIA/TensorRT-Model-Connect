/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <type_traits>

namespace trtmc {

// One output-coordinate row in a flattened quantized antialias resize plan.
// Entries and their Pillow-compatible signed 22-bit fixed-point coefficients
// (stored as int32) are copied to caller-owned device
// buffers before sam3_cuda_preprocess_image is launched.
struct Sam3CudaResizeAxisEntry {
    std::int32_t first;
    std::int32_t weight_offset;
    std::int32_t weight_count;
};
static_assert(std::is_trivial_v<Sam3CudaResizeAxisEntry> &&
                  std::is_standard_layout_v<Sam3CudaResizeAxisEntry>,
              "SAM3 CUDA resize entries must remain POD values");

// Reproduce SAM3's native uint8 image preprocessing entirely on one CUDA
// stream. input_hwc, both scratch buffers, resize plans, output_chw, and
// normalization_lut and nonfinite_status are device pointers. output_offset is measured in
// float elements, so callers can write one lane directly into a batched engine
// input. normalization_lut is a device-resident [3, 256] FP32 table whose
// values reproduce Meta's FP16 image normalization operation order while the
// TensorRT binding remains FP32.
//
// The caller must initialize nonfinite_status to zero on stream. The kernels
// atomically set bit 0 for any non-finite source value and bit 1 for an invalid
// device plan entry; offending values are safely processed as zero so no
// invalid address is dereferenced. When an axis is an identity transform, its
// plan pointers may be null and its counts/precision may be zero.
// horizontal_hwc may likewise be null when input_width == output_width.
// Returns false without launching any kernels for obvious invalid host
// arguments, and true after all kernels have been queued. CUDA launch/runtime
// errors are reported by the caller's ordinary stream error handling.
bool sam3_cuda_preprocess_image(
    const float* input_hwc, std::int32_t input_height, std::int32_t input_width,
    std::uint8_t* quantized_hwc, std::uint8_t* horizontal_hwc,
    const Sam3CudaResizeAxisEntry* horizontal_entries, std::int32_t horizontal_entry_count,
    const std::int32_t* horizontal_weights, std::int32_t horizontal_weight_count,
    unsigned int horizontal_precision, const Sam3CudaResizeAxisEntry* vertical_entries,
    std::int32_t vertical_entry_count, const std::int32_t* vertical_weights,
    std::int32_t vertical_weight_count, unsigned int vertical_precision, float* output_chw,
    std::size_t output_offset, std::int32_t output_height, std::int32_t output_width,
    const float* normalization_lut, int* nonfinite_status, cudaStream_t stream);

// Round FP32 values to the exact BF16-representable FP32 values used by the
// native SAM3 recurrent memory cache. Exact in-place operation
// (source == destination) is supported; partially overlapping ranges are not.
void sam3_round_bfloat16_copy(const float* source, float* destination, std::size_t count,
                              cudaStream_t stream);

} // namespace trtmc
