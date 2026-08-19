/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sam2_engine_contract.h"
#include "sam2_preprocess_cuda.h"

#include <cstddef>
#include <cstdint>

namespace trtmc::sam2 {

namespace {

constexpr std::int64_t kPillowRounding = std::int64_t{1} << (kPillowResizePrecisionBits - 1);
constexpr std::size_t kChannels = 3U;
constexpr std::size_t kOutputArea =
    static_cast<std::size_t>(kPreprocessImageSize) * static_cast<std::size_t>(kPreprocessImageSize);
constexpr std::size_t kHorizontalElements = static_cast<std::size_t>(kOriginalImageHeight) *
                                            static_cast<std::size_t>(kPreprocessImageSize) *
                                            kChannels;
constexpr std::size_t kOutputElements = kOutputArea * kChannels;
constexpr std::uint32_t kThreads = 256U;

__device__ __forceinline__ std::uint8_t applyPillowSpan(const std::uint8_t* source,
                                                        std::int32_t source_stride,
                                                        const PillowResizeSpan& span,
                                                        const std::int32_t* weights) {
    std::int64_t sum = kPillowRounding;
#pragma unroll 1
    for (std::int32_t index = 0; index < span.weight_count; ++index) {
        const auto source_index =
            static_cast<std::size_t>(span.first + index) * static_cast<std::size_t>(source_stride);
        sum += static_cast<std::int64_t>(source[source_index]) *
               static_cast<std::int64_t>(weights[span.weight_offset + index]);
    }
    // Shifting a negative signed integer is implementation-defined in C++.
    // Clamp the negative domain before the shift; this is equivalent to the
    // Pillow uint8 clamp after an arithmetic shift, without relying on it.
    if (sum <= 0)
        return std::uint8_t{0};
    const std::int64_t rounded = sum >> kPillowResizePrecisionBits;
    if (rounded >= 255)
        return std::uint8_t{255};
    return static_cast<std::uint8_t>(rounded);
}

__global__ void pillowHorizontalKernel(const std::uint8_t* source, std::uint8_t* horizontal,
                                       const PillowResizeSpan* spans, const std::int32_t* weights) {
    const auto linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= kHorizontalElements)
        return;

    const auto channel = linear % kChannels;
    const auto pixel = linear / kChannels;
    const auto destination_x = pixel % static_cast<std::size_t>(kPreprocessImageSize);
    const auto source_y = pixel / static_cast<std::size_t>(kPreprocessImageSize);
    const auto source_offset =
        (source_y * static_cast<std::size_t>(kOriginalImageWidth) * kChannels) + channel;
    horizontal[linear] =
        applyPillowSpan(source + source_offset, static_cast<std::int32_t>(kChannels),
                        spans[destination_x], weights);
}

__global__ void pillowVerticalNormalizeKernel(const std::uint8_t* horizontal, float* pixel_values,
                                              const PillowResizeSpan* spans,
                                              const std::int32_t* weights,
                                              const float* normalization_table) {
    const auto linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= kOutputElements)
        return;

    const auto channel = linear % kChannels;
    const auto pixel = linear / kChannels;
    const auto destination_x = pixel % static_cast<std::size_t>(kPreprocessImageSize);
    const auto destination_y = pixel / static_cast<std::size_t>(kPreprocessImageSize);
    const auto horizontal_offset = destination_x * kChannels + channel;
    const auto resized = applyPillowSpan(
        horizontal + horizontal_offset, kPreprocessImageSize * static_cast<std::int32_t>(kChannels),
        spans[destination_y], weights);

    pixel_values[channel * kOutputArea + pixel] =
        normalization_table[channel * kSam2Rgb8ValueCount + resized];
}

std::uint32_t blockCount(std::size_t elements) noexcept {
    return static_cast<std::uint32_t>((elements + kThreads - 1U) / kThreads);
}

} // namespace

cudaError_t enqueueSam2PillowRgb8Preprocess(
    const std::uint8_t* source_rgb_hwc, std::uint8_t* horizontal_rgb_hwc, float* pixel_values_nchw,
    const PillowResizeSpan* horizontal_spans, const std::int32_t* horizontal_weights,
    const PillowResizeSpan* vertical_spans, const std::int32_t* vertical_weights,
    const float* normalization_table, cudaStream_t stream) noexcept {
    if (source_rgb_hwc == nullptr || horizontal_rgb_hwc == nullptr ||
        pixel_values_nchw == nullptr || horizontal_spans == nullptr ||
        horizontal_weights == nullptr || vertical_spans == nullptr || vertical_weights == nullptr ||
        normalization_table == nullptr || stream == nullptr) {
        return cudaErrorInvalidValue;
    }

    pillowHorizontalKernel<<<blockCount(kHorizontalElements), kThreads, 0U, stream>>>(
        source_rgb_hwc, horizontal_rgb_hwc, horizontal_spans, horizontal_weights);
    auto status = cudaPeekAtLastError();
    if (status != cudaSuccess)
        return status;

    pillowVerticalNormalizeKernel<<<blockCount(kOutputElements), kThreads, 0U, stream>>>(
        horizontal_rgb_hwc, pixel_values_nchw, vertical_spans, vertical_weights,
        normalization_table);
    status = cudaPeekAtLastError();
    return status;
}

} // namespace trtmc::sam2
