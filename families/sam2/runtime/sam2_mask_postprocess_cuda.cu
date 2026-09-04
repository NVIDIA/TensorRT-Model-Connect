/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/sam2/runtime/sam2_engine_contract.h"
#include "families/sam2/runtime/sam2_mask_postprocess_cuda.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace trtmc::sam2 {

namespace {

constexpr std::int32_t kSourceHeight = 256;
constexpr std::int32_t kSourceWidth = 256;
constexpr std::size_t kMaskLogitElements = 256U * 256U;
constexpr std::size_t kObjectPointerElements = 256U;
constexpr std::size_t kMemoryFeatureElements = 64U * 64U * 64U;
constexpr std::size_t kOutputMaskElements =
    static_cast<std::size_t>(kOriginalImageHeight) * static_cast<std::size_t>(kOriginalImageWidth);
constexpr std::int32_t kThreads = 256;

__device__ bool bfloat16IsFinite(std::uint16_t value) {
    return (value & UINT16_C(0x7F80)) != UINT16_C(0x7F80);
}

__global__ void validateTrackerOutputsKernel(const float* mask_logits, const float* object_pointer,
                                             const std::uint16_t* memory_features,
                                             std::uint32_t* status) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < kMaskLogitElements && !isfinite(mask_logits[index]))
        atomicOr(status, kSam2DeviceStatusMaskLogitsNonFinite);
    if (index < kObjectPointerElements && !isfinite(object_pointer[index]))
        atomicOr(status, kSam2DeviceStatusObjectPointerNonFinite);
    if (index < kMemoryFeatureElements && !bfloat16IsFinite(memory_features[index]))
        atomicOr(status, kSam2DeviceStatusMemoryFeaturesNonFinite);
}

struct AxisSample {
    std::int32_t low;
    std::int32_t high;
    float high_weight;
};

__device__ AxisSample axisSample(std::int32_t output_index, std::int32_t input_size,
                                 std::int32_t output_size) {
    const float scale = static_cast<float>(input_size) / static_cast<float>(output_size);
    const float source = (static_cast<float>(output_index) + 0.5F) * scale - 0.5F;
    const auto floor_source = static_cast<std::int32_t>(floorf(source));
    const auto unclamped_high = floor_source + 1;
    const auto low = max(0, min(floor_source, input_size - 1));
    const auto high = max(0, min(unclamped_high, input_size - 1));
    float weight = source - static_cast<float>(floor_source);
    if (low == high)
        weight = 0.0F;
    return {low, high, weight};
}

__global__ void resizeAndThresholdMaskKernel(const float* mask_logits, std::uint8_t* output_mask,
                                             std::uint32_t* status) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= kOutputMaskElements)
        return;

    const auto y = static_cast<std::int32_t>(index / kOriginalImageWidth);
    const auto x = static_cast<std::int32_t>(index % kOriginalImageWidth);
    const auto ys = axisSample(y, kSourceHeight, kOriginalImageHeight);
    const auto xs = axisSample(x, kSourceWidth, kOriginalImageWidth);

    const float top_left = mask_logits[static_cast<std::size_t>(ys.low) * kSourceWidth + xs.low];
    const float top_right = mask_logits[static_cast<std::size_t>(ys.low) * kSourceWidth + xs.high];
    const float bottom_left =
        mask_logits[static_cast<std::size_t>(ys.high) * kSourceWidth + xs.low];
    const float bottom_right =
        mask_logits[static_cast<std::size_t>(ys.high) * kSourceWidth + xs.high];

    // Preserve the reference scalar operation order: interpolate x first,
    // then y, and apply the strict source threshold without a sigmoid.
    const float top =
        __fadd_rn(top_left, __fmul_rn(__fsub_rn(top_right, top_left), xs.high_weight));
    const float bottom =
        __fadd_rn(bottom_left, __fmul_rn(__fsub_rn(bottom_right, bottom_left), xs.high_weight));
    const float value = __fadd_rn(top, __fmul_rn(__fsub_rn(bottom, top), ys.high_weight));
    if (!isfinite(value))
        atomicOr(status, kSam2DeviceStatusMaskResizeNonFinite);
    output_mask[index] = static_cast<std::uint8_t>(value > 0.0F);
}

} // namespace

cudaError_t enqueueSam2ValidateAndResizeMask(const float* mask_logits, const float* object_pointer,
                                             const std::uint16_t* memory_features,
                                             std::uint8_t* output_mask, std::uint32_t* status,
                                             cudaStream_t stream) noexcept {
    if (mask_logits == nullptr || object_pointer == nullptr || memory_features == nullptr ||
        output_mask == nullptr || status == nullptr || stream == nullptr) {
        return cudaErrorInvalidValue;
    }

    const auto validation_blocks = static_cast<std::int32_t>(
        (kMemoryFeatureElements + static_cast<std::size_t>(kThreads) - 1U) /
        static_cast<std::size_t>(kThreads));
    validateTrackerOutputsKernel<<<validation_blocks, kThreads, 0, stream>>>(
        mask_logits, object_pointer, memory_features, status);
    auto launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess)
        return launch_status;

    const auto resize_blocks =
        static_cast<std::int32_t>((kOutputMaskElements + static_cast<std::size_t>(kThreads) - 1U) /
                                  static_cast<std::size_t>(kThreads));
    resizeAndThresholdMaskKernel<<<resize_blocks, kThreads, 0, stream>>>(mask_logits, output_mask,
                                                                         status);
    launch_status = cudaGetLastError();
    return launch_status;
}

} // namespace trtmc::sam2
