/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace trtmc::sam2 {

inline constexpr std::uint32_t kSam2DeviceStatusMaskLogitsNonFinite = UINT32_C(1) << 0U;
inline constexpr std::uint32_t kSam2DeviceStatusObjectPointerNonFinite = UINT32_C(1) << 1U;
inline constexpr std::uint32_t kSam2DeviceStatusMemoryFeaturesNonFinite = UINT32_C(1) << 2U;
inline constexpr std::uint32_t kSam2DeviceStatusMaskResizeNonFinite = UINT32_C(1) << 3U;

// Validate the three fixed tracker outputs and resize one [1,1,256,256]
// float32 mask-logit plane to the fixed 1280x1088 binary uint8 result. Both
// kernels are enqueued on stream and never synchronize it. status accumulates
// the bits above and must be zeroed by the caller before a request.
cudaError_t enqueueSam2ValidateAndResizeMask(const float* mask_logits, const float* object_pointer,
                                             const std::uint16_t* memory_features,
                                             std::uint8_t* output_mask, std::uint32_t* status,
                                             cudaStream_t stream) noexcept;

} // namespace trtmc::sam2
