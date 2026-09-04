/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/segformer/runtime/segformer_postprocess_cuda.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace trtmc {

namespace {

constexpr int32_t kBlockSize = 256;

__device__ int32_t clamp_index(int32_t value, int32_t upper_bound) {
    return max(0, min(value, upper_bound - 1));
}

__global__ void segformer_bilinear_argmax_kernel(const float* __restrict__ logits,
                                                 int32_t num_classes, int32_t logits_h,
                                                 int32_t logits_w, int32_t target_h,
                                                 int32_t target_w,
                                                 int32_t* __restrict__ class_map) {
    const int32_t target_index = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const int32_t target_size = target_h * target_w;
    if (target_index >= target_size)
        return;

    const int32_t target_y = target_index / target_w;
    const int32_t target_x = target_index - target_y * target_w;
    const float source_y = (static_cast<float>(target_y) + 0.5F) * static_cast<float>(logits_h) /
                               static_cast<float>(target_h) -
                           0.5F;
    const float source_x = (static_cast<float>(target_x) + 0.5F) * static_cast<float>(logits_w) /
                               static_cast<float>(target_w) -
                           0.5F;
    const int32_t first_y_unclamped = static_cast<int32_t>(floorf(source_y));
    const int32_t first_x_unclamped = static_cast<int32_t>(floorf(source_x));
    const int32_t first_y = clamp_index(first_y_unclamped, logits_h);
    const int32_t second_y = clamp_index(first_y_unclamped + 1, logits_h);
    const int32_t first_x = clamp_index(first_x_unclamped, logits_w);
    const int32_t second_x = clamp_index(first_x_unclamped + 1, logits_w);
    const float y_lerp = source_y - static_cast<float>(first_y_unclamped);
    const float x_lerp = source_x - static_cast<float>(first_x_unclamped);
    const int64_t plane_size = static_cast<int64_t>(logits_h) * logits_w;
    const int64_t first_row = static_cast<int64_t>(first_y) * logits_w;
    const int64_t second_row = static_cast<int64_t>(second_y) * logits_w;

    float best_value = -1.0e30F;
    int32_t best_class = 0;
    for (int32_t class_index = 0; class_index < num_classes; ++class_index) {
        const int64_t class_offset = static_cast<int64_t>(class_index) * plane_size;
        const float top_left = logits[class_offset + first_row + first_x];
        const float top_right = logits[class_offset + first_row + second_x];
        const float bottom_left = logits[class_offset + second_row + first_x];
        const float bottom_right = logits[class_offset + second_row + second_x];
        const float top = top_left * (1.0F - x_lerp) + top_right * x_lerp;
        const float bottom = bottom_left * (1.0F - x_lerp) + bottom_right * x_lerp;
        const float value = top * (1.0F - y_lerp) + bottom * y_lerp;
        if (value > best_value) {
            best_value = value;
            best_class = class_index;
        }
    }
    class_map[target_index] = best_class;
}

} // namespace

cudaError_t launch_segformer_bilinear_argmax(const float* logits, int32_t num_classes,
                                             int32_t logits_h, int32_t logits_w, int32_t target_h,
                                             int32_t target_w, int32_t* class_map,
                                             cudaStream_t stream) {
    if (logits == nullptr || class_map == nullptr || num_classes <= 0 || logits_h <= 0 ||
        logits_w <= 0 || target_h <= 0 || target_w <= 0) {
        return cudaErrorInvalidValue;
    }
    const int64_t target_size = static_cast<int64_t>(target_h) * target_w;
    if (target_size > INT32_MAX)
        return cudaErrorInvalidValue;
    const int32_t grid = static_cast<int32_t>((target_size + kBlockSize - 1) / kBlockSize);
    segformer_bilinear_argmax_kernel<<<grid, kBlockSize, 0, stream>>>(
        logits, num_classes, logits_h, logits_w, target_h, target_w, class_map);
    return cudaGetLastError();
}

} // namespace trtmc
