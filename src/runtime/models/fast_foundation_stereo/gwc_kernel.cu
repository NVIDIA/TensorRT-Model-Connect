/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/fast_foundation_stereo/gwc_kernel.h"

#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace trtmc {
namespace {

constexpr int kChannels = 224;
constexpr int kGroups = 8;
constexpr int kChannelsPerGroup = kChannels / kGroups;
constexpr int kHeight = 176;
constexpr int kWidth = 176;
constexpr int kDisparities = 48;

__global__ void group_norm_kernel(const __half* input, __half* norm) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = kGroups * kHeight * kWidth;
    if (index >= count)
        return;
    const int width_index = index % kWidth;
    const int tmp = index / kWidth;
    const int height_index = tmp % kHeight;
    const int group = tmp / kHeight;
    float sum = 0.0F;
    for (int channel = 0; channel < kChannelsPerGroup; ++channel) {
        const int channel_index = group * kChannelsPerGroup + channel;
        const int input_index = (channel_index * kHeight + height_index) * kWidth + width_index;
        const float value = __half2float(input[input_index]);
        sum += value * value;
    }
    norm[index] = __float2half_rn(sqrtf(sum));
}

__global__ void gwc_kernel(const __half* reference, const __half* target,
                           const __half* reference_norm, const __half* target_norm,
                           __half* output) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int count = kGroups * kDisparities * kHeight * kWidth;
    if (index >= count)
        return;

    int remaining = index;
    const int width_index = remaining % kWidth;
    remaining /= kWidth;
    const int height_index = remaining % kHeight;
    remaining /= kHeight;
    const int disparity = remaining % kDisparities;
    const int group = remaining / kDisparities;
    const int target_width = width_index - disparity;
    if (target_width < 0) {
        output[index] = __float2half(0.0F);
        return;
    }

    float dot = 0.0F;
    for (int channel = 0; channel < kChannelsPerGroup; ++channel) {
        const int channel_index = group * kChannelsPerGroup + channel;
        const int reference_index = (channel_index * kHeight + height_index) * kWidth + width_index;
        const int target_index = (channel_index * kHeight + height_index) * kWidth + target_width;
        dot += __half2float(reference[reference_index]) * __half2float(target[target_index]);
    }
    const int reference_norm_index = (group * kHeight + height_index) * kWidth + width_index;
    const int target_norm_index = (group * kHeight + height_index) * kWidth + target_width;
    const float denominator = __half2float(reference_norm[reference_norm_index]) *
                                  __half2float(target_norm[target_norm_index]) +
                              1.0e-5F;
    output[index] = __float2half_rn(dot / denominator);
}

} // namespace

cudaError_t launch_fast_foundation_stereo_gwc(const void* reference, const void* target,
                                              void* reference_norm, void* target_norm, void* output,
                                              cudaStream_t stream) {
    constexpr int threads = 256;
    constexpr int norm_count = kGroups * kHeight * kWidth;
    constexpr int output_count = kGroups * kDisparities * kHeight * kWidth;
    group_norm_kernel<<<(norm_count + threads - 1) / threads, threads, 0, stream>>>(
        static_cast<const __half*>(reference), static_cast<__half*>(reference_norm));
    group_norm_kernel<<<(norm_count + threads - 1) / threads, threads, 0, stream>>>(
        static_cast<const __half*>(target), static_cast<__half*>(target_norm));
    gwc_kernel<<<(output_count + threads - 1) / threads, threads, 0, stream>>>(
        static_cast<const __half*>(reference), static_cast<const __half*>(target),
        static_cast<const __half*>(reference_norm), static_cast<const __half*>(target_norm),
        static_cast<__half*>(output));
    return cudaGetLastError();
}

} // namespace trtmc
