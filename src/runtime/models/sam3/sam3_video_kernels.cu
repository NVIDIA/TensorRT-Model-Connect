/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sam3/sam3_video_kernels.h"

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <limits>

namespace trtmc {
namespace {

constexpr int32_t kBlockSize = 256;
constexpr int32_t kChannels = 3;
constexpr unsigned int kPillowResizePrecision = 22;
constexpr std::size_t kUint8Values = 256;

bool checked_hwc_elements(int32_t height, int32_t width, std::size_t& elements) {
    if (height <= 0 || width <= 0)
        return false;
    const auto height_size = static_cast<std::size_t>(height);
    const auto width_size = static_cast<std::size_t>(width);
    if (height_size > std::numeric_limits<std::size_t>::max() / width_size)
        return false;
    const auto pixels = height_size * width_size;
    if (pixels > std::numeric_limits<std::size_t>::max() / kChannels)
        return false;
    elements = pixels * kChannels;
    return true;
}

bool valid_grid_size(std::size_t elements) {
    const auto block_size = static_cast<std::size_t>(kBlockSize);
    const auto blocks = elements / block_size + (elements % block_size != 0U ? 1U : 0U);
    return blocks > 0U && blocks <= static_cast<std::size_t>(std::numeric_limits<int32_t>::max());
}

unsigned int grid_size(std::size_t elements) {
    const auto block_size = static_cast<std::size_t>(kBlockSize);
    return static_cast<unsigned int>(elements / block_size +
                                     (elements % block_size != 0U ? 1U : 0U));
}

bool valid_resize_plan(bool identity, const Sam3CudaResizeAxisEntry* entries, int32_t entry_count,
                       const int32_t* weights, int32_t weight_count, unsigned int precision,
                       int32_t expected_entry_count) {
    if (identity)
        return true;
    return entries != nullptr && entry_count == expected_entry_count && weights != nullptr &&
           weight_count > 0 && precision == kPillowResizePrecision;
}

__device__ uint8_t clamp_uint8(int32_t value) {
    return static_cast<uint8_t>(value < 0 ? 0 : (value > 255 ? 255 : value));
}

__device__ bool valid_axis_entry(const Sam3CudaResizeAxisEntry& entry, int32_t input_size,
                                 int32_t total_weight_count) {
    return entry.first >= 0 && entry.first < input_size && entry.weight_offset >= 0 &&
           entry.weight_offset <= total_weight_count && entry.weight_count > 0 &&
           entry.weight_count <= input_size - entry.first &&
           entry.weight_count <= total_weight_count - entry.weight_offset;
}

__device__ uint8_t apply_uint8_resize_weights(const uint8_t* values, std::size_t stride,
                                              const int32_t* weights, int32_t weight_count,
                                              unsigned int precision) {
    int32_t accumulated = 1 << (precision - 1U);
    for (int32_t index = 0; index < weight_count; ++index)
        accumulated += static_cast<int32_t>(values[static_cast<std::size_t>(index) * stride]) *
                       static_cast<int32_t>(weights[index]);
    return clamp_uint8(accumulated >> precision);
}

__global__ void quantize_hwc_kernel(const float* input_hwc, uint8_t* quantized_hwc,
                                    std::size_t elements, int* nonfinite_status) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements)
        return;

    const float value = input_hwc[index];
    if (!isfinite(value)) {
        atomicOr(nonfinite_status, 1);
        quantized_hwc[index] = 0;
        return;
    }
    const float clamped = value < 0.0F ? 0.0F : (value > 1.0F ? 1.0F : value);
    const float rounded = __fmaf_rn(clamped, 255.0F, 0.5F);
    quantized_hwc[index] = clamp_uint8(__float2int_rz(rounded));
}

__global__ void horizontal_resize_hwc_kernel(const uint8_t* quantized_hwc, int32_t input_height,
                                             int32_t input_width, uint8_t* horizontal_hwc,
                                             int32_t output_width,
                                             const Sam3CudaResizeAxisEntry* horizontal_entries,
                                             const int32_t* horizontal_weights,
                                             int32_t horizontal_weight_count,
                                             unsigned int horizontal_precision,
                                             std::size_t output_elements, int* status) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= output_elements)
        return;

    const auto output_pixel = index / kChannels;
    const auto channel = index % kChannels;
    const auto output_x =
        static_cast<int32_t>(output_pixel % static_cast<std::size_t>(output_width));
    const auto output_y =
        static_cast<int32_t>(output_pixel / static_cast<std::size_t>(output_width));
    if (output_y >= input_height) {
        horizontal_hwc[index] = 0;
        return;
    }

    const auto entry = horizontal_entries[output_x];
    if (!valid_axis_entry(entry, input_width, horizontal_weight_count)) {
        atomicOr(status, 2);
        horizontal_hwc[index] = 0;
        return;
    }
    const auto input_offset =
        (static_cast<std::size_t>(output_y) * input_width + entry.first) * kChannels + channel;
    horizontal_hwc[index] = apply_uint8_resize_weights(quantized_hwc + input_offset, kChannels,
                                                       horizontal_weights + entry.weight_offset,
                                                       entry.weight_count, horizontal_precision);
}

__device__ void store_normalized_chw(uint8_t value, std::size_t output_pixel,
                                     std::size_t output_plane, std::size_t channel,
                                     const float* normalization_lut, float* output_chw,
                                     std::size_t output_offset) {
    output_chw[output_offset + channel * output_plane + output_pixel] =
        normalization_lut[channel * kUint8Values + static_cast<std::size_t>(value)];
}

__global__ void normalize_hwc_to_chw_kernel(const uint8_t* input_hwc, float* output_chw,
                                            std::size_t output_offset, std::size_t output_plane,
                                            std::size_t output_elements,
                                            const float* normalization_lut) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= output_elements)
        return;

    const auto output_pixel = index / kChannels;
    const auto channel = index % kChannels;
    store_normalized_chw(input_hwc[index], output_pixel, output_plane, channel, normalization_lut,
                         output_chw, output_offset);
}

__global__ void vertical_resize_normalize_kernel(
    const uint8_t* input_hwc, int32_t input_height, int32_t input_width, float* output_chw,
    std::size_t output_offset, int32_t output_height,
    const Sam3CudaResizeAxisEntry* vertical_entries, const int32_t* vertical_weights,
    int32_t vertical_weight_count, unsigned int vertical_precision, std::size_t output_plane,
    std::size_t output_elements, const float* normalization_lut, int* status) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= output_elements)
        return;

    const auto output_pixel = index / kChannels;
    const auto channel = index % kChannels;
    const auto output_x = output_pixel % static_cast<std::size_t>(input_width);
    const auto output_y =
        static_cast<int32_t>(output_pixel / static_cast<std::size_t>(input_width));
    if (output_y >= output_height)
        return;

    const auto entry = vertical_entries[output_y];
    uint8_t value = 0;
    if (valid_axis_entry(entry, input_height, vertical_weight_count)) {
        const auto input_offset = static_cast<std::size_t>(entry.first) * input_width * kChannels +
                                  output_x * kChannels + channel;
        const auto row_stride = static_cast<std::size_t>(input_width) * kChannels;
        value = apply_uint8_resize_weights(input_hwc + input_offset, row_stride,
                                           vertical_weights + entry.weight_offset,
                                           entry.weight_count, vertical_precision);
    } else {
        atomicOr(status, 2);
    }

    store_normalized_chw(value, output_pixel, output_plane, channel, normalization_lut, output_chw,
                         output_offset);
}

__global__ void round_bfloat16_copy_kernel(const float* source, float* destination,
                                           std::size_t count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count)
        return;

    std::uint32_t bits = __float_as_uint(source[index]);
    const std::uint32_t exponent = bits & 0x7F800000U;
    const std::uint32_t mantissa = bits & 0x007FFFFFU;
    if (exponent == 0x7F800000U && mantissa != 0U) {
        bits |= 0x00010000U;
    } else {
        bits += 0x00007FFFU + ((bits >> 16U) & 1U);
    }
    destination[index] = __uint_as_float(bits & 0xFFFF0000U);
}

} // namespace

bool sam3_cuda_preprocess_image(const float* input_hwc, int32_t input_height, int32_t input_width,
                                uint8_t* quantized_hwc, uint8_t* horizontal_hwc,
                                const Sam3CudaResizeAxisEntry* horizontal_entries,
                                int32_t horizontal_entry_count, const int32_t* horizontal_weights,
                                int32_t horizontal_weight_count, unsigned int horizontal_precision,
                                const Sam3CudaResizeAxisEntry* vertical_entries,
                                int32_t vertical_entry_count, const int32_t* vertical_weights,
                                int32_t vertical_weight_count, unsigned int vertical_precision,
                                float* output_chw, std::size_t output_offset, int32_t output_height,
                                int32_t output_width, const float* normalization_lut,
                                int* nonfinite_status, cudaStream_t stream) {
    std::size_t input_elements = 0;
    std::size_t horizontal_elements = 0;
    std::size_t output_elements = 0;
    if (input_hwc == nullptr || quantized_hwc == nullptr || output_chw == nullptr ||
        normalization_lut == nullptr || nonfinite_status == nullptr ||
        !checked_hwc_elements(input_height, input_width, input_elements) ||
        !checked_hwc_elements(input_height, output_width, horizontal_elements) ||
        !checked_hwc_elements(output_height, output_width, output_elements) ||
        !valid_grid_size(input_elements) || !valid_grid_size(horizontal_elements) ||
        !valid_grid_size(output_elements)) {
        return false;
    }

    const bool horizontal_identity = input_width == output_width;
    const bool vertical_identity = input_height == output_height;
    if ((!horizontal_identity && horizontal_hwc == nullptr) ||
        !valid_resize_plan(horizontal_identity, horizontal_entries, horizontal_entry_count,
                           horizontal_weights, horizontal_weight_count, horizontal_precision,
                           output_width) ||
        !valid_resize_plan(vertical_identity, vertical_entries, vertical_entry_count,
                           vertical_weights, vertical_weight_count, vertical_precision,
                           output_height)) {
        return false;
    }

    constexpr std::size_t channels = static_cast<std::size_t>(kChannels);
    const auto output_plane = output_elements / channels;
    if (output_offset > std::numeric_limits<std::size_t>::max() - output_elements)
        return false;
    quantize_hwc_kernel<<<grid_size(input_elements), kBlockSize, 0, stream>>>(
        input_hwc, quantized_hwc, input_elements, nonfinite_status);

    const uint8_t* vertical_input = quantized_hwc;
    if (!horizontal_identity) {
        horizontal_resize_hwc_kernel<<<grid_size(horizontal_elements), kBlockSize, 0, stream>>>(
            quantized_hwc, input_height, input_width, horizontal_hwc, output_width,
            horizontal_entries, horizontal_weights, horizontal_weight_count, horizontal_precision,
            horizontal_elements, nonfinite_status);
        vertical_input = horizontal_hwc;
    }

    if (vertical_identity) {
        normalize_hwc_to_chw_kernel<<<grid_size(output_elements), kBlockSize, 0, stream>>>(
            vertical_input, output_chw, output_offset, output_plane, output_elements,
            normalization_lut);
    } else {
        vertical_resize_normalize_kernel<<<grid_size(output_elements), kBlockSize, 0, stream>>>(
            vertical_input, input_height, output_width, output_chw, output_offset, output_height,
            vertical_entries, vertical_weights, vertical_weight_count, vertical_precision,
            output_plane, output_elements, normalization_lut, nonfinite_status);
    }
    return true;
}

void sam3_round_bfloat16_copy(const float* source, float* destination, std::size_t count,
                              cudaStream_t stream) {
    if (source == nullptr || destination == nullptr || count == 0)
        return;
    const auto blocks = static_cast<unsigned int>((count + kBlockSize - 1U) / kBlockSize);
    round_bfloat16_copy_kernel<<<blocks, kBlockSize, 0, stream>>>(source, destination, count);
}

} // namespace trtmc
