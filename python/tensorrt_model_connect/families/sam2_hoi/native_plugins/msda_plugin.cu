/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Forward interpolation arithmetic is adapted from MMCV 1.7.2 commit
 * 4c01b026f0afa5a91a5f54aea313788da1e40f95,
 * mmcv/ops/csrc/common/cuda/ms_deform_attn_cuda_kernel.cuh, and its
 * Deformable DETR source:
 * Copyright (c) 2020 SenseTime. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0.
 */

#include "msda_plugin.h"

#include <algorithm>
#include <cstdint>
#include <cuda_bf16.h>

namespace trtmc::sam2_hoi {
namespace {

constexpr int32_t kNumLevels = 3;
constexpr int32_t kNumPoints = 4;
constexpr int32_t kSpatialSize = (128 * 128) + (64 * 64) + (32 * 32);
__device__ constexpr int32_t kLevelHeights[kNumLevels] = {128, 64, 32};
__device__ constexpr int32_t kLevelWidths[kNumLevels] = {128, 64, 32};
__device__ constexpr int32_t kLevelStarts[kNumLevels] = {0, 128 * 128, (128 * 128) + (64 * 64)};

template <typename T>
__device__ __forceinline__ T from_float(float value) {
    return static_cast<T>(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float(float value) {
    return __float2bfloat16_rn(value);
}

template <typename T>
__device__ __forceinline__ float to_float(T value) {
    return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float to_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ T add(T left, T right) {
    return left + right;
}

template <>
__device__ __forceinline__ __nv_bfloat16 add(__nv_bfloat16 left, __nv_bfloat16 right) {
    return __float2bfloat16_rn(__bfloat162float(left) + __bfloat162float(right));
}

template <typename T>
__device__ __forceinline__ T subtract(T left, T right) {
    return left - right;
}

template <>
__device__ __forceinline__ __nv_bfloat16 subtract(__nv_bfloat16 left, __nv_bfloat16 right) {
    return __float2bfloat16_rn(__bfloat162float(left) - __bfloat162float(right));
}

template <typename T>
__device__ __forceinline__ T multiply(T left, T right) {
    return left * right;
}

template <>
__device__ __forceinline__ __nv_bfloat16 multiply(__nv_bfloat16 left, __nv_bfloat16 right) {
    return __float2bfloat16_rn(__bfloat162float(left) * __bfloat162float(right));
}

template <typename T>
__device__ __forceinline__ T bilinear(const T* level_value, int32_t height, int32_t width,
                                      int32_t num_heads, int32_t channels, T h, T w, int32_t head,
                                      int32_t channel) {
    const int32_t h_low = static_cast<int32_t>(floorf(to_float(h)));
    const int32_t w_low = static_cast<int32_t>(floorf(to_float(w)));
    const int32_t h_high = h_low + 1;
    const int32_t w_high = w_low + 1;
    const T lh = subtract(h, from_float<T>(static_cast<float>(h_low)));
    const T lw = subtract(w, from_float<T>(static_cast<float>(w_low)));
    const T hh = subtract(from_float<T>(1.0F), lh);
    const T hw = subtract(from_float<T>(1.0F), lw);

    const int32_t width_stride = num_heads * channels;
    const int32_t height_stride = width * width_stride;
    const int32_t base = head * channels + channel;
    T v1 = from_float<T>(0.0F);
    T v2 = from_float<T>(0.0F);
    T v3 = from_float<T>(0.0F);
    T v4 = from_float<T>(0.0F);
    if (h_low >= 0 && w_low >= 0)
        v1 = level_value[h_low * height_stride + w_low * width_stride + base];
    if (h_low >= 0 && w_high <= width - 1)
        v2 = level_value[h_low * height_stride + w_high * width_stride + base];
    if (h_high <= height - 1 && w_low >= 0)
        v3 = level_value[h_high * height_stride + w_low * width_stride + base];
    if (h_high <= height - 1 && w_high <= width - 1)
        v4 = level_value[h_high * height_stride + w_high * width_stride + base];

    const T w1 = multiply(hh, hw);
    const T w2 = multiply(hh, lw);
    const T w3 = multiply(lh, hw);
    const T w4 = multiply(lh, lw);
    T result = add(multiply(w1, v1), multiply(w2, v2));
    result = add(result, multiply(w3, v3));
    return add(result, multiply(w4, v4));
}

template <typename T>
__global__ void ms_deform_attn_kernel(const T* value, const T* sampling_locations,
                                      const T* attention_weights, int32_t batch,
                                      int32_t num_queries, int32_t num_heads, int32_t channels,
                                      T* output) {
    const int64_t total = static_cast<int64_t>(batch) * num_queries * num_heads * channels;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        int64_t remaining = index;
        const int32_t channel = static_cast<int32_t>(remaining % channels);
        remaining /= channels;
        const int32_t head = static_cast<int32_t>(remaining % num_heads);
        remaining /= num_heads;
        const int32_t query = static_cast<int32_t>(remaining % num_queries);
        remaining /= num_queries;
        const int32_t batch_index = static_cast<int32_t>(remaining);

        T accumulated = from_float<T>(0.0F);
        for (int32_t level = 0; level < kNumLevels; ++level) {
            const int32_t height = kLevelHeights[level];
            const int32_t width = kLevelWidths[level];
            const T* level_value =
                value + (static_cast<int64_t>(batch_index) * kSpatialSize + kLevelStarts[level]) *
                            num_heads * channels;
            for (int32_t point = 0; point < kNumPoints; ++point) {
                const int64_t sample =
                    ((((static_cast<int64_t>(batch_index) * num_queries + query) * num_heads +
                       head) *
                          kNumLevels +
                      level) *
                         kNumPoints +
                     point);
                const T location_w = sampling_locations[sample * 2];
                const T location_h = sampling_locations[sample * 2 + 1];
                const T h_im =
                    subtract(multiply(location_h, from_float<T>(height)), from_float<T>(0.5F));
                const T w_im =
                    subtract(multiply(location_w, from_float<T>(width)), from_float<T>(0.5F));
                if (to_float(h_im) > -1.0F && to_float(w_im) > -1.0F &&
                    to_float(h_im) < static_cast<float>(height) &&
                    to_float(w_im) < static_cast<float>(width)) {
                    const T sampled = bilinear(level_value, height, width, num_heads, channels,
                                               h_im, w_im, head, channel);
                    accumulated = add(accumulated, multiply(sampled, attention_weights[sample]));
                }
            }
        }
        output[index] = accumulated;
    }
}

bool valid_dimensions(nvinfer1::PluginTensorDesc const* input) {
    const auto& value = input[0].dims;
    const auto& locations = input[1].dims;
    const auto& weights = input[2].dims;
    return value.nbDims == 4 && locations.nbDims == 6 && weights.nbDims == 5 &&
           value.d[0] == locations.d[0] && value.d[0] == weights.d[0] &&
           value.d[1] == kSpatialSize && value.d[2] == locations.d[2] &&
           value.d[2] == weights.d[2] && locations.d[1] == weights.d[1] &&
           locations.d[3] == kNumLevels && weights.d[3] == kNumLevels &&
           locations.d[4] == kNumPoints && weights.d[4] == kNumPoints && locations.d[5] == 2;
}

} // namespace

MsDeformAttnPlugin::MsDeformAttnPlugin(const void* data, std::size_t length) {
    (void)data;
    (void)length;
}

char const* MsDeformAttnPlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}
char const* MsDeformAttnPlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}
int32_t MsDeformAttnPlugin::getNbOutputs() const noexcept {
    return 1;
}
int32_t MsDeformAttnPlugin::initialize() noexcept {
    return 0;
}
void MsDeformAttnPlugin::terminate() noexcept {}
void MsDeformAttnPlugin::destroy() noexcept {
    delete this;
}
std::size_t MsDeformAttnPlugin::getSerializationSize() const noexcept {
    return 0;
}
void MsDeformAttnPlugin::serialize(void* buffer) const noexcept {
    (void)buffer;
}

void MsDeformAttnPlugin::setPluginNamespace(char const* plugin_namespace) noexcept {
    namespace_ = plugin_namespace != nullptr ? plugin_namespace : "";
}

char const* MsDeformAttnPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::DataType MsDeformAttnPlugin::getOutputDataType(int32_t index,
                                                         nvinfer1::DataType const* input_types,
                                                         int32_t num_inputs) const noexcept {
    return index == 0 && num_inputs == 3 ? input_types[0] : nvinfer1::DataType::kFLOAT;
}

MsDeformAttnPlugin* MsDeformAttnPlugin::clone() const noexcept {
    auto* cloned = new MsDeformAttnPlugin();
    cloned->setPluginNamespace(namespace_.c_str());
    return cloned;
}

nvinfer1::DimsExprs
MsDeformAttnPlugin::getOutputDimensions(int32_t output_index, nvinfer1::DimsExprs const* inputs,
                                        int32_t num_inputs,
                                        nvinfer1::IExprBuilder& expression_builder) noexcept {
    nvinfer1::DimsExprs output{};
    if (output_index != 0 || num_inputs != 3 || inputs[0].nbDims != 4 || inputs[1].nbDims != 6)
        return output;
    output.nbDims = 3;
    output.d[0] = inputs[0].d[0];
    output.d[1] = inputs[1].d[1];
    output.d[2] = expression_builder.operation(nvinfer1::DimensionOperation::kPROD, *inputs[0].d[2],
                                               *inputs[0].d[3]);
    return output;
}

bool MsDeformAttnPlugin::supportsFormatCombination(int32_t position,
                                                   nvinfer1::PluginTensorDesc const* inputs_outputs,
                                                   int32_t num_inputs,
                                                   int32_t num_outputs) noexcept {
    if (num_inputs != 3 || num_outputs != 1 || position < 0 || position >= 4)
        return false;
    const auto type = inputs_outputs[0].type;
    const bool supported_type =
        type == nvinfer1::DataType::kFLOAT || type == nvinfer1::DataType::kBF16;
    return supported_type && inputs_outputs[position].type == type &&
           inputs_outputs[position].format == nvinfer1::TensorFormat::kLINEAR;
}

void MsDeformAttnPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* inputs,
                                         int32_t num_inputs,
                                         nvinfer1::DynamicPluginTensorDesc const* outputs,
                                         int32_t num_outputs) noexcept {
    (void)inputs;
    (void)num_inputs;
    (void)outputs;
    (void)num_outputs;
}

std::size_t MsDeformAttnPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs,
                                                 int32_t num_inputs,
                                                 nvinfer1::PluginTensorDesc const* outputs,
                                                 int32_t num_outputs) const noexcept {
    (void)inputs;
    (void)num_inputs;
    (void)outputs;
    (void)num_outputs;
    return 0;
}

int32_t MsDeformAttnPlugin::enqueue(nvinfer1::PluginTensorDesc const* input_descriptors,
                                    nvinfer1::PluginTensorDesc const* output_descriptors,
                                    void const* const* inputs, void* const* outputs,
                                    void* workspace, cudaStream_t stream) noexcept {
    (void)output_descriptors;
    (void)workspace;
    if (inputs == nullptr || outputs == nullptr || !valid_dimensions(input_descriptors))
        return 1;
    const auto& value = input_descriptors[0].dims;
    const int32_t batch = value.d[0];
    const int32_t num_queries = input_descriptors[1].dims.d[1];
    const int32_t num_heads = value.d[2];
    const int32_t channels = value.d[3];
    if (batch <= 0 || num_queries <= 0 || num_heads <= 0 || channels <= 0)
        return 1;
    const int64_t total = static_cast<int64_t>(batch) * num_queries * num_heads * channels;
    constexpr int32_t threads = 256;
    const int32_t blocks =
        static_cast<int32_t>(std::min<int64_t>((total + threads - 1) / threads, 65535));
    if (input_descriptors[0].type == nvinfer1::DataType::kBF16) {
        ms_deform_attn_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(inputs[0]),
            static_cast<const __nv_bfloat16*>(inputs[1]),
            static_cast<const __nv_bfloat16*>(inputs[2]), batch, num_queries, num_heads, channels,
            static_cast<__nv_bfloat16*>(outputs[0]));
    } else if (input_descriptors[0].type == nvinfer1::DataType::kFLOAT) {
        ms_deform_attn_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const float*>(inputs[0]), static_cast<const float*>(inputs[1]),
            static_cast<const float*>(inputs[2]), batch, num_queries, num_heads, channels,
            static_cast<float*>(outputs[0]));
    } else {
        return 1;
    }
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
}

} // namespace trtmc::sam2_hoi
