/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "spatial_attention_reduce_plugin.h"

#include <cfloat>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace trtmc {
namespace {

constexpr int32_t kBatch = 1;
constexpr int32_t kChannels = 48;
constexpr int32_t kHeight = 176;
constexpr int32_t kWidth = 176;
constexpr int32_t kPixelCount = kBatch * kHeight * kWidth;

__global__ void spatial_attention_reduce_kernel(const __half* input, __half* average,
                                                __half* maximum) {
    const int32_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= kPixelCount)
        return;

    float sum = 0.0F;
    float largest = -FLT_MAX;
    for (int32_t channel = 0; channel < kChannels; ++channel) {
        const float value = __half2float(input[channel * kPixelCount + pixel]);
        sum += value;
        largest = fmaxf(largest, value);
    }
    average[pixel] = __float2half_rn(sum / static_cast<float>(kChannels));
    maximum[pixel] = __float2half_rn(largest);
}

bool is_exact_input(nvinfer1::PluginTensorDesc const& desc) noexcept {
    return desc.type == nvinfer1::DataType::kHALF &&
           desc.format == nvinfer1::TensorFormat::kLINEAR && desc.dims.nbDims == 4 &&
           desc.dims.d[0] == kBatch && desc.dims.d[1] == kChannels && desc.dims.d[2] == kHeight &&
           desc.dims.d[3] == kWidth;
}

bool is_exact_output(nvinfer1::PluginTensorDesc const& desc) noexcept {
    return desc.type == nvinfer1::DataType::kHALF &&
           desc.format == nvinfer1::TensorFormat::kLINEAR && desc.dims.nbDims == 4 &&
           desc.dims.d[0] == kBatch && desc.dims.d[1] == 1 && desc.dims.d[2] == kHeight &&
           desc.dims.d[3] == kWidth;
}

} // namespace

FastFoundationStereoSpatialAttentionReducePlugin::FastFoundationStereoSpatialAttentionReducePlugin(
    const void*, std::size_t) {}

char const* FastFoundationStereoSpatialAttentionReducePlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}

char const* FastFoundationStereoSpatialAttentionReducePlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}

int32_t FastFoundationStereoSpatialAttentionReducePlugin::getNbOutputs() const noexcept {
    return 2;
}

int32_t FastFoundationStereoSpatialAttentionReducePlugin::initialize() noexcept {
    return 0;
}

void FastFoundationStereoSpatialAttentionReducePlugin::terminate() noexcept {}

void FastFoundationStereoSpatialAttentionReducePlugin::destroy() noexcept {
    delete this;
}

std::size_t
FastFoundationStereoSpatialAttentionReducePlugin::getSerializationSize() const noexcept {
    return 0;
}

void FastFoundationStereoSpatialAttentionReducePlugin::serialize(void*) const noexcept {}

void FastFoundationStereoSpatialAttentionReducePlugin::setPluginNamespace(
    char const* plugin_namespace) noexcept {
    namespace_ = plugin_namespace != nullptr ? plugin_namespace : "";
}

char const* FastFoundationStereoSpatialAttentionReducePlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::DataType FastFoundationStereoSpatialAttentionReducePlugin::getOutputDataType(
    int32_t, nvinfer1::DataType const*, int32_t) const noexcept {
    return nvinfer1::DataType::kHALF;
}

FastFoundationStereoSpatialAttentionReducePlugin*
FastFoundationStereoSpatialAttentionReducePlugin::clone() const noexcept {
    auto* plugin = new FastFoundationStereoSpatialAttentionReducePlugin();
    plugin->namespace_ = namespace_;
    return plugin;
}

nvinfer1::DimsExprs FastFoundationStereoSpatialAttentionReducePlugin::getOutputDimensions(
    int32_t, nvinfer1::DimsExprs const* inputs, int32_t,
    nvinfer1::IExprBuilder& expr_builder) noexcept {
    nvinfer1::DimsExprs output;
    output.nbDims = 4;
    output.d[0] = inputs[0].d[0];
    output.d[1] = expr_builder.constant(1);
    output.d[2] = inputs[0].d[2];
    output.d[3] = inputs[0].d[3];
    return output;
}

bool FastFoundationStereoSpatialAttentionReducePlugin::supportsFormatCombination(
    int32_t position, nvinfer1::PluginTensorDesc const* input_output, int32_t input_count,
    int32_t output_count) noexcept {
    if (input_count != 1 || output_count != 2 || position < 0 || position >= 3)
        return false;
    return position == 0 ? is_exact_input(input_output[position])
                         : is_exact_output(input_output[position]);
}

void FastFoundationStereoSpatialAttentionReducePlugin::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const*, int32_t, nvinfer1::DynamicPluginTensorDesc const*,
    int32_t) noexcept {}

std::size_t FastFoundationStereoSpatialAttentionReducePlugin::getWorkspaceSize(
    nvinfer1::PluginTensorDesc const*, int32_t, nvinfer1::PluginTensorDesc const*,
    int32_t) const noexcept {
    return 0;
}

int32_t FastFoundationStereoSpatialAttentionReducePlugin::enqueue(
    nvinfer1::PluginTensorDesc const* input_desc, nvinfer1::PluginTensorDesc const* output_desc,
    void const* const* inputs, void* const* outputs, void*, cudaStream_t stream) noexcept {
    if (input_desc == nullptr || output_desc == nullptr || inputs == nullptr ||
        outputs == nullptr || inputs[0] == nullptr || outputs[0] == nullptr ||
        outputs[1] == nullptr || !is_exact_input(input_desc[0]) ||
        !is_exact_output(output_desc[0]) || !is_exact_output(output_desc[1])) {
        return -1;
    }

    constexpr int32_t threads = 256;
    spatial_attention_reduce_kernel<<<(kPixelCount + threads - 1) / threads, threads, 0, stream>>>(
        static_cast<const __half*>(inputs[0]), static_cast<__half*>(outputs[0]),
        static_cast<__half*>(outputs[1]));
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

} // namespace trtmc
