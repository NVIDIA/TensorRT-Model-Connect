/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "plugins.h"

#include <cstddef>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <new>

namespace trtmc {
namespace {

using Plugin = FastFoundationStereoFullVolumeLeakyPlugin;

constexpr int32_t kPositions =
    Plugin::kBatch * Plugin::kDisparities * Plugin::kHeight * Plugin::kWidth;
constexpr int32_t kHalfLanesPerVector = 8;
constexpr int32_t kVectorsPerPosition = Plugin::kChannelPitch / kHalfLanesPerVector;
constexpr int32_t kVectorCount = kPositions * kVectorsPerPosition;
constexpr int32_t kThreads = 256;
constexpr float kNegativeSlope = 0.01F;

static_assert(Plugin::kChannels == 28);
static_assert(Plugin::kChannelPitch == 32);
static_assert(Plugin::kChannelPitch % kHalfLanesPerVector == 0);
static_assert(kVectorCount % kThreads == 0);

union alignas(16) Half8Vector {
    uint4 packed;
    __half lanes[kHalfLanesPerVector];
};

static_assert(sizeof(Half8Vector) == 16);
static_assert(alignof(Half8Vector) == 16);

__device__ __forceinline__ __half leaky_relu_fp32(__half input) {
    float const value = __half2float(input);
    if (value >= 0.0F)
        return input;
    // TensorRT LeakyReLU is x >= 0 ? x : alpha*x. NaN therefore follows the
    // FP32 multiply path, while both signs of zero preserve their input bits.
    return __float2half_rn(__fmul_rn(value, kNegativeSlope));
}

__global__ __launch_bounds__(kThreads) void full_volume_leaky_hwc8_kernel(__half const* input,
                                                                          __half* output) {
    int32_t const vector_index =
        static_cast<int32_t>(blockIdx.x) * kThreads + static_cast<int32_t>(threadIdx.x);
    if (vector_index >= kVectorCount)
        return;

    auto const* input_vectors = reinterpret_cast<uint4 const*>(input);
    auto* output_vectors = reinterpret_cast<uint4*>(output);
    Half8Vector values{};
    values.packed = input_vectors[vector_index];
    int32_t const channel_base = (vector_index % kVectorsPerPosition) * kHalfLanesPerVector;
#pragma unroll
    for (int32_t lane = 0; lane < kHalfLanesPerVector; ++lane) {
        int32_t const channel = channel_base + lane;
        values.lanes[lane] = channel < Plugin::kChannels ? leaky_relu_fp32(values.lanes[lane])
                                                         : __float2half_rn(0.0F);
    }
    output_vectors[vector_index] = values.packed;
}

bool has_exact_dims(nvinfer1::Dims const& dims) noexcept {
    return dims.nbDims == 5 && dims.d[0] == Plugin::kBatch && dims.d[1] == Plugin::kChannels &&
           dims.d[2] == Plugin::kDisparities && dims.d[3] == Plugin::kHeight &&
           dims.d[4] == Plugin::kWidth;
}

bool is_exact_desc(nvinfer1::PluginTensorDesc const& desc) noexcept {
    return desc.type == nvinfer1::DataType::kHALF &&
           desc.format == nvinfer1::TensorFormat::kDHWC8 && has_exact_dims(desc.dims);
}

bool is_exact_dynamic_desc(nvinfer1::DynamicPluginTensorDesc const& desc) noexcept {
    return is_exact_desc(desc.desc) && has_exact_dims(desc.min) && has_exact_dims(desc.max);
}

} // namespace

FastFoundationStereoFullVolumeLeakyPlugin::FastFoundationStereoFullVolumeLeakyPlugin(
    nvinfer1::PluginFieldCollection const& fields) noexcept
    : valid_(fields.nbFields == 0) {
    serialization_collection_.nbFields = 0;
    serialization_collection_.fields = nullptr;
}

FastFoundationStereoFullVolumeLeakyPlugin::FastFoundationStereoFullVolumeLeakyPlugin(
    FastFoundationStereoFullVolumeLeakyPlugin const& other) noexcept
    : valid_(other.valid_) {
    serialization_collection_.nbFields = 0;
    serialization_collection_.fields = nullptr;
}

nvinfer1::IPluginCapability* FastFoundationStereoFullVolumeLeakyPlugin::getCapabilityInterface(
    nvinfer1::PluginCapabilityType type) noexcept {
    switch (type) {
    case nvinfer1::PluginCapabilityType::kCORE:
        return static_cast<nvinfer1::IPluginV3OneCore*>(this);
    case nvinfer1::PluginCapabilityType::kBUILD:
        return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
    case nvinfer1::PluginCapabilityType::kRUNTIME:
        return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
    }
    return nullptr;
}

FastFoundationStereoFullVolumeLeakyPlugin*
FastFoundationStereoFullVolumeLeakyPlugin::clone() noexcept {
    auto* plugin = new (std::nothrow) FastFoundationStereoFullVolumeLeakyPlugin(*this);
    if (plugin != nullptr && !plugin->isValid()) {
        delete plugin;
        return nullptr;
    }
    return plugin;
}

nvinfer1::AsciiChar const*
FastFoundationStereoFullVolumeLeakyPlugin::getPluginName() const noexcept {
    return kPLUGIN_NAME;
}

nvinfer1::AsciiChar const*
FastFoundationStereoFullVolumeLeakyPlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}

nvinfer1::AsciiChar const*
FastFoundationStereoFullVolumeLeakyPlugin::getPluginNamespace() const noexcept {
    return "";
}

int32_t FastFoundationStereoFullVolumeLeakyPlugin::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t input_count,
    nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t output_count) noexcept {
    return valid_ && inputs != nullptr && outputs != nullptr && input_count == 1 &&
                   output_count == 1 && is_exact_dynamic_desc(inputs[0]) &&
                   is_exact_dynamic_desc(outputs[0])
               ? 0
               : 1;
}

int32_t FastFoundationStereoFullVolumeLeakyPlugin::getOutputDataTypes(
    nvinfer1::DataType* output_types, int32_t output_count, nvinfer1::DataType const* input_types,
    int32_t input_count) const noexcept {
    if (output_types == nullptr || input_types == nullptr || output_count != 1 ||
        input_count != 1 || input_types[0] != nvinfer1::DataType::kHALF) {
        return 1;
    }
    output_types[0] = nvinfer1::DataType::kHALF;
    return 0;
}

int32_t FastFoundationStereoFullVolumeLeakyPlugin::getOutputShapes(
    nvinfer1::DimsExprs const* inputs, int32_t input_count, nvinfer1::DimsExprs const* shape_inputs,
    int32_t shape_input_count, nvinfer1::DimsExprs* outputs, int32_t output_count,
    nvinfer1::IExprBuilder& expression_builder) noexcept {
    if (inputs == nullptr || outputs == nullptr || input_count != 1 || output_count != 1 ||
        shape_input_count != 0 || inputs[0].nbDims != 5) {
        return 1;
    }
    (void)shape_inputs;
    (void)expression_builder;
    outputs[0] = inputs[0];
    return 0;
}

bool FastFoundationStereoFullVolumeLeakyPlugin::supportsFormatCombination(
    int32_t position, nvinfer1::DynamicPluginTensorDesc const* input_output, int32_t input_count,
    int32_t output_count) noexcept {
    return input_output != nullptr && input_count == 1 && output_count == 1 && position >= 0 &&
           position < 2 && is_exact_desc(input_output[position].desc);
}

int32_t FastFoundationStereoFullVolumeLeakyPlugin::getNbOutputs() const noexcept {
    return 1;
}

std::size_t FastFoundationStereoFullVolumeLeakyPlugin::getWorkspaceSize(
    nvinfer1::DynamicPluginTensorDesc const*, int32_t, nvinfer1::DynamicPluginTensorDesc const*,
    int32_t) const noexcept {
    return 0;
}

char const* FastFoundationStereoFullVolumeLeakyPlugin::getTimingCacheID() noexcept {
    return "full-volume-leaky-28x48x176x176-dhwc8-fp16-uint4-v1";
}

char const* FastFoundationStereoFullVolumeLeakyPlugin::getMetadataString() noexcept {
    return "input=NCDHW:1x28x48x176x176:DHWC8;"
           "output=NCDHW:1x28x48x176x176:DHWC8;pitch=32;vector=uint4;"
           "alpha=0.01;negative=half_rn(fp32(input)*fp32(alpha));tail=zero";
}

int32_t FastFoundationStereoFullVolumeLeakyPlugin::onShapeChange(
    nvinfer1::PluginTensorDesc const* inputs, int32_t input_count,
    nvinfer1::PluginTensorDesc const* outputs, int32_t output_count) noexcept {
    return valid_ && inputs != nullptr && outputs != nullptr && input_count == 1 &&
                   output_count == 1 && is_exact_desc(inputs[0]) && is_exact_desc(outputs[0])
               ? 0
               : 1;
}

int32_t FastFoundationStereoFullVolumeLeakyPlugin::enqueue(
    nvinfer1::PluginTensorDesc const* input_desc, nvinfer1::PluginTensorDesc const* output_desc,
    void const* const* inputs, void* const* outputs, void*, cudaStream_t stream) noexcept {
    if (!valid_ || input_desc == nullptr || output_desc == nullptr || inputs == nullptr ||
        outputs == nullptr || inputs[0] == nullptr || outputs[0] == nullptr ||
        !is_exact_desc(input_desc[0]) || !is_exact_desc(output_desc[0])) {
        return 1;
    }

    constexpr int32_t kBlocks = kVectorCount / kThreads;
    full_volume_leaky_hwc8_kernel<<<kBlocks, kThreads, 0, stream>>>(
        static_cast<__half const*>(inputs[0]), static_cast<__half*>(outputs[0]));
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
}

nvinfer1::IPluginV3* FastFoundationStereoFullVolumeLeakyPlugin::attachToContext(
    nvinfer1::IPluginResourceContext*) noexcept {
    return clone();
}

nvinfer1::PluginFieldCollection const*
FastFoundationStereoFullVolumeLeakyPlugin::getFieldsToSerialize() noexcept {
    return valid_ ? &serialization_collection_ : nullptr;
}

} // namespace trtmc
