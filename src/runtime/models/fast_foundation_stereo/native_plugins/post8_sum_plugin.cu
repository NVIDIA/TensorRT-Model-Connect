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

using Plugin = FastFoundationStereoPost8SumPlugin;

constexpr int32_t kLogicalElements =
    Plugin::kBatch * Plugin::kDisparities * Plugin::kHeight * Plugin::kWidth;
constexpr int32_t kThreads = 256;
constexpr int32_t kTilePositions = 32;

static_assert(Plugin::kChannels == 28);
static_assert(Plugin::kChannelPitch == 32);

bool has_exact_dims(nvinfer1::Dims const& dims) noexcept {
    return dims.nbDims == 5 && dims.d[0] == Plugin::kBatch && dims.d[1] == Plugin::kChannels &&
           dims.d[2] == Plugin::kDisparities && dims.d[3] == Plugin::kHeight &&
           dims.d[4] == Plugin::kWidth;
}

bool is_exact_linear_desc(nvinfer1::PluginTensorDesc const& desc) noexcept {
    return desc.type == nvinfer1::DataType::kHALF &&
           desc.format == nvinfer1::TensorFormat::kLINEAR && has_exact_dims(desc.dims);
}

bool is_exact_dhwc8_desc(nvinfer1::PluginTensorDesc const& desc) noexcept {
    return desc.type == nvinfer1::DataType::kHALF &&
           desc.format == nvinfer1::TensorFormat::kDHWC8 && has_exact_dims(desc.dims);
}

bool is_exact_dynamic_linear_desc(nvinfer1::DynamicPluginTensorDesc const& desc) noexcept {
    return is_exact_linear_desc(desc.desc) && has_exact_dims(desc.min) && has_exact_dims(desc.max);
}

bool is_exact_dynamic_dhwc8_desc(nvinfer1::DynamicPluginTensorDesc const& desc) noexcept {
    return is_exact_dhwc8_desc(desc.desc) && has_exact_dims(desc.min) && has_exact_dims(desc.max);
}

// Each CTA transposes a 28x32 NCDHW tile into a bank-conflict-free
// position-major tile.  The second half of the CTA then performs contiguous
// DHWC8 reads and writes while preserving the original FP32-add/FP16-round
// boundary.  The four padded lanes are explicitly initialized on every tile.
__global__ __launch_bounds__(kThreads) void post8_sum_linear_to_dhwc8_kernel(__half const* linear,
                                                                             __half const* skip,
                                                                             __half* output) {
    constexpr int32_t tile_elements = kTilePositions * Plugin::kChannelPitch;
    static_assert(tile_elements % kThreads == 0);
    __shared__ __half transposed[kTilePositions][Plugin::kChannelPitch + 1];

    int32_t const tile_start = static_cast<int32_t>(blockIdx.x) * kTilePositions;
    for (int32_t index = static_cast<int32_t>(threadIdx.x); index < tile_elements;
         index += kThreads) {
        int32_t const channel = index / kTilePositions;
        int32_t const local_position = index % kTilePositions;
        int32_t const position = tile_start + local_position;
        __half value = __float2half_rn(0.0F);
        if (channel < Plugin::kChannels && position < kLogicalElements) {
            value = linear[static_cast<std::size_t>(channel) * kLogicalElements + position];
        }
        transposed[local_position][channel] = value;
    }
    __syncthreads();

    for (int32_t index = static_cast<int32_t>(threadIdx.x); index < tile_elements;
         index += kThreads) {
        int32_t const local_position = index / Plugin::kChannelPitch;
        int32_t const channel = index % Plugin::kChannelPitch;
        int32_t const position = tile_start + local_position;
        if (position >= kLogicalElements)
            continue;
        std::size_t const packed_index =
            static_cast<std::size_t>(position) * Plugin::kChannelPitch + channel;
        if (channel < Plugin::kChannels) {
            float const linear_value = __half2float(transposed[local_position][channel]);
            float const skip_value = __half2float(skip[packed_index]);
            output[packed_index] = __float2half_rn(__fadd_rn(linear_value, skip_value));
        } else {
            output[packed_index] = __float2half_rn(0.0F);
        }
    }
}

cudaError_t launch_post8_sum(__half const* linear, __half const* skip, __half* output,
                             cudaStream_t stream) noexcept {
    constexpr int32_t blocks = (kLogicalElements + kTilePositions - 1) / kTilePositions;
    post8_sum_linear_to_dhwc8_kernel<<<blocks, kThreads, 0, stream>>>(linear, skip, output);
    return cudaPeekAtLastError();
}

} // namespace

FastFoundationStereoPost8SumPlugin::FastFoundationStereoPost8SumPlugin(
    nvinfer1::PluginFieldCollection const& fields) noexcept
    : valid_(fields.nbFields == 0) {
    serialization_collection_.nbFields = 0;
    serialization_collection_.fields = nullptr;
}

FastFoundationStereoPost8SumPlugin::FastFoundationStereoPost8SumPlugin(
    FastFoundationStereoPost8SumPlugin const& other) noexcept
    : valid_(other.valid_) {
    serialization_collection_.nbFields = 0;
    serialization_collection_.fields = nullptr;
}

nvinfer1::IPluginCapability* FastFoundationStereoPost8SumPlugin::getCapabilityInterface(
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

FastFoundationStereoPost8SumPlugin* FastFoundationStereoPost8SumPlugin::clone() noexcept {
    auto* plugin = new (std::nothrow) FastFoundationStereoPost8SumPlugin(*this);
    if (plugin != nullptr && !plugin->isValid()) {
        delete plugin;
        return nullptr;
    }
    return plugin;
}

nvinfer1::AsciiChar const* FastFoundationStereoPost8SumPlugin::getPluginName() const noexcept {
    return kPLUGIN_NAME;
}

nvinfer1::AsciiChar const* FastFoundationStereoPost8SumPlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}

nvinfer1::AsciiChar const* FastFoundationStereoPost8SumPlugin::getPluginNamespace() const noexcept {
    return "";
}

int32_t FastFoundationStereoPost8SumPlugin::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t input_count,
    nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t output_count) noexcept {
    return valid_ && inputs != nullptr && outputs != nullptr && input_count == 2 &&
                   output_count == 1 && is_exact_dynamic_linear_desc(inputs[0]) &&
                   is_exact_dynamic_dhwc8_desc(inputs[1]) && is_exact_dynamic_dhwc8_desc(outputs[0])
               ? 0
               : 1;
}

int32_t FastFoundationStereoPost8SumPlugin::getOutputDataTypes(
    nvinfer1::DataType* output_types, int32_t output_count, nvinfer1::DataType const* input_types,
    int32_t input_count) const noexcept {
    if (output_types == nullptr || input_types == nullptr || output_count != 1 ||
        input_count != 2 || input_types[0] != nvinfer1::DataType::kHALF ||
        input_types[1] != nvinfer1::DataType::kHALF) {
        return 1;
    }
    output_types[0] = nvinfer1::DataType::kHALF;
    return 0;
}

int32_t FastFoundationStereoPost8SumPlugin::getOutputShapes(
    nvinfer1::DimsExprs const* inputs, int32_t input_count, nvinfer1::DimsExprs const* shape_inputs,
    int32_t shape_input_count, nvinfer1::DimsExprs* outputs, int32_t output_count,
    nvinfer1::IExprBuilder& expression_builder) noexcept {
    if (inputs == nullptr || outputs == nullptr || input_count != 2 || output_count != 1 ||
        shape_input_count != 0 || inputs[0].nbDims != 5 || inputs[1].nbDims != 5) {
        return 1;
    }
    (void)shape_inputs;
    (void)expression_builder;
    outputs[0] = inputs[0];
    return 0;
}

bool FastFoundationStereoPost8SumPlugin::supportsFormatCombination(
    int32_t position, nvinfer1::DynamicPluginTensorDesc const* input_output, int32_t input_count,
    int32_t output_count) noexcept {
    if (input_output == nullptr || input_count != 2 || output_count != 1 || position < 0 ||
        position >= 3) {
        return false;
    }
    return position == 0 ? is_exact_linear_desc(input_output[0].desc)
                         : is_exact_dhwc8_desc(input_output[position].desc);
}

int32_t FastFoundationStereoPost8SumPlugin::getNbOutputs() const noexcept {
    return 1;
}

std::size_t FastFoundationStereoPost8SumPlugin::getWorkspaceSize(
    nvinfer1::DynamicPluginTensorDesc const*, int32_t, nvinfer1::DynamicPluginTensorDesc const*,
    int32_t) const noexcept {
    return 0;
}

char const* FastFoundationStereoPost8SumPlugin::getTimingCacheID() noexcept {
    return "post8-sum-28x48x176x176-linear-dhwc8-fp16-tile32-v1";
}

char const* FastFoundationStereoPost8SumPlugin::getMetadataString() noexcept {
    return "input0=NCDHW:1x28x48x176x176:linear;input1=NCDHW:1x28x48x176x176:DHWC8;"
           "output=NCDHW:1x28x48x176x176:DHWC8;pitch=32;tile=32;"
           "sum=half(fp32(half(linear))+fp32(half(skip)));tail=zero";
}

int32_t FastFoundationStereoPost8SumPlugin::onShapeChange(nvinfer1::PluginTensorDesc const* inputs,
                                                          int32_t input_count,
                                                          nvinfer1::PluginTensorDesc const* outputs,
                                                          int32_t output_count) noexcept {
    return valid_ && inputs != nullptr && outputs != nullptr && input_count == 2 &&
                   output_count == 1 && is_exact_linear_desc(inputs[0]) &&
                   is_exact_dhwc8_desc(inputs[1]) && is_exact_dhwc8_desc(outputs[0])
               ? 0
               : 1;
}

int32_t FastFoundationStereoPost8SumPlugin::enqueue(nvinfer1::PluginTensorDesc const* input_desc,
                                                    nvinfer1::PluginTensorDesc const* output_desc,
                                                    void const* const* inputs, void* const* outputs,
                                                    void*, cudaStream_t stream) noexcept {
    if (!valid_ || input_desc == nullptr || output_desc == nullptr || inputs == nullptr ||
        outputs == nullptr || inputs[0] == nullptr || inputs[1] == nullptr ||
        outputs[0] == nullptr || !is_exact_linear_desc(input_desc[0]) ||
        !is_exact_dhwc8_desc(input_desc[1]) || !is_exact_dhwc8_desc(output_desc[0])) {
        return 1;
    }

    auto const* linear = static_cast<__half const*>(inputs[0]);
    auto const* skip = static_cast<__half const*>(inputs[1]);
    auto* output = static_cast<__half*>(outputs[0]);
    cudaError_t const result = launch_post8_sum(linear, skip, output, stream);
    return result == cudaSuccess ? 0 : 1;
}

nvinfer1::IPluginV3*
FastFoundationStereoPost8SumPlugin::attachToContext(nvinfer1::IPluginResourceContext*) noexcept {
    return clone();
}

nvinfer1::PluginFieldCollection const*
FastFoundationStereoPost8SumPlugin::getFieldsToSerialize() noexcept {
    return valid_ ? &serialization_collection_ : nullptr;
}

} // namespace trtmc
