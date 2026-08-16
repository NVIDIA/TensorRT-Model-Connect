/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gwc_plugin.h"

#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace trtmc {
namespace {

constexpr int32_t kBatch = 1;
constexpr int32_t kFeatureChannels = 224;
constexpr int32_t kProjectedChannels = 12;
constexpr int32_t kGroups = 8;
constexpr int32_t kCombinedChannels = kGroups + 2 * kProjectedChannels;
constexpr int32_t kChannelsPerGroup = kFeatureChannels / kGroups;
constexpr int32_t kHeight = 176;
constexpr int32_t kWidth = 176;
constexpr int32_t kDisparities = 48;
constexpr std::size_t kNormElements = static_cast<std::size_t>(kBatch) * kGroups * kHeight * kWidth;
constexpr std::size_t kWorkspaceBytes = 2 * kNormElements * sizeof(__half);

__global__ void group_norm_kernel(const __half* input, __half* norm) {
    const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= static_cast<int32_t>(kNormElements))
        return;
    const int32_t width_index = index % kWidth;
    const int32_t tmp = index / kWidth;
    const int32_t height_index = tmp % kHeight;
    const int32_t group = tmp / kHeight;
    float sum = 0.0F;
    for (int32_t channel = 0; channel < kChannelsPerGroup; ++channel) {
        const int32_t channel_index = group * kChannelsPerGroup + channel;
        const int32_t input_index = (channel_index * kHeight + height_index) * kWidth + width_index;
        const float value = __half2float(input[input_index]);
        sum += value * value;
    }
    norm[index] = __float2half_rn(sqrtf(sum));
}

__global__ void combined_volume_kernel(const __half* reference, const __half* target,
                                       const __half* left_projected, const __half* right_projected,
                                       const __half* reference_norm, const __half* target_norm,
                                       __half* output) {
    const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int32_t count = kCombinedChannels * kDisparities * kHeight * kWidth;
    if (index >= count)
        return;

    int32_t remaining = index;
    const int32_t width_index = remaining % kWidth;
    remaining /= kWidth;
    const int32_t height_index = remaining % kHeight;
    remaining /= kHeight;
    const int32_t disparity = remaining % kDisparities;
    const int32_t output_channel = remaining / kDisparities;

    if (output_channel >= kGroups) {
        int32_t projected_channel = output_channel - kGroups;
        const __half* projected = left_projected;
        int32_t projected_width = width_index;
        if (projected_channel >= kProjectedChannels) {
            projected_channel -= kProjectedChannels;
            projected = right_projected;
            projected_width -= disparity;
        }
        if (projected_width < 0) {
            output[index] = __float2half_rn(0.0F);
            return;
        }
        const int32_t projected_index =
            (projected_channel * kHeight + height_index) * kWidth + projected_width;
        output[index] = projected[projected_index];
        return;
    }

    const int32_t group = output_channel;
    const int32_t target_width = width_index - disparity;
    if (target_width < 0) {
        output[index] = __float2half_rn(0.0F);
        return;
    }
    float dot = 0.0F;
    for (int32_t channel = 0; channel < kChannelsPerGroup; ++channel) {
        const int32_t channel_index = group * kChannelsPerGroup + channel;
        const int32_t reference_index =
            (channel_index * kHeight + height_index) * kWidth + width_index;
        const int32_t target_index =
            (channel_index * kHeight + height_index) * kWidth + target_width;
        dot += __half2float(reference[reference_index]) * __half2float(target[target_index]);
    }
    const int32_t reference_norm_index = (group * kHeight + height_index) * kWidth + width_index;
    const int32_t target_norm_index = (group * kHeight + height_index) * kWidth + target_width;
    const float denominator = __half2float(reference_norm[reference_norm_index]) *
                                  __half2float(target_norm[target_norm_index]) +
                              1.0e-5F;
    output[index] = __float2half_rn(dot / denominator);
}

bool is_exact_input(nvinfer1::PluginTensorDesc const& desc, int32_t channels) noexcept {
    return desc.type == nvinfer1::DataType::kHALF &&
           desc.format == nvinfer1::TensorFormat::kLINEAR && desc.dims.nbDims == 4 &&
           desc.dims.d[0] == kBatch && desc.dims.d[1] == channels && desc.dims.d[2] == kHeight &&
           desc.dims.d[3] == kWidth;
}

bool is_exact_output(nvinfer1::PluginTensorDesc const& desc) noexcept {
    return desc.type == nvinfer1::DataType::kHALF &&
           desc.format == nvinfer1::TensorFormat::kLINEAR && desc.dims.nbDims == 5 &&
           desc.dims.d[0] == kBatch && desc.dims.d[1] == kCombinedChannels &&
           desc.dims.d[2] == kDisparities && desc.dims.d[3] == kHeight && desc.dims.d[4] == kWidth;
}

} // namespace

FastFoundationStereoCombinedVolumePlugin::FastFoundationStereoCombinedVolumePlugin(const void*,
                                                                                   std::size_t) {}

char const* FastFoundationStereoCombinedVolumePlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}

char const* FastFoundationStereoCombinedVolumePlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}

int32_t FastFoundationStereoCombinedVolumePlugin::getNbOutputs() const noexcept {
    return 1;
}

int32_t FastFoundationStereoCombinedVolumePlugin::initialize() noexcept {
    return 0;
}

void FastFoundationStereoCombinedVolumePlugin::terminate() noexcept {}

void FastFoundationStereoCombinedVolumePlugin::destroy() noexcept {
    delete this;
}

std::size_t FastFoundationStereoCombinedVolumePlugin::getSerializationSize() const noexcept {
    return 0;
}

void FastFoundationStereoCombinedVolumePlugin::serialize(void*) const noexcept {}

void FastFoundationStereoCombinedVolumePlugin::setPluginNamespace(
    char const* plugin_namespace) noexcept {
    namespace_ = plugin_namespace != nullptr ? plugin_namespace : "";
}

char const* FastFoundationStereoCombinedVolumePlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::DataType
FastFoundationStereoCombinedVolumePlugin::getOutputDataType(int32_t, nvinfer1::DataType const*,
                                                            int32_t) const noexcept {
    return nvinfer1::DataType::kHALF;
}

FastFoundationStereoCombinedVolumePlugin*
FastFoundationStereoCombinedVolumePlugin::clone() const noexcept {
    auto* plugin = new FastFoundationStereoCombinedVolumePlugin();
    plugin->namespace_ = namespace_;
    return plugin;
}

nvinfer1::DimsExprs FastFoundationStereoCombinedVolumePlugin::getOutputDimensions(
    int32_t, nvinfer1::DimsExprs const* inputs, int32_t,
    nvinfer1::IExprBuilder& expr_builder) noexcept {
    nvinfer1::DimsExprs output;
    output.nbDims = 5;
    output.d[0] = inputs[0].d[0];
    output.d[1] = expr_builder.constant(kCombinedChannels);
    output.d[2] = expr_builder.constant(kDisparities);
    output.d[3] = inputs[0].d[2];
    output.d[4] = inputs[0].d[3];
    return output;
}

bool FastFoundationStereoCombinedVolumePlugin::supportsFormatCombination(
    int32_t position, nvinfer1::PluginTensorDesc const* input_output, int32_t input_count,
    int32_t output_count) noexcept {
    if (input_count != 4 || output_count != 1 || position < 0 || position >= 5)
        return false;
    if (position < 2)
        return is_exact_input(input_output[position], kFeatureChannels);
    if (position < 4)
        return is_exact_input(input_output[position], kProjectedChannels);
    return is_exact_output(input_output[position]);
}

void FastFoundationStereoCombinedVolumePlugin::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const*, int32_t, nvinfer1::DynamicPluginTensorDesc const*,
    int32_t) noexcept {}

std::size_t FastFoundationStereoCombinedVolumePlugin::getWorkspaceSize(
    nvinfer1::PluginTensorDesc const*, int32_t, nvinfer1::PluginTensorDesc const*,
    int32_t) const noexcept {
    return kWorkspaceBytes;
}

int32_t
FastFoundationStereoCombinedVolumePlugin::enqueue(nvinfer1::PluginTensorDesc const* input_desc,
                                                  nvinfer1::PluginTensorDesc const* output_desc,
                                                  void const* const* inputs, void* const* outputs,
                                                  void* workspace, cudaStream_t stream) noexcept {
    if (input_desc == nullptr || output_desc == nullptr || inputs == nullptr ||
        outputs == nullptr || workspace == nullptr || inputs[0] == nullptr ||
        inputs[1] == nullptr || inputs[2] == nullptr || inputs[3] == nullptr ||
        outputs[0] == nullptr || !is_exact_input(input_desc[0], kFeatureChannels) ||
        !is_exact_input(input_desc[1], kFeatureChannels) ||
        !is_exact_input(input_desc[2], kProjectedChannels) ||
        !is_exact_input(input_desc[3], kProjectedChannels) || !is_exact_output(output_desc[0])) {
        return -1;
    }

    auto* reference_norm = static_cast<__half*>(workspace);
    auto* target_norm = reference_norm + kNormElements;
    constexpr int32_t threads = 256;
    constexpr int32_t output_elements = kCombinedChannels * kDisparities * kHeight * kWidth;
    group_norm_kernel<<<(static_cast<int32_t>(kNormElements) + threads - 1) / threads, threads, 0,
                        stream>>>(static_cast<const __half*>(inputs[0]), reference_norm);
    if (cudaGetLastError() != cudaSuccess)
        return -1;
    group_norm_kernel<<<(static_cast<int32_t>(kNormElements) + threads - 1) / threads, threads, 0,
                        stream>>>(static_cast<const __half*>(inputs[1]), target_norm);
    if (cudaGetLastError() != cudaSuccess)
        return -1;
    combined_volume_kernel<<<(output_elements + threads - 1) / threads, threads, 0, stream>>>(
        static_cast<const __half*>(inputs[0]), static_cast<const __half*>(inputs[1]),
        static_cast<const __half*>(inputs[2]), static_cast<const __half*>(inputs[3]),
        reference_norm, target_norm, static_cast<__half*>(outputs[0]));
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

} // namespace trtmc
