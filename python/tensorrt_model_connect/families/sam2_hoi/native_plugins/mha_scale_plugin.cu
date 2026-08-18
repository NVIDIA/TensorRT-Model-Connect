/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * The explicit FP32 multiply followed by BF16 rounding preserves the
 * PyTorch v2.7.1 MultiheadAttention query-scaling boundary for head width 32.
 */

#include "mha_scale_plugin.h"

#include <algorithm>
#include <cstdint>
#include <cuda_bf16.h>

namespace trtmc::sam2_hoi {
namespace {

template <typename T>
__global__ void mha_scale_kernel(const T* input, int64_t elements, T* output) {
    constexpr float scale = 0.1767766952966369F;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements; index += static_cast<int64_t>(blockDim.x) * gridDim.x)
        output[index] = static_cast<T>(static_cast<float>(input[index]) * scale);
}

template <>
__global__ void mha_scale_kernel(const __nv_bfloat16* input, int64_t elements,
                                 __nv_bfloat16* output) {
    constexpr float scale = 0.1767766952966369F;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements; index += static_cast<int64_t>(blockDim.x) * gridDim.x)
        output[index] = __float2bfloat16_rn(__bfloat162float(input[index]) * scale);
}

int64_t volume(const nvinfer1::Dims& dimensions) {
    int64_t elements = 1;
    for (int32_t index = 0; index < dimensions.nbDims; ++index) {
        if (dimensions.d[index] <= 0)
            return 0;
        elements *= dimensions.d[index];
    }
    return elements;
}

} // namespace

MhaScalePlugin::MhaScalePlugin(const void* data, std::size_t length) {
    (void)data;
    (void)length;
}
char const* MhaScalePlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}
char const* MhaScalePlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}
int32_t MhaScalePlugin::getNbOutputs() const noexcept {
    return 1;
}
int32_t MhaScalePlugin::initialize() noexcept {
    return 0;
}
void MhaScalePlugin::terminate() noexcept {}
void MhaScalePlugin::destroy() noexcept {
    delete this;
}
std::size_t MhaScalePlugin::getSerializationSize() const noexcept {
    return 0;
}
void MhaScalePlugin::serialize(void* buffer) const noexcept {
    (void)buffer;
}
void MhaScalePlugin::setPluginNamespace(char const* plugin_namespace) noexcept {
    namespace_ = plugin_namespace != nullptr ? plugin_namespace : "";
}
char const* MhaScalePlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}
nvinfer1::DataType MhaScalePlugin::getOutputDataType(int32_t index,
                                                     nvinfer1::DataType const* input_types,
                                                     int32_t num_inputs) const noexcept {
    return index == 0 && num_inputs == 1 ? input_types[0] : nvinfer1::DataType::kFLOAT;
}
MhaScalePlugin* MhaScalePlugin::clone() const noexcept {
    auto* cloned = new MhaScalePlugin();
    cloned->setPluginNamespace(namespace_.c_str());
    return cloned;
}
nvinfer1::DimsExprs
MhaScalePlugin::getOutputDimensions(int32_t output_index, nvinfer1::DimsExprs const* inputs,
                                    int32_t num_inputs,
                                    nvinfer1::IExprBuilder& expression_builder) noexcept {
    (void)expression_builder;
    nvinfer1::DimsExprs output{};
    if (output_index == 0 && num_inputs == 1)
        output = inputs[0];
    return output;
}
bool MhaScalePlugin::supportsFormatCombination(int32_t position,
                                               nvinfer1::PluginTensorDesc const* inputs_outputs,
                                               int32_t num_inputs, int32_t num_outputs) noexcept {
    if (num_inputs != 1 || num_outputs != 1 || position < 0 || position >= 2)
        return false;
    const auto type = inputs_outputs[0].type;
    return (type == nvinfer1::DataType::kFLOAT || type == nvinfer1::DataType::kBF16) &&
           inputs_outputs[position].type == type &&
           inputs_outputs[position].format == nvinfer1::TensorFormat::kLINEAR;
}
void MhaScalePlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* inputs,
                                     int32_t num_inputs,
                                     nvinfer1::DynamicPluginTensorDesc const* outputs,
                                     int32_t num_outputs) noexcept {
    (void)inputs;
    (void)num_inputs;
    (void)outputs;
    (void)num_outputs;
}
std::size_t MhaScalePlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs,
                                             int32_t num_inputs,
                                             nvinfer1::PluginTensorDesc const* outputs,
                                             int32_t num_outputs) const noexcept {
    (void)inputs;
    (void)num_inputs;
    (void)outputs;
    (void)num_outputs;
    return 0;
}
int32_t MhaScalePlugin::enqueue(nvinfer1::PluginTensorDesc const* input_descriptors,
                                nvinfer1::PluginTensorDesc const* output_descriptors,
                                void const* const* inputs, void* const* outputs, void* workspace,
                                cudaStream_t stream) noexcept {
    (void)output_descriptors;
    (void)workspace;
    if (inputs == nullptr || outputs == nullptr)
        return 1;
    const int64_t elements = volume(input_descriptors[0].dims);
    if (elements <= 0)
        return 1;
    constexpr int32_t threads = 256;
    const int32_t blocks =
        static_cast<int32_t>(std::min<int64_t>((elements + threads - 1) / threads, 65535));
    if (input_descriptors[0].type == nvinfer1::DataType::kFLOAT) {
        mha_scale_kernel<<<blocks, threads, 0, stream>>>(static_cast<const float*>(inputs[0]),
                                                         elements, static_cast<float*>(outputs[0]));
    } else if (input_descriptors[0].type == nvinfer1::DataType::kBF16) {
        mha_scale_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(inputs[0]), elements,
            static_cast<__nv_bfloat16*>(outputs[0]));
    } else {
        return 1;
    }
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
}

} // namespace trtmc::sam2_hoi
