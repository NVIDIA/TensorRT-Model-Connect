/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hiera_gelu_plugin.h"

#include <cstdint>

namespace trtmc::sam2_hoi {
namespace {

bool valid_dims(const nvinfer1::Dims& dims) {
    return dims.nbDims == 4 &&
           sam2_hoi_hiera_gelu_bf16_shape_allowed(dims.d[0], dims.d[1], dims.d[2], dims.d[3]) != 0;
}

bool valid_descriptors(const nvinfer1::PluginTensorDesc* inputs,
                       const nvinfer1::PluginTensorDesc* outputs) {
    if (inputs == nullptr || outputs == nullptr || !valid_dims(inputs[0].dims) ||
        !valid_dims(outputs[0].dims)) {
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        if (inputs[0].dims.d[i] != outputs[0].dims.d[i])
            return false;
    }
    return inputs[0].type == nvinfer1::DataType::kBF16 &&
           outputs[0].type == nvinfer1::DataType::kBF16 &&
           inputs[0].format == nvinfer1::TensorFormat::kLINEAR &&
           outputs[0].format == nvinfer1::TensorFormat::kLINEAR;
}

} // namespace

HieraGeluErfBf16Plugin::HieraGeluErfBf16Plugin(const void* data, std::size_t length) {
    (void)data;
    (void)length;
}
char const* HieraGeluErfBf16Plugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}
char const* HieraGeluErfBf16Plugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}
int32_t HieraGeluErfBf16Plugin::getNbOutputs() const noexcept {
    return 1;
}
int32_t HieraGeluErfBf16Plugin::initialize() noexcept {
    return 0;
}
void HieraGeluErfBf16Plugin::terminate() noexcept {}
void HieraGeluErfBf16Plugin::destroy() noexcept {
    delete this;
}
std::size_t HieraGeluErfBf16Plugin::getSerializationSize() const noexcept {
    return 0;
}
void HieraGeluErfBf16Plugin::serialize(void* buffer) const noexcept {
    (void)buffer;
}
void HieraGeluErfBf16Plugin::setPluginNamespace(char const* plugin_namespace) noexcept {
    namespace_ = plugin_namespace != nullptr ? plugin_namespace : "";
}
char const* HieraGeluErfBf16Plugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}
nvinfer1::DataType HieraGeluErfBf16Plugin::getOutputDataType(int32_t index,
                                                             nvinfer1::DataType const* input_types,
                                                             int32_t num_inputs) const noexcept {
    if (index == 0 && num_inputs == 1 && input_types != nullptr &&
        input_types[0] == nvinfer1::DataType::kBF16) {
        return nvinfer1::DataType::kBF16;
    }
    return nvinfer1::DataType::kFLOAT;
}
HieraGeluErfBf16Plugin* HieraGeluErfBf16Plugin::clone() const noexcept {
    auto* result = new HieraGeluErfBf16Plugin();
    result->setPluginNamespace(namespace_.c_str());
    return result;
}
nvinfer1::DimsExprs
HieraGeluErfBf16Plugin::getOutputDimensions(int32_t output_index, nvinfer1::DimsExprs const* inputs,
                                            int32_t num_inputs,
                                            nvinfer1::IExprBuilder& expression_builder) noexcept {
    (void)expression_builder;
    if (output_index != 0 || num_inputs != 1 || inputs == nullptr || inputs[0].nbDims != 4) {
        return {};
    }
    return inputs[0];
}
bool HieraGeluErfBf16Plugin::supportsFormatCombination(
    int32_t position, nvinfer1::PluginTensorDesc const* inputs_outputs, int32_t num_inputs,
    int32_t num_outputs) noexcept {
    if (inputs_outputs == nullptr || num_inputs != 1 || num_outputs != 1 || position < 0 ||
        position >= 2) {
        return false;
    }
    return inputs_outputs[position].type == nvinfer1::DataType::kBF16 &&
           inputs_outputs[position].format == nvinfer1::TensorFormat::kLINEAR;
}
void HieraGeluErfBf16Plugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* inputs,
                                             int32_t num_inputs,
                                             nvinfer1::DynamicPluginTensorDesc const* outputs,
                                             int32_t num_outputs) noexcept {
    (void)inputs;
    (void)num_inputs;
    (void)outputs;
    (void)num_outputs;
}
std::size_t HieraGeluErfBf16Plugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs,
                                                     int32_t num_inputs,
                                                     nvinfer1::PluginTensorDesc const* outputs,
                                                     int32_t num_outputs) const noexcept {
    (void)inputs;
    (void)num_inputs;
    (void)outputs;
    (void)num_outputs;
    return 0;
}
int32_t HieraGeluErfBf16Plugin::enqueue(nvinfer1::PluginTensorDesc const* input_descriptors,
                                        nvinfer1::PluginTensorDesc const* output_descriptors,
                                        void const* const* inputs, void* const* outputs,
                                        void* workspace, cudaStream_t stream) noexcept {
    (void)workspace;
    if (inputs == nullptr || outputs == nullptr || inputs[0] == nullptr || outputs[0] == nullptr ||
        !valid_descriptors(input_descriptors, output_descriptors)) {
        return 1;
    }
    const auto& dims = input_descriptors[0].dims;
    const int kElements = dims.d[0] * dims.d[1] * dims.d[2] * dims.d[3];
    return sam2_hoi_hiera_gelu_bf16_launch(inputs[0], outputs[0], kElements, stream) == 0 ? 0 : 1;
}

} // namespace trtmc::sam2_hoi
