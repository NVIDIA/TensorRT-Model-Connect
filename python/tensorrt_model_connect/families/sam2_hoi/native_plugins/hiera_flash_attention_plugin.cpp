/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hiera_flash_attention_plugin.h"

#include <cstdint>

namespace trtmc::sam2_hoi {
namespace {

bool dims4(const nvinfer1::Dims& dims, int& b, int& h, int& s, int& d) {
    if (dims.nbDims != 4)
        return false;
    b = dims.d[0];
    h = dims.d[1];
    s = dims.d[2];
    d = dims.d[3];
    return true;
}

bool valid_descriptors(const nvinfer1::PluginTensorDesc* inputs,
                       const nvinfer1::PluginTensorDesc* outputs) {
    int qb, qh, qs, qd;
    int kb, kh, ks, kd;
    int vb, vh, vs, vd;
    int ob, oh, os, od;
    if (inputs == nullptr || outputs == nullptr) {
        return false;
    }
    for (int32_t index = 0; index < 3; ++index) {
        if (inputs[index].type != nvinfer1::DataType::kBF16 ||
            inputs[index].format != nvinfer1::TensorFormat::kLINEAR) {
            return false;
        }
    }
    if (outputs[0].type != nvinfer1::DataType::kBF16 ||
        outputs[0].format != nvinfer1::TensorFormat::kLINEAR) {
        return false;
    }
    return dims4(inputs[0].dims, qb, qh, qs, qd) && dims4(inputs[1].dims, kb, kh, ks, kd) &&
           dims4(inputs[2].dims, vb, vh, vs, vd) && dims4(outputs[0].dims, ob, oh, os, od) &&
           qd == 96 && kd == 96 && vd == 96 && od == 96 && qb == kb && qb == vb && qb == ob &&
           qh == kh && qh == vh && qh == oh && ks == vs && qs == os &&
           sam2_hoi_hiera_flash_attention96_shape_allowed(qb, qh, qs, ks) != 0;
}

} // namespace

HieraFlashAttention96Plugin::HieraFlashAttention96Plugin(const void* data, std::size_t length) {
    (void)data;
    (void)length;
}
char const* HieraFlashAttention96Plugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}
char const* HieraFlashAttention96Plugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}
int32_t HieraFlashAttention96Plugin::getNbOutputs() const noexcept {
    return 1;
}
int32_t HieraFlashAttention96Plugin::initialize() noexcept {
    return 0;
}
void HieraFlashAttention96Plugin::terminate() noexcept {}
void HieraFlashAttention96Plugin::destroy() noexcept {
    delete this;
}
std::size_t HieraFlashAttention96Plugin::getSerializationSize() const noexcept {
    return 0;
}
void HieraFlashAttention96Plugin::serialize(void* buffer) const noexcept {
    (void)buffer;
}
void HieraFlashAttention96Plugin::setPluginNamespace(char const* plugin_namespace) noexcept {
    namespace_ = plugin_namespace != nullptr ? plugin_namespace : "";
}
char const* HieraFlashAttention96Plugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}
nvinfer1::DataType
HieraFlashAttention96Plugin::getOutputDataType(int32_t index, nvinfer1::DataType const* input_types,
                                               int32_t num_inputs) const noexcept {
    if (index == 0 && num_inputs == 3 && input_types != nullptr &&
        input_types[0] == nvinfer1::DataType::kBF16 &&
        input_types[1] == nvinfer1::DataType::kBF16 &&
        input_types[2] == nvinfer1::DataType::kBF16) {
        return nvinfer1::DataType::kBF16;
    }
    return nvinfer1::DataType::kFLOAT;
}
HieraFlashAttention96Plugin* HieraFlashAttention96Plugin::clone() const noexcept {
    auto* result = new HieraFlashAttention96Plugin();
    result->setPluginNamespace(namespace_.c_str());
    return result;
}
nvinfer1::DimsExprs HieraFlashAttention96Plugin::getOutputDimensions(
    int32_t output_index, nvinfer1::DimsExprs const* inputs, int32_t num_inputs,
    nvinfer1::IExprBuilder& expression_builder) noexcept {
    (void)expression_builder;
    if (output_index != 0 || num_inputs != 3 || inputs == nullptr || inputs[0].nbDims != 4) {
        return {};
    }
    return inputs[0];
}
bool HieraFlashAttention96Plugin::supportsFormatCombination(
    int32_t position, nvinfer1::PluginTensorDesc const* inputs_outputs, int32_t num_inputs,
    int32_t num_outputs) noexcept {
    if (inputs_outputs == nullptr || num_inputs != 3 || num_outputs != 1 || position < 0 ||
        position >= 4) {
        return false;
    }
    return inputs_outputs[position].type == nvinfer1::DataType::kBF16 &&
           inputs_outputs[position].format == nvinfer1::TensorFormat::kLINEAR;
}
void HieraFlashAttention96Plugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* inputs,
                                                  int32_t num_inputs,
                                                  nvinfer1::DynamicPluginTensorDesc const* outputs,
                                                  int32_t num_outputs) noexcept {
    (void)inputs;
    (void)num_inputs;
    (void)outputs;
    (void)num_outputs;
}
std::size_t HieraFlashAttention96Plugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs,
                                                          int32_t num_inputs,
                                                          nvinfer1::PluginTensorDesc const* outputs,
                                                          int32_t num_outputs) const noexcept {
    if (num_inputs != 3 || num_outputs != 1 || !valid_descriptors(inputs, outputs)) {
        return 0;
    }
    const auto& q = inputs[0].dims;
    return static_cast<std::size_t>(q.d[0]) * q.d[1] * q.d[2] * sizeof(float);
}
int32_t HieraFlashAttention96Plugin::enqueue(nvinfer1::PluginTensorDesc const* input_descriptors,
                                             nvinfer1::PluginTensorDesc const* output_descriptors,
                                             void const* const* inputs, void* const* outputs,
                                             void* workspace, cudaStream_t stream) noexcept {
    if (inputs == nullptr || outputs == nullptr || workspace == nullptr || inputs[0] == nullptr ||
        inputs[1] == nullptr || inputs[2] == nullptr || outputs[0] == nullptr ||
        !valid_descriptors(input_descriptors, output_descriptors)) {
        return 1;
    }
    const auto& q = input_descriptors[0].dims;
    const auto& k = input_descriptors[1].dims;
    return sam2_hoi_hiera_flash_attention96_launch(inputs[0], inputs[1], inputs[2], outputs[0],
                                                   workspace, q.d[0], q.d[1], q.d[2], k.d[2],
                                                   stream) == 0
               ? 0
               : 1;
}

} // namespace trtmc::sam2_hoi
