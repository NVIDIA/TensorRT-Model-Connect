/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "layer_norm_plugin.h"

#include <ATen/ATen.h>
#include <ATen/ops/layer_norm.h>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>
#include <cstdio>
#include <exception>
#include <new>
#include <optional>

namespace trtmc::minimax_h3 {
namespace {

bool is_bf16_linear(nvinfer1::PluginTensorDesc const& desc) noexcept {
    return desc.type == nvinfer1::DataType::kBF16 && desc.format == nvinfer1::TensorFormat::kLINEAR;
}

bool row_dimension(int32_t value) noexcept {
    return value == -1 || value > 0;
}

bool has_network_dims(nvinfer1::Dims const& dims, int32_t position) noexcept {
    if (position == 0 || position == 3)
        return dims.nbDims == 2 && row_dimension(dims.d[0]) && dims.d[1] > 0;
    return (position == 1 || position == 2) && dims.nbDims == 1 && dims.d[0] > 0;
}

bool has_runtime_dims(nvinfer1::Dims const& dims, int32_t position) noexcept {
    if (position == 0 || position == 3)
        return dims.nbDims == 2 && dims.d[0] > 0 && dims.d[1] > 0;
    return (position == 1 || position == 2) && dims.nbDims == 1 && dims.d[0] > 0;
}

bool shapes_match(nvinfer1::Dims const& value, nvinfer1::Dims const& weight,
                  nvinfer1::Dims const& bias, nvinfer1::Dims const& output) noexcept {
    int32_t const width = value.d[1];
    return weight.d[0] == width && bias.d[0] == width && output.d[0] == value.d[0] &&
           output.d[1] == width;
}

bool runtime_contract(nvinfer1::PluginTensorDesc const* inputs,
                      nvinfer1::PluginTensorDesc const* outputs) noexcept {
    for (int32_t index = 0; index < 3; ++index) {
        if (!is_bf16_linear(inputs[index]) || !has_runtime_dims(inputs[index].dims, index))
            return false;
    }
    return is_bf16_linear(outputs[0]) && has_runtime_dims(outputs[0].dims, 3) &&
           shapes_match(inputs[0].dims, inputs[1].dims, inputs[2].dims, outputs[0].dims);
}

bool profile_ordered(nvinfer1::DynamicPluginTensorDesc const& desc, int32_t position) noexcept {
    if (!is_bf16_linear(desc.desc) || !has_network_dims(desc.desc.dims, position) ||
        !has_runtime_dims(desc.min, position) || !has_runtime_dims(desc.opt, position) ||
        !has_runtime_dims(desc.max, position)) {
        return false;
    }
    for (int32_t axis = 0; axis < desc.min.nbDims; ++axis) {
        if (desc.min.d[axis] > desc.opt.d[axis] || desc.opt.d[axis] > desc.max.d[axis])
            return false;
    }
    return true;
}

bool dynamic_contract(nvinfer1::DynamicPluginTensorDesc const* inputs,
                      nvinfer1::DynamicPluginTensorDesc const* outputs) noexcept {
    for (int32_t index = 0; index < 3; ++index) {
        if (!profile_ordered(inputs[index], index))
            return false;
    }
    if (!profile_ordered(outputs[0], 3) ||
        !shapes_match(inputs[0].desc.dims, inputs[1].desc.dims, inputs[2].desc.dims,
                      outputs[0].desc.dims)) {
        return false;
    }
    for (auto member :
         {&nvinfer1::DynamicPluginTensorDesc::min, &nvinfer1::DynamicPluginTensorDesc::opt,
          &nvinfer1::DynamicPluginTensorDesc::max}) {
        if (!shapes_match(inputs[0].*member, inputs[1].*member, inputs[2].*member,
                          outputs[0].*member)) {
            return false;
        }
    }
    return true;
}

bool has_expression_contract(nvinfer1::DimsExprs const* inputs) noexcept {
    if (inputs[0].nbDims != 2 || inputs[1].nbDims != 1 || inputs[2].nbDims != 1)
        return false;
    for (int32_t input = 0; input < 3; ++input) {
        for (int32_t axis = 0; axis < inputs[input].nbDims; ++axis) {
            if (inputs[input].d[axis] == nullptr)
                return false;
        }
    }
    return true;
}

void report_layer_norm_error(char const* detail) noexcept {
    std::fprintf(stderr, "[trtmc.minimax_h3.layer_norm] ATen layer_norm failed: %s\n",
                 detail != nullptr ? detail : "unknown error");
}

} // namespace

MiniMaxH3LayerNormPlugin::MiniMaxH3LayerNormPlugin(
    nvinfer1::PluginFieldCollection const& fields) noexcept
    : valid_(fields.nbFields == 0 && fields.fields == nullptr) {
    serialization_collection_.nbFields = 0;
    serialization_collection_.fields = nullptr;
}

MiniMaxH3LayerNormPlugin::MiniMaxH3LayerNormPlugin(MiniMaxH3LayerNormPlugin const& other) noexcept
    : valid_(other.valid_) {
    serialization_collection_.nbFields = 0;
    serialization_collection_.fields = nullptr;
}

nvinfer1::IPluginCapability*
MiniMaxH3LayerNormPlugin::getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept {
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

MiniMaxH3LayerNormPlugin* MiniMaxH3LayerNormPlugin::clone() noexcept {
    auto* plugin = new (std::nothrow) MiniMaxH3LayerNormPlugin(*this);
    if (plugin != nullptr && !plugin->isValid()) {
        delete plugin;
        return nullptr;
    }
    return plugin;
}

nvinfer1::AsciiChar const* MiniMaxH3LayerNormPlugin::getPluginName() const noexcept {
    return kPLUGIN_NAME;
}

nvinfer1::AsciiChar const* MiniMaxH3LayerNormPlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}

nvinfer1::AsciiChar const* MiniMaxH3LayerNormPlugin::getPluginNamespace() const noexcept {
    return "";
}

int32_t MiniMaxH3LayerNormPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* inputs,
                                                  int32_t input_count,
                                                  nvinfer1::DynamicPluginTensorDesc const* outputs,
                                                  int32_t output_count) noexcept {
    return valid_ && inputs != nullptr && outputs != nullptr && input_count == 3 &&
                   output_count == 1 && dynamic_contract(inputs, outputs)
               ? 0
               : 1;
}

int32_t MiniMaxH3LayerNormPlugin::getOutputDataTypes(nvinfer1::DataType* output_types,
                                                     int32_t output_count,
                                                     nvinfer1::DataType const* input_types,
                                                     int32_t input_count) const noexcept {
    if (output_types == nullptr || input_types == nullptr || output_count != 1 ||
        input_count != 3 || input_types[0] != nvinfer1::DataType::kBF16 ||
        input_types[1] != nvinfer1::DataType::kBF16 ||
        input_types[2] != nvinfer1::DataType::kBF16) {
        return 1;
    }
    output_types[0] = nvinfer1::DataType::kBF16;
    return 0;
}

int32_t MiniMaxH3LayerNormPlugin::getOutputShapes(
    nvinfer1::DimsExprs const* inputs, int32_t input_count, nvinfer1::DimsExprs const* shape_inputs,
    int32_t shape_input_count, nvinfer1::DimsExprs* outputs, int32_t output_count,
    nvinfer1::IExprBuilder& expression_builder) noexcept {
    if (inputs == nullptr || outputs == nullptr || input_count != 3 || output_count != 1 ||
        shape_input_count != 0 || !has_expression_contract(inputs)) {
        return 1;
    }
    (void)shape_inputs;
    (void)expression_builder;
    outputs[0] = inputs[0];
    return 0;
}

bool MiniMaxH3LayerNormPlugin::supportsFormatCombination(
    int32_t position, nvinfer1::DynamicPluginTensorDesc const* input_output, int32_t input_count,
    int32_t output_count) noexcept {
    return input_output != nullptr && input_count == 3 && output_count == 1 && position >= 0 &&
           position < 4 && is_bf16_linear(input_output[position].desc) &&
           has_network_dims(input_output[position].desc.dims, position);
}

int32_t MiniMaxH3LayerNormPlugin::getNbOutputs() const noexcept {
    return 1;
}

std::size_t MiniMaxH3LayerNormPlugin::getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const*,
                                                       int32_t,
                                                       nvinfer1::DynamicPluginTensorDesc const*,
                                                       int32_t) const noexcept {
    return 0;
}

char const* MiniMaxH3LayerNormPlugin::getTimingCacheID() noexcept {
    return "aten-layer-norm-bf16-eps-1e-6-v1";
}

char const* MiniMaxH3LayerNormPlugin::getMetadataString() noexcept {
    return "x=[rows,width]:bf16:linear;weight=[width]:bf16:linear;"
           "bias=[width]:bf16:linear;output=x-shape;eps=1e-6;cudnn=true;"
           "backend=aten-layer-norm";
}

int32_t MiniMaxH3LayerNormPlugin::onShapeChange(nvinfer1::PluginTensorDesc const* inputs,
                                                int32_t input_count,
                                                nvinfer1::PluginTensorDesc const* outputs,
                                                int32_t output_count) noexcept {
    return valid_ && inputs != nullptr && outputs != nullptr && input_count == 3 &&
                   output_count == 1 && runtime_contract(inputs, outputs)
               ? 0
               : 1;
}

int32_t MiniMaxH3LayerNormPlugin::enqueue(nvinfer1::PluginTensorDesc const* input_desc,
                                          nvinfer1::PluginTensorDesc const* output_desc,
                                          void const* const* inputs, void* const* outputs, void*,
                                          cudaStream_t stream) noexcept {
    if (!valid_ || input_desc == nullptr || output_desc == nullptr || inputs == nullptr ||
        outputs == nullptr || inputs[0] == nullptr || inputs[1] == nullptr ||
        inputs[2] == nullptr || outputs[0] == nullptr ||
        !runtime_contract(input_desc, output_desc)) {
        return 1;
    }
    try {
        int32_t device_index = 0;
        if (cudaGetDevice(&device_index) != cudaSuccess)
            return 1;
        c10::InferenceMode inference_mode;
        auto const torch_stream = c10::cuda::getStreamFromExternal(stream, device_index);
        c10::cuda::CUDAStreamGuard stream_guard(torch_stream);
        auto const options =
            at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, device_index);
        int64_t const rows = input_desc[0].dims.d[0];
        int64_t const width = input_desc[0].dims.d[1];
        auto value = at::from_blob(const_cast<void*>(inputs[0]), {rows, width}, options);
        auto weight = at::from_blob(const_cast<void*>(inputs[1]), {width}, options);
        auto bias = at::from_blob(const_cast<void*>(inputs[2]), {width}, options);
        auto result = at::layer_norm(value, {width}, std::optional<at::Tensor>(weight),
                                     std::optional<at::Tensor>(bias), kEPSILON, true);
        auto output = at::from_blob(outputs[0], {rows, width}, options);
        output.copy_(result);
        return 0;
    } catch (c10::Error const& error) {
        report_layer_norm_error(error.what());
    } catch (std::exception const& error) {
        report_layer_norm_error(error.what());
    } catch (...) {
        report_layer_norm_error("unknown exception");
    }
    return 1;
}

nvinfer1::IPluginV3*
MiniMaxH3LayerNormPlugin::attachToContext(nvinfer1::IPluginResourceContext*) noexcept {
    return clone();
}

nvinfer1::PluginFieldCollection const* MiniMaxH3LayerNormPlugin::getFieldsToSerialize() noexcept {
    return valid_ ? &serialization_collection_ : nullptr;
}

} // namespace trtmc::minimax_h3
