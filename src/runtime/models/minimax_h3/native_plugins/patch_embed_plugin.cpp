/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "patch_embed_plugin.h"

#include <ATen/ATen.h>
#include <ATen/ops/conv3d.h>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>
#include <cstdio>
#include <exception>
#include <new>

namespace trtmc::minimax_h3 {
namespace {

using Plugin = MiniMaxH3PatchEmbedPlugin;

bool has_pixel_network_dims(nvinfer1::Dims const& dims) noexcept {
    return dims.nbDims == 2 && (dims.d[0] == -1 || dims.d[0] > 0) &&
           dims.d[1] == Plugin::kPIXEL_ROW_WIDTH;
}

bool has_pixel_runtime_dims(nvinfer1::Dims const& dims) noexcept {
    return dims.nbDims == 2 && dims.d[0] > 0 && dims.d[1] == Plugin::kPIXEL_ROW_WIDTH;
}

bool has_output_network_dims(nvinfer1::Dims const& dims) noexcept {
    return dims.nbDims == 2 && (dims.d[0] == -1 || dims.d[0] > 0) &&
           dims.d[1] == Plugin::kOUTPUT_CHANNELS;
}

bool has_output_runtime_dims(nvinfer1::Dims const& dims) noexcept {
    return dims.nbDims == 2 && dims.d[0] > 0 && dims.d[1] == Plugin::kOUTPUT_CHANNELS;
}

bool has_weight_dims(nvinfer1::Dims const& dims) noexcept {
    return dims.nbDims == 5 && dims.d[0] == Plugin::kOUTPUT_CHANNELS &&
           dims.d[1] == Plugin::kINPUT_CHANNELS && dims.d[2] == Plugin::kTEMPORAL_PATCH &&
           dims.d[3] == Plugin::kSPATIAL_PATCH && dims.d[4] == Plugin::kSPATIAL_PATCH;
}

bool has_bias_dims(nvinfer1::Dims const& dims) noexcept {
    return dims.nbDims == 1 && dims.d[0] == Plugin::kOUTPUT_CHANNELS;
}

bool is_bf16_linear(nvinfer1::PluginTensorDesc const& desc) noexcept {
    return desc.type == nvinfer1::DataType::kBF16 && desc.format == nvinfer1::TensorFormat::kLINEAR;
}

bool has_network_contract(nvinfer1::PluginTensorDesc const& desc, int32_t position) noexcept {
    if (!is_bf16_linear(desc))
        return false;
    if (position == 0)
        return has_pixel_network_dims(desc.dims);
    if (position == 1)
        return has_weight_dims(desc.dims);
    if (position == 2)
        return has_bias_dims(desc.dims);
    return position == 3 && has_output_network_dims(desc.dims);
}

bool has_runtime_contract(nvinfer1::PluginTensorDesc const& desc, int32_t position) noexcept {
    if (!is_bf16_linear(desc))
        return false;
    if (position == 0)
        return has_pixel_runtime_dims(desc.dims);
    if (position == 1)
        return has_weight_dims(desc.dims);
    if (position == 2)
        return has_bias_dims(desc.dims);
    return position == 3 && has_output_runtime_dims(desc.dims);
}

bool has_dynamic_contract(nvinfer1::DynamicPluginTensorDesc const& desc,
                          int32_t position) noexcept {
    if (!has_network_contract(desc.desc, position))
        return false;
    if (position == 0) {
        return has_pixel_runtime_dims(desc.min) && has_pixel_runtime_dims(desc.opt) &&
               has_pixel_runtime_dims(desc.max) && desc.min.d[0] <= desc.opt.d[0] &&
               desc.opt.d[0] <= desc.max.d[0];
    }
    if (position == 3) {
        return has_output_runtime_dims(desc.min) && has_output_runtime_dims(desc.opt) &&
               has_output_runtime_dims(desc.max) && desc.min.d[0] <= desc.opt.d[0] &&
               desc.opt.d[0] <= desc.max.d[0];
    }
    auto const shape_matches = position == 1 ? has_weight_dims : has_bias_dims;
    return shape_matches(desc.min) && shape_matches(desc.opt) && shape_matches(desc.max);
}

bool same_dynamic_rows(nvinfer1::DynamicPluginTensorDesc const& pixels,
                       nvinfer1::DynamicPluginTensorDesc const& output) noexcept {
    return pixels.desc.dims.d[0] == output.desc.dims.d[0] && pixels.min.d[0] == output.min.d[0] &&
           pixels.opt.d[0] == output.opt.d[0] && pixels.max.d[0] == output.max.d[0];
}

bool has_constant_expression(nvinfer1::IDimensionExpr const* expression,
                             int64_t expected) noexcept {
    return expression != nullptr && expression->isConstant() &&
           expression->getConstantValue() == expected;
}

bool has_pixel_expression_contract(nvinfer1::DimsExprs const& dims) noexcept {
    return dims.nbDims == 2 && dims.d[0] != nullptr &&
           (!dims.d[0]->isConstant() || dims.d[0]->getConstantValue() > 0) &&
           has_constant_expression(dims.d[1], Plugin::kPIXEL_ROW_WIDTH);
}

bool has_weight_expression_contract(nvinfer1::DimsExprs const& dims) noexcept {
    return dims.nbDims == 5 && has_constant_expression(dims.d[0], Plugin::kOUTPUT_CHANNELS) &&
           has_constant_expression(dims.d[1], Plugin::kINPUT_CHANNELS) &&
           has_constant_expression(dims.d[2], Plugin::kTEMPORAL_PATCH) &&
           has_constant_expression(dims.d[3], Plugin::kSPATIAL_PATCH) &&
           has_constant_expression(dims.d[4], Plugin::kSPATIAL_PATCH);
}

bool has_bias_expression_contract(nvinfer1::DimsExprs const& dims) noexcept {
    return dims.nbDims == 1 && has_constant_expression(dims.d[0], Plugin::kOUTPUT_CHANNELS);
}

bool same_runtime_contract(nvinfer1::PluginTensorDesc const* inputs,
                           nvinfer1::PluginTensorDesc const* outputs) noexcept {
    return has_runtime_contract(inputs[0], 0) && has_runtime_contract(inputs[1], 1) &&
           has_runtime_contract(inputs[2], 2) && has_runtime_contract(outputs[0], 3) &&
           inputs[0].dims.d[0] == outputs[0].dims.d[0];
}

void report_patch_embed_error(char const* category, char const* detail) noexcept {
    std::fprintf(stderr, "[trtmc.minimax_h3.patch_embed] %s: %s\n", category,
                 detail != nullptr ? detail : "unknown error");
}

} // namespace

MiniMaxH3PatchEmbedPlugin::MiniMaxH3PatchEmbedPlugin(
    nvinfer1::PluginFieldCollection const& fields) noexcept
    : valid_(fields.nbFields == 0 && fields.fields == nullptr) {
    serialization_collection_.nbFields = 0;
    serialization_collection_.fields = nullptr;
}

MiniMaxH3PatchEmbedPlugin::MiniMaxH3PatchEmbedPlugin(
    MiniMaxH3PatchEmbedPlugin const& other) noexcept
    : valid_(other.valid_) {
    serialization_collection_.nbFields = 0;
    serialization_collection_.fields = nullptr;
}

nvinfer1::IPluginCapability*
MiniMaxH3PatchEmbedPlugin::getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept {
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

MiniMaxH3PatchEmbedPlugin* MiniMaxH3PatchEmbedPlugin::clone() noexcept {
    auto* plugin = new (std::nothrow) MiniMaxH3PatchEmbedPlugin(*this);
    if (plugin != nullptr && !plugin->isValid()) {
        delete plugin;
        return nullptr;
    }
    return plugin;
}

nvinfer1::AsciiChar const* MiniMaxH3PatchEmbedPlugin::getPluginName() const noexcept {
    return kPLUGIN_NAME;
}

nvinfer1::AsciiChar const* MiniMaxH3PatchEmbedPlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}

nvinfer1::AsciiChar const* MiniMaxH3PatchEmbedPlugin::getPluginNamespace() const noexcept {
    return "";
}

int32_t MiniMaxH3PatchEmbedPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* inputs,
                                                   int32_t input_count,
                                                   nvinfer1::DynamicPluginTensorDesc const* outputs,
                                                   int32_t output_count) noexcept {
    return valid_ && inputs != nullptr && outputs != nullptr && input_count == 3 &&
                   output_count == 1 && has_dynamic_contract(inputs[0], 0) &&
                   has_dynamic_contract(inputs[1], 1) && has_dynamic_contract(inputs[2], 2) &&
                   has_dynamic_contract(outputs[0], 3) && same_dynamic_rows(inputs[0], outputs[0])
               ? 0
               : 1;
}

int32_t MiniMaxH3PatchEmbedPlugin::getOutputDataTypes(nvinfer1::DataType* output_types,
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

int32_t MiniMaxH3PatchEmbedPlugin::getOutputShapes(
    nvinfer1::DimsExprs const* inputs, int32_t input_count, nvinfer1::DimsExprs const* shape_inputs,
    int32_t shape_input_count, nvinfer1::DimsExprs* outputs, int32_t output_count,
    nvinfer1::IExprBuilder& expression_builder) noexcept {
    if (inputs == nullptr || outputs == nullptr || input_count != 3 || output_count != 1 ||
        shape_input_count != 0 || !has_pixel_expression_contract(inputs[0]) ||
        !has_weight_expression_contract(inputs[1]) || !has_bias_expression_contract(inputs[2])) {
        return 1;
    }
    (void)shape_inputs;
    outputs[0].nbDims = 2;
    outputs[0].d[0] = inputs[0].d[0];
    outputs[0].d[1] = expression_builder.constant(kOUTPUT_CHANNELS);
    return outputs[0].d[1] != nullptr ? 0 : 1;
}

bool MiniMaxH3PatchEmbedPlugin::supportsFormatCombination(
    int32_t position, nvinfer1::DynamicPluginTensorDesc const* input_output, int32_t input_count,
    int32_t output_count) noexcept {
    return input_output != nullptr && input_count == 3 && output_count == 1 && position >= 0 &&
           position < 4 && has_network_contract(input_output[position].desc, position);
}

int32_t MiniMaxH3PatchEmbedPlugin::getNbOutputs() const noexcept {
    return 1;
}

std::size_t MiniMaxH3PatchEmbedPlugin::getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const*,
                                                        int32_t,
                                                        nvinfer1::DynamicPluginTensorDesc const*,
                                                        int32_t) const noexcept {
    return 0;
}

char const* MiniMaxH3PatchEmbedPlugin::getTimingCacheID() noexcept {
    return "aten-conv3d-bf16-patch-3x2x16x16-out1152-v1";
}

char const* MiniMaxH3PatchEmbedPlugin::getMetadataString() noexcept {
    return "pixels=[rows,1536]:bf16:linear;weight=[1152,3,2,16,16]:bf16:linear;"
           "bias=[1152]:bf16:linear;output=[rows,1152]:bf16:linear;"
           "conv3d_stride=[2,16,16];padding=[0,0,0];dilation=[1,1,1];groups=1;"
           "backend=aten-conv3d";
}

int32_t MiniMaxH3PatchEmbedPlugin::onShapeChange(nvinfer1::PluginTensorDesc const* inputs,
                                                 int32_t input_count,
                                                 nvinfer1::PluginTensorDesc const* outputs,
                                                 int32_t output_count) noexcept {
    return valid_ && inputs != nullptr && outputs != nullptr && input_count == 3 &&
                   output_count == 1 && same_runtime_contract(inputs, outputs)
               ? 0
               : 1;
}

int32_t MiniMaxH3PatchEmbedPlugin::enqueue(nvinfer1::PluginTensorDesc const* input_desc,
                                           nvinfer1::PluginTensorDesc const* output_desc,
                                           void const* const* inputs, void* const* outputs, void*,
                                           cudaStream_t stream) noexcept {
    if (!valid_ || input_desc == nullptr || output_desc == nullptr || inputs == nullptr ||
        outputs == nullptr || inputs[0] == nullptr || inputs[1] == nullptr ||
        inputs[2] == nullptr || outputs[0] == nullptr ||
        !same_runtime_contract(input_desc, output_desc)) {
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

        auto pixels = at::from_blob(
            const_cast<void*>(inputs[0]),
            {rows, kINPUT_CHANNELS, kTEMPORAL_PATCH, kSPATIAL_PATCH, kSPATIAL_PATCH}, options);
        auto weight = at::from_blob(
            const_cast<void*>(inputs[1]),
            {kOUTPUT_CHANNELS, kINPUT_CHANNELS, kTEMPORAL_PATCH, kSPATIAL_PATCH, kSPATIAL_PATCH},
            options);
        auto bias = at::from_blob(const_cast<void*>(inputs[2]), {kOUTPUT_CHANNELS}, options);
        auto result =
            at::conv3d(pixels, weight, bias, {kTEMPORAL_PATCH, kSPATIAL_PATCH, kSPATIAL_PATCH},
                       {0, 0, 0}, {1, 1, 1}, 1)
                .reshape({rows, kOUTPUT_CHANNELS});
        auto output = at::from_blob(outputs[0], {rows, kOUTPUT_CHANNELS}, options);
        output.copy_(result);
        return 0;
    } catch (c10::Error const& error) {
        report_patch_embed_error("ATen conv3d failed", error.what());
    } catch (std::exception const& error) {
        report_patch_embed_error("native patch embed failed", error.what());
    } catch (...) {
        report_patch_embed_error("native patch embed failed", "unknown exception");
    }
    return 1;
}

nvinfer1::IPluginV3*
MiniMaxH3PatchEmbedPlugin::attachToContext(nvinfer1::IPluginResourceContext*) noexcept {
    return clone();
}

nvinfer1::PluginFieldCollection const* MiniMaxH3PatchEmbedPlugin::getFieldsToSerialize() noexcept {
    return valid_ ? &serialization_collection_ : nullptr;
}

} // namespace trtmc::minimax_h3
