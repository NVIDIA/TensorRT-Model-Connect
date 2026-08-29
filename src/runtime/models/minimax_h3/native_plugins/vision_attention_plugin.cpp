/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vision_attention_plugin.h"

#include <ATen/ATen.h>
#include <ATen/ops/scaled_dot_product_attention.h>
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

using VisionPlugin = MiniMaxH3VisionAttentionPlugin;

// Python's ``72**-0.5`` result, rather than the adjacent result produced by
// ``1.0 / math.sqrt(72)``. This is the exact optional scale passed by the
// released Hugging Face Qwen3-VL vision attention implementation.
constexpr double kVisionAttentionScale = 0x1.e2b7dddfefa66p-4;

bool has_network_dims(nvinfer1::Dims const& dims, int32_t row_width) noexcept {
    return dims.nbDims == 2 && (dims.d[0] == -1 || dims.d[0] > 0) && dims.d[1] == row_width;
}

bool has_runtime_dims(nvinfer1::Dims const& dims, int32_t row_width) noexcept {
    return dims.nbDims == 2 && dims.d[0] > 0 && dims.d[1] == row_width;
}

bool same_dims(nvinfer1::Dims const& lhs, nvinfer1::Dims const& rhs) noexcept {
    if (lhs.nbDims != rhs.nbDims)
        return false;
    for (int32_t index = 0; index < lhs.nbDims; ++index) {
        if (lhs.d[index] != rhs.d[index])
            return false;
    }
    return true;
}

bool is_network_desc(nvinfer1::PluginTensorDesc const& desc, int32_t row_width) noexcept {
    return desc.type == nvinfer1::DataType::kBF16 &&
           desc.format == nvinfer1::TensorFormat::kLINEAR && has_network_dims(desc.dims, row_width);
}

bool is_runtime_desc(nvinfer1::PluginTensorDesc const& desc, int32_t row_width) noexcept {
    return desc.type == nvinfer1::DataType::kBF16 &&
           desc.format == nvinfer1::TensorFormat::kLINEAR && has_runtime_dims(desc.dims, row_width);
}

bool is_dynamic_desc(nvinfer1::DynamicPluginTensorDesc const& desc, int32_t row_width) noexcept {
    return is_network_desc(desc.desc, row_width) && has_runtime_dims(desc.min, row_width) &&
           has_runtime_dims(desc.opt, row_width) && has_runtime_dims(desc.max, row_width) &&
           desc.min.d[0] <= desc.opt.d[0] && desc.opt.d[0] <= desc.max.d[0];
}

bool same_dynamic_rows(nvinfer1::DynamicPluginTensorDesc const& lhs,
                       nvinfer1::DynamicPluginTensorDesc const& rhs) noexcept {
    return lhs.desc.dims.d[0] == rhs.desc.dims.d[0] && lhs.min.d[0] == rhs.min.d[0] &&
           lhs.opt.d[0] == rhs.opt.d[0] && lhs.max.d[0] == rhs.max.d[0];
}

bool has_shape_expression_contract(nvinfer1::DimsExprs const& dims, int32_t row_width) noexcept {
    if (dims.nbDims != 2 || dims.d[0] == nullptr || dims.d[1] == nullptr ||
        !dims.d[1]->isConstant() || dims.d[1]->getConstantValue() != row_width) {
        return false;
    }
    return !dims.d[0]->isConstant() || dims.d[0]->getConstantValue() > 0;
}

bool same_vision_runtime_contract(nvinfer1::PluginTensorDesc const* inputs,
                                  nvinfer1::PluginTensorDesc const* outputs) noexcept {
    if (!is_runtime_desc(inputs[0], VisionPlugin::kROW_WIDTH) ||
        !is_runtime_desc(inputs[1], VisionPlugin::kROW_WIDTH) ||
        !is_runtime_desc(inputs[2], VisionPlugin::kROW_WIDTH) ||
        !is_runtime_desc(outputs[0], VisionPlugin::kROW_WIDTH)) {
        return false;
    }
    return same_dims(inputs[0].dims, inputs[1].dims) && same_dims(inputs[0].dims, inputs[2].dims) &&
           same_dims(inputs[0].dims, outputs[0].dims);
}

void report_attention_error(char const* role, char const* category, char const* detail) noexcept {
    std::fprintf(stderr, "[trtmc.minimax_h3.%s_attention] %s: %s\n", role, category,
                 detail != nullptr ? detail : "unknown error");
}

} // namespace

MiniMaxH3VisionAttentionPlugin::MiniMaxH3VisionAttentionPlugin(
    nvinfer1::PluginFieldCollection const& fields) noexcept
    : valid_(fields.nbFields == 0 && fields.fields == nullptr) {
    serialization_collection_.nbFields = 0;
    serialization_collection_.fields = nullptr;
}

MiniMaxH3VisionAttentionPlugin::MiniMaxH3VisionAttentionPlugin(
    MiniMaxH3VisionAttentionPlugin const& other) noexcept
    : valid_(other.valid_) {
    serialization_collection_.nbFields = 0;
    serialization_collection_.fields = nullptr;
}

nvinfer1::IPluginCapability* MiniMaxH3VisionAttentionPlugin::getCapabilityInterface(
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

MiniMaxH3VisionAttentionPlugin* MiniMaxH3VisionAttentionPlugin::clone() noexcept {
    auto* plugin = new (std::nothrow) MiniMaxH3VisionAttentionPlugin(*this);
    if (plugin != nullptr && !plugin->isValid()) {
        delete plugin;
        return nullptr;
    }
    return plugin;
}

nvinfer1::AsciiChar const* MiniMaxH3VisionAttentionPlugin::getPluginName() const noexcept {
    return kPLUGIN_NAME;
}

nvinfer1::AsciiChar const* MiniMaxH3VisionAttentionPlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}

nvinfer1::AsciiChar const* MiniMaxH3VisionAttentionPlugin::getPluginNamespace() const noexcept {
    return "";
}

int32_t MiniMaxH3VisionAttentionPlugin::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t input_count,
    nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t output_count) noexcept {
    if (!valid_ || inputs == nullptr || outputs == nullptr || input_count != 3 ||
        output_count != 1) {
        return 1;
    }
    for (int32_t index = 0; index < input_count; ++index) {
        if (!is_dynamic_desc(inputs[index], kROW_WIDTH) ||
            !same_dynamic_rows(inputs[0], inputs[index]))
            return 1;
    }
    return is_dynamic_desc(outputs[0], kROW_WIDTH) && same_dynamic_rows(inputs[0], outputs[0]) ? 0
                                                                                               : 1;
}

int32_t MiniMaxH3VisionAttentionPlugin::getOutputDataTypes(nvinfer1::DataType* output_types,
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

int32_t MiniMaxH3VisionAttentionPlugin::getOutputShapes(
    nvinfer1::DimsExprs const* inputs, int32_t input_count, nvinfer1::DimsExprs const* shape_inputs,
    int32_t shape_input_count, nvinfer1::DimsExprs* outputs, int32_t output_count,
    nvinfer1::IExprBuilder& expression_builder) noexcept {
    if (inputs == nullptr || outputs == nullptr || input_count != 3 || output_count != 1 ||
        shape_input_count != 0 || !has_shape_expression_contract(inputs[0], kROW_WIDTH) ||
        !has_shape_expression_contract(inputs[1], kROW_WIDTH) ||
        !has_shape_expression_contract(inputs[2], kROW_WIDTH)) {
        return 1;
    }
    (void)shape_inputs;
    (void)expression_builder;
    outputs[0] = inputs[0];
    return 0;
}

bool MiniMaxH3VisionAttentionPlugin::supportsFormatCombination(
    int32_t position, nvinfer1::DynamicPluginTensorDesc const* input_output, int32_t input_count,
    int32_t output_count) noexcept {
    return input_output != nullptr && input_count == 3 && output_count == 1 && position >= 0 &&
           position < 4 && is_network_desc(input_output[position].desc, kROW_WIDTH);
}

int32_t MiniMaxH3VisionAttentionPlugin::getNbOutputs() const noexcept {
    return 1;
}

std::size_t
MiniMaxH3VisionAttentionPlugin::getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                                 nvinfer1::DynamicPluginTensorDesc const*,
                                                 int32_t) const noexcept {
    return 0;
}

char const* MiniMaxH3VisionAttentionPlugin::getTimingCacheID() noexcept {
    return "aten-sdpa-bf16-row-major-h16-d72-scale-0x1.e2b7dddfefa66p-4-v1";
}

char const* MiniMaxH3VisionAttentionPlugin::getMetadataString() noexcept {
    return "qkv=[rows,1152]:bf16:linear;output=q-shape;heads=16;head_dim=72;"
           "scale=0x1.e2b7dddfefa66p-4;mask=null;dropout=0;causal=false;gqa=true;"
           "q_prescale=false;backend=aten-sdpa";
}

int32_t MiniMaxH3VisionAttentionPlugin::onShapeChange(nvinfer1::PluginTensorDesc const* inputs,
                                                      int32_t input_count,
                                                      nvinfer1::PluginTensorDesc const* outputs,
                                                      int32_t output_count) noexcept {
    return valid_ && inputs != nullptr && outputs != nullptr && input_count == 3 &&
                   output_count == 1 && same_vision_runtime_contract(inputs, outputs)
               ? 0
               : 1;
}

int32_t MiniMaxH3VisionAttentionPlugin::enqueue(nvinfer1::PluginTensorDesc const* input_desc,
                                                nvinfer1::PluginTensorDesc const* output_desc,
                                                void const* const* inputs, void* const* outputs,
                                                void*, cudaStream_t stream) noexcept {
    if (!valid_ || input_desc == nullptr || output_desc == nullptr || inputs == nullptr ||
        outputs == nullptr || inputs[0] == nullptr || inputs[1] == nullptr ||
        inputs[2] == nullptr || outputs[0] == nullptr ||
        !same_vision_runtime_contract(input_desc, output_desc)) {
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

        auto query = at::from_blob(const_cast<void*>(inputs[0]), {rows, kHEADS, kHEAD_DIM}, options)
                         .permute({1, 0, 2})
                         .unsqueeze(0);
        auto key = at::from_blob(const_cast<void*>(inputs[1]), {rows, kHEADS, kHEAD_DIM}, options)
                       .permute({1, 0, 2})
                       .unsqueeze(0);
        auto value = at::from_blob(const_cast<void*>(inputs[2]), {rows, kHEADS, kHEAD_DIM}, options)
                         .permute({1, 0, 2})
                         .unsqueeze(0);

        auto context =
            at::scaled_dot_product_attention(query, key, value, std::nullopt, 0.0, false,
                                             std::optional<double>(kVisionAttentionScale), true);
        auto result = context.transpose(1, 2).reshape({rows, kROW_WIDTH});
        auto output = at::from_blob(outputs[0], {rows, kROW_WIDTH}, options);
        output.copy_(result);
        return 0;
    } catch (c10::Error const& error) {
        report_attention_error("vision", "ATen scaled_dot_product_attention failed", error.what());
    } catch (std::exception const& error) {
        report_attention_error("vision", "native attention failed", error.what());
    } catch (...) {
        report_attention_error("vision", "native attention failed", "unknown exception");
    }
    return 1;
}

nvinfer1::IPluginV3*
MiniMaxH3VisionAttentionPlugin::attachToContext(nvinfer1::IPluginResourceContext*) noexcept {
    return clone();
}

nvinfer1::PluginFieldCollection const*
MiniMaxH3VisionAttentionPlugin::getFieldsToSerialize() noexcept {
    return valid_ ? &serialization_collection_ : nullptr;
}

} // namespace trtmc::minimax_h3
