/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "plugins/runtime_kv/native_kv_append_plugin.h"

#include <cstdint>
#include <cuda_runtime_api.h>
#include <new>

namespace trtmc::runtime_kv {
namespace {

template <typename Storage>
__global__ void append_rows_kernel(Storage const* new_rows, Storage* cache_rows,
                                   int32_t const* write_index, int32_t const* active_length,
                                   int64_t bound_rows, int64_t new_row_count, int64_t row_width) {
    const int64_t element = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t element_count = new_row_count * row_width;
    if (element >= element_count)
        return;

    // These values intentionally stay on device. Invalid requests become a
    // no-write operation, preventing an out-of-bounds mutation even if a
    // caller bypasses the host-side admission checks.
    const int64_t first_row = static_cast<int64_t>(*write_index);
    const int64_t active_rows = static_cast<int64_t>(*active_length);
    if (first_row < 0 || active_rows < first_row || first_row > bound_rows ||
        active_rows > bound_rows || new_row_count > bound_rows - first_row ||
        active_rows != first_row + new_row_count) {
        return;
    }

    cache_rows[first_row * row_width + element] = new_rows[element];
}

bool is_cache_type(nvinfer1::DataType type) noexcept {
    return type == nvinfer1::DataType::kFLOAT || type == nvinfer1::DataType::kHALF ||
           type == nvinfer1::DataType::kBF16;
}

bool is_linear(nvinfer1::DynamicPluginTensorDesc const& desc) noexcept {
    return desc.desc.format == nvinfer1::TensorFormat::kLINEAR;
}

bool valid_runtime_shapes(nvinfer1::PluginTensorDesc const* inputs, int32_t nb_inputs,
                          nvinfer1::PluginTensorDesc const* outputs, int32_t nb_outputs) noexcept {
    if (inputs == nullptr || outputs == nullptr || nb_inputs != kNativeKvAppendInputCount ||
        nb_outputs != kNativeKvAppendOutputCount) {
        return false;
    }
    for (int32_t index = 0; index < 4; ++index) {
        if (inputs[index].dims.nbDims != 2 ||
            inputs[index].format != nvinfer1::TensorFormat::kLINEAR) {
            return false;
        }
    }
    if (inputs[4].dims.nbDims != 1 || inputs[4].dims.d[0] != 1 || inputs[5].dims.nbDims != 1 ||
        inputs[5].dims.d[0] != 1 || inputs[4].type != nvinfer1::DataType::kINT32 ||
        inputs[5].type != nvinfer1::DataType::kINT32) {
        return false;
    }
    if (inputs[0].dims.d[0] != inputs[1].dims.d[0] || inputs[0].dims.d[1] != inputs[1].dims.d[1] ||
        inputs[2].dims.d[0] != inputs[3].dims.d[0] || inputs[2].dims.d[1] != inputs[3].dims.d[1] ||
        inputs[0].dims.d[1] != inputs[2].dims.d[1]) {
        return false;
    }
    for (int32_t index = 0; index < 2; ++index) {
        if (outputs[index].dims.nbDims != 2 ||
            outputs[index].format != nvinfer1::TensorFormat::kLINEAR ||
            outputs[index].dims.d[0] != inputs[index].dims.d[0] ||
            outputs[index].dims.d[1] != inputs[index].dims.d[1]) {
            return false;
        }
    }
    return inputs[0].type == inputs[1].type && inputs[0].type == inputs[2].type &&
           inputs[0].type == inputs[3].type && outputs[0].type == inputs[0].type &&
           outputs[1].type == inputs[1].type && is_cache_type(inputs[0].type);
}

template <typename Storage>
cudaError_t launch_append_pair(nvinfer1::PluginTensorDesc const* input_desc,
                               void const* const* inputs, void* const* outputs,
                               cudaStream_t stream) noexcept {
    const int64_t bound_rows = input_desc[0].dims.d[0];
    const int64_t new_rows = input_desc[2].dims.d[0];
    const int64_t row_width = input_desc[0].dims.d[1];
    const int64_t count = new_rows * row_width;
    if (bound_rows <= 0 || new_rows <= 0 || row_width <= 0 || count <= 0)
        return cudaErrorInvalidValue;

    constexpr int32_t threads = 256;
    const auto blocks = static_cast<uint32_t>((count + threads - 1) / threads);
    auto const* write_index = static_cast<int32_t const*>(inputs[4]);
    auto const* active_length = static_cast<int32_t const*>(inputs[5]);

    append_rows_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<Storage const*>(inputs[2]), static_cast<Storage*>(outputs[0]), write_index,
        active_length, bound_rows, new_rows, row_width);
    auto status = cudaPeekAtLastError();
    if (status != cudaSuccess)
        return status;

    append_rows_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<Storage const*>(inputs[3]), static_cast<Storage*>(outputs[1]), write_index,
        active_length, bound_rows, new_rows, row_width);
    return cudaPeekAtLastError();
}

} // namespace

NativeKvAppendPlugin::NativeKvAppendPlugin(int32_t abi_version) noexcept
    : abi_version_(abi_version) {}

nvinfer1::IPluginCapability*
NativeKvAppendPlugin::getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept {
    switch (type) {
    case nvinfer1::PluginCapabilityType::kCORE:
        return static_cast<nvinfer1::IPluginV3OneCore*>(this);
    case nvinfer1::PluginCapabilityType::kBUILD:
        return static_cast<nvinfer1::IPluginV3OneBuildV2*>(this);
    case nvinfer1::PluginCapabilityType::kRUNTIME:
        return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
    }
    return nullptr;
}

nvinfer1::IPluginV3* NativeKvAppendPlugin::clone() noexcept {
    return new (std::nothrow) NativeKvAppendPlugin(abi_version_);
}

nvinfer1::AsciiChar const* NativeKvAppendPlugin::getPluginName() const noexcept {
    return kNativeKvAppendPluginName;
}

nvinfer1::AsciiChar const* NativeKvAppendPlugin::getPluginVersion() const noexcept {
    return kNativeKvAppendPluginVersion;
}

nvinfer1::AsciiChar const* NativeKvAppendPlugin::getPluginNamespace() const noexcept {
    return "";
}

int32_t NativeKvAppendPlugin::getNbOutputs() const noexcept {
    return kNativeKvAppendOutputCount;
}

int32_t NativeKvAppendPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* inputs,
                                              int32_t nb_inputs,
                                              nvinfer1::DynamicPluginTensorDesc const* outputs,
                                              int32_t nb_outputs) noexcept {
    if (inputs == nullptr || outputs == nullptr || nb_inputs != kNativeKvAppendInputCount ||
        nb_outputs != kNativeKvAppendOutputCount || abi_version_ != kNativeKvAppendPluginAbi) {
        return -1;
    }
    return 0;
}

int32_t NativeKvAppendPlugin::getOutputDataTypes(nvinfer1::DataType* output_types,
                                                 int32_t nb_outputs,
                                                 nvinfer1::DataType const* input_types,
                                                 int32_t nb_inputs) const noexcept {
    if (output_types == nullptr || input_types == nullptr ||
        nb_inputs != kNativeKvAppendInputCount || nb_outputs != kNativeKvAppendOutputCount) {
        return -1;
    }
    output_types[0] = input_types[0];
    output_types[1] = input_types[1];
    return 0;
}

int32_t NativeKvAppendPlugin::getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nb_inputs,
                                              nvinfer1::DimsExprs const*, int32_t nb_shape_inputs,
                                              nvinfer1::DimsExprs* outputs, int32_t nb_outputs,
                                              nvinfer1::IExprBuilder&) noexcept {
    if (inputs == nullptr || outputs == nullptr || nb_inputs != kNativeKvAppendInputCount ||
        nb_shape_inputs != 0 || nb_outputs != kNativeKvAppendOutputCount) {
        return -1;
    }
    outputs[0] = inputs[0];
    outputs[1] = inputs[1];
    return 0;
}

bool NativeKvAppendPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::DynamicPluginTensorDesc const* in_out, int32_t nb_inputs,
    int32_t nb_outputs) noexcept {
    if (in_out == nullptr || nb_inputs != kNativeKvAppendInputCount ||
        nb_outputs != kNativeKvAppendOutputCount || pos < 0 || pos >= nb_inputs + nb_outputs ||
        !is_linear(in_out[pos])) {
        return false;
    }
    if (pos == 0)
        return is_cache_type(in_out[0].desc.type);
    if (pos <= 3)
        return in_out[pos].desc.type == in_out[0].desc.type;
    if (pos <= 5)
        return in_out[pos].desc.type == nvinfer1::DataType::kINT32;
    if (pos == 6)
        return in_out[pos].desc.type == in_out[0].desc.type;
    return in_out[pos].desc.type == in_out[1].desc.type;
}

size_t NativeKvAppendPlugin::getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                              nvinfer1::DynamicPluginTensorDesc const*,
                                              int32_t) const noexcept {
    return 0;
}

int32_t NativeKvAppendPlugin::getAliasedInput(int32_t output_index) noexcept {
    return output_index >= 0 && output_index < kNativeKvAppendOutputCount ? output_index : -1;
}

nvinfer1::AsciiChar const* NativeKvAppendPlugin::getMetadataString() noexcept {
    return "abi=1;layout=contiguous_runtime_v1;workspace=0;writes=new_rows_only";
}

int32_t NativeKvAppendPlugin::onShapeChange(nvinfer1::PluginTensorDesc const* inputs,
                                            int32_t nb_inputs,
                                            nvinfer1::PluginTensorDesc const* outputs,
                                            int32_t nb_outputs) noexcept {
    return valid_runtime_shapes(inputs, nb_inputs, outputs, nb_outputs) ? 0 : -1;
}

int32_t NativeKvAppendPlugin::enqueue(nvinfer1::PluginTensorDesc const* input_desc,
                                      nvinfer1::PluginTensorDesc const* output_desc,
                                      void const* const* inputs, void* const* outputs, void*,
                                      cudaStream_t stream) noexcept {
    if (inputs == nullptr || outputs == nullptr || inputs[2] == nullptr || inputs[3] == nullptr ||
        inputs[4] == nullptr || inputs[5] == nullptr || outputs[0] == nullptr ||
        outputs[1] == nullptr ||
        onShapeChange(input_desc, kNativeKvAppendInputCount, output_desc,
                      kNativeKvAppendOutputCount) != 0) {
        return -1;
    }

    cudaError_t status = cudaErrorInvalidValue;
    switch (input_desc[0].type) {
    case nvinfer1::DataType::kFLOAT:
        status = launch_append_pair<uint32_t>(input_desc, inputs, outputs, stream);
        break;
    case nvinfer1::DataType::kHALF:
    case nvinfer1::DataType::kBF16:
        status = launch_append_pair<uint16_t>(input_desc, inputs, outputs, stream);
        break;
    default:
        return -1;
    }
    return status == cudaSuccess ? 0 : -1;
}

nvinfer1::IPluginV3*
NativeKvAppendPlugin::attachToContext(nvinfer1::IPluginResourceContext*) noexcept {
    return clone();
}

nvinfer1::PluginFieldCollection const* NativeKvAppendPlugin::getFieldsToSerialize() noexcept {
    serialized_fields_[0] = {
        "abi_version",
        &abi_version_,
        nvinfer1::PluginFieldType::kINT32,
        1,
    };
    serialized_fields_collection_.nbFields = static_cast<int32_t>(serialized_fields_.size());
    serialized_fields_collection_.fields = serialized_fields_.data();
    return &serialized_fields_collection_;
}

} // namespace trtmc::runtime_kv

extern "C" void trtmc_native_kv_append_fixture_force_link() noexcept {}
