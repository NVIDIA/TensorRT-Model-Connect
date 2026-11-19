/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/blip/runtime/fp32_plugin.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <new>
#include <stdexcept>

namespace trtmc::blip {
namespace {
constexpr const char* kName = "BlipFp32";
constexpr const char* kVersion = "1";

int volume(const nvinfer1::Dims& shape) noexcept {
    if (shape.nbDims < 1 || shape.nbDims > nvinfer1::Dims::MAX_DIMS)
        return 0;
    int result = 1;
    for (int i = 0; i < shape.nbDims; ++i) {
        if (shape.d[i] <= 0 || shape.d[i] > INT_MAX / result)
            return 0;
        result *= static_cast<int>(shape.d[i]);
    }
    return result;
}

bool same_shape(const nvinfer1::Dims& a, const nvinfer1::Dims& b) noexcept {
    return a.nbDims == b.nbDims && std::equal(a.d, a.d + a.nbDims, b.d);
}

bool float_linear(const nvinfer1::PluginTensorDesc& descriptor) noexcept {
    return descriptor.type == nvinfer1::DataType::kFLOAT &&
           descriptor.format == nvinfer1::TensorFormat::kLINEAR && volume(descriptor.dims) > 0;
}

class Creator final : public nvinfer1::IPluginCreatorV3One {
  public:
    const char* getPluginName() const noexcept override { return kName; }
    const char* getPluginVersion() const noexcept override { return kVersion; }
    const char* getPluginNamespace() const noexcept override { return ""; }
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override { return &fields_; }
    nvinfer1::IPluginV3* createPlugin(const char*, const nvinfer1::PluginFieldCollection* fields,
                                      nvinfer1::TensorRTPhase) noexcept override {
        if (!fields || fields->nbFields != 2 || !fields->fields)
            return nullptr;
        std::int32_t operation = -1;
        float epsilon = 0.0F;
        bool found_operation = false, found_epsilon = false;
        for (int i = 0; i < fields->nbFields; ++i) {
            const auto& field = fields->fields[i];
            if (!field.name || !field.data || field.length != 1)
                return nullptr;
            if (std::strcmp(field.name, "operation") == 0 && !found_operation &&
                field.type == nvinfer1::PluginFieldType::kINT32) {
                std::memcpy(&operation, field.data, sizeof(operation));
                found_operation = true;
            } else if (std::strcmp(field.name, "epsilon") == 0 && !found_epsilon &&
                       field.type == nvinfer1::PluginFieldType::kFLOAT32) {
                std::memcpy(&epsilon, field.data, sizeof(epsilon));
                found_epsilon = true;
            } else {
                return nullptr;
            }
        }
        if (!found_operation || !found_epsilon)
            return nullptr;
        auto* plugin = new (std::nothrow) Fp32Plugin(operation, epsilon);
        if (plugin && !plugin->valid()) {
            delete plugin;
            return nullptr;
        }
        return plugin;
    }

  private:
    std::array<nvinfer1::PluginField, 2> names_{{
        {"operation", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
        {"epsilon", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1},
    }};
    nvinfer1::PluginFieldCollection fields_{2, names_.data()};
};
} // namespace

Fp32Plugin::Fp32Plugin(std::int32_t operation, float epsilon) noexcept
    : operation_(operation), epsilon_(epsilon),
      fields_{{{"operation", &operation_, nvinfer1::PluginFieldType::kINT32, 1},
               {"epsilon", &epsilon_, nvinfer1::PluginFieldType::kFLOAT32, 1}}},
      collection_{2, fields_.data()} {}

bool Fp32Plugin::valid() const noexcept {
    return operation_ >= 0 && operation_ <= static_cast<int>(Fp32Operation::Gelu) &&
           std::isfinite(epsilon_) && epsilon_ > 0.0F;
}

int Fp32Plugin::input_count() const noexcept {
    return operation_ == static_cast<int>(Fp32Operation::Norm) ||
                   operation_ == static_cast<int>(Fp32Operation::Linear)
               ? 3
               : 1;
}

nvinfer1::IPluginCapability*
Fp32Plugin::getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept {
    switch (type) {
    case nvinfer1::PluginCapabilityType::kCORE:
        return static_cast<nvinfer1::IPluginV3OneCore*>(this);
    case nvinfer1::PluginCapabilityType::kBUILD:
        return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
    case nvinfer1::PluginCapabilityType::kRUNTIME:
        return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
    default:
        return nullptr;
    }
}

Fp32Plugin* Fp32Plugin::clone() noexcept {
    return new (std::nothrow) Fp32Plugin(operation_, epsilon_);
}
const char* Fp32Plugin::getPluginName() const noexcept {
    return kName;
}
const char* Fp32Plugin::getPluginVersion() const noexcept {
    return kVersion;
}
const char* Fp32Plugin::getPluginNamespace() const noexcept {
    return "";
}
std::int32_t Fp32Plugin::getNbOutputs() const noexcept {
    return 1;
}

std::int32_t Fp32Plugin::getOutputDataTypes(nvinfer1::DataType* outputs, std::int32_t output_count,
                                            const nvinfer1::DataType* inputs,
                                            std::int32_t count) const noexcept {
    if (!valid() || !inputs || !outputs || count != input_count() || output_count != 1)
        return -1;
    for (int i = 0; i < count; ++i)
        if (inputs[i] != nvinfer1::DataType::kFLOAT)
            return -1;
    outputs[0] = nvinfer1::DataType::kFLOAT;
    return 0;
}

std::int32_t Fp32Plugin::getOutputShapes(const nvinfer1::DimsExprs* inputs, std::int32_t count,
                                         const nvinfer1::DimsExprs*, std::int32_t shape_count,
                                         nvinfer1::DimsExprs* outputs, std::int32_t output_count,
                                         nvinfer1::IExprBuilder&) noexcept {
    if (!valid() || !inputs || !outputs || count != input_count() || shape_count != 0 ||
        output_count != 1 || inputs[0].nbDims < 1)
        return -1;
    outputs[0] = inputs[0];
    if (operation_ == static_cast<int>(Fp32Operation::Linear)) {
        if (inputs[0].nbDims != 2 || inputs[1].nbDims != 2)
            return -1;
        outputs[0].d[1] = inputs[1].d[0];
    }
    return 0;
}

bool Fp32Plugin::supportsFormatCombination(std::int32_t position,
                                           const nvinfer1::DynamicPluginTensorDesc* descriptors,
                                           std::int32_t count, std::int32_t output_count) noexcept {
    return valid() && descriptors && count == input_count() && output_count == 1 && position >= 0 &&
           position < count + 1 && descriptors[position].desc.type == nvinfer1::DataType::kFLOAT &&
           descriptors[position].desc.format == nvinfer1::TensorFormat::kLINEAR;
}

bool Fp32Plugin::descriptors_valid(const nvinfer1::PluginTensorDesc* inputs, int count,
                                   const nvinfer1::PluginTensorDesc* outputs,
                                   int output_count) const noexcept {
    if (!valid() || !inputs || !outputs || count != input_count() || output_count != 1)
        return false;
    for (int i = 0; i < count; ++i)
        if (!float_linear(inputs[i]))
            return false;
    if (!float_linear(outputs[0]))
        return false;
    const auto& shape = inputs[0].dims;
    if (operation_ == static_cast<int>(Fp32Operation::Linear))
        return shape.nbDims == 2 && inputs[1].dims.nbDims == 2 &&
               inputs[1].dims.d[1] == shape.d[1] && volume(inputs[2].dims) == inputs[1].dims.d[0] &&
               outputs[0].dims.nbDims == 2 && outputs[0].dims.d[0] == shape.d[0] &&
               outputs[0].dims.d[1] == inputs[1].dims.d[0];
    if (!same_shape(shape, outputs[0].dims))
        return false;
    if (operation_ == static_cast<int>(Fp32Operation::Norm))
        return shape.nbDims == 2 && shape.d[1] <= 0x1000000 &&
               volume(inputs[1].dims) == shape.d[1] && volume(inputs[2].dims) == shape.d[1];
    return true;
}

std::int32_t Fp32Plugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* inputs,
                                         std::int32_t count,
                                         const nvinfer1::DynamicPluginTensorDesc* outputs,
                                         std::int32_t output_count) noexcept {
    if (!inputs || !outputs || count != input_count() || output_count != 1)
        return -1;
    std::array<nvinfer1::PluginTensorDesc, 3> descriptors{};
    for (int i = 0; i < count; ++i)
        descriptors[i] = inputs[i].desc;
    const auto output = outputs[0].desc;
    return descriptors_valid(descriptors.data(), count, &output, output_count) ? 0 : -1;
}

std::size_t Fp32Plugin::getWorkspaceSize(const nvinfer1::DynamicPluginTensorDesc*,
                                         std::int32_t count,
                                         const nvinfer1::DynamicPluginTensorDesc*,
                                         std::int32_t output_count) const noexcept {
    return valid() && count == input_count() && output_count == 1 &&
                   operation_ == static_cast<int>(Fp32Operation::Linear)
               ? kLinearWorkspaceBytes
               : 0;
}

const char* Fp32Plugin::getTimingCacheID() noexcept {
    static constexpr const char* ids[]{"blip.norm.fp32.1", "blip.linear.fp32.1",
                                       "blip.softmax.fp32.1", "blip.gelu.fp32.1"};
    return valid() ? ids[operation_] : nullptr;
}
const char* Fp32Plugin::getMetadataString() noexcept {
    return "BLIP native FP32 CUDA math";
}

std::int32_t Fp32Plugin::onShapeChange(const nvinfer1::PluginTensorDesc* inputs, std::int32_t count,
                                       const nvinfer1::PluginTensorDesc* outputs,
                                       std::int32_t output_count) noexcept {
    return descriptors_valid(inputs, count, outputs, output_count) ? 0 : -1;
}

std::int32_t Fp32Plugin::enqueue(const nvinfer1::PluginTensorDesc* input_descriptors,
                                 const nvinfer1::PluginTensorDesc* output_descriptors,
                                 const void* const* inputs, void* const* outputs, void* workspace,
                                 cudaStream_t stream) noexcept {
    if (!descriptors_valid(input_descriptors, input_count(), output_descriptors, 1) || !inputs ||
        !outputs || !outputs[0])
        return -1;
    for (int i = 0; i < input_count(); ++i)
        if (!inputs[i])
            return -1;
    const auto* input = static_cast<const float*>(inputs[0]);
    auto* output = static_cast<float*>(outputs[0]);
    const auto& shape = input_descriptors[0].dims;
    const int count = volume(shape);
    const int width = static_cast<int>(shape.d[shape.nbDims - 1]);
    switch (static_cast<Fp32Operation>(operation_)) {
    case Fp32Operation::Norm:
        return norm_fp32(input, static_cast<const float*>(inputs[1]),
                         static_cast<const float*>(inputs[2]), output, count / width, width,
                         epsilon_, stream);
    case Fp32Operation::Linear:
        try {
            if (!linear_)
                linear_ = create_linear_context();
            return linear_fp32(*linear_, input, static_cast<const float*>(inputs[1]),
                               static_cast<const float*>(inputs[2]), output, count / width, width,
                               static_cast<int>(input_descriptors[1].dims.d[0]), workspace,
                               workspace ? kLinearWorkspaceBytes : 0, stream);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "BLIP CUDA projection: %s\n", error.what());
            return -1;
        }
    case Fp32Operation::Softmax:
        return softmax_fp32(input, output, count / width, width, stream);
    case Fp32Operation::Gelu:
        return gelu_fp32(input, output, count, stream);
    }
    return -1;
}

nvinfer1::IPluginV3* Fp32Plugin::attachToContext(nvinfer1::IPluginResourceContext*) noexcept {
    auto* result = clone();
    if (result && operation_ == static_cast<int>(Fp32Operation::Linear)) {
        try {
            result->linear_ = create_linear_context();
        } catch (const std::exception& error) {
            std::fprintf(stderr, "BLIP CUDA projection initialization: %s\n", error.what());
            delete result;
            return nullptr;
        }
    }
    return result;
}
const nvinfer1::PluginFieldCollection* Fp32Plugin::getFieldsToSerialize() noexcept {
    return &collection_;
}

void initialize_fp32_kernels() {
    static std::once_flag once;
    std::call_once(once, [] {
        auto* registry = getPluginRegistry();
        if (registry->getCreator(kName, kVersion, ""))
            return;
        static Creator creator;
        if (!registry->registerCreator(creator, "") && !registry->getCreator(kName, kVersion, ""))
            throw std::runtime_error("BLIP could not register its native CUDA TensorRT plugin");
    });
}
} // namespace trtmc::blip
