/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Host-only identity boundary for an already-computed BF16 projection.
#include <NvInferPlugin.h>
#include <NvInferRuntime.h>
#include <cstdint>
#include <new>

#ifndef HSTU_PLUGIN_NAMESPACE
#error "HSTU_PLUGIN_NAMESPACE must identify the model-owned plugin library"
#endif

namespace {
constexpr char kName[] = "HstuProjectionBarrier";
constexpr char kVersion[] = "1";
constexpr char kNamespace[] = HSTU_PLUGIN_NAMESPACE;
constexpr std::int64_t kWidth = 1024;

bool descriptor(const nvinfer1::PluginTensorDesc& value, bool dynamic) noexcept {
    return value.type == nvinfer1::DataType::kBF16 &&
           value.format == nvinfer1::TensorFormat::kLINEAR && value.dims.nbDims == 2 &&
           value.dims.d[1] == kWidth && (value.dims.d[0] > 0 || (dynamic && value.dims.d[0] == -1));
}

bool same_storage(const void* const* input, void* const* output) noexcept {
    return input && output && input[0] && input[0] == output[0];
}

class ProjectionBarrier final : public nvinfer1::IPluginV3,
                                public nvinfer1::IPluginV3OneCore,
                                public nvinfer1::IPluginV3OneBuildV2,
                                public nvinfer1::IPluginV3OneRuntime {
  public:
    nvinfer1::IPluginCapability*
    getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override {
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
    ProjectionBarrier* clone() noexcept override {
        // A clone must receive its own successful onShapeChange callback.
        return new (std::nothrow) ProjectionBarrier;
    }
    const char* getPluginName() const noexcept override { return kName; }
    const char* getPluginVersion() const noexcept override { return kVersion; }
    const char* getPluginNamespace() const noexcept override { return kNamespace; }
    int32_t getNbOutputs() const noexcept override { return 1; }
    int32_t getAliasedInput(int32_t output) noexcept override { return output == 0 ? 0 : -1; }
    int32_t getOutputDataTypes(nvinfer1::DataType* output, int32_t outputs,
                               const nvinfer1::DataType* input,
                               int32_t inputs) const noexcept override {
        if (!input || !output || inputs != 1 || outputs != 1 ||
            input[0] != nvinfer1::DataType::kBF16)
            return 1;
        output[0] = input[0];
        return 0;
    }
    int32_t getOutputShapes(const nvinfer1::DimsExprs* input, int32_t inputs,
                            const nvinfer1::DimsExprs*, int32_t shape_inputs,
                            nvinfer1::DimsExprs* output, int32_t outputs,
                            nvinfer1::IExprBuilder&) noexcept override {
        if (!input || !output || inputs != 1 || outputs != 1 || shape_inputs != 0 ||
            input[0].nbDims != 2)
            return 1;
        output[0] = input[0];
        return 0;
    }
    bool supportsFormatCombination(int32_t position,
                                   const nvinfer1::DynamicPluginTensorDesc* values, int32_t inputs,
                                   int32_t outputs) noexcept override {
        return values && inputs == 1 && outputs == 1 && position >= 0 && position < 2 &&
               descriptor(values[position].desc, true);
    }
    int32_t configurePlugin(const nvinfer1::DynamicPluginTensorDesc* input, int32_t inputs,
                            const nvinfer1::DynamicPluginTensorDesc* output,
                            int32_t outputs) noexcept override {
        ready_ = false;
        rows_ = 0;
        return input && output && inputs == 1 && outputs == 1 && descriptor(input[0].desc, true) &&
                       descriptor(output[0].desc, true) &&
                       input[0].desc.dims.d[0] == output[0].desc.dims.d[0]
                   ? 0
                   : 1;
    }
    std::size_t getWorkspaceSize(const nvinfer1::DynamicPluginTensorDesc*, int32_t,
                                 const nvinfer1::DynamicPluginTensorDesc*,
                                 int32_t) const noexcept override {
        return 0;
    }
    const char* getTimingCacheID() noexcept override { return "identity_bf16_t1024_alias_v1"; }
    const char* getMetadataString() noexcept override {
        return "BF16[T,1024] identity; strict same pointer; no GPU work; requires aliased-plugin "
               "preview";
    }
    int32_t onShapeChange(const nvinfer1::PluginTensorDesc* input, int32_t inputs,
                          const nvinfer1::PluginTensorDesc* output,
                          int32_t outputs) noexcept override {
        ready_ = false;
        rows_ = 0;
        if (!input || !output || inputs != 1 || outputs != 1 || !descriptor(input[0], false) ||
            !descriptor(output[0], false) || input[0].dims.d[0] != output[0].dims.d[0])
            return 1;
        rows_ = input[0].dims.d[0];
        ready_ = true;
        return 0;
    }
    int32_t enqueue(const nvinfer1::PluginTensorDesc* input,
                    const nvinfer1::PluginTensorDesc* output, const void* const* input_data,
                    void* const* output_data, void*, cudaStream_t) noexcept override {
        if (!valid_call(input, output, input_data, output_data)) {
            ready_ = false;
            return 1;
        }
        return 0;
    }
    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext*) noexcept override {
        return clone();
    }
    const nvinfer1::PluginFieldCollection* getFieldsToSerialize() noexcept override {
        return &fields_;
    }

  private:
    bool valid_call(const nvinfer1::PluginTensorDesc* input,
                    const nvinfer1::PluginTensorDesc* output, const void* const* input_data,
                    void* const* output_data) const noexcept {
        if (!ready_ || !input || !output)
            return false;
        return same_storage(input_data, output_data) && descriptor(input[0], false) &&
               descriptor(output[0], false) && input[0].dims.d[0] == rows_ &&
               output[0].dims.d[0] == rows_;
    }
    bool ready_{false};
    std::int64_t rows_{0};
    nvinfer1::PluginFieldCollection fields_{0, nullptr};
};

class Creator final : public nvinfer1::IPluginCreatorV3One {
  public:
    const char* getPluginName() const noexcept override { return kName; }
    const char* getPluginVersion() const noexcept override { return kVersion; }
    const char* getPluginNamespace() const noexcept override { return kNamespace; }
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override { return &fields_; }
    nvinfer1::IPluginV3* createPlugin(const char*, const nvinfer1::PluginFieldCollection* fields,
                                      nvinfer1::TensorRTPhase) noexcept override {
        if (fields && fields->nbFields != 0)
            return nullptr;
        return new (std::nothrow) ProjectionBarrier;
    }

  private:
    const nvinfer1::PluginFieldCollection fields_{0, nullptr};
};
Creator creator;
} // namespace

nvinfer1::IPluginCreatorInterface* hstu_projection_barrier_creator() noexcept {
    return &creator;
}
