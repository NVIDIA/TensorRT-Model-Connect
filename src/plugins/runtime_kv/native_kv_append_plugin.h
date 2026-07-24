/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvInfer.h>
#include <array>
#include <cstdint>

namespace trtmc::runtime_kv {

inline constexpr char kNativeKvAppendPluginName[] = "NativeKvAppend";
inline constexpr char kNativeKvAppendPluginVersion[] = "1";
inline constexpr int32_t kNativeKvAppendPluginAbi = 1;
inline constexpr int32_t kNativeKvAppendInputCount = 6;
inline constexpr int32_t kNativeKvAppendOutputCount = 2;

// Input order is part of ABI v1.
enum class NativeKvAppendInput : int32_t {
    kCacheK = 0,
    kCacheV = 1,
    kNewK = 2,
    kNewV = 3,
    kWriteIndex = 4,
    kActiveLength = 5,
};

class NativeKvAppendPlugin final : public nvinfer1::IPluginV3,
                                   public nvinfer1::IPluginV3OneCore,
                                   public nvinfer1::IPluginV3OneBuildV2,
                                   public nvinfer1::IPluginV3OneRuntime {
  public:
    explicit NativeKvAppendPlugin(int32_t abi_version = kNativeKvAppendPluginAbi) noexcept;

    // IPluginV3
    nvinfer1::IPluginCapability*
    getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;
    nvinfer1::IPluginV3* clone() noexcept override;

    // IPluginV3OneCore
    nvinfer1::AsciiChar const* getPluginName() const noexcept override;
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override;
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override;

    // IPluginV3OneBuildV2
    int32_t getNbOutputs() const noexcept override;
    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nb_inputs,
                            nvinfer1::DynamicPluginTensorDesc const* outputs,
                            int32_t nb_outputs) noexcept override;
    int32_t getOutputDataTypes(nvinfer1::DataType* output_types, int32_t nb_outputs,
                               nvinfer1::DataType const* input_types,
                               int32_t nb_inputs) const noexcept override;
    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nb_inputs,
                            nvinfer1::DimsExprs const* shape_inputs, int32_t nb_shape_inputs,
                            nvinfer1::DimsExprs* outputs, int32_t nb_outputs,
                            nvinfer1::IExprBuilder& expr_builder) noexcept override;
    bool supportsFormatCombination(int32_t pos, nvinfer1::DynamicPluginTensorDesc const* in_out,
                                   int32_t nb_inputs, int32_t nb_outputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nb_inputs,
                            nvinfer1::DynamicPluginTensorDesc const* outputs,
                            int32_t nb_outputs) const noexcept override;
    int32_t getAliasedInput(int32_t output_index) noexcept override;
    nvinfer1::AsciiChar const* getMetadataString() noexcept override;

    // IPluginV3OneRuntime
    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* inputs, int32_t nb_inputs,
                          nvinfer1::PluginTensorDesc const* outputs,
                          int32_t nb_outputs) noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* input_desc,
                    nvinfer1::PluginTensorDesc const* output_desc, void const* const* inputs,
                    void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    nvinfer1::IPluginV3*
    attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

    int32_t abi_version() const noexcept { return abi_version_; }

  private:
    int32_t abi_version_{kNativeKvAppendPluginAbi};
    std::array<nvinfer1::PluginField, 1> serialized_fields_{};
    nvinfer1::PluginFieldCollection serialized_fields_collection_{};
};

} // namespace trtmc::runtime_kv

extern "C" void trtmc_native_kv_append_fixture_force_link() noexcept;
