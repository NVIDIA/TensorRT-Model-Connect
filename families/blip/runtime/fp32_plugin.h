/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "families/blip/runtime/fp32_kernels.h"

#include <NvInfer.h>
#include <array>
#include <cstdint>

namespace trtmc::blip {
enum class Fp32Operation : std::int32_t { Norm = 0, Linear = 1, Softmax = 2, Gelu = 3 };

class Fp32Plugin final : public nvinfer1::IPluginV3,
                         public nvinfer1::IPluginV3OneCore,
                         public nvinfer1::IPluginV3OneBuild,
                         public nvinfer1::IPluginV3OneRuntime {
  public:
    Fp32Plugin(std::int32_t operation, float epsilon) noexcept;
    ~Fp32Plugin() override = default;
    bool valid() const noexcept;
    nvinfer1::IPluginCapability*
    getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;
    Fp32Plugin* clone() noexcept override;
    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const char* getPluginNamespace() const noexcept override;
    std::int32_t getNbOutputs() const noexcept override;
    std::int32_t getOutputDataTypes(nvinfer1::DataType* outputs, std::int32_t output_count,
                                    const nvinfer1::DataType* inputs,
                                    std::int32_t input_count) const noexcept override;
    std::int32_t getOutputShapes(const nvinfer1::DimsExprs* inputs, std::int32_t input_count,
                                 const nvinfer1::DimsExprs* shape_inputs,
                                 std::int32_t shape_input_count, nvinfer1::DimsExprs* outputs,
                                 std::int32_t output_count,
                                 nvinfer1::IExprBuilder& builder) noexcept override;
    bool supportsFormatCombination(std::int32_t position,
                                   const nvinfer1::DynamicPluginTensorDesc* descriptors,
                                   std::int32_t input_count,
                                   std::int32_t output_count) noexcept override;
    std::int32_t configurePlugin(const nvinfer1::DynamicPluginTensorDesc* inputs,
                                 std::int32_t input_count,
                                 const nvinfer1::DynamicPluginTensorDesc* outputs,
                                 std::int32_t output_count) noexcept override;
    std::size_t getWorkspaceSize(const nvinfer1::DynamicPluginTensorDesc* inputs,
                                 std::int32_t input_count,
                                 const nvinfer1::DynamicPluginTensorDesc* outputs,
                                 std::int32_t output_count) const noexcept override;
    const char* getTimingCacheID() noexcept override;
    const char* getMetadataString() noexcept override;
    std::int32_t onShapeChange(const nvinfer1::PluginTensorDesc* inputs, std::int32_t input_count,
                               const nvinfer1::PluginTensorDesc* outputs,
                               std::int32_t output_count) noexcept override;
    std::int32_t enqueue(const nvinfer1::PluginTensorDesc* input_descriptors,
                         const nvinfer1::PluginTensorDesc* output_descriptors,
                         const void* const* inputs, void* const* outputs, void* workspace,
                         cudaStream_t stream) noexcept override;
    nvinfer1::IPluginV3*
    attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldsToSerialize() noexcept override;

  private:
    int input_count() const noexcept;
    bool descriptors_valid(const nvinfer1::PluginTensorDesc* inputs, int input_count,
                           const nvinfer1::PluginTensorDesc* outputs,
                           int output_count) const noexcept;
    std::int32_t operation_;
    float epsilon_;
    std::array<nvinfer1::PluginField, 2> fields_;
    nvinfer1::PluginFieldCollection collection_;
    std::shared_ptr<LinearContext> linear_;
};
} // namespace trtmc::blip
