/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvInferRuntime.h>
#include <cstddef>
#include <cuda_runtime_api.h>
#include <memory>

namespace trtmc::minimax_h3 {

class MiniMaxH3AudioEncoderPlugin final : public nvinfer1::IPluginV3,
                                          public nvinfer1::IPluginV3OneCore,
                                          public nvinfer1::IPluginV3OneBuild,
                                          public nvinfer1::IPluginV3OneRuntime {
  public:
    explicit MiniMaxH3AudioEncoderPlugin(nvinfer1::PluginFieldCollection const& fields) noexcept;
    MiniMaxH3AudioEncoderPlugin(MiniMaxH3AudioEncoderPlugin const& other) noexcept;
    ~MiniMaxH3AudioEncoderPlugin() override;

    nvinfer1::IPluginCapability*
    getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;
    MiniMaxH3AudioEncoderPlugin* clone() noexcept override;

    nvinfer1::AsciiChar const* getPluginName() const noexcept override;
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override;
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override;

    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t input_count,
                            nvinfer1::DynamicPluginTensorDesc const* outputs,
                            int32_t output_count) noexcept override;
    int32_t getOutputDataTypes(nvinfer1::DataType* output_types, int32_t output_count,
                               nvinfer1::DataType const* input_types,
                               int32_t input_count) const noexcept override;
    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t input_count,
                            nvinfer1::DimsExprs const* shape_inputs, int32_t shape_input_count,
                            nvinfer1::DimsExprs* outputs, int32_t output_count,
                            nvinfer1::IExprBuilder& expression_builder) noexcept override;
    bool supportsFormatCombination(int32_t position,
                                   nvinfer1::DynamicPluginTensorDesc const* input_output,
                                   int32_t input_count, int32_t output_count) noexcept override;
    int32_t getNbOutputs() const noexcept override;
    std::size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs,
                                 int32_t input_count,
                                 nvinfer1::DynamicPluginTensorDesc const* outputs,
                                 int32_t output_count) const noexcept override;
    char const* getTimingCacheID() noexcept override;
    char const* getMetadataString() noexcept override;

    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* inputs, int32_t input_count,
                          nvinfer1::PluginTensorDesc const* outputs,
                          int32_t output_count) noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* input_desc,
                    nvinfer1::PluginTensorDesc const* output_desc, void const* const* inputs,
                    void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    nvinfer1::IPluginV3*
    attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

    bool isValid() const noexcept { return valid_; }

    static constexpr nvinfer1::AsciiChar const* kPLUGIN_NAME = "MiniMaxH3AudioEncoder";
    static constexpr nvinfer1::AsciiChar const* kPLUGIN_VERSION = "1";
    static constexpr int32_t kBATCH = 2;
    static constexpr int32_t kINPUT_CHANNELS = 1;
    static constexpr int32_t kOUTPUT_CHANNELS = 32;
    static constexpr int32_t kHOP_LENGTH = 800;
    static constexpr int32_t kMIN_SAMPLES = 64000;
    static constexpr int32_t kOPT_SAMPLES = 165600;
    static constexpr int32_t kMAX_SAMPLES = 480000;

  private:
    struct RuntimeState;

    std::unique_ptr<RuntimeState> runtime_;
    nvinfer1::PluginFieldCollection serialization_collection_{};
    bool valid_{false};
};

} // namespace trtmc::minimax_h3
