/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvInfer.h>
#include <cstddef>
#include <cuda_runtime_api.h>
#include <string>

namespace trtmc::sam2_hoi {

class SoftmaxPlugin final : public nvinfer1::IPluginV2DynamicExt {
  public:
    SoftmaxPlugin() = default;
    SoftmaxPlugin(const void* data, std::size_t length);
    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    void destroy() noexcept override;
    std::size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void setPluginNamespace(char const* plugin_namespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;
    nvinfer1::DataType getOutputDataType(int32_t index, nvinfer1::DataType const* input_types,
                                         int32_t num_inputs) const noexcept override;
    SoftmaxPlugin* clone() const noexcept override;
    nvinfer1::DimsExprs
    getOutputDimensions(int32_t output_index, nvinfer1::DimsExprs const* inputs, int32_t num_inputs,
                        nvinfer1::IExprBuilder& expression_builder) noexcept override;
    bool supportsFormatCombination(int32_t position,
                                   nvinfer1::PluginTensorDesc const* inputs_outputs,
                                   int32_t num_inputs, int32_t num_outputs) noexcept override;
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t num_inputs,
                         nvinfer1::DynamicPluginTensorDesc const* outputs,
                         int32_t num_outputs) noexcept override;
    std::size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t num_inputs,
                                 nvinfer1::PluginTensorDesc const* outputs,
                                 int32_t num_outputs) const noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* input_descriptors,
                    nvinfer1::PluginTensorDesc const* output_descriptors, void const* const* inputs,
                    void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    static constexpr char const* kPLUGIN_NAME = "Sam2HoiSoftmax";
    static constexpr char const* kPLUGIN_VERSION = "1";

  private:
    std::string namespace_;
};

} // namespace trtmc::sam2_hoi
