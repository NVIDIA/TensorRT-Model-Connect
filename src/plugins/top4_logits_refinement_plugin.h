/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#if TRTMC_HAS_TRT

#include <NvInferRuntime.h>
#include <cstddef>
#include <cstdint>
#include <string>

namespace trtmc {

class Top4LogitsRefinementPlugin final : public nvinfer1::IPluginV2DynamicExt {
  public:
    Top4LogitsRefinementPlugin() = default;
    ~Top4LogitsRefinementPlugin() override = default;

    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    void destroy() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void setPluginNamespace(char const* plugin_namespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

    nvinfer1::DataType getOutputDataType(int32_t index, nvinfer1::DataType const* input_types,
                                         int32_t input_count) const noexcept override;

    Top4LogitsRefinementPlugin* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int32_t output_index, nvinfer1::DimsExprs const* inputs,
                                            int32_t input_count,
                                            nvinfer1::IExprBuilder& builder) noexcept override;
    bool supportsFormatCombination(int32_t position, nvinfer1::PluginTensorDesc const* input_output,
                                   int32_t input_count, int32_t output_count) noexcept override;
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t input_count,
                         nvinfer1::DynamicPluginTensorDesc const* outputs,
                         int32_t output_count) noexcept override;
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t input_count,
                            nvinfer1::PluginTensorDesc const* outputs,
                            int32_t output_count) const noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* input_desc,
                    nvinfer1::PluginTensorDesc const* output_desc, void const* const* inputs,
                    void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    static constexpr char const* kPLUGIN_NAME = "Top4LogitsRefinement";
    static constexpr char const* kPLUGIN_VERSION = "1";

  private:
    std::string namespace_;
};

int32_t launch_top4_logits_refinement(void const* logits, void const* hidden, void const* weights,
                                      void const* bias, void* output, void* workspace,
                                      int32_t hidden_size, int32_t vocab_size,
                                      cudaStream_t stream) noexcept;
size_t top4_logits_refinement_workspace_size() noexcept;

} // namespace trtmc

#endif // TRTMC_HAS_TRT
