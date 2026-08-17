/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvInferRuntime.h>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <string>

namespace trtmc {

class FastFoundationStereoGeometryVolumeConvc1Plugin final : public nvinfer1::IPluginV2DynamicExt {
  public:
    FastFoundationStereoGeometryVolumeConvc1Plugin() = default;
    FastFoundationStereoGeometryVolumeConvc1Plugin(const void* data, std::size_t length);

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
                                         int32_t input_count) const noexcept override;

    FastFoundationStereoGeometryVolumeConvc1Plugin* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int32_t output_index, nvinfer1::DimsExprs const* inputs,
                                            int32_t input_count,
                                            nvinfer1::IExprBuilder& expr_builder) noexcept override;
    bool supportsFormatCombination(int32_t position, nvinfer1::PluginTensorDesc const* input_output,
                                   int32_t input_count, int32_t output_count) noexcept override;
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t input_count,
                         nvinfer1::DynamicPluginTensorDesc const* outputs,
                         int32_t output_count) noexcept override;
    std::size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t input_count,
                                 nvinfer1::PluginTensorDesc const* outputs,
                                 int32_t output_count) const noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* input_desc,
                    nvinfer1::PluginTensorDesc const* output_desc, void const* const* inputs,
                    void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    static constexpr const char* kPLUGIN_NAME = "FastFoundationStereoGeometryVolumeConvc1";
    static constexpr const char* kPLUGIN_VERSION = "1";

  private:
    std::string namespace_;
    bool valid_{true};
};

} // namespace trtmc
