/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvInferRuntime.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <string>

namespace trtmc::sam3 {

class TrackerStepFfiPlugin final : public nvinfer1::IPluginV2DynamicExt {
  public:
    TrackerStepFfiPlugin(std::string kernel_name, int32_t batch_size);
    TrackerStepFfiPlugin(const void* data, std::size_t length);

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

    TrackerStepFfiPlugin* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int32_t output_index, nvinfer1::DimsExprs const* inputs,
                                            int32_t input_count,
                                            nvinfer1::IExprBuilder& builder) noexcept override;
    bool supportsFormatCombination(int32_t position, nvinfer1::PluginTensorDesc const* in_out,
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

    static constexpr const char* kPluginName = "Sam3TrackerStepFfi";
    static constexpr const char* kPluginVersion = "1";
    static constexpr int32_t kInputCount = 10;
    static constexpr int32_t kOutputCount = 1;
    static constexpr int32_t kPackedWidth = 288 * 288 + 256 + 1 + 1;

  private:
    bool valid_contract() const noexcept;
    bool resolve_kernel() noexcept;
    void release_kernel() noexcept;

    std::string kernel_name_;
    std::string namespace_;
    int32_t batch_size_{0};
    void* cached_function_{nullptr};
    std::atomic_bool configuration_checked_{false};
    std::atomic_bool configuration_valid_{true};
};

} // namespace trtmc::sam3
