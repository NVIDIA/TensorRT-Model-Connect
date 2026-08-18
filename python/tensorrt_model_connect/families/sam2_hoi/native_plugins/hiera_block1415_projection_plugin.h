/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvInfer.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cublasLt.h>
#include <string>

namespace trtmc::sam2_hoi {

class HieraBlock1415ProjectionPlugin final : public nvinfer1::IPluginV2DynamicExt {
  public:
    HieraBlock1415ProjectionPlugin() = default;
    HieraBlock1415ProjectionPlugin(const void* data, std::size_t length);
    ~HieraBlock1415ProjectionPlugin() override;

    HieraBlock1415ProjectionPlugin(const HieraBlock1415ProjectionPlugin&) = delete;
    HieraBlock1415ProjectionPlugin& operator=(const HieraBlock1415ProjectionPlugin&) = delete;

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
    HieraBlock1415ProjectionPlugin* clone() const noexcept override;
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

    static constexpr char const* kPLUGIN_NAME = "Sam2HoiHieraBlock1415Projection";
    static constexpr char const* kPLUGIN_VERSION = "1";
    static constexpr int32_t kM = 1225;
    static constexpr int32_t kN = 768;
    static constexpr int32_t kK = 768;
    static constexpr std::size_t kWORKSPACE_BYTES = std::size_t{1} << 20;
    static constexpr int32_t kALGORITHM_ID = 30;
    static constexpr uint32_t kCTA_SWIZZLING = 1;

  private:
    class LockGuard {
      public:
        explicit LockGuard(std::atomic_flag& lock) noexcept;
        ~LockGuard();
        LockGuard(const LockGuard&) = delete;
        LockGuard& operator=(const LockGuard&) = delete;

      private:
        std::atomic_flag& lock_;
    };

    bool initializeLocked() noexcept;
    bool selectExactAlgorithmLocked() noexcept;
    void releaseLocked() noexcept;

    std::string namespace_;
    std::atomic_flag lock_ = ATOMIC_FLAG_INIT;
    cublasLtHandle_t handle_{nullptr};
    cublasLtMatmulDesc_t operation_{nullptr};
    cublasLtMatrixLayout_t weight_layout_{nullptr};
    cublasLtMatrixLayout_t input_layout_{nullptr};
    cublasLtMatrixLayout_t output_layout_{nullptr};
    cublasLtMatmulPreference_t preference_{nullptr};
    cublasLtMatmulAlgo_t algorithm_{};
    bool algorithm_ready_{false};
    bool configured_{false};
    bool serialization_valid_{true};
};

} // namespace trtmc::sam2_hoi
