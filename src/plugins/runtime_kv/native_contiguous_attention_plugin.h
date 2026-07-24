/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvInfer.h>
#include <array>
#include <cstdint>
#include <memory>

namespace trtmc::runtime_kv {

class CudnnAttentionExecutor;

inline constexpr char kNativeContiguousAttentionPluginName[] = "NativeContiguousAttention";
inline constexpr char kNativeContiguousAttentionPluginVersion[] = "2";
inline constexpr int32_t kNativeContiguousAttentionPluginAbi = 2;
inline constexpr int32_t kNativeContiguousAttentionInputCount = 6;
inline constexpr int32_t kNativeContiguousAttentionOutputCount = 1;

// ABI v2 tensor order:
//   history K/V:       [T, Hkv*D] BF16 token-major, read-only
//   new Q:             [1, Hq, Sq, D] BF16 head-major
//   current K/V:       [1, Hkv, Sq, D] BF16 head-major
//   H:                 [1] INT32. H==0 requires the T==1 cold sentinel;
//                      H>0 requires 2<=T and H<=T.
//   context:           [1, Hq, Sq, D] BF16
//
// The plugin performs two normalized SDPA segments (history noncausal and
// current lower-right causal) and combines them from log-sum-exp statistics.
// It never mutates history. The engine exposes current K/V as
// exact-Sq staging outputs; the native runtime appends only those rows after
// enqueue.
enum class NativeContiguousAttentionInput : int32_t {
    kHistoryK = 0,
    kHistoryV = 1,
    kNewQ = 2,
    kCurrentK = 3,
    kCurrentV = 4,
    kHistoryLength = 5,
};

class NativeContiguousAttentionPlugin final : public nvinfer1::IPluginV3,
                                              public nvinfer1::IPluginV3OneCore,
                                              public nvinfer1::IPluginV3OneBuildV2,
                                              public nvinfer1::IPluginV3OneRuntime {
  public:
    NativeContiguousAttentionPlugin(int32_t abi_version, int32_t num_query_heads,
                                    int32_t num_kv_heads, int32_t head_dim,
                                    int32_t chunk_limit) noexcept;
    ~NativeContiguousAttentionPlugin() override;

    nvinfer1::IPluginCapability*
    getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;
    nvinfer1::IPluginV3* clone() noexcept override;

    nvinfer1::AsciiChar const* getPluginName() const noexcept override;
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override;
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override;

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

    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* inputs, int32_t nb_inputs,
                          nvinfer1::PluginTensorDesc const* outputs,
                          int32_t nb_outputs) noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* input_desc,
                    nvinfer1::PluginTensorDesc const* output_desc, void const* const* inputs,
                    void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    nvinfer1::IPluginV3*
    attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

  private:
    int32_t abi_version_{};
    int32_t num_query_heads_{};
    int32_t num_kv_heads_{};
    int32_t head_dim_{};
    int32_t chunk_limit_{};
    std::unique_ptr<CudnnAttentionExecutor> attention_;
    std::array<nvinfer1::PluginField, 5> serialized_fields_{};
    nvinfer1::PluginFieldCollection serialized_fields_collection_{};
};

} // namespace trtmc::runtime_kv
