/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "sam2_trt_layers.h"

#include <NvInfer.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace trtmc::sam2::native {

inline constexpr std::size_t kDeliveredCheckpointTensorCount = 603;
inline constexpr std::size_t kImageNetworkReferencedTensorCount = 282;
inline constexpr int32_t kImageNetworkLayerCount = 1139;
inline constexpr int32_t kImageNetworkConvolutionLayerCount = 23;
inline constexpr int32_t kImageNetworkActivationLayerCount = 28;
inline constexpr int32_t kImageNetworkPoolingLayerCount = 6;
inline constexpr int32_t kImageNetworkElementWiseLayerCount = 130;
inline constexpr int32_t kImageNetworkShuffleLayerCount = 313;
inline constexpr int32_t kImageNetworkConstantLayerCount = 216;
inline constexpr int32_t kImageNetworkSliceLayerCount = 67;
inline constexpr int32_t kImageNetworkResizeLayerCount = 2;
inline constexpr int32_t kImageNetworkNormalizationLayerCount = 32;
inline constexpr int32_t kImageNetworkCastLayerCount = 223;
inline constexpr int32_t kImageNetworkMatrixMultiplyLayerCount = 67;
inline constexpr int32_t kImageNetworkSoftmaxLayerCount = 0;
inline constexpr int32_t kImageNetworkPluginV3LayerCount = 0;
inline constexpr int32_t kImageNetworkAttentionInputLayerCount = 16;
inline constexpr int32_t kImageNetworkAttentionOutputLayerCount = 16;
inline constexpr int32_t kSam2ImageSize = 1024;
inline constexpr int32_t kHieraSmallBlockCount = 16;

struct StaticTensorContract {
    std::string_view name;
    nvinfer1::DataType type;
    std::array<int32_t, 4> dimensions;
};

inline constexpr StaticTensorContract kImageInputContract{
    "pixel_values", nvinfer1::DataType::kFLOAT, {1, 3, 1024, 1024}};

inline constexpr std::array<StaticTensorContract, 3> kTrackerFpnContracts = {{
    {"tracker_fpn_0", nvinfer1::DataType::kBF16, {1, 256, 256, 256}},
    {"tracker_fpn_1", nvinfer1::DataType::kBF16, {1, 256, 128, 128}},
    {"tracker_fpn_2", nvinfer1::DataType::kFLOAT, {1, 256, 64, 64}},
}};

inline constexpr std::array<StaticTensorContract, 6> kBboxMapContracts = {{
    {"bbox_cls_stride_8", nvinfer1::DataType::kBF16, {1, 2, 128, 128}},
    {"bbox_cls_stride_16", nvinfer1::DataType::kBF16, {1, 2, 64, 64}},
    {"bbox_cls_stride_32", nvinfer1::DataType::kBF16, {1, 2, 32, 32}},
    {"bbox_reg_stride_8", nvinfer1::DataType::kBF16, {1, 4, 128, 128}},
    {"bbox_reg_stride_16", nvinfer1::DataType::kBF16, {1, 4, 64, 64}},
    {"bbox_reg_stride_32", nvinfer1::DataType::kBF16, {1, 4, 32, 32}},
}};

struct HieraBlockContract {
    int32_t input_channels;
    int32_t output_channels;
    int32_t heads;
    int32_t input_height;
    int32_t window_size;
    bool query_pool;
};

inline constexpr std::array<HieraBlockContract, kHieraSmallBlockCount> kHieraSmallBlocks = {{
    {96, 96, 1, 256, 8, false},
    {96, 192, 2, 256, 8, true},
    {192, 192, 2, 128, 4, false},
    {192, 384, 4, 128, 4, true},
    {384, 384, 4, 64, 14, false},
    {384, 384, 4, 64, 14, false},
    {384, 384, 4, 64, 14, false},
    {384, 384, 4, 64, 0, false},
    {384, 384, 4, 64, 14, false},
    {384, 384, 4, 64, 14, false},
    {384, 384, 4, 64, 0, false},
    {384, 384, 4, 64, 14, false},
    {384, 384, 4, 64, 14, false},
    {384, 384, 4, 64, 0, false},
    {384, 768, 8, 64, 14, true},
    {768, 768, 8, 32, 7, false},
}};

struct Sam2ImageNetworkOutputs {
    nvinfer1::ITensor* pixel_values{nullptr};
    std::array<nvinfer1::ITensor*, 3> tracker_fpn{};
    std::array<nvinfer1::ITensor*, 3> bbox_classification{};
    std::array<nvinfer1::ITensor*, 3> bbox_regression{};
    std::size_t checkpoint_tensor_count{0};
    std::size_t referenced_tensor_count{0};
    std::size_t unreferenced_tensor_count{0};
    int32_t added_layer_count{0};
    int32_t convolution_layer_count{0};
    int32_t activation_layer_count{0};
    int32_t pooling_layer_count{0};
    int32_t element_wise_layer_count{0};
    int32_t shuffle_layer_count{0};
    int32_t constant_layer_count{0};
    int32_t slice_layer_count{0};
    int32_t resize_layer_count{0};
    int32_t normalization_layer_count{0};
    int32_t cast_layer_count{0};
    int32_t matrix_multiply_layer_count{0};
    int32_t softmax_layer_count{0};
    int32_t plugin_v3_layer_count{0};
    int32_t attention_input_layer_count{0};
    int32_t attention_output_layer_count{0};
};

// Keep this object and its CheckpointReader alive through plan serialization.
// build() is intentionally single-use and rejects a nonempty network.
class Sam2ImageNetworkBuilder final {
  public:
    Sam2ImageNetworkBuilder(nvinfer1::INetworkDefinition& network,
                            const CheckpointReader& checkpoint);

    Sam2ImageNetworkOutputs build();

  private:
    std::array<nvinfer1::ITensor*, 4> buildHiera(nvinfer1::ITensor& pixel_values);
    std::array<nvinfer1::ITensor*, 4>
    buildFpn(const std::array<nvinfer1::ITensor*, 4>& trunk_features);
    nvinfer1::ITensor& buildHieraBlock(nvinfer1::ITensor& input, int32_t index,
                                       const HieraBlockContract& contract);
    nvinfer1::ITensor& buildAttention(nvinfer1::ITensor& input, int32_t index,
                                      int32_t output_channels, int32_t heads, bool query_pool);
    void buildBboxHead(const std::array<nvinfer1::ITensor*, 4>& fpn,
                       std::array<nvinfer1::ITensor*, 3>& classification,
                       std::array<nvinfer1::ITensor*, 3>& regression);
    void markCheckedOutput(nvinfer1::ITensor& tensor, const StaticTensorContract& contract);
    void validateNetworkMode() const;

    nvinfer1::INetworkDefinition& network_;
    const CheckpointReader& checkpoint_;
    TrtLayers layers_;
    bool built_{false};
};

} // namespace trtmc::sam2::native
