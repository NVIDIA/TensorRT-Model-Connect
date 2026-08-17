/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "checkpoint_reader.h"

#include <NvInfer.h>
#include <cstdint>
#include <deque>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace trtmc::sam2::native {

constexpr uint32_t sam2NetworkCreationFlags() noexcept {
#if NV_TENSORRT_MAJOR >= 11
    // TensorRT 11 networks are unconditionally strongly typed.
    return 0U;
#else
    return 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
#endif
}

class NetworkBuildError final : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

struct WindowTensor {
    nvinfer1::ITensor* tensor{nullptr};
    int32_t input_height{0};
    int32_t input_width{0};
    int32_t padded_height{0};
    int32_t padded_width{0};
    int32_t window_size{0};
};

// Adds only native TensorRT layers. The CheckpointReader and this object must
// both outlive engine construction because TensorRT may retain host weight
// pointers until the engine is serialized.
class TrtLayers final {
  public:
    TrtLayers(nvinfer1::INetworkDefinition& network, const CheckpointReader& checkpoint);

    nvinfer1::ITensor& cast(nvinfer1::ITensor& input, nvinfer1::DataType type,
                            std::string_view name);
    nvinfer1::ITensor& shuffle(nvinfer1::ITensor& input, std::initializer_list<int64_t> reshape,
                               std::string_view name);
    nvinfer1::ITensor& shuffle(nvinfer1::ITensor& input, const std::vector<int64_t>& reshape,
                               std::string_view name);
    nvinfer1::ITensor& transpose(nvinfer1::ITensor& input, std::initializer_list<int32_t> order,
                                 std::string_view name);
    nvinfer1::ITensor& slice(nvinfer1::ITensor& input, std::initializer_list<int64_t> start,
                             std::initializer_list<int64_t> size,
                             std::initializer_list<int64_t> stride, nvinfer1::SampleMode mode,
                             std::string_view name);
    nvinfer1::ITensor& concatenate(const std::vector<nvinfer1::ITensor*>& inputs, int32_t axis,
                                   std::string_view name);
    nvinfer1::ITensor& elementWise(nvinfer1::ITensor& lhs, nvinfer1::ITensor& rhs,
                                   nvinfer1::ElementWiseOperation operation, std::string_view name);
    nvinfer1::ITensor& matrixMultiply(nvinfer1::ITensor& lhs,
                                      nvinfer1::MatrixOperation lhs_operation,
                                      nvinfer1::ITensor& rhs,
                                      nvinfer1::MatrixOperation rhs_operation,
                                      std::string_view name);
    nvinfer1::ITensor& softmax(nvinfer1::ITensor& input, uint32_t axes, std::string_view name);

    nvinfer1::ITensor& convolution(nvinfer1::ITensor& input, std::string_view weight_name,
                                   std::string_view bias_name, int32_t input_channels,
                                   int32_t output_channels, int32_t kernel, int32_t stride,
                                   int32_t padding, int32_t groups, std::string_view name);
    nvinfer1::ITensor& convolutionBatchNormSilu(nvinfer1::ITensor& input,
                                                std::string_view module_name,
                                                int32_t input_channels, int32_t output_channels,
                                                int32_t kernel, int32_t stride, int32_t padding,
                                                int32_t groups, float epsilon,
                                                std::string_view name);
    nvinfer1::ITensor& linearAutocastBf16(nvinfer1::ITensor& input, std::string_view module_name,
                                          int32_t input_features, int32_t output_features,
                                          std::string_view name);
    nvinfer1::ITensor& layerNormFp32(nvinfer1::ITensor& input, std::string_view module_name,
                                     int32_t channels, float epsilon, std::string_view name);
    nvinfer1::ITensor& gelu(nvinfer1::ITensor& input, std::string_view name);
    nvinfer1::ITensor& silu(nvinfer1::ITensor& input, std::string_view name);
    nvinfer1::ITensor& maxPoolNchw(nvinfer1::ITensor& input, int32_t kernel, int32_t stride,
                                   std::string_view name);
    nvinfer1::ITensor& maxPoolNhwc(nvinfer1::ITensor& input, int32_t kernel, int32_t stride,
                                   std::string_view name);
    nvinfer1::ITensor& resizeNchw(nvinfer1::ITensor& input, int32_t output_height,
                                  int32_t output_width, nvinfer1::InterpolationMode interpolation,
                                  nvinfer1::ResizeCoordinateTransformation coordinates,
                                  std::string_view name, float cubic_coefficient = -0.75F);

    WindowTensor windowPartition(nvinfer1::ITensor& input, int32_t height, int32_t width,
                                 int32_t channels, int32_t window_size, std::string_view name);
    nvinfer1::ITensor& windowUnpartition(const WindowTensor& windows, nvinfer1::ITensor& input,
                                         int32_t output_height, int32_t output_width,
                                         int32_t channels, int32_t output_window_size,
                                         std::string_view name);

    nvinfer1::ITensor& constant(std::string_view checkpoint_name,
                                std::initializer_list<int64_t> checkpoint_shape,
                                std::initializer_list<int64_t> tensor_shape, std::string_view name);
    nvinfer1::ITensor& scalar(float value, int32_t rank, nvinfer1::DataType type,
                              std::string_view name);

    std::size_t referencedTensorCount() const noexcept;

  private:
    struct FoldedWeights {
        nvinfer1::Weights kernel;
        nvinfer1::Weights bias;
    };

    nvinfer1::Weights requiredWeights(std::string_view name, const std::vector<int64_t>& shape);
    nvinfer1::Weights convertedWeights(nvinfer1::Weights source, nvinfer1::DataType type);
    FoldedWeights foldBatchNorm(std::string_view module_name, int32_t input_channels,
                                int32_t output_channels, int32_t kernel, int32_t groups,
                                float epsilon);
    nvinfer1::ITensor& ownedConstant(const std::vector<float>& values,
                                     std::initializer_list<int64_t> tensor_shape,
                                     nvinfer1::DataType type, std::string_view name);
    nvinfer1::ILayer& requireLayer(nvinfer1::ILayer* layer, std::string_view operation,
                                   std::string_view name);
    static nvinfer1::Dims dims(std::initializer_list<int64_t> values);
    static nvinfer1::Dims dims(const std::vector<int64_t>& values);
    static nvinfer1::Permutation permutation(std::initializer_list<int32_t> values);
    static std::string joined(std::string_view prefix, std::string_view suffix);

    nvinfer1::INetworkDefinition& network_;
    const CheckpointReader& checkpoint_;
    std::deque<std::vector<float>> owned_float_weights_;
    std::deque<std::vector<uint16_t>> owned_bf16_weights_;
    std::unordered_set<std::string> referenced_tensors_;
};

} // namespace trtmc::sam2::native
