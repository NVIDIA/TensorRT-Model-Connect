/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/dinov3/pipeline.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {
namespace {

std::size_t validated_numel(const Tensor& tensor, const char* name) {
    if (tensor.data == nullptr)
        throw std::runtime_error(std::string("DINOv3 output '") + name + "' has no data");
    if (tensor.shape.empty())
        throw std::runtime_error(std::string("DINOv3 output '") + name + "' has no shape");
    std::size_t count = 1;
    for (int64_t dim : tensor.shape) {
        if (dim <= 0 ||
            static_cast<uint64_t>(dim) > std::numeric_limits<std::size_t>::max() / count) {
            throw std::runtime_error(std::string("DINOv3 output '") + name +
                                     "' has an invalid shape");
        }
        count *= static_cast<std::size_t>(dim);
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(float))
        throw std::runtime_error(std::string("DINOv3 output '") + name + "' is too large");
    return count;
}

std::vector<float> tensor_to_floats(const Tensor& tensor, const char* name) {
    const auto count = validated_numel(tensor, name);
    if (tensor.dtype != DType::kFloat32)
        throw std::runtime_error(std::string("DINOv3 output '") + name + "' must be float32");

    std::vector<float> result(count);
    std::copy_n(static_cast<const float*>(tensor.data), count, result.data());
    return result;
}

std::vector<float> copy_device_tensor_to_floats(const void* data, std::size_t count,
                                                cudaStream_t stream, const char* name) {
    std::vector<float> result(count);
    const auto status =
        cudaMemcpyAsync(result.data(), data, count * sizeof(float), cudaMemcpyDeviceToHost, stream);
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("DINOv3 output '") + name +
                                 "' D2H copy failed: " + cudaGetErrorString(status));
    }
    return result;
}

bool supports_direct_output(const TrtModule& model) {
    return model.has_input("pixel_values") && !model.input_is_dynamic("pixel_values") &&
           model.has_output("last_hidden_state") && model.has_output("pooler_output");
}

bool has_float32_outputs(const TrtModule& model) {
    return model.tensor_dtype("last_hidden_state") == DType::kFloat32 &&
           model.tensor_dtype("pooler_output") == DType::kFloat32;
}

bool has_supported_shapes(const std::vector<int64_t>& hidden, const std::vector<int64_t>& pooler) {
    return hidden.size() == 3 && pooler.size() == 2 && hidden[0] == 1 && pooler[0] == 1 &&
           hidden[2] == pooler[1];
}

const Tensor& require_output(const TensorMap& outputs, const char* name) {
    const auto output = outputs.find(name);
    if (output == outputs.end())
        throw std::runtime_error(std::string("DINOv3 engine did not return required output '") +
                                 name + "'");
    return output->second;
}

} // namespace

Dinov3ImageFeaturePipeline::Dinov3ImageFeaturePipeline(std::unique_ptr<TrtModule> model,
                                                       Dinov3PreprocessConfig preprocess_config,
                                                       std::string model_id)
    : model_(std::move(model)), preprocess_config_(std::move(preprocess_config)),
      model_id_(std::move(model_id)) {
    if (!model_ || !model_->ok())
        throw std::runtime_error("Dinov3ImageFeaturePipeline: invalid model");

    if (!supports_direct_output(*model_))
        return;

    auto hidden_shape = model_->tensor_shape("last_hidden_state");
    auto pooler_shape = model_->tensor_shape("pooler_output");
    const auto hidden = model_->device_ptr("last_hidden_state");
    const auto pooler = model_->device_ptr("pooler_output");
    if (hidden == nullptr || pooler == nullptr)
        return;
    if (!has_float32_outputs(*model_) || !has_supported_shapes(hidden_shape, pooler_shape))
        return;

    const auto hidden_count =
        validated_numel({hidden, hidden_shape, DType::kFloat32}, "last_hidden_state");
    const auto pooler_count =
        validated_numel({pooler, pooler_shape, DType::kFloat32}, "pooler_output");
    if (pooler_count > hidden_count)
        return;
    hidden_count_ = hidden_count;
    pooler_count_ = pooler_count;
    hidden_shape_ = std::move(hidden_shape);
    pooler_shape_ = std::move(pooler_shape);
}

ImageFeaturesResult Dinov3ImageFeaturePipeline::extract_image_features(const float* pixels,
                                                                       int32_t height,
                                                                       int32_t width) {
    auto pixel_values = preprocess_dinov3_image(pixels, height, width, preprocess_config_);
    const std::vector<int64_t> input_shape{1, 3, preprocess_config_.input_image_h,
                                           preprocess_config_.input_image_w};
    Tensor input{pixel_values.data(), input_shape, DType::kFloat32};
    if (hidden_count_ != 0) {
        model_->forward_async({{"pixel_values", input}});
        ImageFeaturesResult result;
        try {
            result.last_hidden_state =
                copy_device_tensor_to_floats(model_->device_ptr("last_hidden_state"), hidden_count_,
                                             model_->stream(), "last_hidden_state");
        } catch (...) {
            model_->sync();
            throw;
        }
        model_->sync();
        result.last_hidden_state_shape = hidden_shape_;
        result.pooler_output.assign(result.last_hidden_state.begin(),
                                    result.last_hidden_state.begin() + pooler_count_);
        result.pooler_output_shape = pooler_shape_;
        return result;
    }

    const auto outputs = model_->forward({{"pixel_values", input}});

    const auto& hidden = require_output(outputs, "last_hidden_state");
    const auto& pooled = require_output(outputs, "pooler_output");
    ImageFeaturesResult result;
    result.last_hidden_state = tensor_to_floats(hidden, "last_hidden_state");
    result.last_hidden_state_shape = hidden.shape;
    result.pooler_output = tensor_to_floats(pooled, "pooler_output");
    result.pooler_output_shape = pooled.shape;
    return result;
}

} // namespace trtmc
