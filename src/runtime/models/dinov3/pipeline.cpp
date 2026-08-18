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
}

ImageFeaturesResult Dinov3ImageFeaturePipeline::extract_image_features(const float* pixels,
                                                                       int32_t height,
                                                                       int32_t width) {
    auto pixel_values = preprocess_dinov3_image(pixels, height, width, preprocess_config_);
    const std::vector<int64_t> input_shape{1, 3, preprocess_config_.input_image_h,
                                           preprocess_config_.input_image_w};
    Tensor input{pixel_values.data(), input_shape, DType::kFloat32};
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
