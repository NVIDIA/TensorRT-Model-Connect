/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/timm_densenet/runtime/pipeline.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace trtmc {

namespace {

const Tensor& require_logits(const TensorMap& outputs) {
    const auto found = outputs.find("logits");
    if (found == outputs.end())
        throw std::runtime_error("timm DenseNet engine did not return logits");
    if (found->second.numel() <= 0)
        throw std::runtime_error("timm DenseNet engine returned empty logits");
    return found->second;
}

} // namespace

TimmDensenetImageClassificationPipeline::TimmDensenetImageClassificationPipeline(
    std::unique_ptr<ITrtModule> model, TimmDensenetPreprocessConfig preprocess_config)
    : model_(std::move(model)), preprocess_config_(std::move(preprocess_config)) {
    if (!model_ || !model_->ok())
        throw std::runtime_error("TimmDensenetImageClassificationPipeline: invalid model");
}

ClassificationResult TimmDensenetImageClassificationPipeline::classify(const float* pixels,
                                                                       int32_t height,
                                                                       int32_t width) {
    auto pixel_values = preprocess_timm_densenet_image(pixels, height, width, preprocess_config_);

    Tensor img_t;
    img_t.data = pixel_values.data();
    img_t.shape = {1, 3, preprocess_config_.input_image_h, preprocess_config_.input_image_w};
    img_t.dtype = DType::kFloat32;

    auto outputs = model_->forward({{"pixel_values", img_t}});
    ClassificationResult result;

    const auto& logits_tensor = require_logits(outputs);
    const auto n = logits_tensor.numel();

    result.logits.resize(static_cast<std::size_t>(n));
    std::memcpy(result.logits.data(), logits_tensor.data,
                static_cast<std::size_t>(n) * sizeof(float));

    auto best = std::max_element(result.logits.begin(), result.logits.end());
    result.top_class = static_cast<int32_t>(std::distance(result.logits.begin(), best));
    result.top_score = (best == result.logits.end()) ? 0.0F : *best;
    return result;
}

} // namespace trtmc
