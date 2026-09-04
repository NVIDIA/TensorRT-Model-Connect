/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/timm_repvgg/runtime/pipeline.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace trtmc {
namespace {

const Tensor* find_logits(const TensorMap& outputs) {
    for (const auto& [name, tensor] : outputs) {
        if (name.find("logits") != std::string::npos || outputs.size() == 1)
            return &tensor;
    }
    return nullptr;
}

} // namespace

TimmRepvggImageClassificationPipeline::TimmRepvggImageClassificationPipeline(
    std::unique_ptr<ITrtModule> model, TimmRepvggPreprocessConfig preprocess_config)
    : model_(std::move(model)), preprocess_config_(std::move(preprocess_config)) {
    if (!model_ || !model_->ok())
        throw std::runtime_error("TimmRepvggImageClassificationPipeline: invalid model");
}

ClassificationResult TimmRepvggImageClassificationPipeline::classify(const float* pixels,
                                                                     int32_t height,
                                                                     int32_t width) {
    auto values = preprocess_timm_repvgg_image(pixels, height, width, preprocess_config_);
    Tensor input;
    input.data = values.data();
    input.shape = {1, 3, preprocess_config_.input_image_h, preprocess_config_.input_image_w};
    input.dtype = DType::kFloat32;
    const auto outputs = model_->forward({{"pixel_values", input}});
    const Tensor* logits = find_logits(outputs);
    if (logits == nullptr || logits->numel() <= 0)
        throw std::runtime_error("timm RepVGG engine returned no logits");
    if (logits->dtype != DType::kFloat32)
        throw std::runtime_error("timm RepVGG logits must be float32");
    ClassificationResult result;
    result.logits.resize(static_cast<std::size_t>(logits->numel()));
    std::memcpy(result.logits.data(), logits->data, result.logits.size() * sizeof(float));
    const auto best = std::max_element(result.logits.begin(), result.logits.end());
    result.top_class = static_cast<int32_t>(std::distance(result.logits.begin(), best));
    result.top_score = *best;
    return result;
}

} // namespace trtmc
