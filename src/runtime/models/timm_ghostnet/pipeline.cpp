/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/timm_ghostnet/pipeline.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace trtmc {

namespace {

const Tensor* find_logits_output(const TensorMap& outputs) {
    for (const auto& [name, tensor] : outputs) {
        if (name.find("logits") != std::string::npos || outputs.size() == 1)
            return &tensor;
    }
    return nullptr;
}

} // namespace

TimmGhostnetImageClassificationPipeline::TimmGhostnetImageClassificationPipeline(
    std::unique_ptr<TrtModule> model, TimmGhostnetPreprocessConfig preprocess_config,
    std::string model_id_str)
    : model_(std::move(model)), preprocess_config_(std::move(preprocess_config)),
      model_id_(std::move(model_id_str)) {
    if (!model_ || !model_->ok())
        throw std::runtime_error("TimmGhostnetImageClassificationPipeline: invalid model");
}

ClassificationResult TimmGhostnetImageClassificationPipeline::classify(const float* pixels,
                                                                       int32_t height,
                                                                       int32_t width) {
    auto pixel_values = preprocess_timm_ghostnet_image(pixels, height, width, preprocess_config_);

    Tensor img_t;
    img_t.data = pixel_values.data();
    img_t.shape = {1, 3, preprocess_config_.input_image_h, preprocess_config_.input_image_w};
    img_t.dtype = DType::kFloat32;

    auto outputs = model_->forward({{"pixel_values", img_t}});
    ClassificationResult result;

    const Tensor* logits_tensor = find_logits_output(outputs);
    if (!logits_tensor)
        return result;

    const auto n = logits_tensor->numel();
    if (n <= 0)
        return result;

    result.logits.resize(static_cast<std::size_t>(n));
    std::memcpy(result.logits.data(), logits_tensor->data,
                static_cast<std::size_t>(n) * sizeof(float));

    auto best = std::max_element(result.logits.begin(), result.logits.end());
    result.top_class = static_cast<int32_t>(std::distance(result.logits.begin(), best));
    result.top_score = (best == result.logits.end()) ? 0.0F : *best;
    return result;
}

} // namespace trtmc
