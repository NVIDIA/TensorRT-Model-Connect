/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/timm_vit/pipeline.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

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

ImageClassificationPipeline::ImageClassificationPipeline(std::unique_ptr<TrtModule> model,
                                                         std::string model_id_str)
    : model_(std::move(model)), model_id_(std::move(model_id_str)) {
    if (!model_ || !model_->ok())
        throw std::runtime_error("ImageClassificationPipeline: invalid model");
}

ClassificationResult ImageClassificationPipeline::classify(const float* pixels, int32_t height,
                                                           int32_t width) {
    Tensor img_t;
    img_t.data = const_cast<float*>(pixels);
    img_t.shape = {1, 3, height, width};
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
