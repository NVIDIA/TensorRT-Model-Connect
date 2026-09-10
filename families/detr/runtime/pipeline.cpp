/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/detr/runtime/pipeline.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace trtmc {
namespace {

const Tensor* find(const TensorMap& outputs, const std::string& name) {
    const auto entry = outputs.find(name);
    return entry == outputs.end() ? nullptr : &entry->second;
}

} // namespace

DetrObjectDetectionPipeline::DetrObjectDetectionPipeline(std::unique_ptr<ITrtModule> model,
                                                         DetrPreprocessConfig preprocess_config,
                                                         float score_threshold)
    : model_(std::move(model)), preprocess_config_(std::move(preprocess_config)),
      score_threshold_(score_threshold) {
    if (!model_ || !model_->ok())
        throw std::runtime_error("DetrObjectDetectionPipeline: invalid model");
}

ObjectDetectionResult DetrObjectDetectionPipeline::detect(const float* pixels, int32_t height,
                                                          int32_t width) {
    auto values = preprocess_detr_image(pixels, height, width, preprocess_config_);
    Tensor input;
    input.data = values.data();
    input.shape = {1, 3, preprocess_config_.input_image_h, preprocess_config_.input_image_w};
    input.dtype = DType::kFloat32;
    const auto outputs = model_->forward({{"pixel_values", input}});

    const Tensor* boxes = find(outputs, "boxes");
    const Tensor* scores = find(outputs, "scores");
    const Tensor* classes = find(outputs, "classes");
    if (boxes == nullptr || scores == nullptr || classes == nullptr)
        throw std::runtime_error("DETR engine did not return boxes, scores and classes");
    if (boxes->dtype != DType::kFloat32 || scores->dtype != DType::kFloat32)
        throw std::runtime_error("DETR boxes and scores must be float32");
    if (classes->dtype != DType::kInt32)
        throw std::runtime_error("DETR classes must be int32");

    const auto count = static_cast<std::size_t>(scores->numel());
    if (static_cast<std::size_t>(boxes->numel()) != count * 4U ||
        static_cast<std::size_t>(classes->numel()) != count)
        throw std::runtime_error("DETR detection outputs disagree on their length");

    const auto* box_values = static_cast<const float*>(boxes->data);
    const auto* score_values = static_cast<const float*>(scores->data);
    const auto* class_values = static_cast<const int32_t*>(classes->data);
    const auto image_w = static_cast<float>(width);
    const auto image_h = static_cast<float>(height);

    ObjectDetectionResult result;
    result.image_height = height;
    result.image_width = width;
    result.boxes.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        const float score = score_values[index];
        // The head emits a fixed set ordered by score, so the first slot below
        // the threshold ends the useful part of the output.
        if (!(score >= score_threshold_))
            break;
        DetectionBox box;
        // The engine works in fractions of its own input. Preprocessing is a
        // plain resize, so one multiply per axis returns source pixels.
        box.x_min = std::clamp(box_values[index * 4U + 0U] * image_w, 0.0F, image_w);
        box.y_min = std::clamp(box_values[index * 4U + 1U] * image_h, 0.0F, image_h);
        box.x_max = std::clamp(box_values[index * 4U + 2U] * image_w, 0.0F, image_w);
        box.y_max = std::clamp(box_values[index * 4U + 3U] * image_h, 0.0F, image_h);
        box.score = score;
        box.class_id = class_values[index];
        result.boxes.push_back(box);
    }
    return result;
}

} // namespace trtmc
