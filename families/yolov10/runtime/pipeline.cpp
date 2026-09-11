/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/yolov10/runtime/pipeline.h"

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

Yolov10ObjectDetectionPipeline::Yolov10ObjectDetectionPipeline(
    std::unique_ptr<ITrtModule> model, Yolov10PreprocessConfig preprocess_config,
    float score_threshold)
    : model_(std::move(model)), preprocess_config_(std::move(preprocess_config)),
      score_threshold_(score_threshold) {
    if (!model_ || !model_->ok())
        throw std::runtime_error("Yolov10ObjectDetectionPipeline: invalid model");
}

ObjectDetectionResult Yolov10ObjectDetectionPipeline::detect(const float* pixels, int32_t height,
                                                             int32_t width) {
    Yolov10Letterbox letterbox;
    auto values = preprocess_yolov10_image(pixels, height, width, preprocess_config_, letterbox);
    Tensor input;
    input.data = values.data();
    input.shape = {1, 3, preprocess_config_.input_image_h, preprocess_config_.input_image_w};
    input.dtype = DType::kFloat32;
    const auto outputs = model_->forward({{"pixel_values", input}});

    const Tensor* boxes = find(outputs, "boxes");
    const Tensor* scores = find(outputs, "scores");
    const Tensor* classes = find(outputs, "classes");
    if (boxes == nullptr || scores == nullptr || classes == nullptr)
        throw std::runtime_error("YOLOv10 engine did not return boxes, scores and classes");
    if (boxes->dtype != DType::kFloat32 || scores->dtype != DType::kFloat32)
        throw std::runtime_error("YOLOv10 boxes and scores must be float32");
    if (classes->dtype != DType::kInt32)
        throw std::runtime_error("YOLOv10 classes must be int32");

    const auto count = static_cast<std::size_t>(scores->numel());
    if (static_cast<std::size_t>(boxes->numel()) != count * 4U ||
        static_cast<std::size_t>(classes->numel()) != count)
        throw std::runtime_error("YOLOv10 detection outputs disagree on their length");

    const auto* box_values = static_cast<const float*>(boxes->data);
    const auto* score_values = static_cast<const float*>(scores->data);
    const auto* class_values = static_cast<const int32_t*>(classes->data);

    ObjectDetectionResult result;
    result.image_height = height;
    result.image_width = width;
    result.boxes.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        const float score = score_values[index];
        // The head emits a fixed number of slots ordered by score, so the first
        // slot below the threshold ends the useful part of the output.
        if (!(score >= score_threshold_))
            break;
        DetectionBox box;
        // Undo the letterbox: remove the padding, then the scale. The engine
        // works in network-input pixels and the caller expects its own.
        const float left = (box_values[index * 4U + 0U] - letterbox.pad_x) / letterbox.scale;
        const float top = (box_values[index * 4U + 1U] - letterbox.pad_y) / letterbox.scale;
        const float right = (box_values[index * 4U + 2U] - letterbox.pad_x) / letterbox.scale;
        const float bottom = (box_values[index * 4U + 3U] - letterbox.pad_y) / letterbox.scale;
        box.x_min = std::clamp(left, 0.0F, static_cast<float>(width));
        box.y_min = std::clamp(top, 0.0F, static_cast<float>(height));
        box.x_max = std::clamp(right, 0.0F, static_cast<float>(width));
        box.y_max = std::clamp(bottom, 0.0F, static_cast<float>(height));
        box.score = score;
        box.class_id = class_values[index];
        result.boxes.push_back(box);
    }
    return result;
}

} // namespace trtmc
