/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/yolos/runtime/pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace trtmc {
namespace {

constexpr float kConfidenceThreshold = 0.5F;

const Tensor& require_output(const std::unordered_map<std::string, Tensor>& outputs,
                             const char* name) {
    const auto found = outputs.find(name);
    if (found == outputs.end())
        throw std::runtime_error("yolos engine did not produce " + std::string(name));
    return found->second;
}

// The trailing class is "no object", so it is scored but never reported.
int32_t argmax_softmax(const float* logits, int32_t class_count, int32_t label_count,
                       float* score_out) {
    float max_logit = logits[0];
    for (int32_t c = 1; c < class_count; ++c)
        max_logit = std::max(max_logit, logits[c]);

    float sum_exp = 0.0F;
    for (int32_t c = 0; c < class_count; ++c)
        sum_exp += std::exp(logits[c] - max_logit);

    int32_t best_class = 0;
    float best_logit = logits[0];
    for (int32_t c = 1; c < label_count; ++c) {
        if (logits[c] > best_logit) {
            best_logit = logits[c];
            best_class = c;
        }
    }
    *score_out = std::exp(best_logit - max_logit) / sum_exp;
    return best_class;
}

} // namespace

YolosObjectDetectionPipeline::YolosObjectDetectionPipeline(std::unique_ptr<ITrtModule> model,
                                                           YolosPreprocessConfig preprocess_config)
    : model_(std::move(model)), preprocess_config_(std::move(preprocess_config)) {
    if (!model_ || !model_->ok())
        throw std::runtime_error("YolosObjectDetectionPipeline: invalid model");
}

ObjectDetectionResult YolosObjectDetectionPipeline::detect(const float* pixels, int32_t height,
                                                           int32_t width) {
    auto pixel_values = preprocess_yolos_image(pixels, height, width, preprocess_config_);

    Tensor image;
    image.data = pixel_values.data();
    image.shape = {1, 3, preprocess_config_.input_image_h, preprocess_config_.input_image_w};
    image.dtype = DType::kFloat32;

    auto outputs = model_->forward({{"pixel_values", image}});

    ObjectDetectionResult result;
    result.image_height = height;
    result.image_width = width;

    const Tensor& logits_tensor = require_output(outputs, "logits");
    const Tensor& boxes_tensor = require_output(outputs, "pred_boxes");
    if (logits_tensor.shape.size() < 3 || boxes_tensor.shape.size() < 3)
        throw std::runtime_error("yolos engine outputs have unexpected rank");

    const auto detections = static_cast<int32_t>(logits_tensor.shape[1]);
    const auto class_count = static_cast<int32_t>(logits_tensor.shape[2]);
    const int32_t label_count = class_count - 1;
    if (detections <= 0 || label_count <= 0 ||
        boxes_tensor.numel() < static_cast<std::size_t>(detections) * 4U)
        throw std::runtime_error("yolos engine outputs have unexpected shapes");

    const auto* logits = static_cast<const float*>(logits_tensor.data);
    const auto* boxes = static_cast<const float*>(boxes_tensor.data);
    const auto scale_x = static_cast<float>(width);
    const auto scale_y = static_cast<float>(height);

    for (int32_t index = 0; index < detections; ++index) {
        float score = 0.0F;
        const int32_t best_class =
            argmax_softmax(logits + static_cast<std::size_t>(index) * class_count, class_count,
                           label_count, &score);
        if (score <= kConfidenceThreshold)
            continue;
        // The head emits normalized cxcywh; the task contract wants absolute xyxy.
        const float* box = boxes + static_cast<std::size_t>(index) * 4;
        const float cx = box[0];
        const float cy = box[1];
        const float box_w = box[2];
        const float box_h = box[3];
        result.boxes.push_back(DetectionBox{
            (cx - box_w * 0.5F) * scale_x, (cy - box_h * 0.5F) * scale_y,
            (cx + box_w * 0.5F) * scale_x, (cy + box_h * 0.5F) * scale_y, score, best_class});
    }
    std::sort(result.boxes.begin(), result.boxes.end(),
              [](const DetectionBox& left, const DetectionBox& right) {
                  return left.score > right.score;
              });
    return result;
}

} // namespace trtmc
