/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/detr/runtime/pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace trtmc {

namespace {

constexpr float kConfidenceThreshold = 0.5F;

const Tensor& require_output(const TensorMap& outputs, const char* name) {
    const auto found = outputs.find(name);
    if (found == outputs.end())
        throw std::runtime_error(std::string("DETR engine did not return ") + name);
    if (found->second.numel() <= 0)
        throw std::runtime_error(std::string("DETR engine returned empty ") + name);
    return found->second;
}

int32_t argmax_softmax(const float* logits, int32_t class_count, int32_t label_count,
                       float* score_out) {
    float max_logit = logits[0];
    for (int32_t c = 1; c < class_count; ++c) {
        if (logits[c] > max_logit)
            max_logit = logits[c];
    }
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

DetrObjectDetectionPipeline::DetrObjectDetectionPipeline(std::unique_ptr<ITrtModule> model,
                                                         DetrPreprocessConfig preprocess_config)
    : model_(std::move(model)), preprocess_config_(std::move(preprocess_config)) {
    if (!model_ || !model_->ok())
        throw std::runtime_error("DetrObjectDetectionPipeline: invalid model");
}

ObjectDetectionResult DetrObjectDetectionPipeline::detect(const float* pixels, int32_t height,
                                                          int32_t width) {
    auto pixel_values = preprocess_detr_image(pixels, height, width, preprocess_config_);

    Tensor img_t;
    img_t.data = pixel_values.data();
    img_t.shape = {1, 3, preprocess_config_.input_image_h, preprocess_config_.input_image_w};
    img_t.dtype = DType::kFloat32;

    auto outputs = model_->forward({{"pixel_values", img_t}});
    ObjectDetectionResult result;
    result.image_height = height;
    result.image_width = width;

    const Tensor& logits_tensor = require_output(outputs, "logits");
    const Tensor& boxes_tensor = require_output(outputs, "pred_boxes");
    if (logits_tensor.shape.size() < 3 || boxes_tensor.shape.size() < 3)
        throw std::runtime_error("DETR engine outputs have unexpected rank");

    const int32_t num_queries = static_cast<int32_t>(logits_tensor.shape[1]);
    const int32_t num_classes = static_cast<int32_t>(logits_tensor.shape[2]);
    const int32_t num_labels = num_classes - 1;
    if (num_queries <= 0 || num_labels <= 0 || boxes_tensor.numel() < num_queries * 4U)
        throw std::runtime_error("DETR engine outputs have unexpected shapes");

    const float* logits = static_cast<const float*>(logits_tensor.data);
    const float* boxes = static_cast<const float*>(boxes_tensor.data);
    const float scale_x = static_cast<float>(width);
    const float scale_y = static_cast<float>(height);

    for (int32_t q = 0; q < num_queries; ++q) {
        float score = 0.0F;
        const int32_t best_class = argmax_softmax(
            logits + static_cast<std::size_t>(q) * num_classes, num_classes, num_labels, &score);
        if (score <= kConfidenceThreshold)
            continue;
        const float* query_boxes = boxes + static_cast<std::size_t>(q) * 4;
        const float cx = query_boxes[0];
        const float cy = query_boxes[1];
        const float bw = query_boxes[2];
        const float bh = query_boxes[3];
        result.boxes.push_back(DetectionBox{(cx - bw * 0.5F) * scale_x, (cy - bh * 0.5F) * scale_y,
                                            (cx + bw * 0.5F) * scale_x, (cy + bh * 0.5F) * scale_y,
                                            score, best_class});
    }
    std::sort(result.boxes.begin(), result.boxes.end(),
              [](const DetectionBox& left, const DetectionBox& right) {
                  return left.score > right.score;
              });
    return result;
}

} // namespace trtmc
