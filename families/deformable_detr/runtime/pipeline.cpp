/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/deformable_detr/runtime/pipeline.h"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace trtmc {
namespace {

const Tensor& require_output(const std::unordered_map<std::string, Tensor>& outputs,
                             const char* name) {
    const auto found = outputs.find(name);
    if (found == outputs.end())
        throw std::runtime_error("deformable_detr engine did not produce " + std::string(name));
    return found->second;
}

} // namespace

DeformableDetrObjectDetectionPipeline::DeformableDetrObjectDetectionPipeline(
    std::unique_ptr<ITrtModule> model, DeformableDetrRuntimeConfig config)
    : model_(std::move(model)), config_(std::move(config)) {
    if (!model_ || !model_->ok())
        throw std::runtime_error("DeformableDetrObjectDetectionPipeline: engine failed to load");
}

ObjectDetectionResult DeformableDetrObjectDetectionPipeline::detect(const float* pixels,
                                                                    std::int32_t height,
                                                                    std::int32_t width) {
    const auto resized = deformable_detr_resize_shape(height, width, config_.resize_shortest_edge,
                                                      config_.resize_longest_edge);
    auto preprocessing = config_.preprocess;
    preprocessing.input_image_h = resized[0];
    preprocessing.input_image_w = resized[1];
    auto prepared = preprocess_deformable_detr_image(pixels, height, width, preprocessing);

    Tensor input;
    input.data = prepared.data();
    input.shape = {1, 3, resized[0], resized[1]};
    input.dtype = DType::kFloat32;
    auto outputs = model_->forward({{"pixel_values", input}});

    const Tensor& logits = require_output(outputs, "logits");
    const Tensor& boxes = require_output(outputs, "boxes");

    // Restore coordinates in the original image after the aspect-preserving resize.
    const auto detections = decode_deformable_detr_boxes(
        static_cast<const float*>(logits.data), static_cast<const float*>(boxes.data),
        config_.num_queries, config_.num_labels, height, width, config_.score_threshold,
        config_.top_k);

    ObjectDetectionResult result;
    result.image_height = height;
    result.image_width = width;
    result.boxes.reserve(detections.size());
    for (const auto& detection : detections) {
        DetectionBox box;
        box.x_min = detection.x_min;
        box.y_min = detection.y_min;
        box.x_max = detection.x_max;
        box.y_max = detection.y_max;
        box.score = detection.score;
        box.class_id = detection.label;
        result.boxes.push_back(box);
    }
    return result;
}

} // namespace trtmc
