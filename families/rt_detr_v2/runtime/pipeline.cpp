/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/rt_detr_v2/runtime/pipeline.h"

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
        throw std::runtime_error("rt_detr_v2 engine did not produce " + std::string(name));
    return found->second;
}

} // namespace

RtDetrV2ObjectDetectionPipeline::RtDetrV2ObjectDetectionPipeline(std::unique_ptr<ITrtModule> model,
                                                                 RtDetrRuntimeConfig config)
    : model_(std::move(model)), config_(std::move(config)) {
    if (!model_ || !model_->ok())
        throw std::runtime_error("RtDetrV2ObjectDetectionPipeline: engine failed to load");
}

ObjectDetectionResult RtDetrV2ObjectDetectionPipeline::detect(const float* pixels,
                                                              std::int32_t height,
                                                              std::int32_t width) {
    auto prepared = preprocess_rt_detr_image(pixels, height, width, config_.preprocess);

    Tensor input;
    input.data = prepared.data();
    input.shape = {1, 3, config_.preprocess.input_image_h, config_.preprocess.input_image_w};
    input.dtype = DType::kFloat32;
    auto outputs = model_->forward({{"pixel_values", input}});

    const Tensor& logits = require_output(outputs, "logits");
    const Tensor& boxes = require_output(outputs, "boxes");

    // Boxes come back normalised to the original image, not the square the
    // network ran on, so the source size is what scales them.
    const auto detections = decode_rt_detr_boxes(
        static_cast<const float*>(logits.data), static_cast<const float*>(boxes.data),
        config_.num_queries, config_.num_labels, height, width, config_.score_threshold);

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
