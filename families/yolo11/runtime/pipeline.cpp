/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/yolo11/runtime/pipeline.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace trtmc {
namespace {

const Tensor* find(const TensorMap& outputs, const std::string& name) {
    const auto entry = outputs.find(name);
    return entry == outputs.end() ? nullptr : &entry->second;
}

// Intersection over union of two corner-form boxes.
float overlap(const DetectionBox& left, const DetectionBox& right) {
    const float x0 = std::max(left.x_min, right.x_min);
    const float y0 = std::max(left.y_min, right.y_min);
    const float x1 = std::min(left.x_max, right.x_max);
    const float y1 = std::min(left.y_max, right.y_max);
    const float shared = std::max(0.0F, x1 - x0) * std::max(0.0F, y1 - y0);
    if (shared <= 0.0F)
        return 0.0F;
    const auto area = [](const DetectionBox& box) {
        return std::max(0.0F, box.x_max - box.x_min) * std::max(0.0F, box.y_max - box.y_min);
    };
    const float total = area(left) + area(right) - shared;
    return total > 0.0F ? shared / total : 0.0F;
}

// Greedy non-maximum suppression, per class. YOLO11's head reports one
// prediction per anchor and leaves the overlaps in, so the runtime removes
// them. Boxes of different classes never suppress each other, which is what
// the reference does unless it is asked for the class-agnostic variant.
} // namespace

std::vector<DetectionBox> suppress_yolo11_boxes(std::vector<DetectionBox> boxes,
                                                float iou_threshold, std::size_t max_detections) {
    std::stable_sort(boxes.begin(), boxes.end(), [](const DetectionBox& a, const DetectionBox& b) {
        return a.score > b.score;
    });
    std::vector<DetectionBox> kept;
    std::vector<bool> dropped(boxes.size(), false);
    for (std::size_t index = 0; index < boxes.size(); ++index) {
        if (dropped[index])
            continue;
        kept.push_back(boxes[index]);
        if (kept.size() >= max_detections)
            break;
        for (std::size_t other = index + 1; other < boxes.size(); ++other) {
            if (dropped[other] || boxes[other].class_id != boxes[index].class_id)
                continue;
            if (overlap(boxes[index], boxes[other]) > iou_threshold)
                dropped[other] = true;
        }
    }
    return kept;
}

Yolo11ObjectDetectionPipeline::Yolo11ObjectDetectionPipeline(
    std::unique_ptr<ITrtModule> model, Yolo11PreprocessConfig preprocess_config,
    float score_threshold, float iou_threshold, std::int32_t max_detections)
    : model_(std::move(model)), preprocess_config_(std::move(preprocess_config)),
      score_threshold_(score_threshold), iou_threshold_(iou_threshold),
      max_detections_(max_detections) {
    if (!model_ || !model_->ok())
        throw std::runtime_error("Yolo11ObjectDetectionPipeline: invalid model");
}

ObjectDetectionResult Yolo11ObjectDetectionPipeline::detect(const float* pixels, int32_t height,
                                                            int32_t width) {
    Yolo11Letterbox letterbox;
    auto values = preprocess_yolo11_image(pixels, height, width, preprocess_config_, letterbox);
    Tensor input;
    input.data = values.data();
    input.shape = {1, 3, preprocess_config_.input_image_h, preprocess_config_.input_image_w};
    input.dtype = DType::kFloat32;
    const auto outputs = model_->forward({{"pixel_values", input}});

    const Tensor* boxes = find(outputs, "boxes");
    const Tensor* scores = find(outputs, "scores");
    const Tensor* classes = find(outputs, "classes");
    if (boxes == nullptr || scores == nullptr || classes == nullptr)
        throw std::runtime_error("YOLO11 engine did not return boxes, scores and classes");
    if (boxes->dtype != DType::kFloat32 || scores->dtype != DType::kFloat32)
        throw std::runtime_error("YOLO11 boxes and scores must be float32");
    if (classes->dtype != DType::kInt32)
        throw std::runtime_error("YOLO11 classes must be int32");

    const auto count = static_cast<std::size_t>(scores->numel());
    if (static_cast<std::size_t>(boxes->numel()) != count * 4U ||
        static_cast<std::size_t>(classes->numel()) != count)
        throw std::runtime_error("YOLO11 detection outputs disagree on their length");

    const auto* box_values = static_cast<const float*>(boxes->data);
    const auto* score_values = static_cast<const float*>(scores->data);
    const auto* class_values = static_cast<const int32_t*>(classes->data);

    std::vector<DetectionBox> candidates;
    candidates.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        const float score = score_values[index];
        // Every anchor is reported, in no particular order, so the whole set
        // has to be walked rather than stopped at the first weak one.
        if (!(score >= score_threshold_))
            continue;
        DetectionBox box;
        // Undo the letterbox: remove the padding, then the scale.
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
        candidates.push_back(box);
    }

    ObjectDetectionResult result;
    result.image_height = height;
    result.image_width = width;
    result.boxes = suppress_yolo11_boxes(std::move(candidates), iou_threshold_,
                                         static_cast<std::size_t>(max_detections_));
    return result;
}

} // namespace trtmc
