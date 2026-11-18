/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/table_transformer/runtime/pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace trtmc {

namespace {

constexpr float kConfidenceThreshold = 0.5F;

const Tensor& require_output(const TensorMap& outputs, const char* name) {
    const auto found = outputs.find(name);
    if (found == outputs.end())
        throw std::runtime_error(std::string("Table Transformer engine did not return ") + name);
    if (found->second.data == nullptr || found->second.dtype != DType::kFloat32)
        throw std::runtime_error(
            std::string("Table Transformer engine requires FP32 storage for ") + name);
    return found->second;
}

int32_t argmax_softmax(const float* logits, int32_t class_count, int32_t label_count,
                       float* score_out) {
    float max_logit = logits[0];
    for (int32_t c = 0; c < class_count; ++c) {
        if (!std::isfinite(logits[c]))
            throw std::runtime_error("Table Transformer engine returned non-finite logits");
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
    if (!std::isfinite(*score_out))
        throw std::runtime_error("Table Transformer engine returned a non-finite score");
    return best_class;
}

} // namespace

TableTransformerObjectDetectionPipeline::TableTransformerObjectDetectionPipeline(
    std::unique_ptr<ITrtModule> model, TableTransformerPreprocessConfig preprocess_config)
    : model_(std::move(model)), preprocess_config_(std::move(preprocess_config)) {
    if (!model_ || !model_->ok())
        throw std::runtime_error("TableTransformerObjectDetectionPipeline: invalid model");
}

internal::DetectedBoxesResult
TableTransformerObjectDetectionPipeline::run(const internal::ImageToBoxesRequest& request,
                                             internal::ConfigView config) {
    if (!config.empty())
        throw internal::ConfigError("Table Transformer has no runtime configuration");
    const auto& image = request.image;
    constexpr auto kInt32Max = static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max());
    if (image.format != internal::ImageFormat::Float32 || image.channels != 3 ||
        image.data == nullptr || image.height == 0 || image.width == 0 ||
        image.height > kInt32Max || image.width > kInt32Max ||
        static_cast<std::uint64_t>(image.height) >
            std::numeric_limits<std::size_t>::max() / image.width / 3U / sizeof(float) ||
        image.byte_size !=
            static_cast<std::size_t>(image.height) * image.width * 3U * sizeof(float))
        throw std::invalid_argument("Table Transformer requires one contiguous float32 RGB image");
    return detect(static_cast<const float*>(image.data), static_cast<std::int32_t>(image.height),
                  static_cast<std::int32_t>(image.width));
}

internal::DetectedBoxesResult TableTransformerObjectDetectionPipeline::detect(const float* pixels,
                                                                              std::int32_t height,
                                                                              std::int32_t width) {
    auto pixel_values =
        preprocess_table_transformer_image(pixels, height, width, preprocess_config_);
    const auto image_shape =
        compute_table_transformer_resize_shape(height, width, preprocess_config_);

    Tensor img_t;
    img_t.data = pixel_values.data();
    img_t.shape = {1, 3, image_shape.height, image_shape.width};
    img_t.dtype = DType::kFloat32;

    auto outputs = model_->forward({{"pixel_values", img_t}});
    internal::DetectedBoxesResult result;
    result.image_height = height;
    result.image_width = width;

    const Tensor& logits_tensor = require_output(outputs, "logits");
    const Tensor& boxes_tensor = require_output(outputs, "pred_boxes");
    if (logits_tensor.shape.size() != 3 || boxes_tensor.shape.size() != 3)
        throw std::runtime_error("Table Transformer engine outputs have unexpected rank");
    const auto queries = logits_tensor.shape[1];
    const auto classes = logits_tensor.shape[2];
    if (logits_tensor.shape[0] != 1 || boxes_tensor.shape[0] != 1 || queries <= 0 ||
        queries > std::numeric_limits<int32_t>::max() || classes <= 1 ||
        classes > std::numeric_limits<int32_t>::max() || boxes_tensor.shape[1] != queries ||
        boxes_tensor.shape[2] != 4 ||
        static_cast<std::uint64_t>(queries) >
            std::numeric_limits<std::size_t>::max() / sizeof(float) / classes)
        throw std::runtime_error("Table Transformer engine outputs have unexpected shapes");
    const int32_t num_queries = static_cast<int32_t>(queries);
    const int32_t num_classes = static_cast<int32_t>(classes);
    const int32_t num_labels = num_classes - 1;

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
