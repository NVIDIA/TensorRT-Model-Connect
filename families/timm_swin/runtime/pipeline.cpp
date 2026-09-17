/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/timm_swin/runtime/pipeline.h"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace trtmc {

namespace {

const Tensor& require_logits(const TensorMap& outputs) {
    for (const auto& [name, tensor] : outputs) {
        if (name.find("logits") == std::string::npos && outputs.size() != 1)
            continue;
        if (tensor.data == nullptr || tensor.dtype != DType::kFloat32 || tensor.numel() <= 0)
            throw std::runtime_error("timm Swin engine must return nonempty float32 logits");
        return tensor;
    }
    throw std::runtime_error("timm Swin engine did not return logits");
}

} // namespace

TimmSwinImageClassificationPipeline::TimmSwinImageClassificationPipeline(
    std::unique_ptr<ITrtModule> model, TimmSwinPreprocessConfig preprocess_config,
    std::int32_t num_classes, std::string vocabulary_id, std::vector<std::string> labels)
    : model_(std::move(model)), preprocess_config_(std::move(preprocess_config)),
      num_classes_(num_classes), vocabulary_id_(std::move(vocabulary_id)),
      labels_(std::move(labels)) {
    if (!model_ || !model_->ok())
        throw std::runtime_error("TimmSwinImageClassificationPipeline: invalid model");
    if (num_classes_ <= 0 ||
        (!labels_.empty() && labels_.size() != static_cast<std::size_t>(num_classes_)))
        throw std::runtime_error("timm Swin class metadata does not match its output size");
}

internal::LabelScoresResult
TimmSwinImageClassificationPipeline::run(const internal::ImageToClassScoresRequest& request,
                                         internal::ConfigView config) {
    if (!config.empty())
        throw internal::ConfigError("timm Swin has no runtime configuration");
    const auto& image = request.image;
    if (image.format != internal::ImageFormat::Float32 || image.channels != 3 ||
        image.data == nullptr || image.height == 0 || image.width == 0 ||
        image.height > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        image.width > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        static_cast<std::uint64_t>(image.height) >
            std::numeric_limits<std::size_t>::max() / image.width / 3 / sizeof(float) ||
        image.byte_size != static_cast<std::size_t>(image.height) * image.width * 3 * sizeof(float))
        throw std::invalid_argument("timm Swin requires contiguous float32 RGB input");
    auto pixel_values = preprocess_timm_swin_image(
        static_cast<const float*>(image.data), static_cast<std::int32_t>(image.height),
        static_cast<std::int32_t>(image.width), preprocess_config_);

    Tensor img_t;
    img_t.data = pixel_values.data();
    img_t.shape = {1, 3, preprocess_config_.input_image_h, preprocess_config_.input_image_w};
    img_t.dtype = DType::kFloat32;

    auto outputs = model_->forward({{"pixel_values", img_t}});
    internal::LabelScoresResult result;

    const auto& logits_tensor = require_logits(outputs);
    const auto n = logits_tensor.numel();
    if (n != static_cast<std::size_t>(num_classes_))
        throw std::runtime_error("timm Swin logits do not match its configured class count");

    result.scores.resize(static_cast<std::size_t>(n));
    std::memcpy(result.scores.data(), logits_tensor.data,
                static_cast<std::size_t>(n) * sizeof(float));
    result.kind = internal::ScoreKind::Logit;
    result.vocabulary_id = vocabulary_id_;
    result.labels = labels_;
    return result;
}

} // namespace trtmc
