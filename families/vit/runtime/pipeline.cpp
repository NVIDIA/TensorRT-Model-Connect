/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/vit/runtime/pipeline.h"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace trtmc {

namespace {

const Tensor& require_logits(const TensorMap& outputs, std::int32_t expected) {
    const auto found = outputs.find("logits");
    if (found == outputs.end())
        throw std::runtime_error("vit engine did not return logits");
    const auto& tensor = found->second;
    if (tensor.data == nullptr || tensor.dtype != DType::kFloat32 ||
        tensor.numel() != static_cast<std::size_t>(expected)) {
        throw std::runtime_error("vit engine returned unusable logits");
    }
    return tensor;
}

} // namespace

VitPipeline::VitPipeline(std::unique_ptr<ITrtModule> engine, VitRuntimeConfig config,
                         std::string primary_task)
    : engine_(std::move(engine)), config_(std::move(config)),
      primary_task_(std::move(primary_task)) {
    if (!engine_ || !engine_->ok())
        throw std::runtime_error("VitPipeline: invalid model");
    if (config_.num_classes <= 0)
        throw std::runtime_error("vit runtime config has no classes");
    if (!config_.labels.empty() &&
        config_.labels.size() != static_cast<std::size_t>(config_.num_classes)) {
        // Either every class is named or none is. A partial vocabulary would
        // line names up against the wrong scores.
        throw std::runtime_error("vit class metadata does not match its output size");
    }
}

std::vector<internal::TaskInstance> VitPipeline::task_bindings() {
    return {internal::bind<internal::IImageToClassScores>(*this)};
}

internal::LabelScoresResult VitPipeline::run(const internal::ImageToClassScoresRequest& request,
                                             internal::ConfigView config) {
    if (!config.empty())
        throw internal::ConfigError("vit has no runtime configuration");
    const auto& image = request.image;
    if (image.format != internal::ImageFormat::Float32 || image.channels != 3 ||
        image.data == nullptr || image.height == 0 || image.width == 0 ||
        image.height > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        image.width > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        static_cast<std::uint64_t>(image.height) >
            std::numeric_limits<std::size_t>::max() / image.width / 3 / sizeof(float) ||
        image.byte_size !=
            static_cast<std::size_t>(image.height) * image.width * 3 * sizeof(float)) {
        throw std::invalid_argument("vit requires contiguous float32 RGB input");
    }

    auto pixel_values = preprocess_vit_image(
        static_cast<const float*>(image.data), static_cast<std::int32_t>(image.height),
        static_cast<std::int32_t>(image.width), config_.preprocess);
    Tensor pixels;
    pixels.data = pixel_values.data();
    pixels.shape = {1, 3, config_.preprocess.input_image_h, config_.preprocess.input_image_w};
    pixels.dtype = DType::kFloat32;

    auto outputs = engine_->forward({{"pixel_values", pixels}});
    const auto& logits = require_logits(outputs, config_.num_classes);

    internal::LabelScoresResult result;
    result.scores.resize(static_cast<std::size_t>(config_.num_classes));
    std::memcpy(result.scores.data(), logits.data,
                static_cast<std::size_t>(config_.num_classes) * sizeof(float));
    // The head emits logits and nothing here softmaxes them. Saying Logit
    // keeps that visible rather than implying a probability this model's
    // output is not.
    result.kind = internal::ScoreKind::Logit;
    result.labels = config_.labels;
    result.vocabulary_id = config_.vocabulary_id;
    return result;
}

} // namespace trtmc
