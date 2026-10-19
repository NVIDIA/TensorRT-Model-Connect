/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/grounding_dino/runtime/pipeline.h"

#include "families/grounding_dino/runtime/text_inputs.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace trtmc {

namespace {

const internal::ConfigField kFields[] = {
    {"box_threshold", internal::ConfigKind::F64, std::nullopt,
     "Minimum per-query score; the bundle's value is used when unset"},
    {"text_threshold", internal::ConfigKind::F64, std::nullopt,
     "Minimum per-token score for a token to join the predicted phrase"},
};

const Tensor& require_output(const TensorMap& outputs, const char* name) {
    const auto found = outputs.find(name);
    if (found == outputs.end())
        throw std::runtime_error(std::string("grounding_dino engine did not return ") + name);
    if (found->second.data == nullptr || found->second.dtype != DType::kFloat32 ||
        found->second.numel() <= 0) {
        throw std::runtime_error(std::string("grounding_dino engine returned an unusable ") + name);
    }
    return found->second;
}

float sigmoid(float value) {
    return 1.0F / (1.0F + std::exp(-value));
}

float threshold_from(internal::ConfigView supplied, std::string_view name, float fallback) {
    const auto value = internal::config_get<double>(supplied, kFields, name);
    if (!value.has_value())
        return fallback;
    if (!std::isfinite(*value) || *value < 0.0 || *value > 1.0)
        throw internal::ConfigError("grounding_dino thresholds must lie in [0, 1]");
    return static_cast<float>(*value);
}

} // namespace

GroundingDinoPipeline::GroundingDinoPipeline(std::unique_ptr<ITrtModule> model,
                                             std::shared_ptr<ITokenizer> tokenizer,
                                             GroundingDinoRuntimeConfig config)
    : model_(std::move(model)), tokenizer_(std::move(tokenizer)), config_(std::move(config)) {
    if (!model_ || !model_->ok())
        throw std::runtime_error("GroundingDinoPipeline: invalid model");
    if (!tokenizer_)
        throw std::runtime_error("GroundingDinoPipeline: missing tokenizer");
    if (config_.max_text_len <= 0 || config_.num_queries <= 0 || config_.d_model <= 0)
        throw std::runtime_error("grounding_dino runtime config has invalid dimensions");
}

std::vector<internal::TaskInstance> GroundingDinoPipeline::task_bindings() {
    return {internal::bind<internal::IImageTextToBoxes>(*this, kFields)};
}

internal::GroundedBoxesResult
GroundingDinoPipeline::run(const internal::ImageTextToBoxesRequest& request,
                           internal::ConfigView config) {
    const float box_threshold = threshold_from(config, "box_threshold", config_.box_threshold);
    const float text_threshold = threshold_from(config, "text_threshold", config_.text_threshold);

    const auto& image = request.image;
    if (image.format != internal::ImageFormat::Float32 || image.channels != 3 ||
        image.data == nullptr || image.height == 0 || image.width == 0 ||
        image.height > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        image.width > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        static_cast<std::uint64_t>(image.height) >
            std::numeric_limits<std::size_t>::max() / image.width / 3 / sizeof(float) ||
        image.byte_size !=
            static_cast<std::size_t>(image.height) * image.width * 3 * sizeof(float)) {
        throw std::invalid_argument("grounding_dino requires contiguous float32 RGB input");
    }
    if (request.text.empty())
        throw std::invalid_argument("grounding_dino requires a text prompt");

    // The reference lower-cases and requires a trailing separator; without one
    // the last phrase has no delimiter and is dropped by the block mask.
    std::string prompt(request.text);
    std::transform(prompt.begin(), prompt.end(), prompt.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    while (!prompt.empty() && std::isspace(static_cast<unsigned char>(prompt.back())))
        prompt.pop_back();
    if (prompt.empty())
        throw std::invalid_argument("grounding_dino prompt is blank");
    if (prompt.back() != '.' && prompt.back() != '?')
        prompt.push_back('.');

    const auto tokens = tokenizer_->encode(prompt);
    auto text = grounding_dino_text_inputs(tokens, config_.max_text_len, config_.d_model);

    auto pixel_values = preprocess_grounding_dino_image(
        static_cast<const float*>(image.data), static_cast<std::int32_t>(image.height),
        static_cast<std::int32_t>(image.width), config_.preprocess);

    const std::int32_t length = config_.max_text_len;
    Tensor pixels_t{pixel_values.data(),
                    {1, 3, config_.preprocess.input_image_h, config_.preprocess.input_image_w},
                    DType::kFloat32};
    Tensor ids_t{text.input_ids.data(), {length}, DType::kInt32};
    Tensor types_t{text.token_type_ids.data(), {length}, DType::kInt32};
    Tensor positions_t{text.position_ids.data(), {length}, DType::kInt32};
    Tensor self_t{text.self_attention_mask.data(), {1, length, length}, DType::kFloat32};
    Tensor key_t{text.key_mask.data(), {1, 1, length}, DType::kFloat32};
    Tensor text_pos_t{
        text.position_embedding.data(), {1, length, config_.d_model}, DType::kFloat32};

    auto outputs = model_->forward({{"pixel_values", pixels_t},
                                    {"input_ids", ids_t},
                                    {"token_type_ids", types_t},
                                    {"position_ids", positions_t},
                                    {"text_self_attention_mask", self_t},
                                    {"text_key_mask", key_t},
                                    {"text_position_embedding", text_pos_t}});

    const Tensor& logits_tensor = require_output(outputs, "logits");
    const Tensor& boxes_tensor = require_output(outputs, "pred_boxes");
    if (logits_tensor.shape.size() != 3 || boxes_tensor.shape.size() != 3)
        throw std::runtime_error("grounding_dino engine outputs have unexpected rank");
    const auto queries = static_cast<std::int32_t>(logits_tensor.shape[1]);
    const auto columns = static_cast<std::int32_t>(logits_tensor.shape[2]);
    if (queries != config_.num_queries || columns != length ||
        static_cast<std::int32_t>(boxes_tensor.shape[1]) != queries ||
        static_cast<std::int32_t>(boxes_tensor.shape[2]) != 4) {
        throw std::runtime_error("grounding_dino engine outputs have unexpected shapes");
    }

    const float* logits = static_cast<const float*>(logits_tensor.data);
    const float* boxes = static_cast<const float*>(boxes_tensor.data);
    const float scale_x = static_cast<float>(image.width);
    const float scale_y = static_cast<float>(image.height);

    internal::GroundedBoxesResult result;
    for (std::int32_t q = 0; q < queries; ++q) {
        const float* row = logits + static_cast<std::size_t>(q) * columns;
        // Only the prompt's own tokens count; the padded tail is masked in the
        // graph and carries no label.
        float best = 0.0F;
        for (std::int32_t t = 0; t < text.real_length; ++t)
            best = std::max(best, sigmoid(row[t]));
        if (best <= box_threshold)
            continue;

        // The label is the phrase the query actually matched: the tokens whose
        // own score clears text_threshold, decoded back to text.
        std::vector<std::int32_t> phrase;
        for (std::int32_t t = 0; t < text.real_length; ++t) {
            const std::int32_t token = text.input_ids[static_cast<std::size_t>(t)];
            if (grounding_dino_is_special_token(token))
                continue;
            if (sigmoid(row[t]) > text_threshold)
                phrase.push_back(token);
        }

        const float* box = boxes + static_cast<std::size_t>(q) * 4;
        const float cx = box[0];
        const float cy = box[1];
        const float bw = box[2];
        const float bh = box[3];
        internal::GroundedBox detection;
        detection.box = {(cx - bw * 0.5F) * scale_x, (cy - bh * 0.5F) * scale_y,
                         (cx + bw * 0.5F) * scale_x, (cy + bh * 0.5F) * scale_y};
        detection.label = phrase.empty() ? std::string() : tokenizer_->decode(phrase);
        detection.confidence = best;
        result.boxes.push_back(std::move(detection));
    }
    std::sort(result.boxes.begin(), result.boxes.end(),
              [](const internal::GroundedBox& left, const internal::GroundedBox& right) {
                  return left.confidence.value_or(0.0F) > right.confidence.value_or(0.0F);
              });
    // Nothing is generated, so there is no raw response to retain and nothing
    // that could have been parsed incompletely.
    result.parse_complete = true;
    return result;
}

} // namespace trtmc
