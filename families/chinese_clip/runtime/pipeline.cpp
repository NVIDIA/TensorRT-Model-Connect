/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/chinese_clip/runtime/pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace trtmc {

namespace {

// Both towers project into one space, which is the only reason a dot product
// between them means anything. Naming it lets a caller see that two results
// are comparable; it does not claim compatibility with any other checkpoint.

const Tensor& require_output(const TensorMap& outputs, const char* name,
                             std::int32_t expected_width) {
    const auto found = outputs.find(name);
    if (found == outputs.end())
        throw std::runtime_error(std::string("chinese_clip engine did not return ") + name);
    const auto& tensor = found->second;
    if (tensor.data == nullptr || tensor.dtype != DType::kFloat32 ||
        tensor.numel() != static_cast<std::size_t>(expected_width)) {
        throw std::runtime_error(std::string("chinese_clip engine returned an unusable ") + name);
    }
    return tensor;
}

std::vector<float> l2_normalise(const Tensor& tensor) {
    std::vector<float> values(tensor.numel());
    std::memcpy(values.data(), tensor.data, values.size() * sizeof(float));
    const double total = std::inner_product(values.begin(), values.end(), values.begin(), 0.0);
    const double norm = std::sqrt(total);
    if (!std::isfinite(norm) || norm <= 0.0)
        throw std::runtime_error("chinese_clip produced an embedding with no direction");
    for (float& value : values)
        value = static_cast<float>(value / norm);
    return values;
}

float dot(const std::vector<float>& left, const std::vector<float>& right) {
    if (left.size() != right.size())
        throw std::runtime_error("chinese_clip embeddings have mismatched widths");
    return static_cast<float>(std::inner_product(left.begin(), left.end(), right.begin(), 0.0));
}

} // namespace

ChineseClipPipeline::ChineseClipPipeline(std::unique_ptr<ITrtModule> vision,
                                         std::unique_ptr<ITrtModule> text,
                                         std::shared_ptr<ITokenizer> tokenizer,
                                         ChineseClipRuntimeConfig config, std::string primary_task)
    : vision_(std::move(vision)), text_(std::move(text)), tokenizer_(std::move(tokenizer)),
      config_(std::move(config)), primary_task_(std::move(primary_task)) {
    if (!vision_ || !vision_->ok() || !text_ || !text_->ok())
        throw std::runtime_error("ChineseClipPipeline: invalid model");
    if (!tokenizer_)
        throw std::runtime_error("ChineseClipPipeline: missing tokenizer");
    if (config_.max_text_len <= 0 || config_.projection_dim <= 0 ||
        !std::isfinite(config_.logit_scale) || config_.logit_scale <= 0.0F) {
        throw std::runtime_error("chinese_clip runtime config has invalid dimensions");
    }
}

std::vector<internal::TaskInstance> ChineseClipPipeline::task_bindings() {
    return {internal::bind<internal::IImageToEmbedding>(*this),
            internal::bind<internal::ITextToEmbedding>(*this),
            internal::bind<internal::ITextImageToRelevance>(*this)};
}

std::vector<float> ChineseClipPipeline::embed_image(const internal::ImageView& image) {
    if (image.format != internal::ImageFormat::Float32 || image.channels != 3 ||
        image.data == nullptr || image.height == 0 || image.width == 0 ||
        image.height > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        image.width > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        static_cast<std::uint64_t>(image.height) >
            std::numeric_limits<std::size_t>::max() / image.width / 3 / sizeof(float) ||
        image.byte_size !=
            static_cast<std::size_t>(image.height) * image.width * 3 * sizeof(float)) {
        throw std::invalid_argument("chinese_clip requires contiguous float32 RGB input");
    }
    auto pixel_values = preprocess_chinese_clip_image(
        static_cast<const float*>(image.data), static_cast<std::int32_t>(image.height),
        static_cast<std::int32_t>(image.width), config_.preprocess);
    Tensor pixels;
    pixels.data = pixel_values.data();
    pixels.shape = {1, 3, config_.preprocess.input_image_h, config_.preprocess.input_image_w};
    pixels.dtype = DType::kFloat32;

    auto outputs = vision_->forward({{"pixel_values", pixels}});
    return l2_normalise(require_output(outputs, "image_embeds", config_.projection_dim));
}

std::vector<float> ChineseClipPipeline::embed_text(std::string_view text) {
    auto ids = tokenizer_->encode(std::string(text));
    const auto count = ids.size();
    if (count > static_cast<std::size_t>(config_.max_text_len))
        throw std::invalid_argument("Chinese-CLIP prompt exceeds the built text profile");
    ids.resize(config_.max_text_len, 0);
    std::vector<int32_t> mask(config_.max_text_len, 0);
    std::fill(mask.begin(), mask.begin() + count, 1);
    Tensor input_ids, attention_mask;
    input_ids.data = ids.data();
    input_ids.shape = {config_.max_text_len};
    input_ids.dtype = DType::kInt32;
    attention_mask.data = mask.data();
    attention_mask.shape = {config_.max_text_len};
    attention_mask.dtype = DType::kInt32;
    auto outputs = text_->forward({{"input_ids", input_ids}, {"attention_mask", attention_mask}});
    return l2_normalise(require_output(outputs, "text_embeds", config_.projection_dim));
}

internal::SemanticEmbeddingResult
ChineseClipPipeline::run(const internal::ImageToEmbeddingRequest& request,
                         internal::ConfigView config) {
    if (!config.empty())
        throw internal::ConfigError("chinese_clip has no runtime configuration");
    internal::SemanticEmbeddingResult result;
    result.values = embed_image(request.image);
    result.pooling = "class_token";
    result.normalization = "l2";
    return result;
}

internal::SemanticEmbeddingResult
ChineseClipPipeline::run(const internal::TextToEmbeddingRequest& request,
                         internal::ConfigView config) {
    if (!config.empty())
        throw internal::ConfigError("chinese_clip has no runtime configuration");
    if (request.role != internal::EmbeddingRole::Default) {
        // One text tower, trained without role prompts: a query embedding and
        // a document embedding would be the same vector, so claiming to honour
        // the distinction would be a lie rather than a limitation.
        throw std::invalid_argument("chinese_clip embeds queries and documents the same way");
    }
    internal::SemanticEmbeddingResult result;
    result.values = embed_text(request.text);
    result.pooling = "class_token";
    result.normalization = "l2";
    return result;
}

internal::RelevanceResult
ChineseClipPipeline::run(const internal::TextImageToRelevanceRequest& request,
                         internal::ConfigView config) {
    if (!config.empty())
        throw internal::ConfigError("chinese_clip has no runtime configuration");
    const auto image = embed_image(request.document);
    const auto text = embed_text(request.query);
    internal::RelevanceResult result;
    // The reference scales the cosine by exp(logit_scale) before its softmax;
    // the softmax itself needs the other candidates, which a single call does
    // not have, so the scaled cosine is what this task can honestly return.
    result.score = config_.logit_scale * dot(image, text);
    result.kind = internal::ScoreKind::Logit;
    return result;
}

} // namespace trtmc
