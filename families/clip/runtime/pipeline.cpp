/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/clip/runtime/pipeline.h"

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
constexpr const char* kEmbeddingSpace = "clip";

const Tensor& require_output(const TensorMap& outputs, const char* name,
                             std::int32_t expected_width) {
    const auto found = outputs.find(name);
    if (found == outputs.end())
        throw std::runtime_error(std::string("clip engine did not return ") + name);
    const auto& tensor = found->second;
    if (tensor.data == nullptr || tensor.dtype != DType::kFloat32 ||
        tensor.numel() != static_cast<std::size_t>(expected_width)) {
        throw std::runtime_error(std::string("clip engine returned an unusable ") + name);
    }
    return tensor;
}

std::vector<float> l2_normalise(const Tensor& tensor) {
    std::vector<float> values(tensor.numel());
    std::memcpy(values.data(), tensor.data, values.size() * sizeof(float));
    const double total = std::inner_product(values.begin(), values.end(), values.begin(), 0.0);
    const double norm = std::sqrt(total);
    if (!std::isfinite(norm) || norm <= 0.0)
        throw std::runtime_error("clip produced an embedding with no direction");
    for (float& value : values)
        value = static_cast<float>(value / norm);
    return values;
}

float dot(const std::vector<float>& left, const std::vector<float>& right) {
    if (left.size() != right.size())
        throw std::runtime_error("clip embeddings have mismatched widths");
    return static_cast<float>(std::inner_product(left.begin(), left.end(), right.begin(), 0.0));
}

} // namespace

ClipPipeline::ClipPipeline(std::unique_ptr<ITrtModule> vision, std::unique_ptr<ITrtModule> text,
                           std::shared_ptr<ITokenizer> tokenizer, ClipRuntimeConfig config,
                           std::string primary_task)
    : vision_(std::move(vision)), text_(std::move(text)), tokenizer_(std::move(tokenizer)),
      config_(std::move(config)), primary_task_(std::move(primary_task)) {
    if (!vision_ || !vision_->ok() || !text_ || !text_->ok())
        throw std::runtime_error("ClipPipeline: invalid model");
    if (!tokenizer_)
        throw std::runtime_error("ClipPipeline: missing tokenizer");
    if (config_.max_text_len <= 0 || config_.projection_dim <= 0 ||
        !std::isfinite(config_.logit_scale) || config_.logit_scale <= 0.0F) {
        throw std::runtime_error("clip runtime config has invalid dimensions");
    }
}

std::vector<internal::TaskInstance> ClipPipeline::task_bindings() {
    return {internal::bind<internal::IImageToEmbedding>(*this),
            internal::bind<internal::ITextToEmbedding>(*this),
            internal::bind<internal::ITextImageToRelevance>(*this)};
}

std::vector<float> ClipPipeline::embed_image(const internal::ImageView& image) {
    if (image.format != internal::ImageFormat::Float32 || image.channels != 3 ||
        image.data == nullptr || image.height == 0 || image.width == 0 ||
        image.height > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        image.width > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        static_cast<std::uint64_t>(image.height) >
            std::numeric_limits<std::size_t>::max() / image.width / 3 / sizeof(float) ||
        image.byte_size !=
            static_cast<std::size_t>(image.height) * image.width * 3 * sizeof(float)) {
        throw std::invalid_argument("clip requires contiguous float32 RGB input");
    }
    auto pixel_values = preprocess_clip_image(
        static_cast<const float*>(image.data), static_cast<std::int32_t>(image.height),
        static_cast<std::int32_t>(image.width), config_.preprocess);
    Tensor pixels;
    pixels.data = pixel_values.data();
    pixels.shape = {1, 3, config_.preprocess.input_image_h, config_.preprocess.input_image_w};
    pixels.dtype = DType::kFloat32;

    auto outputs = vision_->forward({{"pixel_values", pixels}});
    return l2_normalise(require_output(outputs, "image_embeds", config_.projection_dim));
}

std::vector<float> ClipPipeline::embed_text(std::string_view text) {
    auto ids = tokenizer_->encode(std::string(text));
    if (ids.empty())
        throw std::runtime_error("clip tokenizer returned no tokens");

    // The tokenizer adds the start and end markers. The engine's pooled row is
    // the first end-of-text token, so a prompt longer than the table keeps its
    // head and ends on that marker rather than losing it off the end.
    const auto end_id = ids.back();
    const auto limit = static_cast<std::size_t>(config_.max_text_len);
    if (ids.size() > limit) {
        ids.resize(limit);
        ids.back() = end_id;
    }
    std::int32_t eos_index = -1;
    for (std::size_t index = 0; index < ids.size(); ++index) {
        if (ids[index] == end_id) {
            eos_index = static_cast<std::int32_t>(index);
            break;
        }
    }
    if (eos_index < 0)
        throw std::runtime_error("clip tokenizer produced no end-of-text token");
    // Padding is the end marker too, as the reference pads. Nothing after the
    // pooled row can reach it through a causal mask, so the value only has to
    // be a real token id.
    ids.resize(limit, end_id);

    Tensor input_ids;
    input_ids.data = ids.data();
    input_ids.shape = {config_.max_text_len};
    input_ids.dtype = DType::kInt32;
    Tensor eos;
    eos.data = &eos_index;
    eos.shape = {1};
    eos.dtype = DType::kInt32;

    auto outputs = text_->forward({{"input_ids", input_ids}, {"eos_index", eos}});
    return l2_normalise(require_output(outputs, "text_embeds", config_.projection_dim));
}

internal::SemanticEmbeddingResult
ClipPipeline::run(const internal::ImageToEmbeddingRequest& request, internal::ConfigView config) {
    if (!config.empty())
        throw internal::ConfigError("clip has no runtime configuration");
    internal::SemanticEmbeddingResult result;
    result.values = embed_image(request.image);
    result.embedding_space = kEmbeddingSpace;
    result.pooling = "class_token";
    result.normalization = "l2";
    return result;
}

internal::SemanticEmbeddingResult ClipPipeline::run(const internal::TextToEmbeddingRequest& request,
                                                    internal::ConfigView config) {
    if (!config.empty())
        throw internal::ConfigError("clip has no runtime configuration");
    if (request.role != internal::EmbeddingRole::Default) {
        // One text tower, trained without role prompts: a query embedding and
        // a document embedding would be the same vector, so claiming to honour
        // the distinction would be a lie rather than a limitation.
        throw std::invalid_argument("clip embeds queries and documents the same way");
    }
    internal::SemanticEmbeddingResult result;
    result.values = embed_text(request.text);
    result.embedding_space = kEmbeddingSpace;
    result.pooling = "end_of_text_token";
    result.normalization = "l2";
    return result;
}

internal::RelevanceResult ClipPipeline::run(const internal::TextImageToRelevanceRequest& request,
                                            internal::ConfigView config) {
    if (!config.empty())
        throw internal::ConfigError("clip has no runtime configuration");
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
