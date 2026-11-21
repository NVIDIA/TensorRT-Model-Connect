/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/siglip/runtime/pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace trtmc {

namespace {

// Both towers land in one space, which is the only reason a dot product
// between them means anything. Naming it lets a caller see that two results
// are comparable; it claims nothing about any other checkpoint.
constexpr const char* kEmbeddingSpace = "siglip";

const Tensor& require_output(const TensorMap& outputs, const char* name,
                             std::int32_t expected_width) {
    const auto found = outputs.find(name);
    if (found == outputs.end())
        throw std::runtime_error(std::string("siglip engine did not return ") + name);
    const auto& tensor = found->second;
    if (tensor.data == nullptr || tensor.dtype != DType::kFloat32 ||
        tensor.numel() != static_cast<std::size_t>(expected_width)) {
        throw std::runtime_error(std::string("siglip engine returned an unusable ") + name);
    }
    return tensor;
}

std::vector<float> l2_normalise(const Tensor& tensor) {
    std::vector<float> values(tensor.numel());
    std::memcpy(values.data(), tensor.data, values.size() * sizeof(float));
    const double total = std::inner_product(values.begin(), values.end(), values.begin(), 0.0);
    const double norm = std::sqrt(total);
    if (!std::isfinite(norm) || norm <= 0.0)
        throw std::runtime_error("siglip produced an embedding with no direction");
    for (float& value : values)
        value = static_cast<float>(value / norm);
    return values;
}

float dot(const std::vector<float>& left, const std::vector<float>& right) {
    if (left.size() != right.size())
        throw std::runtime_error("siglip embeddings have mismatched widths");
    return static_cast<float>(std::inner_product(left.begin(), left.end(), right.begin(), 0.0));
}

} // namespace

SiglipPipeline::SiglipPipeline(std::unique_ptr<ITrtModule> vision, std::unique_ptr<ITrtModule> text,
                               std::shared_ptr<ITokenizer> tokenizer, SiglipRuntimeConfig config,
                               std::string primary_task)
    : vision_(std::move(vision)), text_(std::move(text)), tokenizer_(std::move(tokenizer)),
      config_(std::move(config)), primary_task_(std::move(primary_task)) {
    if (!vision_ || !vision_->ok() || !text_ || !text_->ok())
        throw std::runtime_error("SiglipPipeline: invalid model");
    if (!tokenizer_)
        throw std::runtime_error("SiglipPipeline: missing tokenizer");
    if (config_.max_text_len <= 0 || config_.embedding_dim <= 0 || config_.pad_token_id < 0 ||
        !std::isfinite(config_.logit_scale) || config_.logit_scale <= 0.0F ||
        !std::isfinite(config_.logit_bias)) {
        throw std::runtime_error("siglip runtime config has invalid dimensions");
    }
    if (config_.resample != 3) {
        // SigLIP 2 resamples bilinear. This family reproduces Pillow's bicubic
        // only, so a checkpoint asking for anything else is refused here rather
        // than resampled with the wrong filter.
        throw std::runtime_error("siglip is qualified for bicubic resize only");
    }
}

std::vector<internal::TaskInstance> SiglipPipeline::task_bindings() {
    return {internal::bind<internal::IImageToEmbedding>(*this),
            internal::bind<internal::ITextToEmbedding>(*this),
            internal::bind<internal::ITextImageToRelevance>(*this)};
}

std::vector<float> SiglipPipeline::embed_image(const internal::ImageView& image) {
    if (image.format != internal::ImageFormat::Float32 || image.channels != 3 ||
        image.data == nullptr || image.height == 0 || image.width == 0 ||
        image.height > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        image.width > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        static_cast<std::uint64_t>(image.height) >
            std::numeric_limits<std::size_t>::max() / image.width / 3 / sizeof(float) ||
        image.byte_size !=
            static_cast<std::size_t>(image.height) * image.width * 3 * sizeof(float)) {
        throw std::invalid_argument("siglip requires contiguous float32 RGB input");
    }
    auto pixel_values = preprocess_siglip_image(
        static_cast<const float*>(image.data), static_cast<std::int32_t>(image.height),
        static_cast<std::int32_t>(image.width), config_.preprocess);
    Tensor pixels;
    pixels.data = pixel_values.data();
    pixels.shape = {1, 3, config_.preprocess.input_image_h, config_.preprocess.input_image_w};
    pixels.dtype = DType::kFloat32;

    auto outputs = vision_->forward({{"pixel_values", pixels}});
    return l2_normalise(require_output(outputs, "image_embeds", config_.embedding_dim));
}

std::vector<float> SiglipPipeline::embed_text(std::string_view text) {
    auto ids = tokenizer_->encode(std::string(text));
    if (ids.empty())
        throw std::runtime_error("siglip tokenizer returned no tokens");

    // The window is always filled. The engine pools its LAST row, so a short
    // prompt that is not padded out would be pooled at a different row than
    // the reference pools - measured at 1.98 maxabs for a 32-row window - and
    // an over-long prompt must lose its tail rather than its head.
    const auto limit = static_cast<std::size_t>(config_.max_text_len);
    const auto end_id = ids.back();
    if (ids.size() > limit) {
        ids.resize(limit);
        ids.back() = end_id;
    }
    ids.resize(limit, config_.pad_token_id);

    Tensor input_ids;
    input_ids.data = ids.data();
    input_ids.shape = {config_.max_text_len};
    input_ids.dtype = DType::kInt32;

    auto outputs = text_->forward({{"input_ids", input_ids}});
    return l2_normalise(require_output(outputs, "text_embeds", config_.embedding_dim));
}

internal::SemanticEmbeddingResult
SiglipPipeline::run(const internal::ImageToEmbeddingRequest& request, internal::ConfigView config) {
    if (!config.empty())
        throw internal::ConfigError("siglip has no runtime configuration");
    internal::SemanticEmbeddingResult result;
    result.values = embed_image(request.image);
    result.embedding_space = kEmbeddingSpace;
    // Not a row of the sequence: a learned query attends over the whole grid.
    result.pooling = "attention_probe";
    result.normalization = "l2";
    return result;
}

internal::SemanticEmbeddingResult
SiglipPipeline::run(const internal::TextToEmbeddingRequest& request, internal::ConfigView config) {
    if (!config.empty())
        throw internal::ConfigError("siglip has no runtime configuration");
    if (request.role != internal::EmbeddingRole::Default) {
        // One text tower, trained without role prompts: a query embedding and
        // a document embedding would be the same vector, so honouring the
        // distinction would be a lie rather than a limitation.
        throw std::invalid_argument("siglip embeds queries and documents the same way");
    }
    internal::SemanticEmbeddingResult result;
    result.values = embed_text(request.text);
    result.embedding_space = kEmbeddingSpace;
    // The last row of the padded window, which is itself padding. That is the
    // checkpoint's rule, not an accident of how this runtime pads.
    result.pooling = "last_token";
    result.normalization = "l2";
    return result;
}

internal::RelevanceResult SiglipPipeline::run(const internal::TextImageToRelevanceRequest& request,
                                              internal::ConfigView config) {
    if (!config.empty())
        throw internal::ConfigError("siglip has no runtime configuration");
    const auto image = embed_image(request.document);
    const auto text = embed_text(request.query);
    internal::RelevanceResult result;
    // SigLIP's score is a genuine per-pair probability: the reference trains
    // it with a sigmoid, so the scale and bias calibrate one pair on its own
    // and nothing here needs the other candidates. That is why this returns a
    // Probability where a softmax-trained model can honestly return only a
    // Logit.
    const float logit = config_.logit_scale * dot(image, text) + config_.logit_bias;
    result.score = 1.0F / (1.0F + std::exp(-logit));
    result.kind = internal::ScoreKind::Probability;
    return result;
}

} // namespace trtmc
