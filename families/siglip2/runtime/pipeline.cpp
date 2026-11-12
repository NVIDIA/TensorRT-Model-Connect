/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/siglip2/runtime/pipeline.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace trtmc {

namespace {

// Both towers land in one space, which is the only reason a dot product between them
// means anything. Naming it lets a caller see that two results are comparable; it
// claims nothing about any other checkpoint.
constexpr const char* kEmbeddingSpace = "siglip2";

const Tensor& require_output(const TensorMap& outputs, const char* name,
                             std::int32_t expected_width) {
    const auto found = outputs.find(name);
    if (found == outputs.end())
        throw std::runtime_error(std::string("siglip2 engine did not return ") + name);
    const auto& tensor = found->second;
    if (tensor.data == nullptr || tensor.dtype != DType::kFloat32 ||
        tensor.numel() != static_cast<std::size_t>(expected_width)) {
        throw std::runtime_error(std::string("siglip2 engine returned an unusable ") + name);
    }
    return tensor;
}

std::vector<float> l2_normalise(const Tensor& tensor) {
    std::vector<float> values(tensor.numel());
    std::memcpy(values.data(), tensor.data, values.size() * sizeof(float));
    const double total = std::inner_product(values.begin(), values.end(), values.begin(), 0.0);
    const double norm = std::sqrt(total);
    if (!std::isfinite(norm) || norm <= 0.0)
        throw std::runtime_error("siglip2 produced an embedding with no direction");
    for (float& value : values)
        value = static_cast<float>(value / norm);
    return values;
}

float dot(const std::vector<float>& left, const std::vector<float>& right) {
    if (left.size() != right.size())
        throw std::runtime_error("siglip2 embeddings have mismatched widths");
    return static_cast<float>(std::inner_product(left.begin(), left.end(), right.begin(), 0.0));
}

void validate_image(const internal::ImageView& image) {
    const bool shaped = image.format == internal::ImageFormat::Float32 && image.channels == 3 &&
                        image.data != nullptr && image.height != 0 && image.width != 0;
    const auto limit = static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max());
    if (!shaped || image.height > limit || image.width > limit ||
        static_cast<std::uint64_t>(image.height) >
            std::numeric_limits<std::size_t>::max() / image.width / 3 / sizeof(float) ||
        image.byte_size !=
            static_cast<std::size_t>(image.height) * image.width * 3 * sizeof(float)) {
        throw std::invalid_argument("siglip2 requires contiguous float32 RGB input");
    }
}

} // namespace

Siglip2Pipeline::Siglip2Pipeline(std::unique_ptr<ITrtModule> vision,
                                 std::unique_ptr<ITrtModule> text,
                                 std::shared_ptr<ITokenizer> tokenizer, Siglip2RuntimeConfig config,
                                 std::string primary_task)
    : vision_(std::move(vision)), text_(std::move(text)), tokenizer_(std::move(tokenizer)),
      config_(std::move(config)), primary_task_(std::move(primary_task)) {
    if (!vision_ || !vision_->ok() || !text_ || !text_->ok())
        throw std::runtime_error("Siglip2Pipeline: invalid model");
    if (!tokenizer_)
        throw std::runtime_error("Siglip2Pipeline: missing tokenizer");
    if (config_.max_text_len <= 0 || config_.embedding_dim <= 0 || config_.pad_token_id < 0 ||
        !std::isfinite(config_.logit_scale) || config_.logit_scale <= 0.0F ||
        !std::isfinite(config_.logit_bias)) {
        throw std::runtime_error("siglip2 runtime config has invalid dimensions");
    }
    if (config_.resample != 2) {
        // The runtime reproduces Pillow's bilinear resize only; another filter would move
        // every pixel a little and nothing loudly.
        throw std::runtime_error("siglip2 is qualified for bilinear resize only");
    }
}

std::vector<internal::TaskInstance> Siglip2Pipeline::task_bindings() {
    return {internal::bind<internal::IImageToEmbedding>(*this),
            internal::bind<internal::ITextToEmbedding>(*this),
            internal::bind<internal::ITextImageToRelevance>(*this)};
}

std::vector<float> Siglip2Pipeline::embed_image(const internal::ImageView& image) {
    validate_image(image);
    auto patches = preprocess_siglip2_image(
        static_cast<const float*>(image.data), static_cast<std::int32_t>(image.height),
        static_cast<std::int32_t>(image.width), config_.preprocess);
    const auto count = static_cast<std::int64_t>(patches.grid_height) *
                       static_cast<std::int64_t>(patches.grid_width);
    const auto table = static_cast<std::int64_t>(config_.preprocess.position_grid) *
                       config_.preprocess.position_grid;

    Tensor pixels;
    pixels.data = patches.pixel_values.data();
    pixels.shape = {count, static_cast<std::int64_t>(patches.pixel_values.size()) / count};
    pixels.dtype = DType::kFloat32;
    Tensor taps;
    taps.data = patches.position_weights.data();
    taps.shape = {count, table};
    taps.dtype = DType::kFloat32;

    auto outputs = vision_->forward({{"pixel_values", pixels}, {"position_weights", taps}});
    return l2_normalise(require_output(outputs, "image_embeds", config_.embedding_dim));
}

std::vector<std::int32_t> Siglip2Pipeline::text_window(std::string_view text) const {
    auto ids = tokenizer_->encode(std::string(text));
    if (ids.empty())
        throw std::runtime_error("siglip2 tokenizer returned no tokens");

    // The window is always filled: the engine pools its LAST row, so an unpadded short
    // prompt would be pooled at a different row than the reference pools. An over-long
    // prompt keeps its head and its end token, as the reference's truncation does.
    const auto limit = static_cast<std::size_t>(config_.max_text_len);
    const auto end_id = ids.back();
    if (ids.size() > limit) {
        ids.resize(limit);
        ids.back() = end_id;
    }
    ids.resize(limit, config_.pad_token_id);
    return ids;
}

std::vector<float> Siglip2Pipeline::embed_text(std::string_view text) {
    auto ids = text_window(text);
    Tensor input_ids;
    input_ids.data = ids.data();
    input_ids.shape = {config_.max_text_len};
    input_ids.dtype = DType::kInt32;

    auto outputs = text_->forward({{"input_ids", input_ids}});
    return l2_normalise(require_output(outputs, "text_embeds", config_.embedding_dim));
}

internal::SemanticEmbeddingResult
Siglip2Pipeline::run(const internal::ImageToEmbeddingRequest& request,
                     internal::ConfigView config) {
    if (!config.empty())
        throw internal::ConfigError("siglip2 has no runtime configuration");
    internal::SemanticEmbeddingResult result;
    result.values = embed_image(request.image);
    result.embedding_space = kEmbeddingSpace;
    // Not a row of the sequence: a learned query attends over every patch.
    result.pooling = "attention_probe";
    result.normalization = "l2";
    return result;
}

internal::SemanticEmbeddingResult
Siglip2Pipeline::run(const internal::TextToEmbeddingRequest& request, internal::ConfigView config) {
    if (!config.empty())
        throw internal::ConfigError("siglip2 has no runtime configuration");
    if (request.role != internal::EmbeddingRole::Default) {
        // One text tower, trained without role prompts: a query and a document
        // embedding would be the same vector.
        throw std::invalid_argument("siglip2 embeds queries and documents the same way");
    }
    internal::SemanticEmbeddingResult result;
    result.values = embed_text(request.text);
    result.embedding_space = kEmbeddingSpace;
    // The last row of the padded window, which is padding for a short prompt. That is
    // the checkpoint's rule, not an accident of how this runtime pads.
    result.pooling = "last_token";
    result.normalization = "l2";
    return result;
}

internal::RelevanceResult Siglip2Pipeline::run(const internal::TextImageToRelevanceRequest& request,
                                               internal::ConfigView config) {
    if (!config.empty())
        throw internal::ConfigError("siglip2 has no runtime configuration");
    const auto image = embed_image(request.document);
    const auto text = embed_text(request.query);
    internal::RelevanceResult result;
    // SigLIP 2 is trained with a per-pair sigmoid, so its scale and bias calibrate one
    // pair on its own: the score is a probability that needs no other candidate.
    const float logit = config_.logit_scale * dot(image, text) + config_.logit_bias;
    result.score = 1.0F / (1.0F + std::exp(-logit));
    result.kind = internal::ScoreKind::Probability;
    return result;
}

} // namespace trtmc
