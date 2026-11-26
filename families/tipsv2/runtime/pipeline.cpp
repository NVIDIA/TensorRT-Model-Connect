/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/tipsv2/runtime/pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace trtmc::tipsv2 {
namespace {

// Both towers land in one space, which is the only reason a dot product between them means
// anything. Naming it lets a caller see that two results are comparable; it claims nothing about
// any other checkpoint.
constexpr const char* kEmbeddingSpace = "tipsv2";

// How much longer than the window's own worth of text a prompt may be before it is refused.
// Normalisation can lengthen a prompt (one malformed byte becomes a three-byte U+FFFD), and a
// caller may reasonably send a paragraph and expect its first 64 tokens, so the slack is wide.
constexpr std::size_t kPromptBytesPerTokenSlack = 64;

const Tensor& require_output(const TensorMap& outputs, const char* name,
                             std::int32_t expected_width) {
    const auto found = outputs.find(name);
    if (found == outputs.end()) {
        throw std::runtime_error(std::string("tipsv2 engine did not return ") + name);
    }
    const auto& tensor = found->second;
    if (tensor.data == nullptr || tensor.dtype != DType::kFloat32 ||
        tensor.numel() != static_cast<std::size_t>(expected_width)) {
        throw std::runtime_error(std::string("tipsv2 engine returned an unusable ") + name);
    }
    return tensor;
}

// Both of these promote each operand to double **before** multiplying. `std::inner_product` with
// a double accumulator does not: it still multiplies the two floats in float and only then widens,
// so a finite embedding near 1e20 squares to an infinity and one near 1e-30 squares to zero.
double sum_of_products(const std::vector<float>& left, const std::vector<float>& right) {
    if (left.size() != right.size()) {
        throw std::runtime_error("tipsv2 embeddings have mismatched widths");
    }
    double total = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        total += static_cast<double>(left[index]) * static_cast<double>(right[index]);
    }
    return total;
}

std::vector<float> l2_normalise(const Tensor& tensor) {
    std::vector<float> values(tensor.numel());
    std::memcpy(values.data(), tensor.data, values.size() * sizeof(float));
    if (!std::all_of(values.begin(), values.end(),
                     [](float value) { return std::isfinite(value); })) {
        throw std::runtime_error("tipsv2 produced a non-finite embedding");
    }
    const double norm = std::sqrt(sum_of_products(values, values));
    if (!std::isfinite(norm) || norm <= 0.0) {
        throw std::runtime_error("tipsv2 produced an embedding with no direction");
    }
    for (float& value : values) {
        value = static_cast<float>(static_cast<double>(value) / norm);
    }
    return values;
}

double dot(const std::vector<float>& left, const std::vector<float>& right) {
    return sum_of_products(left, right);
}

} // namespace

Pipeline::Pipeline(std::unique_ptr<ITrtModule> vision, std::unique_ptr<ITrtModule> text,
                   std::shared_ptr<Tokenizer> tokenizer, RuntimeConfig config,
                   std::string primary_task)
    : vision_(std::move(vision)), text_(std::move(text)), tokenizer_(std::move(tokenizer)),
      config_(std::move(config)), primary_task_(std::move(primary_task)) {
    if (!vision_ || !vision_->ok() || !text_ || !text_->ok()) {
        throw std::runtime_error("tipsv2 pipeline: invalid model");
    }
    if (!tokenizer_) {
        throw std::runtime_error("tipsv2 pipeline: missing tokenizer");
    }
    if (config_.max_text_len <= 0 || config_.embedding_dim <= 0 || config_.pad_token_id < 0 ||
        !std::isfinite(config_.temperature) || config_.temperature <= 0.0F) {
        throw std::runtime_error("tipsv2 runtime config has invalid dimensions");
    }
    if (config_.resample != 2) {
        // 2 is Pillow's bilinear, which `image_preprocess.cpp` reproduces coefficient for
        // coefficient. Any other filter would move every pixel a little and nothing loudly.
        throw std::runtime_error("tipsv2 is qualified for the bilinear resize only");
    }
    // The engine gathers rows of a table this wide; an id past it would index out of the table.
    // The builder refuses such a pair, so a bundle that reaches here with one was not built by
    // this family.
    if (static_cast<std::int64_t>(tokenizer_->asset().pieces.size()) >
        static_cast<std::int64_t>(config_.vocab_size)) {
        throw std::runtime_error("tipsv2 tokenizer has more pieces than the text engine's "
                                 "embedding table has rows");
    }
    if (tokenizer_->asset().pad_id != config_.pad_token_id) {
        // The text engine masks on this id; a tokenizer that pads with another one would leave
        // the real padding rows inside the attention and the average.
        throw std::runtime_error("tipsv2 tokenizer and runtime disagree on the padding id");
    }
}

std::vector<internal::TaskInstance> Pipeline::task_bindings() {
    return {internal::bind<internal::IImageToEmbedding>(*this),
            internal::bind<internal::ITextToEmbedding>(*this),
            internal::bind<internal::ITextImageToRelevance>(*this)};
}

std::vector<float> Pipeline::embed_image(const internal::ImageView& image) {
    if (image.format != internal::ImageFormat::Float32 || image.channels != 3 ||
        image.data == nullptr || image.height == 0 || image.width == 0 ||
        image.height > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        image.width > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        static_cast<std::uint64_t>(image.height) >
            std::numeric_limits<std::size_t>::max() / image.width / 3 / sizeof(float) ||
        image.byte_size !=
            static_cast<std::size_t>(image.height) * image.width * 3 * sizeof(float)) {
        throw std::invalid_argument("tipsv2 requires contiguous float32 RGB input");
    }
    auto pixel_values = preprocess_tipsv2_image(
        static_cast<const float*>(image.data), static_cast<std::int32_t>(image.height),
        static_cast<std::int32_t>(image.width), config_.preprocess);
    Tensor pixels;
    pixels.data = pixel_values.data();
    pixels.shape = {1, 3, config_.preprocess.input_image_h, config_.preprocess.input_image_w};
    pixels.dtype = DType::kFloat32;

    // `forward` hands back tensors that point into the module's own staging buffer, which the
    // next `forward` on that module overwrites. The copy out of it therefore has to happen
    // before any other call reaches the same tower: one lock, held across both. Each engine runs
    // once per request here, so serialising them costs nothing a caller would notice.
    const std::lock_guard<std::mutex> guard(vision_mutex_);
    auto outputs = vision_->forward({{"pixel_values", pixels}});
    return l2_normalise(require_output(outputs, "image_embeds", config_.embedding_dim));
}

std::vector<float> Pipeline::embed_text(std::string_view text) {
    // Bound the work before doing it. Tokenizing normalises the whole prompt and then holds one
    // symbol per character of it, so an arbitrarily long query would be an arbitrarily large
    // allocation for a result that is then cut to 64 tokens. The limit is generous - the window
    // can never need more than `max_text_len` times the longest piece - so no prompt a caller
    // would write reaches it, and one that does is refused rather than silently truncated to a
    // byte boundary that would change the merges.
    const std::size_t budget =
        static_cast<std::size_t>(config_.max_text_len) * tokenizer_->longest_piece_bytes();
    const std::size_t byte_limit = std::max<std::size_t>(budget * kPromptBytesPerTokenSlack, 1024);
    if (text.size() > byte_limit) {
        throw std::invalid_argument("tipsv2 prompt is longer than " + std::to_string(byte_limit) +
                                    " bytes");
    }
    auto ids = tokenizer_->encode(text);
    if (ids.empty()) {
        // No token survives: an empty prompt, or one that normalises to nothing. The reference
        // averages over zero rows and returns the zero vector, which has no direction to compare,
        // so this is refused rather than turned into a meaningless score.
        throw std::invalid_argument("tipsv2 prompt has no tokens");
    }

    const auto limit = static_cast<std::size_t>(config_.max_text_len);
    if (ids.size() > limit) {
        // The reference's tokenizer keeps the first `max_len` ids and drops the tail. Nothing is
        // appended after the cut, so there is no end token to preserve.
        ids.resize(limit);
    }
    ids.resize(limit, config_.pad_token_id);

    Tensor input_ids;
    input_ids.data = ids.data();
    input_ids.shape = {config_.max_text_len};
    input_ids.dtype = DType::kInt32;

    // The same rule as the image tower: the read out of the module's staging buffer is inside
    // the lock that the `forward` writing it holds.
    const std::lock_guard<std::mutex> guard(text_mutex_);
    auto outputs = text_->forward({{"input_ids", input_ids}});
    return l2_normalise(require_output(outputs, "text_embeds", config_.embedding_dim));
}

internal::SemanticEmbeddingResult Pipeline::run(const internal::ImageToEmbeddingRequest& request,
                                                internal::ConfigView config) {
    if (!config.empty()) {
        throw internal::ConfigError("tipsv2 has no runtime configuration");
    }
    internal::SemanticEmbeddingResult result;
    result.values = embed_image(request.image);
    result.embedding_space = kEmbeddingSpace;
    // Not an average over the grid: a learned class token that rides through every block and is
    // read out of row 0 after the final norm.
    result.pooling = "class_token";
    result.normalization = "l2";
    return result;
}

internal::SemanticEmbeddingResult Pipeline::run(const internal::TextToEmbeddingRequest& request,
                                                internal::ConfigView config) {
    if (!config.empty()) {
        throw internal::ConfigError("tipsv2 has no runtime configuration");
    }
    if (request.role != internal::EmbeddingRole::Default) {
        // One text tower, trained without role prompts: a query embedding and a document
        // embedding would be the same vector, so honouring the distinction would be a lie rather
        // than a limitation.
        throw std::invalid_argument("tipsv2 embeds queries and documents the same way");
    }
    internal::SemanticEmbeddingResult result;
    result.values = embed_text(request.text);
    result.embedding_space = kEmbeddingSpace;
    // The mean of the non-padding rows after the final norm, which is what GlobalAvgPooling does.
    result.pooling = "mean";
    result.normalization = "l2";
    return result;
}

internal::RelevanceResult Pipeline::run(const internal::TextImageToRelevanceRequest& request,
                                        internal::ConfigView config) {
    if (!config.empty()) {
        throw internal::ConfigError("tipsv2 has no runtime configuration");
    }
    const auto image = embed_image(request.document);
    const auto text = embed_text(request.query);
    internal::RelevanceResult result;
    // TIPSv2 is trained with a softmax contrastive loss, so its temperature calibrates a cosine
    // against the *other* candidates. A single call does not have them, which is why this is a
    // Logit and not a Probability: divided by the learned temperature it is exactly the number
    // the reference would feed its softmax, and softmaxing a set of these reproduces its
    // zero-shot classifier.
    result.score = static_cast<float>(dot(image, text) / static_cast<double>(config_.temperature));
    result.kind = internal::ScoreKind::Logit;
    return result;
}

} // namespace trtmc::tipsv2
