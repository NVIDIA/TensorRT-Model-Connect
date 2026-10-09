/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/modernbert/runtime/pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <variant>

namespace trtmc::modernbert {

EncoderPipeline::EncoderPipeline(std::unique_ptr<ITrtModule> encoder, std::string task,
                                 std::shared_ptr<ITokenizer> tokenizer, std::int64_t vocab_size,
                                 std::int64_t max_sequence_length)
    : encoder_(std::move(encoder)), task_(std::move(task)), tokenizer_(std::move(tokenizer)),
      vocab_size_(vocab_size), max_sequence_length_(max_sequence_length) {
    if (!encoder_ || !encoder_->ok() || !tokenizer_)
        throw std::invalid_argument("ModernBERT requires an encoder and tokenizer");
    if (task_ != internal::ITextToPooledFeatures::kTask &&
        task_ != internal::ITextToEmbedding::kTask &&
        task_ != internal::ITextPairToRelevance::kTask)
        throw internal::UnsupportedTask("ModernBERT unsupported task: " + task_);
    if (vocab_size_ <= 0 || max_sequence_length_ <= 0)
        throw std::invalid_argument("ModernBERT requires positive vocabulary and sequence bounds");
    const bool dynamic = encoder_->input_is_dynamic("input_ids");
    const auto capacity = dynamic
                              ? encoder_->input_profile_shape("input_ids", encoder_->profile_idx(),
                                                              ProfileShapeSelector::kMax)
                              : encoder_->tensor_shape("input_ids");
    if (capacity.size() != 1 || capacity[0] < max_sequence_length_ ||
        encoder_->tensor_dtype("input_ids") != DType::kInt32)
        throw std::invalid_argument("ModernBERT requires a compatible input_ids profile");
    if (!dynamic)
        fixed_sequence_length_ = static_cast<std::size_t>(capacity[0]);
    mask_dtype_ = encoder_->tensor_dtype("attention_mask");
    if (mask_dtype_ != DType::kInt32 && mask_dtype_ != DType::kFloat32)
        throw std::invalid_argument("ModernBERT attention_mask must be int32 or float32");
    const auto shape = encoder_->tensor_shape("hidden_states");
    if (shape.size() != 2 || shape[1] <= 0 ||
        encoder_->tensor_dtype("hidden_states") != DType::kFloat32)
        throw std::invalid_argument("ModernBERT requires FP32 hidden_states [sequence, hidden]");
    hidden_size_ = static_cast<std::size_t>(shape[1]);
}

void EncoderPipeline::require_task(std::string_view task, internal::ConfigView config) const {
    if (task_ != task)
        throw internal::UnsupportedTask("ModernBERT bundle does not implement " +
                                        std::string(task));
    if (!config.empty())
        throw internal::ConfigError("ModernBERT exposes no runtime configuration");
}

std::vector<internal::TaskInstance> EncoderPipeline::task_bindings() {
    if (task_ == internal::ITextToEmbedding::kTask)
        return {internal::bind<internal::ITextToEmbedding>(*this)};
    if (task_ == internal::ITextPairToRelevance::kTask)
        return {internal::bind<internal::ITextPairToRelevance>(*this),
                internal::bind<internal::ITextQueryDocumentsToRelevance>(*this)};
    return {internal::bind<internal::ITextToPooledFeatures>(*this)};
}

std::vector<std::int32_t> EncoderPipeline::resolve_ids(const internal::TextSource& source) const {
    if (const auto* text = std::get_if<std::string_view>(&source))
        return tokenizer_->encode(std::string(*text));
    const auto ids = std::get<Span<const std::int32_t>>(source);
    if (ids.empty())
        throw std::invalid_argument("ModernBERT token input must be nonempty");
    return {ids.begin(), ids.end()};
}

std::vector<float> EncoderPipeline::forward(const std::vector<std::int32_t>& ids) {
    if (ids.empty() || ids.size() > static_cast<std::uint64_t>(max_sequence_length_))
        throw std::invalid_argument("ModernBERT input length is outside the bundle profile");
    for (const auto id : ids)
        if (id < 0 || id >= vocab_size_)
            throw std::invalid_argument("ModernBERT token ID is outside the vocabulary");
    auto ids_copy = ids;
    const auto input_size = fixed_sequence_length_ ? fixed_sequence_length_ : ids.size();
    const auto length = static_cast<std::int64_t>(input_size);
    // Note (Jiaxin Deng): Fixed TP plans need a fresh zeroed tail on every request.
    ids_copy.resize(input_size, 0);
    std::vector<std::int32_t> mask_i32;
    std::vector<float> mask_f32;
    Tensor mask;
    mask.shape = {length};
    mask.dtype = mask_dtype_;
    if (mask_dtype_ == DType::kInt32) {
        mask_i32.assign(input_size, 0);
        std::fill_n(mask_i32.begin(), ids.size(), 1);
        mask.data = mask_i32.data();
    } else {
        mask_f32.assign(input_size, 0.0f);
        std::fill_n(mask_f32.begin(), ids.size(), 1.0f);
        mask.data = mask_f32.data();
    }
    const auto outputs = encoder_->forward(
        {{"input_ids", {ids_copy.data(), {length}, DType::kInt32}}, {"attention_mask", mask}});
    const auto found = outputs.find("hidden_states");
    if (found == outputs.end())
        throw std::runtime_error("ModernBERT encoder returned no hidden_states");
    const auto& output = found->second;
    if (!output.data || output.dtype != DType::kFloat32 || output.shape.size() != 2 ||
        output.shape[0] != length || output.shape[1] != static_cast<std::int64_t>(hidden_size_) ||
        hidden_size_ > std::numeric_limits<std::size_t>::max() / ids.size() / sizeof(float))
        throw std::runtime_error("ModernBERT encoder returned invalid hidden_states");
    std::vector<float> values(ids.size() * hidden_size_);
    std::memcpy(values.data(), output.data, values.size() * sizeof(float));
    return values;
}

internal::PooledFeaturesResult
EncoderPipeline::run(const internal::TextToPooledFeaturesRequest& request,
                     internal::ConfigView config) {
    require_task(internal::ITextToPooledFeatures::kTask, config);
    auto values = forward(resolve_ids(request.text));
    values.resize(hidden_size_);
    return {std::move(values), "cls", "none"};
}

internal::SemanticEmbeddingResult
EncoderPipeline::run(const internal::TextToEmbeddingRequest& request, internal::ConfigView config) {
    require_task(internal::ITextToEmbedding::kTask, config);
    if (request.role != internal::EmbeddingRole::Default &&
        request.role != internal::EmbeddingRole::Query &&
        request.role != internal::EmbeddingRole::Document)
        throw std::invalid_argument("ModernBERT unknown embedding role");
    const auto ids = resolve_ids(internal::TextSource{request.text});
    const auto states = forward(ids);
    std::vector<float> values(hidden_size_, 0.0f);
    for (std::size_t row = 0; row < ids.size(); ++row)
        for (std::size_t column = 0; column < hidden_size_; ++column)
            values[column] += states[row * hidden_size_ + column];
    float norm = 0.0f;
    for (auto& value : values) {
        value /= static_cast<float>(ids.size());
        norm += value * value;
    }
    norm = std::sqrt(norm);
    if (norm > 1e-12f)
        for (auto& value : values)
            value /= norm;
    return {std::move(values), "", "mean", "l2"};
}

internal::RelevanceResult EncoderPipeline::run(const internal::TextPairToRelevanceRequest& request,
                                               internal::ConfigView config) {
    require_task(internal::ITextPairToRelevance::kTask, config);
    const auto text =
        "question:" + std::string(request.query) + "   passage:" + std::string(request.document);
    const auto values = forward(resolve_ids(internal::TextSource{std::string_view(text)}));
    // Note (Jiaxin Deng): Preserve the legacy first-feature score, without claiming a trained head.
    return {values.front(), internal::ScoreKind::Unbounded};
}

internal::DocumentRelevanceResult
EncoderPipeline::run(const internal::TextQueryDocumentsToRelevanceRequest& request,
                     internal::ConfigView config) {
    require_task(internal::ITextPairToRelevance::kTask, config);
    internal::DocumentRelevanceResult result;
    result.kind = internal::ScoreKind::Unbounded;
    result.scores.reserve(request.documents.size());
    for (const auto document : request.documents)
        result.scores.push_back(
            run(internal::TextPairToRelevanceRequest{request.query, document}, {}).score);
    return result;
}

} // namespace trtmc::modernbert
