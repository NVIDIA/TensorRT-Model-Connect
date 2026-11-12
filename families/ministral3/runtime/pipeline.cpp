/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/ministral3/runtime/pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

namespace trtmc::ministral3 {
namespace {

void require_empty(internal::ConfigView config) {
    if (!config.empty())
        throw internal::ConfigError("ministral3 exposes no runtime configuration");
}

} // namespace

EncoderPipeline::EncoderPipeline(std::unique_ptr<ITrtModule> encoder,
                                 std::shared_ptr<const TekkenTokenizer> tokenizer,
                                 EncoderSettings settings)
    : encoder_(std::move(encoder)), tokenizer_(std::move(tokenizer)),
      settings_(std::move(settings)) {
    if (!encoder_ || !encoder_->ok() || !tokenizer_)
        throw std::invalid_argument("ministral3 requires an encoder engine and a tokenizer");
    if (settings_.task != internal::ITextToEmbedding::kTask &&
        settings_.task != internal::ITextToPooledFeatures::kTask &&
        settings_.task != internal::ITextToTokenFeatures::kTask)
        throw internal::UnsupportedTask("ministral3 does not implement task: " + settings_.task);
    if (settings_.hidden_size <= 0 || settings_.max_sequence_length <= 0 ||
        settings_.vocab_size != tokenizer_->vocab_size())
        throw std::invalid_argument("ministral3 runtime settings disagree with the tokenizer");
    if (encoder_->tensor_dtype("input_ids") != DType::kInt32 ||
        encoder_->input_rank("input_ids") != 1 ||
        encoder_->input_profile_shape("input_ids", encoder_->profile_idx(),
                                      ProfileShapeSelector::kMax) !=
            std::vector<std::int64_t>{settings_.max_sequence_length})
        throw std::invalid_argument(
            "ministral3 engine input_ids profile disagrees with runtime.json");
    const auto shape = encoder_->tensor_shape("hidden_states");
    if (encoder_->tensor_dtype("hidden_states") != DType::kFloat32 || shape.size() != 2 ||
        shape[1] != settings_.hidden_size)
        throw std::invalid_argument("ministral3 engine must return FP32 hidden_states [S, hidden]");
    hidden_ = static_cast<std::size_t>(settings_.hidden_size);
}

std::vector<internal::TaskInstance> EncoderPipeline::task_bindings() {
    return {internal::bind<internal::ITextToEmbedding>(*this),
            internal::bind<internal::ITextToPooledFeatures>(*this),
            internal::bind<internal::ITextToTokenFeatures>(*this)};
}

EncoderPipeline::Encoded EncoderPipeline::resolve(const internal::TextSource& source) const {
    Encoded encoded;
    if (const auto* text = std::get_if<std::string_view>(&source)) {
        encoded.tokens = tokenizer_->encode(*text);
        encoded.ids.reserve(encoded.tokens.size());
        for (const auto& token : encoded.tokens)
            encoded.ids.push_back(token.id);
        return encoded;
    }
    const auto ids = std::get<Span<const std::int32_t>>(source);
    encoded.ids.assign(ids.begin(), ids.end());
    return encoded;
}

std::vector<float> EncoderPipeline::forward(const std::vector<std::int32_t>& ids) {
    if (ids.empty())
        throw std::invalid_argument("ministral3 input must contain at least one token");
    if (ids.size() > static_cast<std::uint64_t>(settings_.max_sequence_length))
        throw std::invalid_argument("ministral3 input has " + std::to_string(ids.size()) +
                                    " tokens; the bundle accepts at most " +
                                    std::to_string(settings_.max_sequence_length));
    for (const auto id : ids)
        if (id < 0 || id >= settings_.vocab_size)
            throw std::invalid_argument("ministral3 token ID is outside the vocabulary");
    auto input = ids;
    const auto length = static_cast<std::int64_t>(ids.size());
    const auto outputs =
        encoder_->forward({{"input_ids", {input.data(), {length}, DType::kInt32}}});
    const auto found = outputs.find("hidden_states");
    if (found == outputs.end())
        throw std::runtime_error("ministral3 encoder returned no hidden_states");
    const auto& output = found->second;
    if (output.data == nullptr || output.dtype != DType::kFloat32 ||
        output.shape != std::vector<std::int64_t>{length, settings_.hidden_size})
        throw std::runtime_error("ministral3 encoder returned invalid hidden_states");
    std::vector<float> states(ids.size() * hidden_);
    std::memcpy(states.data(), output.data, states.size() * sizeof(float));
    for (const auto value : states)
        if (!std::isfinite(value))
            throw std::runtime_error("ministral3 encoder returned a non-finite hidden state");
    return states;
}

std::vector<float> EncoderPipeline::mean(const std::vector<float>& states, std::size_t rows) const {
    std::vector<double> sums(hidden_, 0.0);
    for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t column = 0; column < hidden_; ++column)
            sums[column] += states[row * hidden_ + column];
    std::vector<float> values(hidden_);
    for (std::size_t column = 0; column < hidden_; ++column)
        values[column] = static_cast<float>(sums[column] / static_cast<double>(rows));
    return values;
}

internal::SemanticEmbeddingResult
EncoderPipeline::run(const internal::TextToEmbeddingRequest& request, internal::ConfigView config) {
    require_empty(config);
    std::string text;
    switch (request.role) {
    case internal::EmbeddingRole::Default:
        break;
    case internal::EmbeddingRole::Query:
        text = settings_.query_prompt;
        break;
    case internal::EmbeddingRole::Document:
        text = settings_.document_prompt;
        break;
    default:
        throw std::invalid_argument("ministral3 unknown embedding role");
    }
    text.append(request.text);
    const auto encoded = resolve(internal::TextSource{std::string_view(text)});
    auto values = mean(forward(encoded.ids), encoded.ids.size());
    double norm = 0.0;
    for (const auto value : values)
        norm += static_cast<double>(value) * value;
    // Sentence Transformers' Normalize divides by max(||x||, 1e-12).
    const auto scale = 1.0 / std::max(std::sqrt(norm), 1e-12);
    for (auto& value : values)
        value = static_cast<float>(value * scale);
    return {std::move(values), "", "mean", "l2"};
}

internal::PooledFeaturesResult
EncoderPipeline::run(const internal::TextToPooledFeaturesRequest& request,
                     internal::ConfigView config) {
    require_empty(config);
    const auto encoded = resolve(request.text);
    return {mean(forward(encoded.ids), encoded.ids.size()), "mean", "none"};
}

internal::TokenFeaturesResult
EncoderPipeline::run(const internal::TextToTokenFeaturesRequest& request,
                     internal::ConfigView config) {
    require_empty(config);
    const auto encoded = resolve(request.text);
    internal::TokenFeaturesResult result;
    result.features.values = forward(encoded.ids);
    result.features.rows = encoded.ids.size();
    result.features.columns = hidden_;
    result.tokens.reserve(encoded.ids.size());
    for (std::size_t index = 0; index < encoded.ids.size(); ++index) {
        internal::FeatureToken token;
        token.token_id = encoded.ids[index];
        token.input_index = 0;
        token.token_index = index;
        if (!encoded.tokens.empty()) {
            token.has_byte_offsets = true;
            token.byte_begin = encoded.tokens[index].byte_begin;
            token.byte_end = encoded.tokens[index].byte_end;
        }
        result.tokens.push_back(token);
    }
    return result;
}

} // namespace trtmc::ministral3
