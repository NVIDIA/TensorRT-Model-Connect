/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/xlnet/runtime/pipeline.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <variant>

namespace trtmc {

namespace {

// Infer the hidden dimension from the last axis of the first output tensor.
int32_t infer_output_hidden_dim(const ITrtModule& module) {
    for (const auto& info : module.output_info())
        if (!info.shape.empty())
            return static_cast<int32_t>(info.shape.back());
    return 0;
}

// Mean-pool [seq_len, hidden] over the first actual_len positions,
// then L2-normalize. Returns the pooled vector of size hidden.
std::vector<float> mean_pool_and_normalize(const float* data, int32_t actual_len, int32_t hidden) {
    std::vector<float> pooled(static_cast<std::size_t>(hidden), 0.0f);
    const float inv_len = 1.0f / static_cast<float>(actual_len);
    for (int32_t s = 0; s < actual_len; ++s)
        for (int32_t h = 0; h < hidden; ++h)
            pooled[h] += data[static_cast<std::size_t>(s) * hidden + h];
    for (int32_t h = 0; h < hidden; ++h)
        pooled[h] *= inv_len;

    float norm = 0.0f;
    for (int32_t h = 0; h < hidden; ++h)
        norm += pooled[h] * pooled[h];
    norm = std::sqrt(norm);
    if (norm > 1e-12f)
        for (int32_t h = 0; h < hidden; ++h)
            pooled[h] /= norm;

    return pooled;
}

// Check whether the engine's attention_mask input expects int32.
bool engine_mask_is_int32(const ITrtModule& module) {
    for (const auto& info : module.input_info())
        if (info.name == "attention_mask")
            return info.dtype == DType::kInt32;
    return false;
}

// Largest sequence the loaded engine accepts for input_ids, taken from its
// active optimization profile (dynamic input) or its fixed shape.
std::size_t input_capacity(const ITrtModule& module) {
    const std::string name = "input_ids";
    const auto shape =
        module.input_is_dynamic(name)
            ? module.input_profile_shape(name, module.profile_idx(), ProfileShapeSelector::kMax)
            : module.tensor_shape(name);
    if (shape.empty() || shape.back() <= 0)
        throw std::runtime_error("EncoderPipeline: engine reports no usable input_ids capacity");
    return static_cast<std::size_t>(shape.back());
}

} // namespace

// ─── EncoderPipeline ───

EncoderPipeline::EncoderPipeline(std::unique_ptr<ITrtModule> encoder, std::string mode,
                                 std::shared_ptr<ITokenizer> tokenizer)
    : encoder_(std::move(encoder)), mode_(std::move(mode)), tokenizer_(std::move(tokenizer)) {
    if (!encoder_ || !encoder_->ok())
        throw std::runtime_error("EncoderPipeline: invalid encoder module");
    if (!tokenizer_)
        throw std::runtime_error("EncoderPipeline: no tokenizer configured");
}

void EncoderPipeline::require_mode(std::string_view expected) const {
    if (mode_ != expected)
        throw internal::UnsupportedTask("EncoderPipeline: this bundle was built for " + mode_ +
                                        ", not " + std::string(expected));
}

std::vector<internal::TaskInstance> EncoderPipeline::task_bindings() {
    if (mode_ == internal::ITextToEmbedding::kTask)
        return {internal::bind<internal::ITextToEmbedding>(*this)};
    if (mode_ == internal::ITextPairToRelevance::kTask) {
        return {internal::bind<internal::ITextPairToRelevance>(*this),
                internal::bind<internal::ITextQueryDocumentsToRelevance>(*this)};
    }
    return {internal::bind<internal::ITextToPooledFeatures>(*this)};
}

std::vector<std::int32_t> EncoderPipeline::resolve_ids(const internal::TextSource& text) const {
    if (const auto* view = std::get_if<std::string_view>(&text))
        return tokenizer_->encode(std::string(*view));
    const auto ids = std::get<Span<const std::int32_t>>(text);
    return {ids.begin(), ids.end()};
}

std::pair<std::vector<float>, std::int32_t>
EncoderPipeline::run_encoder(const std::vector<std::int32_t>& input_ids) const {
    const auto n = input_ids.size();
    if (n == 0)
        throw std::invalid_argument("EncoderPipeline: text produced no tokens");
    if (n > input_capacity(*encoder_))
        throw std::invalid_argument("EncoderPipeline: input exceeds engine capacity");
    std::vector<int32_t> mask_i32(n, 1);
    std::vector<float> mask_f32(n, 1.0f);

    auto ids_copy = input_ids;
    Tensor ids_t;
    ids_t.data = ids_copy.data();
    ids_t.shape = {static_cast<int64_t>(n)};
    ids_t.dtype = DType::kInt32;

    // Match the engine's expected dtype for the attention mask.
    Tensor mask_t;
    if (engine_mask_is_int32(*encoder_)) {
        mask_t.data = mask_i32.data();
        mask_t.shape = {static_cast<int64_t>(n)};
        mask_t.dtype = DType::kInt32;
    } else {
        mask_t.data = mask_f32.data();
        mask_t.shape = {static_cast<int64_t>(n)};
        mask_t.dtype = DType::kFloat32;
    }

    TensorMap inputs;
    inputs["input_ids"] = ids_t;
    inputs["attention_mask"] = mask_t;

    auto outputs = encoder_->forward(inputs);

    for (auto& [name, tensor] : outputs) {
        if (name.find("logits") != std::string::npos || name.find("embed") != std::string::npos ||
            name.find("output") != std::string::npos || name.find("hidden") != std::string::npos ||
            name.find("score") != std::string::npos) {
            const auto count = tensor.numel();
            std::vector<float> data(static_cast<std::size_t>(count));
            std::memcpy(data.data(), tensor.data, static_cast<std::size_t>(count) * sizeof(float));
            return {std::move(data), static_cast<int32_t>(count)};
        }
    }
    throw std::runtime_error("EncoderPipeline: engine returned no recognizable output tensor");
}

internal::SemanticEmbeddingResult
EncoderPipeline::run(const internal::TextToEmbeddingRequest& request, internal::ConfigView config) {
    require_mode(internal::ITextToEmbedding::kTask);
    if (!config.empty())
        throw internal::ConfigError(
            "EncoderPipeline: text_to_embedding has no runtime configuration");
    auto ids = tokenizer_->encode(std::string(request.text));
    auto [data, dim] = run_encoder(ids);

    // For embedding models: the TRT engine returns [max_seq, hidden] hidden
    // states. Mean-pool over actual input positions and L2-normalize.
    internal::SemanticEmbeddingResult result;
    const auto actual_len = static_cast<int32_t>(ids.size());
    const int32_t hidden = infer_output_hidden_dim(*encoder_);
    if (hidden > 0 && actual_len > 0 && dim >= actual_len * hidden) {
        result.values = mean_pool_and_normalize(data.data(), actual_len, hidden);
        result.pooling = "mean";
        result.normalization = "l2";
    } else {
        result.values = std::move(data);
    }
    return result;
}

internal::PooledFeaturesResult
EncoderPipeline::run(const internal::TextToPooledFeaturesRequest& request,
                     internal::ConfigView config) {
    require_mode(internal::ITextToPooledFeatures::kTask);
    if (!config.empty()) {
        throw internal::ConfigError(
            "EncoderPipeline: text_to_pooled_features has no runtime configuration");
    }
    auto ids = resolve_ids(request.text);
    auto [data, dim] = run_encoder(ids);

    // Extract the CLS token (first hidden_dim values) from the full hidden
    // state matrix [max_seq, hidden]. Matches HF model(**inputs)
    // .last_hidden_state[0, 0] for encoder-only models (BERT, RoBERTa, etc.).
    const int32_t hidden = infer_output_hidden_dim(*encoder_);
    if (hidden > 0 && dim > hidden)
        data.resize(static_cast<std::size_t>(hidden));

    internal::PooledFeaturesResult result;
    result.values = std::move(data);
    result.pooling = "first_token";
    result.normalization = "none";
    return result;
}

internal::RelevanceResult EncoderPipeline::run(const internal::TextPairToRelevanceRequest& request,
                                               internal::ConfigView config) {
    require_mode(internal::ITextPairToRelevance::kTask);
    if (!config.empty()) {
        throw internal::ConfigError(
            "EncoderPipeline: text_pair_to_relevance has no runtime configuration");
    }
    // Match the text-only reranking template documented by the supported
    // Nemotron rerank cross-encoder model card.
    const std::string combined =
        "question:" + std::string(request.query) + "   passage:" + std::string(request.document);
    auto ids = tokenizer_->encode(combined);
    auto [data, dim] = run_encoder(ids);
    (void)dim;

    internal::RelevanceResult result;
    result.score = data.empty() ? 0.0f : data[0];
    result.kind = internal::ScoreKind::Unbounded;
    return result;
}

internal::DocumentRelevanceResult
EncoderPipeline::run(const internal::TextQueryDocumentsToRelevanceRequest& request,
                     internal::ConfigView config) {
    require_mode(internal::ITextPairToRelevance::kTask);
    internal::DocumentRelevanceResult result;
    result.scores.reserve(request.documents.size());
    for (const auto& document : request.documents) {
        result.scores.push_back(
            run(internal::TextPairToRelevanceRequest{request.query, document}, config).score);
    }
    result.kind = internal::ScoreKind::Unbounded;
    return result;
}

} // namespace trtmc
