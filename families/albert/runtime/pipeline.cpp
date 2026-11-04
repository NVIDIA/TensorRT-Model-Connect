/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/albert/runtime/pipeline.h"

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

// Mean-pool [seq_len, hidden] over actual_len positions then L2-normalise.
// Returns the pooled vector of size hidden.
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

void require_encoder_output(const std::vector<float>& values, int32_t actual_len, int32_t hidden) {
    if (hidden <= 0 || actual_len <= 0 ||
        static_cast<std::size_t>(actual_len) > values.size() / static_cast<std::size_t>(hidden)) {
        throw std::runtime_error("EncoderPipeline: encoder output shape mismatch");
    }
}

} // namespace

// ─── EncoderPipeline ───

EncoderPipeline::EncoderPipeline(std::unique_ptr<ITrtModule> encoder, std::string primary_task,
                                 std::shared_ptr<ITokenizer> tokenizer, std::string model_id_str)
    : encoder_(std::move(encoder)), primary_task_(std::move(primary_task)),
      tokenizer_(std::move(tokenizer)), model_id_(std::move(model_id_str)) {
    if (!encoder_ || !encoder_->ok())
        throw std::runtime_error("EncoderPipeline: invalid encoder module");
}

const char* EncoderPipeline::task() const noexcept {
    return primary_task_.c_str();
}

std::vector<trtmc::internal::TaskInstance> EncoderPipeline::task_bindings() {
    return {
        trtmc::internal::bind<trtmc::internal::ITextToPooledFeatures>(*this),
        trtmc::internal::bind<trtmc::internal::ITextToTokenFeatures>(*this),
        trtmc::internal::bind<trtmc::internal::ITextToEmbedding>(*this),
        trtmc::internal::bind<trtmc::internal::ITextPairToRelevance>(*this),
    };
}

// ─── ITextToPooledFeatures (preserves legacy encoding CLS output) ───
// Extracts the CLS token representation (first hidden dimension elements).
trtmc::internal::PooledFeaturesResult
EncoderPipeline::run(const trtmc::internal::TextToPooledFeaturesRequest& request,
                     trtmc::internal::ConfigView /*config*/) {
    std::vector<int32_t> ids;
    if (const auto* sv = std::get_if<std::string_view>(&request.text)) {
        if (!tokenizer_)
            throw std::runtime_error("EncoderPipeline: no tokenizer configured");
        ids = tokenizer_->encode(std::string(*sv));
    } else if (const auto* span = std::get_if<Span<const std::int32_t>>(&request.text)) {
        ids.assign(span->begin(), span->end());
    } else {
        throw std::runtime_error("EncoderPipeline: unsupported text source");
    }
    const auto raw_floats = encode_ids(ids);

    trtmc::internal::PooledFeaturesResult result;
    result.pooling = "cls";
    result.normalization = "none";

    const int32_t hidden = infer_output_hidden_dim(*encoder_);
    const auto actual_len = static_cast<int32_t>(ids.size());
    require_encoder_output(raw_floats, actual_len, hidden);

    result.values.assign(raw_floats.begin(), raw_floats.begin() + hidden);
    return result;
}

// ─── ITextToTokenFeatures (was: encoding) ───
// Tokenise, run the encoder, extract all valid token hidden states.
// Returns [actual_seq_len, hidden_size] where row 0 is the CLS token.
trtmc::internal::TokenFeaturesResult
EncoderPipeline::run(const trtmc::internal::TextToTokenFeaturesRequest& request,
                     trtmc::internal::ConfigView /*config*/) {
    std::vector<int32_t> ids;
    if (const auto* sv = std::get_if<std::string_view>(&request.text)) {
        if (!tokenizer_)
            throw std::runtime_error("EncoderPipeline: no tokenizer configured");
        ids = tokenizer_->encode(std::string(*sv));
    } else if (const auto* span = std::get_if<Span<const std::int32_t>>(&request.text)) {
        ids.assign(span->begin(), span->end());
    } else {
        throw std::runtime_error("EncoderPipeline: unsupported text source");
    }
    const auto raw_floats = encode_ids(ids);

    const int32_t hidden = infer_output_hidden_dim(*encoder_);
    const auto actual_len = static_cast<int32_t>(ids.size());

    trtmc::internal::TokenFeaturesResult result;
    require_encoder_output(raw_floats, actual_len, hidden);

    // Build the [actual_len, hidden] feature matrix.
    result.features.values.assign(raw_floats.begin(), raw_floats.begin() + actual_len * hidden);
    result.features.rows = static_cast<uint64_t>(actual_len);
    result.features.columns = static_cast<uint64_t>(hidden);

    // Build the per-token metadata.
    result.tokens.reserve(static_cast<std::size_t>(actual_len));
    for (int32_t i = 0; i < actual_len; ++i) {
        trtmc::internal::FeatureToken tok;
        tok.token_id = ids[static_cast<std::size_t>(i)];
        tok.input_index = 0;
        tok.token_index = static_cast<uint64_t>(i);
        tok.has_byte_offsets = false;
        result.tokens.push_back(tok);
    }
    return result;
}

// ─── ITextToEmbedding (was: embedding) ───
// Mean-pools all token hidden states and L2-normalises the result.
trtmc::internal::SemanticEmbeddingResult
EncoderPipeline::run(const trtmc::internal::TextToEmbeddingRequest& request,
                     trtmc::internal::ConfigView /*config*/) {
    if (!tokenizer_)
        throw std::runtime_error("EncoderPipeline: no tokenizer configured");

    const std::string text_str(request.text);
    auto ids = tokenizer_->encode(text_str);
    const auto raw_floats = encode_ids(ids);

    trtmc::internal::SemanticEmbeddingResult result;
    result.pooling = "mean";
    result.normalization = "l2";

    const int32_t hidden = infer_output_hidden_dim(*encoder_);
    const auto actual_len = static_cast<int32_t>(ids.size());
    require_encoder_output(raw_floats, actual_len, hidden);

    result.values = mean_pool_and_normalize(raw_floats.data(), actual_len, hidden);
    return result;
}

// ─── ITextPairToRelevance (was: reranking) ───
// Combines query and document with the Nemotron rerank template, returns the
// first output scalar as a relevance score.
trtmc::internal::RelevanceResult
EncoderPipeline::run(const trtmc::internal::TextPairToRelevanceRequest& request,
                     trtmc::internal::ConfigView /*config*/) {
    if (!tokenizer_)
        throw std::runtime_error("EncoderPipeline: no tokenizer configured");

    // Match the text-only reranking template documented by the supported
    // Nemotron rerank cross-encoder model card.
    std::string combined =
        "question:" + std::string(request.query) + "   passage:" + std::string(request.document);
    auto ids = tokenizer_->encode(combined);
    const auto raw_floats = encode_ids(ids);

    trtmc::internal::RelevanceResult result;
    if (!raw_floats.empty())
        result.score = raw_floats[0];
    return result;
}

// ─── encode_ids (private helper) ───
std::vector<float> EncoderPipeline::encode_ids(const std::vector<int32_t>& input_ids) {
    const auto n = input_ids.size();
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

    std::vector<float> raw_floats;
    for (auto& [name, tensor] : outputs) {
        if (name.find("logits") != std::string::npos || name.find("embed") != std::string::npos ||
            name.find("output") != std::string::npos || name.find("hidden") != std::string::npos ||
            name.find("score") != std::string::npos) {
            auto count = tensor.numel();
            raw_floats.resize(static_cast<std::size_t>(count));
            std::memcpy(raw_floats.data(), tensor.data, count * sizeof(float));
            break;
        }
    }
    return raw_floats;
}

} // namespace trtmc
