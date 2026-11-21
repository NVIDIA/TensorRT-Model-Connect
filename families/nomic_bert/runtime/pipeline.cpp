/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/nomic_bert/runtime/pipeline.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc {

namespace {

// This checkpoint is trained with a task prefix on every input, and the two
// prefixes put a query and a document into comparable positions. They are
// part of the model, not a convention the caller may vary, so they live here
// rather than being left for the caller to remember.
constexpr const char* kQueryPrefix = "search_query: ";
constexpr const char* kDocumentPrefix = "search_document: ";

std::string prefix_for(internal::EmbeddingRole role) {
    switch (role) {
    case internal::EmbeddingRole::Query:
        return kQueryPrefix;
    case internal::EmbeddingRole::Document:
        return kDocumentPrefix;
    case internal::EmbeddingRole::Default:
        break;
    }
    // There is no unprefixed mode. Embedding without a prefix returns a
    // vector that is neither a query nor a document and is not comparable
    // with either, so the caller is asked which one it meant instead of
    // being handed a plausible wrong answer.
    throw std::invalid_argument(
        "nomic_bert needs a role: this checkpoint embeds queries and documents differently");
}

const Tensor& require_rows(const TensorMap& outputs, std::int32_t seq_len, std::int32_t hidden) {
    const auto found = outputs.find("last_hidden_state");
    if (found == outputs.end())
        throw std::runtime_error("nomic_bert engine did not return last_hidden_state");
    const auto& tensor = found->second;
    const auto expected = static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(hidden);
    if (tensor.data == nullptr || tensor.dtype != DType::kFloat32 || tensor.numel() != expected)
        throw std::runtime_error("nomic_bert engine returned unusable rows");
    return tensor;
}

} // namespace

NomicBertPipeline::NomicBertPipeline(std::unique_ptr<ITrtModule> encoder,
                                     std::shared_ptr<ITokenizer> tokenizer,
                                     NomicBertRuntimeConfig config, std::string primary_task)
    : encoder_(std::move(encoder)), tokenizer_(std::move(tokenizer)), config_(config),
      primary_task_(std::move(primary_task)) {
    if (!encoder_ || !encoder_->ok())
        throw std::runtime_error("NomicBertPipeline: invalid model");
    if (!tokenizer_)
        throw std::runtime_error("NomicBertPipeline: missing tokenizer");
    if (config_.max_sequence_length < 2 || config_.hidden_size <= 0 || config_.pad_token_id < 0)
        throw std::runtime_error("nomic_bert runtime config has invalid dimensions");
}

std::vector<internal::TaskInstance> NomicBertPipeline::task_bindings() {
    return {internal::bind<internal::ITextToEmbedding>(*this)};
}

std::vector<float> NomicBertPipeline::embed(std::string_view text, internal::EmbeddingRole role) {
    auto ids = tokenizer_->encode(prefix_for(role) + std::string(text));
    if (ids.empty())
        throw std::runtime_error("nomic_bert tokenizer returned no tokens");

    const auto limit = static_cast<std::size_t>(config_.max_sequence_length);
    const auto separator = ids.back();
    if (ids.size() > limit) {
        // Keep the closing [SEP]: the encoder is bidirectional, so dropping
        // it changes every row rather than only the tail.
        ids.resize(limit);
        ids.back() = separator;
    }
    const auto real = static_cast<std::int32_t>(ids.size());
    std::vector<std::int32_t> mask(ids.size(), 1);
    ids.resize(limit, config_.pad_token_id);
    mask.resize(limit, 0);

    Tensor input_ids;
    input_ids.data = ids.data();
    input_ids.shape = {config_.max_sequence_length};
    input_ids.dtype = DType::kInt32;
    Tensor attention_mask;
    attention_mask.data = mask.data();
    attention_mask.shape = {config_.max_sequence_length};
    attention_mask.dtype = DType::kInt32;

    auto outputs =
        encoder_->forward({{"input_ids", input_ids}, {"attention_mask", attention_mask}});
    const auto& rows = require_rows(outputs, config_.max_sequence_length, config_.hidden_size);
    const auto* data = static_cast<const float*>(rows.data);

    // Mean over the real rows only. The padded rows are masked out of the
    // attention, so they carry no information, but they are not zero and
    // averaging them in would still move the result.
    const auto hidden = static_cast<std::size_t>(config_.hidden_size);
    std::vector<double> total(hidden, 0.0);
    for (std::int32_t row = 0; row < real; ++row) {
        const auto* source = data + static_cast<std::size_t>(row) * hidden;
        for (std::size_t column = 0; column < hidden; ++column)
            total[column] += source[column];
    }
    std::vector<float> pooled(hidden);
    for (std::size_t column = 0; column < hidden; ++column)
        pooled[column] = static_cast<float>(total[column] / static_cast<double>(real));
    return pooled;
}

internal::SemanticEmbeddingResult
NomicBertPipeline::run(const internal::TextToEmbeddingRequest& request,
                       internal::ConfigView config) {
    if (!config.empty())
        throw internal::ConfigError("nomic_bert has no runtime configuration");
    internal::SemanticEmbeddingResult result;
    result.values = embed(request.text, request.role);
    // Left empty on purpose. Two nomic-embed checkpoints share this family
    // and do not share a space, and nothing in either checkpoint names the
    // one it belongs to. A name here would claim their vectors can be
    // compared with each other.
    result.embedding_space = "";
    result.pooling = "mean";
    // The published module list for this checkpoint is Transformer then
    // Pooling, with no Normalize step, so the vector is returned with its
    // length intact and the caller normalises if it wants to.
    result.normalization = "none";
    return result;
}

} // namespace trtmc
