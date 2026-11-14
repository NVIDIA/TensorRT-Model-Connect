/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/gte/runtime/pipeline.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc {

namespace {

const Tensor& require_rows(const TensorMap& outputs, std::int32_t seq_len, std::int32_t hidden) {
    const auto found = outputs.find("last_hidden_state");
    if (found == outputs.end())
        throw std::runtime_error("gte engine did not return last_hidden_state");
    const auto& tensor = found->second;
    const auto expected = static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(hidden);
    if (tensor.data == nullptr || tensor.dtype != DType::kFloat32 || tensor.numel() != expected)
        throw std::runtime_error("gte engine returned unusable rows");
    return tensor;
}

} // namespace

GtePipeline::GtePipeline(std::unique_ptr<ITrtModule> encoder, std::shared_ptr<ITokenizer> tokenizer,
                         GteRuntimeConfig config, std::string primary_task)
    : encoder_(std::move(encoder)), tokenizer_(std::move(tokenizer)), config_(config),
      primary_task_(std::move(primary_task)) {
    if (!encoder_ || !encoder_->ok())
        throw std::runtime_error("GtePipeline: invalid model");
    if (!tokenizer_)
        throw std::runtime_error("GtePipeline: missing tokenizer");
    if (config_.max_sequence_length < 2 || config_.hidden_size <= 0 || config_.pad_token_id < 0)
        throw std::runtime_error("gte runtime config has invalid dimensions");
}

std::vector<internal::TaskInstance> GtePipeline::task_bindings() {
    return {internal::bind<internal::ITextToEmbedding>(*this)};
}

std::vector<float> GtePipeline::embed(std::string_view text) {
    auto ids = tokenizer_->encode(std::string(text));
    if (ids.empty())
        throw std::runtime_error("gte tokenizer returned no tokens");

    const auto limit = static_cast<std::size_t>(config_.max_sequence_length);
    const auto separator = ids.back();
    if (ids.size() > limit) {
        // Keep the closing [SEP]: the encoder is bidirectional, so dropping
        // it changes every row rather than only the tail - including row 0,
        // which is the one that becomes the embedding.
        ids.resize(limit);
        ids.back() = separator;
    }
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

    // Row 0, the [CLS] row, and nothing else. 1_Pooling/config.json sets
    // pooling_mode_cls_token and clears every other mode, so mean pooling -
    // which is what the rotary encoder next door does and what a reader
    // coming from BERT expects - is the wrong vector here, not a variant.
    const auto hidden = static_cast<std::size_t>(config_.hidden_size);
    return std::vector<float>(data, data + hidden);
}

internal::SemanticEmbeddingResult GtePipeline::run(const internal::TextToEmbeddingRequest& request,
                                                   internal::ConfigView config) {
    if (!config.empty())
        throw internal::ConfigError("gte has no runtime configuration");
    // There is no query/document asymmetry in this checkpoint: no prompt
    // template ships with it, the published module list is Transformer then
    // Pooling with no prompt, and the README embeds both sides with the
    // same call. Accepting a role would claim this family does something
    // with it, so a role is refused rather than silently ignored.
    if (request.role != internal::EmbeddingRole::Default)
        throw std::invalid_argument(
            "gte embeds queries and documents identically: this checkpoint ships no role prompt");

    internal::SemanticEmbeddingResult result;
    result.values = embed(request.text);
    // Left empty on purpose. The base, large and small GTE-v1.5 checkpoints
    // share this family and do not share a space, and nothing in any of
    // them names the one it belongs to.
    result.embedding_space = "";
    result.pooling = "cls";
    // modules.json lists Transformer then Pooling and no Normalize step, so
    // the vector is returned with its length intact. The README's own
    // snippet calls F.normalize afterwards; that is the caller's step, and
    // doing it here would make this report a contract the checkpoint does
    // not publish.
    result.normalization = "none";
    return result;
}

} // namespace trtmc
