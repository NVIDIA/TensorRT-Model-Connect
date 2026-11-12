/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/esm/runtime/pipeline.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

namespace trtmc {

namespace {

const Tensor& require_logits(const TensorMap& outputs, std::int32_t seq_len, std::int32_t vocab) {
    const auto found = outputs.find("logits");
    if (found == outputs.end())
        throw std::runtime_error("esm engine did not return logits");
    const auto& tensor = found->second;
    const auto expected = static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(vocab);
    if (tensor.data == nullptr || tensor.dtype != DType::kFloat32 || tensor.numel() != expected)
        throw std::runtime_error("esm engine returned unusable logits");
    return tensor;
}

} // namespace

EsmPipeline::EsmPipeline(std::unique_ptr<ITrtModule> encoder,
                         std::shared_ptr<IEsmTokenizer> tokenizer, EsmRuntimeConfig config,
                         std::string primary_task)
    : encoder_(std::move(encoder)), tokenizer_(std::move(tokenizer)), config_(std::move(config)),
      primary_task_(std::move(primary_task)) {
    if (!encoder_ || !encoder_->ok())
        throw std::runtime_error("EsmPipeline: invalid model");
    if (!tokenizer_)
        throw std::runtime_error("EsmPipeline: missing tokenizer");
    if (config_.max_sequence_length < 2 || config_.vocab_size <= 0 || config_.pad_token_id < 0 ||
        config_.cls_token_id < 0 || config_.eos_token_id < 0 || config_.mask_token_id < 0 ||
        config_.unk_token_id < 0) {
        throw std::runtime_error("esm runtime config has invalid dimensions");
    }
}

std::vector<internal::TaskInstance> EsmPipeline::task_bindings() {
    return {internal::bind<internal::IMaskedTextToTokenScores>(*this)};
}

std::vector<std::int32_t> EsmPipeline::token_ids(const internal::TextSource& source) const {
    // Ids supplied by the caller are taken as given, framing included: the
    // caller that tokenized is the one that knows what it meant. A residue
    // string is tokenized and framed here with <cls> and <eos>, which is the
    // framing this checkpoint was trained with.
    if (const auto* ids = std::get_if<Span<const std::int32_t>>(&source)) {
        if (ids->size() == 0)
            throw std::invalid_argument("esm was given no token ids");
        return std::vector<std::int32_t>(ids->begin(), ids->end());
    }
    const auto text = std::get<std::string_view>(source);
    auto body = tokenizer_->encode(std::string(text));
    if (body.empty())
        throw std::invalid_argument("esm was given no residues to score");
    std::vector<std::int32_t> framed;
    framed.reserve(body.size() + 2);
    framed.push_back(config_.cls_token_id);
    framed.insert(framed.end(), body.begin(), body.end());
    framed.push_back(config_.eos_token_id);
    return framed;
}

internal::VocabularyScoresResult
EsmPipeline::run(const internal::MaskedTextToTokenScoresRequest& request,
                 internal::ConfigView config) {
    if (!config.empty())
        throw internal::ConfigError("esm has no runtime configuration");

    auto ids = token_ids(request.text);
    const auto limit = static_cast<std::size_t>(config_.max_sequence_length);
    if (ids.size() > limit) {
        // Keep the closing <eos>: the encoder is bidirectional, so losing it
        // changes every row rather than only the tail.
        const auto terminator = ids.back();
        ids.resize(limit);
        ids.back() = terminator;
    }
    const auto real = static_cast<std::int32_t>(ids.size());
    std::vector<std::int32_t> mask(ids.size(), 1);
    std::vector<std::int32_t> padded = ids;
    padded.resize(limit, config_.pad_token_id);
    mask.resize(limit, 0);

    Tensor input_ids;
    input_ids.data = padded.data();
    input_ids.shape = {config_.max_sequence_length};
    input_ids.dtype = DType::kInt32;
    Tensor attention_mask;
    attention_mask.data = mask.data();
    attention_mask.shape = {config_.max_sequence_length};
    attention_mask.dtype = DType::kInt32;

    auto outputs =
        encoder_->forward({{"input_ids", input_ids}, {"attention_mask", attention_mask}});
    const auto& scores = require_logits(outputs, config_.max_sequence_length, config_.vocab_size);
    const auto* data = static_cast<const float*>(scores.data);

    // The masked positions are what this Task is named for. With none marked
    // the whole sequence is returned rather than nothing, because an empty
    // matrix would be indistinguishable from a model that failed to run.
    std::vector<std::int32_t> selected;
    for (std::int32_t index = 0; index < real; ++index) {
        if (ids[static_cast<std::size_t>(index)] == config_.mask_token_id)
            selected.push_back(index);
    }
    if (selected.empty()) {
        selected.reserve(static_cast<std::size_t>(real));
        for (std::int32_t index = 0; index < real; ++index)
            selected.push_back(index);
    }

    const auto vocab = static_cast<std::size_t>(config_.vocab_size);
    internal::VocabularyScoresResult result;
    result.vocabulary_id = config_.vocabulary_id;
    result.logits.rows = static_cast<std::uint64_t>(selected.size());
    result.logits.columns = static_cast<std::uint64_t>(vocab);
    result.logits.values.reserve(selected.size() * vocab);
    result.positions.reserve(selected.size());
    for (const auto index : selected) {
        const auto* row = data + static_cast<std::size_t>(index) * vocab;
        result.logits.values.insert(result.logits.values.end(), row, row + vocab);
        internal::FeatureToken token;
        token.token_id = ids[static_cast<std::size_t>(index)];
        token.input_index = 0;
        token.token_index = static_cast<std::uint64_t>(index);
        // The tokenizer drops whitespace, so a byte range into the caller's
        // string would be a guess.
        token.has_byte_offsets = false;
        result.positions.push_back(token);
    }
    return result;
}

} // namespace trtmc
