/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/mobilebert/runtime/pipeline.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

namespace trtmc {

namespace {

const Tensor& require_output(const TensorMap& outputs, const char* name, std::size_t count) {
    const auto found = outputs.find(name);
    if (found == outputs.end())
        throw std::runtime_error(std::string("mobilebert engine did not return ") + name);
    const auto& tensor = found->second;
    if (tensor.data == nullptr || tensor.dtype != DType::kFloat32 || tensor.numel() != count)
        throw std::runtime_error(std::string("mobilebert engine returned an unusable ") + name);
    return tensor;
}

Tensor int32_input(std::vector<std::int32_t>& values) {
    Tensor tensor;
    tensor.data = values.data();
    tensor.shape = {static_cast<std::int64_t>(values.size())};
    tensor.dtype = DType::kInt32;
    return tensor;
}

} // namespace

MobileBertPipeline::MobileBertPipeline(std::unique_ptr<ITrtModule> engine,
                                       std::shared_ptr<IMobileBertTokenizer> tokenizer,
                                       MobileBertRuntimeConfig config, std::string primary_task)
    : engine_(std::move(engine)), tokenizer_(std::move(tokenizer)), config_(std::move(config)),
      primary_task_(std::move(primary_task)) {
    if (!engine_ || !engine_->ok())
        throw std::runtime_error("MobileBertPipeline: invalid engine");
    if (!tokenizer_)
        throw std::runtime_error("MobileBertPipeline: missing tokenizer");
    const auto valid_id = [this](std::int32_t id) { return id >= 0 && id < config_.vocab_size; };
    if (config_.max_sequence_length < 3 || config_.vocab_size <= 0 || config_.type_vocab_size < 2 ||
        !valid_id(config_.pad_token_id) || !valid_id(config_.unk_token_id) ||
        !valid_id(config_.cls_token_id) || !valid_id(config_.sep_token_id) ||
        !valid_id(config_.mask_token_id) || config_.vocabulary_id.empty() ||
        config_.relation_labels.size() != 2) {
        throw std::runtime_error("mobilebert runtime config has invalid dimensions or ids");
    }
}

std::vector<internal::TaskInstance> MobileBertPipeline::task_bindings() {
    return {internal::bind<internal::IMaskedTextToTokenScores>(*this),
            internal::bind<internal::ITextPairToPretrainingRelationScores>(*this)};
}

std::vector<std::int32_t> MobileBertPipeline::single_ids(const internal::TextSource& source) const {
    // Ids supplied by the caller are taken as given, framing included: the caller
    // that tokenized knows what it meant. They index the embedding table, so an
    // id outside the vocabulary is refused instead of read out of bounds.
    if (const auto* ids = std::get_if<Span<const std::int32_t>>(&source)) {
        if (ids->size() == 0)
            throw std::invalid_argument("mobilebert was given no token ids");
        for (const auto id : *ids) {
            if (id < 0 || id >= config_.vocab_size)
                throw std::invalid_argument("mobilebert token id outside the vocabulary");
        }
        return {ids->begin(), ids->end()};
    }
    // [CLS] and [SEP] take two places of the window, so the text is read only
    // until it overflows the rest; forward() then refuses it.
    const auto room = static_cast<std::size_t>(config_.max_sequence_length) - 2;
    const auto body = tokenizer_->encode(std::get<std::string_view>(source), room);
    std::vector<std::int32_t> framed;
    framed.reserve(body.size() + 2);
    framed.push_back(config_.cls_token_id);
    framed.insert(framed.end(), body.begin(), body.end());
    framed.push_back(config_.sep_token_id);
    return framed;
}

MobileBertPipeline::Scores MobileBertPipeline::forward(const std::vector<std::int32_t>& ids,
                                                       const std::vector<std::int32_t>& segments,
                                                       const std::vector<std::size_t>& rows) {
    const auto window = static_cast<std::size_t>(config_.max_sequence_length);
    if (ids.size() > window) {
        throw std::invalid_argument("mobilebert input is longer than the engine window of " +
                                    std::to_string(window) + " tokens");
    }
    // The engine is fixed-length and the runtime keeps its input buffers between
    // calls, so the whole window is sent every time: ids padded with [PAD],
    // segment 0 and attention mask 0 beyond the real tokens.
    std::vector<std::int32_t> input_ids(window, config_.pad_token_id);
    std::vector<std::int32_t> token_type_ids(window, 0);
    std::vector<std::int32_t> attention_mask(window, 0);
    std::copy(ids.begin(), ids.end(), input_ids.begin());
    std::copy(segments.begin(), segments.end(), token_type_ids.begin());
    std::fill_n(attention_mask.begin(), ids.size(), 1);

    const auto outputs = engine_->forward({{"input_ids", int32_input(input_ids)},
                                           {"token_type_ids", int32_input(token_type_ids)},
                                           {"attention_mask", int32_input(attention_mask)}});
    const auto vocab = static_cast<std::size_t>(config_.vocab_size);
    const auto& logits = require_output(outputs, "prediction_logits", window * vocab);
    const auto& relation = require_output(outputs, "seq_relationship_logits", 2);

    Scores scores;
    const auto* relation_data = static_cast<const float*>(relation.data);
    scores.relation = {relation_data[0], relation_data[1]};
    // Only the requested rows are copied out of the module output buffer.
    const auto* data = static_cast<const float*>(logits.data);
    scores.token_logits.reserve(rows.size() * vocab);
    for (const auto row : rows)
        scores.token_logits.insert(scores.token_logits.end(), data + row * vocab,
                                   data + (row + 1) * vocab);
    return scores;
}

internal::VocabularyScoresResult
MobileBertPipeline::run(const internal::MaskedTextToTokenScoresRequest& request,
                        internal::ConfigView config) {
    if (!config.empty())
        throw internal::ConfigError("mobilebert has no runtime configuration");
    const bool framed_here = std::holds_alternative<std::string_view>(request.text);
    const auto ids = single_ids(request.text);

    // The [MASK] positions are what this Task is named for. With none marked the
    // whole sequence is returned rather than nothing, because an empty matrix
    // would be indistinguishable from a model that failed to run.
    std::vector<std::size_t> selected;
    for (std::size_t index = 0; index < ids.size(); ++index) {
        if (ids[index] == config_.mask_token_id)
            selected.push_back(index);
    }
    if (selected.empty()) {
        for (std::size_t index = 0; index < ids.size(); ++index)
            selected.push_back(index);
    }
    auto scores = forward(ids, std::vector<std::int32_t>(ids.size(), 0), selected);

    internal::VocabularyScoresResult result;
    result.vocabulary_id = config_.vocabulary_id;
    result.logits.rows = selected.size();
    result.logits.columns = static_cast<std::uint64_t>(config_.vocab_size);
    result.logits.values = std::move(scores.token_logits);
    result.positions.reserve(selected.size());
    for (const auto index : selected) {
        internal::FeatureToken token;
        token.token_id = ids[index];
        // [CLS] and [SEP] added here are inserted tokens; caller ids are all input.
        const bool inserted = framed_here && (index == 0 || index + 1 == ids.size());
        token.input_index = inserted ? -1 : 0;
        token.token_index = index;
        // The tokenizer lowercases, strips accents and drops whitespace, so a byte
        // range into the caller's string would be a reconstruction, not a fact.
        token.has_byte_offsets = false;
        result.positions.push_back(token);
    }
    return result;
}

internal::LabelScoresResult
MobileBertPipeline::run(const internal::TextPairToPretrainingRelationScoresRequest& request,
                        internal::ConfigView config) {
    if (!config.empty())
        throw internal::ConfigError("mobilebert has no runtime configuration");
    // [CLS] A [SEP] B [SEP]: three places of the window are framing. Each text is
    // read only until the pair overflows the rest; forward() then refuses it.
    const auto room = static_cast<std::size_t>(config_.max_sequence_length) - 3;
    const auto first = tokenizer_->encode(request.first, room);
    const auto second = first.size() > room
                            ? std::vector<std::int32_t>{}
                            : tokenizer_->encode(request.second, room - first.size());
    std::vector<std::int32_t> ids;
    std::vector<std::int32_t> segments;
    ids.reserve(first.size() + second.size() + 3);
    ids.push_back(config_.cls_token_id);
    ids.insert(ids.end(), first.begin(), first.end());
    ids.push_back(config_.sep_token_id);
    segments.assign(ids.size(), 0);
    ids.insert(ids.end(), second.begin(), second.end());
    ids.push_back(config_.sep_token_id);
    segments.resize(ids.size(), 1);

    const auto scores = forward(ids, segments, {});
    internal::LabelScoresResult result;
    result.scores.assign(scores.relation.begin(), scores.relation.end());
    result.labels = config_.relation_labels;
    result.kind = internal::ScoreKind::Logit;
    result.vocabulary_id = config_.relation_vocabulary_id;
    return result;
}

} // namespace trtmc
