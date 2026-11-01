/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/deberta_v2/runtime/pipeline.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

namespace trtmc {

namespace {

const Tensor& require_rows(const TensorMap& outputs, std::int32_t seq_len, std::int32_t hidden) {
    const auto found = outputs.find("last_hidden_state");
    if (found == outputs.end())
        throw std::runtime_error("deberta_v2 engine did not return last_hidden_state");
    const auto& tensor = found->second;
    const auto expected = static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(hidden);
    if (tensor.data == nullptr || tensor.dtype != DType::kFloat32 || tensor.numel() != expected)
        throw std::runtime_error("deberta_v2 engine returned unusable rows");
    return tensor;
}

} // namespace

DebertaV2Pipeline::DebertaV2Pipeline(std::unique_ptr<ITrtModule> encoder,
                                     std::shared_ptr<ITokenizer> tokenizer,
                                     DebertaV2RuntimeConfig config, std::string primary_task)
    : encoder_(std::move(encoder)), tokenizer_(std::move(tokenizer)), config_(config),
      primary_task_(std::move(primary_task)) {
    if (!encoder_ || !encoder_->ok())
        throw std::runtime_error("DebertaV2Pipeline: invalid model");
    if (!tokenizer_)
        throw std::runtime_error("DebertaV2Pipeline: missing tokenizer");
    if (config_.max_sequence_length < 2 || config_.hidden_size <= 0 || config_.pad_token_id < 0 ||
        config_.cls_token_id < 0 || config_.sep_token_id < 0) {
        throw std::runtime_error("deberta_v2 runtime config has invalid dimensions");
    }
}

std::vector<internal::TaskInstance> DebertaV2Pipeline::task_bindings() {
    return {internal::bind<internal::ITextToTokenFeatures>(*this)};
}

std::vector<std::int32_t> DebertaV2Pipeline::token_ids(const internal::TextSource& source) const {
    // Ids supplied by the caller are taken as given, framing included: the
    // caller that tokenized is the one that knows what it meant. Text is
    // tokenized and framed here, because this checkpoint's tokenizer.json
    // declares a post-processor with no special tokens, so nothing downstream
    // would add them.
    if (const auto* ids = std::get_if<Span<const std::int32_t>>(&source)) {
        if (ids->size() == 0)
            throw std::invalid_argument("deberta_v2 was given no token ids");
        return std::vector<std::int32_t>(ids->begin(), ids->end());
    }
    const auto text = std::get<std::string_view>(source);
    auto body = tokenizer_->encode(std::string(text));
    std::vector<std::int32_t> framed;
    framed.reserve(body.size() + 2);
    framed.push_back(config_.cls_token_id);
    framed.insert(framed.end(), body.begin(), body.end());
    framed.push_back(config_.sep_token_id);
    return framed;
}

internal::TokenFeaturesResult
DebertaV2Pipeline::run(const internal::TextToTokenFeaturesRequest& request,
                       internal::ConfigView config) {
    if (!config.empty())
        throw internal::ConfigError("deberta_v2 has no runtime configuration");

    auto ids = token_ids(request.text);
    const auto limit = static_cast<std::size_t>(config_.max_sequence_length);
    if (ids.size() > limit) {
        // Keep the closing [SEP]: the encoder is bidirectional, so losing it
        // changes every row rather than only the tail.
        const auto separator = ids.back();
        ids.resize(limit);
        ids.back() = separator;
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
    const auto& rows = require_rows(outputs, config_.max_sequence_length, config_.hidden_size);
    const auto* data = static_cast<const float*>(rows.data);

    const auto hidden = static_cast<std::size_t>(config_.hidden_size);
    internal::TokenFeaturesResult result;
    result.features.rows = static_cast<std::uint64_t>(real);
    result.features.columns = static_cast<std::uint64_t>(hidden);
    result.features.values.assign(data, data + static_cast<std::size_t>(real) * hidden);
    result.tokens.reserve(static_cast<std::size_t>(real));
    for (std::int32_t index = 0; index < real; ++index) {
        internal::FeatureToken token;
        token.token_id = ids[static_cast<std::size_t>(index)];
        token.input_index = 0;
        token.token_index = static_cast<std::uint64_t>(index);
        // The tokenizer works on a normalised copy of the input, so a byte
        // range into the caller's string would be a guess.
        token.has_byte_offsets = false;
        result.tokens.push_back(token);
    }
    return result;
}

} // namespace trtmc
