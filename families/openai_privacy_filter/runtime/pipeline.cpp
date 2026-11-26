/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/openai_privacy_filter/runtime/pipeline.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

namespace trtmc::openai_privacy_filter {
namespace {

constexpr const char* kInput = "input_ids";
constexpr const char* kOutput = "logits";

} // namespace

TokenClassifier::TokenClassifier(std::unique_ptr<ITrtModule> engine, Encoder encode,
                                 RuntimeConfig config)
    : engine_(std::move(engine)), encode_(std::move(encode)), config_(config) {
    if (!engine_ || !engine_->ok() || !encode_)
        throw std::invalid_argument("openai_privacy_filter requires an engine and a tokenizer");
    if (config_.vocab_size <= 0 || config_.max_sequence_length <= 0 || config_.num_labels <= 0)
        throw std::invalid_argument("openai_privacy_filter runtime bounds must be positive");
    if (!engine_->input_is_dynamic(kInput) || engine_->tensor_dtype(kInput) != DType::kInt32)
        throw std::invalid_argument("openai_privacy_filter requires a dynamic int32 input_ids");
    const auto capacity =
        engine_->input_profile_shape(kInput, engine_->profile_idx(), ProfileShapeSelector::kMax);
    if (capacity.size() != 1 || capacity[0] < config_.max_sequence_length)
        throw std::invalid_argument(
            "openai_privacy_filter engine profile is below the bundle bound");
    const auto shape = engine_->tensor_shape(kOutput);
    if (shape.size() != 2 || shape[1] != config_.num_labels ||
        engine_->tensor_dtype(kOutput) != DType::kFloat32)
        throw std::invalid_argument("openai_privacy_filter requires FP32 logits [tokens, labels]");
}

std::vector<internal::TaskInstance> TokenClassifier::task_bindings() {
    return {internal::bind<internal::ITextToHeadScores>(*this)};
}

std::vector<std::int32_t> TokenClassifier::resolve_ids(const internal::TextSource& source) const {
    std::vector<std::int32_t> ids;
    if (const auto* text = std::get_if<std::string_view>(&source)) {
        ids = encode_(*text);
    } else {
        const auto given = std::get<Span<const std::int32_t>>(source);
        ids.assign(given.begin(), given.end());
    }
    if (ids.empty())
        throw std::invalid_argument("openai_privacy_filter input has no tokens");
    if (ids.size() > static_cast<std::uint64_t>(config_.max_sequence_length))
        throw std::invalid_argument(
            "openai_privacy_filter input has " + std::to_string(ids.size()) +
            " tokens; this bundle accepts at most " + std::to_string(config_.max_sequence_length));
    for (const auto id : ids)
        if (id < 0 || id >= config_.vocab_size)
            throw std::invalid_argument("openai_privacy_filter token ID is outside the vocabulary");
    return ids;
}

internal::HeadScoresResult TokenClassifier::run(const internal::TextToHeadScoresRequest& request,
                                                internal::ConfigView config) {
    if (!config.empty())
        throw internal::ConfigError("openai_privacy_filter exposes no runtime configuration");
    auto ids = resolve_ids(request.text);
    const auto length = static_cast<std::int64_t>(ids.size());
    const auto outputs = engine_->forward({{kInput, {ids.data(), {length}, DType::kInt32}}});
    const auto found = outputs.find(kOutput);
    if (found == outputs.end())
        throw std::runtime_error("openai_privacy_filter engine returned no logits");
    const auto& logits = found->second;
    if (logits.data == nullptr || logits.dtype != DType::kFloat32 ||
        logits.shape != std::vector<std::int64_t>{length, config_.num_labels})
        throw std::runtime_error("openai_privacy_filter engine returned invalid logits");
    std::vector<float> values(logits.numel());
    std::memcpy(values.data(), logits.data, values.size() * sizeof(float));
    return {std::move(values),
            {static_cast<std::uint64_t>(length), static_cast<std::uint64_t>(config_.num_labels)},
            internal::ScoreKind::Logit,
            "none",
            "none"};
}

} // namespace trtmc::openai_privacy_filter
