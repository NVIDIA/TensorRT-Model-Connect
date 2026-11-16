/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/llama/runtime/forward.h"

#include "families/llama/runtime/plugin_helpers.h"

#include <limits>
#include <nlohmann/json.hpp>

namespace trtmc::llama {
namespace {
const internal::ConfigField fields[] = {
    {"causal", internal::ConfigKind::Bool, internal::ConfigValue{true},
     "Apply a causal triangle in addition to caller masks (default true)."},
    {"position_ids", internal::ConfigKind::I64List, std::nullopt,
     "RoPE positions, one per input token; default 0..L-1."},
};
}

ForwardPipeline::ForwardPipeline(std::unique_ptr<ITrtModule> module, int64_t vocab_size,
                                 int64_t max_sequence_length, std::string vocabulary_id)
    : module_(std::move(module)), vocab_size_(vocab_size),
      max_sequence_length_(max_sequence_length), vocabulary_id_(std::move(vocabulary_id)) {
    if (!module_ || !module_->ok() || vocabulary_id_.empty() || vocab_size_ <= 0 ||
        vocab_size_ > std::numeric_limits<int32_t>::max() || max_sequence_length_ <= 0 ||
        max_sequence_length_ > std::numeric_limits<int32_t>::max())
        throw std::runtime_error("invalid Llama forward bundle");
}

std::vector<internal::TaskInstance> ForwardPipeline::task_bindings() {
    return {internal::bind<internal::ITextPredictionPositionsToTokenScores>(*this, fields)};
}

internal::VocabularyScoresResult
ForwardPipeline::run(const internal::TextPredictionPositionsToTokenScoresRequest& request,
                     internal::ConfigView config) {
    internal::validate_config(fields, config);
    const auto length = request.token_ids.size();
    if (length == 0 || length > static_cast<size_t>(max_sequence_length_))
        throw std::invalid_argument("Llama forward requires 1..max_sequence_length tokens");
    if (!request.segment_ids.empty())
        throw std::invalid_argument("Llama forward does not use segment IDs");
    if ((!request.attention_mask.empty() && request.attention_mask.size() != length) ||
        (request.blocked_attention.size() / length != length) ||
        (request.blocked_attention.size() % length != 0))
        throw std::invalid_argument("Llama forward mask shape mismatch");
    bool causal = true;
    Span<const int64_t> explicit_positions;
    for (const auto& entry : config) {
        if (entry.name == "causal")
            causal = internal::config_value_as<bool>(entry.value);
        else if (entry.name == "position_ids")
            explicit_positions = internal::config_value_as<Span<const int64_t>>(entry.value);
    }
    if (!explicit_positions.empty() && explicit_positions.size() != length)
        throw std::invalid_argument("Llama forward position_ids must have one entry per token");
    std::vector<int32_t> ids(length), positions(length);
    for (size_t i = 0; i < length; ++i) {
        const auto token = request.token_ids[i];
        const auto position =
            explicit_positions.empty() ? static_cast<int64_t>(i) : explicit_positions[i];
        if (token < 0 || token >= vocab_size_ || position < 0 || position >= max_sequence_length_)
            throw std::invalid_argument("Llama forward token or position outside bundle bounds");
        ids[i] = static_cast<int32_t>(token);
        positions[i] = static_cast<int32_t>(position);
    }
    for (auto mask : request.attention_mask)
        if (mask > 1)
            throw std::invalid_argument("Llama forward attention_mask must contain 0 or 1");
    for (auto mask : request.blocked_attention)
        if (mask > 1)
            throw std::invalid_argument("Llama forward blocked_attention must contain 0 or 1");
    if (request.prediction_positions.empty())
        throw std::invalid_argument("Llama forward requires prediction positions");
    for (auto row : request.prediction_positions)
        if (row >= length)
            throw std::invalid_argument("Llama forward prediction position outside input");

    // No cache inputs, outputs, or persistent state. Masked query rows are still
    // returned; padding masks exclude keys, matching the HF attention convention.
    std::vector<float> mask(length * length);
    for (size_t q = 0; q < length; ++q) {
        bool has_key = false;
        for (size_t k = 0; k < length; ++k) {
            const bool blocked =
                (causal && k > q) ||
                (!request.attention_mask.empty() && !request.attention_mask[k]) ||
                (!request.blocked_attention.empty() && request.blocked_attention[q * length + k]);
            mask[q * length + k] = blocked ? -std::numeric_limits<float>::infinity() : 0.0F;
            has_key |= !blocked;
        }
        if (!has_key)
            throw std::invalid_argument("Llama forward requires an unblocked key in every row");
    }
    auto outputs = module_->forward({
        {"token_id", {ids.data(), {static_cast<int64_t>(length)}, DType::kInt32}},
        {"position_id", {positions.data(), {static_cast<int64_t>(length)}, DType::kInt32}},
        {"attention_mask",
         {mask.data(),
          {static_cast<int64_t>(length), static_cast<int64_t>(length)},
          DType::kFloat32}},
    });
    const auto& logits = outputs.at("logits");
    if (logits.dtype != DType::kFloat32 ||
        logits.shape != std::vector<int64_t>{static_cast<int64_t>(length), vocab_size_} ||
        !logits.data)
        throw std::runtime_error("Llama full-forward engine returned invalid logits");
    internal::VocabularyScoresResult result;
    result.vocabulary_id = vocabulary_id_;
    result.logits.columns = static_cast<uint64_t>(vocab_size_);
    result.logits.rows = request.prediction_positions.size();
    result.logits.values.reserve(result.logits.rows * result.logits.columns);
    for (auto row : request.prediction_positions) {
        const auto* first = static_cast<const float*>(logits.data) + row * vocab_size_;
        result.logits.values.insert(result.logits.values.end(), first, first + vocab_size_);
        result.positions.push_back({request.token_ids[row], 0, row, false, 0, 0});
    }
    return result;
}

ITask* create_forward(const FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("stateless Llama forward has no KV cache");
    const auto data = context.reader.read_section("runtime.json");
    const auto config = nlohmann::json::parse(data.begin(), data.end());
    const auto plan = context.reader.read_section("engine.plan");
    auto module = load_engine(context.backend, plan, "engine.plan");
    return new ForwardPipeline(std::move(module), config.at("vocab_size").get<int64_t>(),
                               config.at("max_sequence_length").get<int64_t>(),
                               config.at("vocabulary_id").get<std::string>());
}

} // namespace trtmc::llama
