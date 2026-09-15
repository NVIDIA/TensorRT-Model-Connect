/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/hstu/runtime/pipeline.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

namespace trtmc::hstu {
namespace {

struct Sequence {
    std::vector<std::int32_t> tokens;
    std::int32_t contextual_length{0};
    std::int32_t history_end{0};
    std::int32_t candidates{0};
    std::int32_t query_position{0};
};

const EmbeddingTable* find_role(const RuntimeConfig& config, const char* role) {
    const EmbeddingTable* found = nullptr;
    for (const auto& table : config.embedding_tables) {
        if (table.role != role)
            continue;
        if (found != nullptr)
            throw std::invalid_argument(std::string("hstu requires at most one ") + role +
                                        " table");
        found = &table;
    }
    return found;
}

std::int32_t lookup(const EmbeddingTable& table, std::int64_t id) {
    std::int64_t row = id;
    if (!table.keys.empty()) {
        const auto found = std::lower_bound(table.keys.begin(), table.keys.end(), id);
        if (found == table.keys.end() || *found != id)
            throw std::invalid_argument("hstu unknown ID in embedding table " + table.name);
        row = std::distance(table.keys.begin(), found);
    }
    if (row < 0 || row >= table.num_embeddings)
        throw std::invalid_argument("hstu ID outside embedding table " + table.name);
    return table.offset + static_cast<std::int32_t>(row);
}

void append_context(const RecommendationSequence& request, const RuntimeConfig& config,
                    Sequence& sequence) {
    std::set<std::string> supplied;
    for (const auto& feature : request.contextual_features) {
        const auto table =
            std::find_if(config.embedding_tables.begin(), config.embedding_tables.end(),
                         [&](const auto& t) { return t.name == feature.name; });
        if (table == config.embedding_tables.end() || table->role != "context")
            throw std::invalid_argument("hstu unknown contextual feature " + feature.name);
        if (!supplied.insert(feature.name).second)
            throw std::invalid_argument("hstu duplicate contextual feature " + feature.name);
    }
    // Checkpoint table order is the reference feature order; request order cannot change it.
    for (const auto& table : config.embedding_tables) {
        if (table.role != "context")
            continue;
        for (const auto& feature : request.contextual_features) {
            if (feature.name == table.name) {
                for (const auto id : feature.ids)
                    sequence.tokens.push_back(lookup(table, id));
            }
        }
    }
    sequence.contextual_length = static_cast<std::int32_t>(sequence.tokens.size());
}

std::size_t sequence_length(const RecommendationSequence& request, bool include_candidates) {
    std::size_t count = request.history_item_ids.size() + request.history_action_ids.size();
    if (include_candidates)
        count += request.candidate_item_ids.size();
    for (const auto& feature : request.contextual_features) {
        if (feature.ids.size() > std::numeric_limits<std::size_t>::max() - count)
            throw std::invalid_argument("hstu sequence length overflow");
        count += feature.ids.size();
    }
    return count;
}

void validate_sequence(const RecommendationSequence& request, const RuntimeConfig& config) {
    const auto count = sequence_length(request, config.mode == "ranking");
    if (count == 0 || count > static_cast<std::size_t>(config.max_sequence_length))
        throw std::invalid_argument("hstu sequence length is outside the built profile");
    if (request.candidate_item_ids.size() > static_cast<std::size_t>(config.max_sequence_length))
        throw std::invalid_argument("hstu candidate count is outside the built profile");
    const auto* action = find_role(config, "action");
    if (action == nullptr && !request.history_action_ids.empty())
        throw std::invalid_argument("hstu bundle has no action embedding table");
    if (action != nullptr && request.history_action_ids.size() != request.history_item_ids.size())
        throw std::invalid_argument("hstu requires one history action per history item");
    if (config.mode == "retrieval" && request.history_item_ids.empty())
        throw std::invalid_argument("hstu retrieval requires history items");
}

Sequence assemble(const RecommendationSequence& request, const RuntimeConfig& config) {
    validate_sequence(request, config);
    const auto* item = find_role(config, "item");
    const auto* action = find_role(config, "action");
    Sequence sequence;
    sequence.tokens.reserve(sequence_length(request, config.mode == "ranking"));
    append_context(request, config, sequence);
    for (std::size_t index = 0; index < request.history_item_ids.size(); ++index) {
        sequence.tokens.push_back(lookup(*item, request.history_item_ids[index]));
        if (action != nullptr)
            sequence.tokens.push_back(lookup(*action, request.history_action_ids[index]));
    }
    sequence.history_end = static_cast<std::int32_t>(sequence.tokens.size());
    if (config.mode == "retrieval") {
        const auto stride = action == nullptr ? 1 : 2;
        sequence.query_position =
            sequence.contextual_length +
            stride * static_cast<std::int32_t>(request.history_item_ids.size() - 1);
    }
    sequence.candidates = static_cast<std::int32_t>(request.candidate_item_ids.size());
    if (config.mode == "ranking") {
        for (const auto id : request.candidate_item_ids)
            sequence.tokens.push_back(lookup(*item, id));
    }
    return sequence;
}

std::int32_t contextual_position(std::int32_t index, std::int32_t contextual) {
    return contextual > 0 ? std::max(index - contextual + 1, 0) : index;
}

bool target_attention_allowed(std::int32_t row, std::int32_t column, std::int32_t history_end,
                              std::int32_t group_size) {
    const auto row_group = row < history_end ? -1 : (row - history_end) / group_size;
    const auto col_group = column < history_end ? -1 : (column - history_end) / group_size;
    return row_group == col_group || row_group < 0 || col_group < 0;
}

bool attention_allowed(std::int32_t row, std::int32_t column, const Sequence& sequence,
                       const RuntimeConfig& config) {
    const auto contextual = config.disable_contextual_mask ? 0 : sequence.contextual_length;
    const auto row_id = contextual_position(row, contextual);
    const auto col_id = contextual_position(column, contextual);
    const auto history_end = contextual_position(sequence.history_end, contextual);
    const auto distance = config.is_causal ? row_id - col_id : std::abs(row_id - col_id);
    bool valid = row == column || distance > 0;
    valid =
        valid && target_attention_allowed(row_id, col_id, history_end, config.target_group_size);
    if (contextual > 0 && row_id == 0 && col_id < history_end)
        valid = true;
    return valid;
}

void fill_positions(const Sequence& sequence, const RuntimeConfig& config, std::int32_t* target) {
    for (std::int32_t index = 0; index < static_cast<std::int32_t>(sequence.tokens.size());
         ++index) {
        const auto position = config.time_buckets > 0 ? std::max(sequence.history_end - index, 0)
                                                      : std::min(index, sequence.history_end);
        target[index] = std::min(position, config.position_buckets - 1);
    }
}

void fill_times(const RecommendationSequence& request, const Sequence& sequence,
                const RuntimeConfig& config, std::int32_t* target) {
    if (request.token_timestamps.size() != sequence.tokens.size())
        throw std::invalid_argument("hstu requires one timestamp per assembled token");
    const auto query_time = request.token_timestamps.back();
    for (std::size_t index = 0; index < request.token_timestamps.size(); ++index) {
        const long double elapsed = static_cast<long double>(query_time) -
                                    static_cast<long double>(request.token_timestamps[index]);
        // The reference Triton boundary promotes elapsed seconds to FP32 before sqrt.
        const float bucket = std::sqrt(std::max(static_cast<float>(elapsed), 0.000001F) / 60.0F);
        target[index] =
            static_cast<std::int32_t>(std::min(bucket, static_cast<float>(config.time_buckets)));
    }
}

void validate_tensor(ITrtModule& engine, const std::string& name, bool input, DType dtype,
                     std::size_t rank, std::int32_t width = 0) {
    if (!(input ? engine.has_input(name) : engine.has_output(name)))
        throw std::invalid_argument("hstu engine is missing tensor " + name);
    const auto shape = engine.tensor_shape(name);
    if (engine.tensor_dtype(name) != dtype || shape.size() != rank)
        throw std::invalid_argument("hstu engine has invalid dtype or rank for " + name);
    if (width > 0 && shape.back() != width)
        throw std::invalid_argument("hstu engine has invalid output width for " + name);
}

const float* output_data(const TensorMap& outputs, const char* name, std::int64_t batch,
                         std::int64_t length, std::int64_t width) {
    const auto found = outputs.find(name);
    if (found == outputs.end())
        throw std::runtime_error(std::string("hstu missing output ") + name);
    const auto& tensor = found->second;
    if (tensor.data == nullptr || tensor.dtype != DType::kFloat32 ||
        tensor.shape != std::vector<std::int64_t>{batch, length, width})
        throw std::runtime_error(std::string("hstu malformed output ") + name);
    return static_cast<const float*>(tensor.data);
}

void require_finite(const std::vector<float>& values) {
    if (!std::all_of(values.begin(), values.end(),
                     [](float value) { return std::isfinite(value); }))
        throw std::runtime_error("hstu engine returned nonfinite values");
}

RecommendationSequenceResult collect(const RecommendationSequence& request,
                                     const Sequence& sequence, const RuntimeConfig& config,
                                     const float* embeddings, const float* logits,
                                     const float* items) {
    RecommendationSequenceResult result;
    result.candidate_item_ids = request.candidate_item_ids;
    result.num_candidates = sequence.candidates;
    result.embedding_dim = config.hidden_size;
    result.output_dim = config.output_dim;
    result.sequence_length = static_cast<std::int32_t>(sequence.tokens.size());
    result.sequence_embeddings.assign(
        embeddings,
        embeddings + sequence.tokens.size() * static_cast<std::size_t>(config.hidden_size));
    if (config.mode == "ranking") {
        const auto* first =
            embeddings + static_cast<std::size_t>(sequence.history_end) * config.hidden_size;
        result.embeddings.assign(first, first + static_cast<std::size_t>(sequence.candidates) *
                                                    config.hidden_size);
        const auto* begin =
            logits + static_cast<std::size_t>(sequence.history_end) * config.output_dim;
        result.logits.assign(begin, begin + static_cast<std::size_t>(sequence.candidates) *
                                                config.output_dim);
    } else {
        result.embeddings.assign(items, items + static_cast<std::size_t>(sequence.candidates) *
                                                    config.hidden_size);
        const auto* query =
            embeddings + static_cast<std::size_t>(sequence.query_position) * config.hidden_size;
        for (std::int32_t candidate = 0; candidate < sequence.candidates; ++candidate) {
            const auto* item = items + static_cast<std::size_t>(candidate) * config.hidden_size;
            float score = 0.0F;
            for (std::int32_t column = 0; column < config.hidden_size; ++column)
                score += query[column] * item[column];
            result.scores.push_back(score);
        }
    }
    require_finite(result.embeddings);
    require_finite(result.sequence_embeddings);
    require_finite(result.logits);
    require_finite(result.scores);
    return result;
}

struct Inputs {
    std::vector<std::int32_t> tokens;
    std::vector<std::int32_t> candidate_tokens;
    std::vector<std::int32_t> positions;
    std::vector<std::int32_t> times;
    std::vector<float> mask;
    float scaling{1.0F};
};

void fill_attention(const Sequence& sequence, const RuntimeConfig& config, std::size_t length,
                    float* mask) {
    for (std::int32_t row = 0; row < static_cast<std::int32_t>(sequence.tokens.size()); ++row) {
        for (std::int32_t column = 0; column < static_cast<std::int32_t>(sequence.tokens.size());
             ++column)
            mask[static_cast<std::size_t>(row) * length + column] =
                attention_allowed(row, column, sequence, config) ? 1.0F : 0.0F;
    }
}

void fill_candidates(const RecommendationSequence& request, const RuntimeConfig& config,
                     std::int32_t* tokens) {
    const auto* table = find_role(config, "item");
    for (std::size_t index = 0; index < request.candidate_item_ids.size(); ++index)
        tokens[index] = lookup(*table, request.candidate_item_ids[index]);
}

Inputs prepare_inputs(const RecommendationRequest& request, const std::vector<Sequence>& sequences,
                      const RuntimeConfig& config, std::size_t length, std::size_t candidates) {
    const auto batch = sequences.size();
    if (length > std::numeric_limits<std::size_t>::max() / length / batch)
        throw std::invalid_argument("hstu attention mask size overflows");
    Inputs inputs;
    inputs.tokens.resize(batch * length, 0);
    if (config.mode == "retrieval")
        inputs.candidate_tokens.resize(batch * candidates, 0);
    inputs.positions.resize(batch * length, 0);
    inputs.times.resize(batch * length, 0);
    inputs.mask.resize(batch * length * length, 0.0F);
    inputs.scaling = config.scaling_seqlen > 0 ? static_cast<float>(config.scaling_seqlen)
                                               : static_cast<float>(length);
    for (std::size_t sample = 0; sample < batch; ++sample) {
        const auto& sequence = sequences[sample];
        if (config.mode == "retrieval")
            fill_candidates(request.sequences[sample], config,
                            inputs.candidate_tokens.data() + sample * candidates);
        std::copy(sequence.tokens.begin(), sequence.tokens.end(),
                  inputs.tokens.begin() + sample * length);
        if (config.position_buckets > 0)
            fill_positions(sequence, config, inputs.positions.data() + sample * length);
        if (config.time_buckets > 0)
            fill_times(request.sequences[sample], sequence, config,
                       inputs.times.data() + sample * length);
        else if (!request.sequences[sample].token_timestamps.empty())
            throw std::invalid_argument("hstu bundle does not use timestamp embeddings");
        fill_attention(sequence, config, length, inputs.mask.data() + sample * length * length);
    }
    return inputs;
}

void validate_config(const RuntimeConfig& config) {
    for (const auto value : {config.hidden_size, config.output_dim, config.max_sequence_length,
                             config.max_batch_size, config.target_group_size}) {
        if (value <= 0)
            throw std::invalid_argument("hstu runtime dimensions must be positive");
    }
    if (config.mode != "ranking" && config.mode != "retrieval")
        throw std::invalid_argument("hstu runtime mode must be ranking or retrieval");
    if (find_role(config, "item") == nullptr)
        throw std::invalid_argument("hstu requires one item embedding table");
    (void)find_role(config, "action");
}

TensorMap input_tensors(Inputs& data, const RuntimeConfig& config, std::int64_t batch,
                        std::int64_t length, std::int64_t candidates) {
    TensorMap inputs = {
        {"token_ids", {data.tokens.data(), {batch, length}, DType::kInt32}},
        {"attention_mask", {data.mask.data(), {batch, 1, length, length}, DType::kFloat32}},
        {"scaling_seqlen", {&data.scaling, {1}, DType::kFloat32}},
    };
    if (config.position_buckets > 0)
        inputs["position_ids"] = {data.positions.data(), {batch, length}, DType::kInt32};
    if (config.time_buckets > 0)
        inputs["time_ids"] = {data.times.data(), {batch, length}, DType::kInt32};
    if (config.mode == "retrieval")
        inputs["candidate_token_ids"] = {
            data.candidate_tokens.data(), {batch, candidates}, DType::kInt32};
    return inputs;
}

} // namespace

Pipeline::Pipeline(std::unique_ptr<ITrtModule> engine, RuntimeConfig config)
    : engine_(std::move(engine)), config_(std::move(config)) {
    if (!engine_ || !engine_->ok())
        throw std::invalid_argument("hstu requires a valid TensorRT engine");
    validate_config(config_);
    validate_tensor(*engine_, "token_ids", true, DType::kInt32, 2);
    validate_tensor(*engine_, "attention_mask", true, DType::kFloat32, 4);
    validate_tensor(*engine_, "scaling_seqlen", true, DType::kFloat32, 1);
    validate_tensor(*engine_, "embeddings", false, DType::kFloat32, 3, config_.hidden_size);
    if (config_.position_buckets > 0)
        validate_tensor(*engine_, "position_ids", true, DType::kInt32, 2);
    if (config_.time_buckets > 0)
        validate_tensor(*engine_, "time_ids", true, DType::kInt32, 2);
    if (config_.mode == "ranking")
        validate_tensor(*engine_, "logits", false, DType::kFloat32, 3, config_.output_dim);
    else {
        validate_tensor(*engine_, "candidate_token_ids", true, DType::kInt32, 2);
        validate_tensor(*engine_, "item_embeddings", false, DType::kFloat32, 3,
                        config_.hidden_size);
    }
}

RecommendationResult Pipeline::recommend(const RecommendationRequest& request) {
    if (request.sequences.empty() ||
        request.sequences.size() > static_cast<std::size_t>(config_.max_batch_size))
        throw std::invalid_argument("hstu batch size is outside the built profile");
    std::vector<Sequence> sequences;
    std::size_t length = 0;
    std::size_t candidates = 1;
    for (const auto& sequence : request.sequences) {
        sequences.push_back(assemble(sequence, config_));
        length = std::max(length, sequences.back().tokens.size());
        candidates = std::max(candidates, sequence.candidate_item_ids.size());
    }
    auto data = prepare_inputs(request, sequences, config_, length, candidates);
    const auto batch = static_cast<std::int64_t>(sequences.size());
    const auto width = static_cast<std::int64_t>(length);
    auto inputs = input_tensors(data, config_, batch, width, static_cast<std::int64_t>(candidates));
    const auto outputs = engine_->forward(inputs);
    const auto* embeddings = output_data(outputs, "embeddings", batch, width, config_.hidden_size);
    const auto* logits = config_.mode == "ranking"
                             ? output_data(outputs, "logits", batch, width, config_.output_dim)
                             : nullptr;
    const auto* items =
        config_.mode == "retrieval"
            ? output_data(outputs, "item_embeddings", batch, static_cast<std::int64_t>(candidates),
                          config_.hidden_size)
            : nullptr;
    RecommendationResult result;
    for (std::size_t sample = 0; sample < sequences.size(); ++sample) {
        const auto offset = sample * length;
        result.sequences.push_back(collect(
            request.sequences[sample], sequences[sample], config_,
            embeddings + offset * config_.hidden_size,
            logits == nullptr ? nullptr : logits + offset * config_.output_dim,
            items == nullptr ? nullptr : items + sample * candidates * config_.hidden_size));
    }
    return result;
}

} // namespace trtmc::hstu
