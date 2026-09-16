/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/hstu/runtime/request.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

namespace trtmc::hstu {
namespace {

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

std::int32_t contextual_position(std::int32_t index, std::int32_t contextual) {
    return contextual > 0 ? std::max(index - contextual + 1, 0) : index;
}

bool target_attention_allowed(std::int32_t row, std::int32_t column, std::int32_t history_end,
                              std::int32_t group_size) {
    const auto row_group = row < history_end ? -1 : (row - history_end) / group_size;
    const auto col_group = column < history_end ? -1 : (column - history_end) / group_size;
    return row_group == col_group || row_group < 0 || col_group < 0;
}

} // namespace

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

} // namespace trtmc::hstu
