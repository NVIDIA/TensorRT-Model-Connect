/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/hstu/runtime/request.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
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

namespace {

inline bool attention_allowed_impl(std::int32_t row, std::int32_t column, const Sequence& sequence,
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

} // namespace

bool attention_allowed(std::int32_t row, std::int32_t column, const Sequence& sequence,
                       const RuntimeConfig& config) {
    return attention_allowed_impl(row, column, sequence, config);
}

namespace {

void validate_dense_sequence(const Sequence& sequence, std::size_t rows) {
    if (sequence.tokens.empty() || sequence.contextual_length != 0 || sequence.history_end < 0 ||
        sequence.candidates < 0)
        throw std::invalid_argument(
            "hstu dense metadata requires nonempty context-free ranking rows");
    const auto history = static_cast<std::size_t>(sequence.history_end);
    if (sequence.tokens.size() > rows || history > sequence.tokens.size() ||
        static_cast<std::size_t>(sequence.candidates) != sequence.tokens.size() - history)
        throw std::invalid_argument(
            "hstu dense metadata has inconsistent history/candidate extents");
}

} // namespace

std::vector<std::int32_t> make_dense_attention_metadata(const std::vector<Sequence>& sequences,
                                                        std::size_t rows) {
    const auto batch = sequences.size();
    const auto limit = static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
    if (batch == 0 || rows == 0 || batch > limit / rows)
        throw std::invalid_argument("hstu dense query/key offsets exceed INT32 or are empty");
    if (batch > std::numeric_limits<std::size_t>::max() / (40 * sizeof(std::int32_t)) - 1)
        throw std::invalid_argument("hstu dense metadata allocation overflows");
    const auto stride = 8 * (batch + 1);
    std::vector<std::int32_t> metadata(5 * stride, 0);
    for (std::size_t sample = 0; sample < batch; ++sample) {
        const auto& sequence = sequences[sample];
        validate_dense_sequence(sequence, rows);
        const auto offset = static_cast<std::int32_t>((sample + 1) * rows);
        metadata[sample + 1] = metadata[stride + sample + 1] = offset;
        // Padding is additional independent group-1 targets, never history.
        metadata[2 * stride + sample] =
            static_cast<std::int32_t>(rows - static_cast<std::size_t>(sequence.history_end));
    }
    return metadata;
}
bool all_finite(const std::vector<float>& values) {
    static_assert(sizeof(float) == sizeof(std::uint32_t));
    static_assert(std::numeric_limits<float>::is_iec559);
    std::uint32_t nonfinite = 0;
    for (const auto& value : values) {
        std::uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        nonfinite |= static_cast<std::uint32_t>((bits & 0x7F800000U) == 0x7F800000U);
    }
    if (nonfinite == 0)
        return true;
    // Keep the original predicate's signaling-NaN and first-error behavior.
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
}

namespace {

std::size_t attention_mask_elements(std::size_t batch, std::size_t rows, std::size_t keys) {
    const auto limit = std::numeric_limits<std::size_t>::max();
    if ((batch && rows > limit / batch) || (keys && batch * rows > limit / keys))
        throw std::invalid_argument("hstu attention mask size overflows");
    return batch * rows * keys;
}

} // namespace

AttentionMask make_attention_mask(std::size_t batch, std::size_t rows, std::size_t keys,
                                  DType dtype, float allowed, bool transposed) {
    if (!std::isfinite(allowed) || allowed < 0.0F)
        throw std::invalid_argument("hstu attention weight must be finite and nonnegative");
    const auto count = attention_mask_elements(batch, rows, keys);
    AttentionMask mask;
    mask.dtype = dtype;
    mask.batch = batch;
    mask.rows = rows;
    mask.keys = keys;
    mask.transposed = transposed;
    mask.allowed_float = allowed;
    if (dtype == DType::kFloat32) {
        mask.floats.resize(count, 0.0F);
    } else if (dtype == DType::kFloat16 || dtype == DType::kBFloat16) {
        if (dtype == DType::kFloat16) {
            const auto value = __float2half_rn(allowed);
            std::memcpy(&mask.allowed_packed, &value, sizeof(mask.allowed_packed));
        } else {
            const auto value = __float2bfloat16_rn(allowed);
            std::memcpy(&mask.allowed_packed, &value, sizeof(mask.allowed_packed));
        }
        mask.packed.resize(count, 0);
    } else {
        throw std::invalid_argument("hstu attention weights require FP32, FP16, or BF16");
    }
    return mask;
}

namespace {

template <class Value>
void fill_attention_rows(Value* output, std::size_t keys, const Sequence& sequence,
                         const RuntimeConfig& config, std::size_t first_query, Value allowed) {
    for (std::size_t row = first_query; row < sequence.tokens.size(); ++row) {
        auto* target = output + (row - first_query) * keys;
        for (std::size_t column = 0; column < sequence.tokens.size(); ++column)
            target[column] =
                attention_allowed_impl(static_cast<std::int32_t>(row),
                                       static_cast<std::int32_t>(column), sequence, config)
                    ? allowed
                    : Value{};
    }
}

template <class Value>
void fill_query_interval(Value* target, std::size_t first_query, std::size_t length,
                         std::size_t begin, std::size_t end, Value allowed) {
    begin = std::max(begin, first_query);
    end = std::min(end, length);
    if (begin < end)
        std::fill(target + begin - first_query, target + end - first_query, allowed);
}

template <class Value>
void fill_attention_columns(Value* output, std::size_t rows, const Sequence& sequence,
                            const RuntimeConfig& config, std::size_t first_query, Value allowed) {
    const auto length = sequence.tokens.size();
    const auto history = static_cast<std::size_t>(sequence.history_end);
    const auto contextual =
        config.disable_contextual_mask ? 0 : static_cast<std::size_t>(sequence.contextual_length);
    const auto group = static_cast<std::size_t>(config.target_group_size);
    // The mask starts at zero. Each key exposes contiguous query intervals,
    // stored directly as [key, query] without per-element predicate evaluation.
    for (std::size_t column = 0; column < length; ++column) {
        auto* target = output + column * rows;
        if (column < contextual) {
            fill_query_interval(target, first_query, length, 0, length, allowed);
        } else if (column < history) {
            fill_query_interval(target, first_query, length, config.is_causal ? column : 0, length,
                                allowed);
            if (config.is_causal)
                fill_query_interval(target, first_query, length, 0, contextual, allowed);
        } else {
            const auto start = history + (column - history) / group * group;
            const auto end = start + std::min(group, length - start);
            fill_query_interval(target, first_query, length, config.is_causal ? column : start, end,
                                allowed);
            if (!config.is_causal)
                fill_query_interval(target, first_query, length, 0, history, allowed);
        }
    }
}

} // namespace

void fill_attention_mask(AttentionMask& mask, std::size_t sample, const Sequence& sequence,
                         const RuntimeConfig& config, std::size_t first_query) {
    if (sample >= mask.batch || first_query > sequence.tokens.size() ||
        sequence.tokens.size() - first_query > mask.rows || sequence.tokens.size() > mask.keys)
        throw std::invalid_argument("hstu attention mask dimensions do not cover the sequence");
    if (first_query == sequence.tokens.size())
        return;
    const auto offset = sample * mask.rows * mask.keys;
    if (mask.transposed && mask.dtype == DType::kFloat32)
        fill_attention_columns(mask.floats.data() + offset, mask.rows, sequence, config,
                               first_query, mask.allowed_float);
    else if (mask.transposed)
        fill_attention_columns(mask.packed.data() + offset, mask.rows, sequence, config,
                               first_query, mask.allowed_packed);
    else if (mask.dtype == DType::kFloat32)
        fill_attention_rows(mask.floats.data() + offset, mask.keys, sequence, config, first_query,
                            mask.allowed_float);
    else
        fill_attention_rows(mask.packed.data() + offset, mask.keys, sequence, config, first_query,
                            mask.allowed_packed);
}

bool prepared_attention(const ITrtModule& engine) {
    return engine.has_input("attention_weights_transposed");
}

const char* attention_input_name(const ITrtModule& engine) {
    return prepared_attention(engine) ? "attention_weights_transposed" : "attention_mask";
}

std::size_t attention_key_width(const ITrtModule& engine, std::size_t logical_length) {
    const auto* name = attention_input_name(engine);
    const auto shape = engine.tensor_shape(name);
    if (shape.size() != 4)
        throw std::invalid_argument("hstu attention input must have rank four");
    const auto minimum =
        engine.input_profile_shape(name, engine.profile_idx(), ProfileShapeSelector::kMin);
    const auto maximum =
        engine.input_profile_shape(name, engine.profile_idx(), ProfileShapeSelector::kMax);
    const auto key_axis = std::strcmp(name, "attention_weights_transposed") == 0 ? 2 : 3;
    const bool dynamic = shape[key_axis] == -1 || (minimum.size() == 4 && maximum.size() == 4 &&
                                                   minimum[key_axis] != maximum[key_axis]);
    const auto width = dynamic ? logical_length : static_cast<std::size_t>(shape[key_axis]);
    if (width < logical_length || (!dynamic && shape[key_axis] <= 0))
        throw std::invalid_argument("hstu attention key width is smaller than the request");
    return width;
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
