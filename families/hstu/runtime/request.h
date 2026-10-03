/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/hstu/runtime/pipeline.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace trtmc::hstu {

struct Sequence {
    std::vector<std::int32_t> tokens;
    std::int32_t contextual_length{0};
    std::int32_t history_end{0};
    std::int32_t candidates{0};
    std::int32_t query_position{0};
};

bool all_finite(const std::vector<float>& values);

// Original dense attention metadata, INT32[5,B+1,8]. Queries/keys include
// padded rows; padding is an independent target under causal group-1 attention.
// Context-free ranking sequences only. Page planes remain entirely zero.
std::vector<std::int32_t> make_dense_attention_metadata(const std::vector<Sequence>& sequences,
                                                        std::size_t rows);

// Host encoding of either a legacy FP32 visibility mask or prepared attention
// weights. The allowed scalar is converted once, rather than per matrix entry.
struct AttentionMask {
    DType dtype{DType::kFloat32};
    std::size_t batch{0}, rows{0}, keys{0};
    bool transposed{false};
    float allowed_float{0.0F};
    std::uint16_t allowed_packed{0};
    std::vector<float> floats;
    std::vector<std::uint16_t> packed;

    void* data() {
        return dtype == DType::kFloat32 ? static_cast<void*>(floats.data())
                                        : static_cast<void*>(packed.data());
    }

    Tensor tensor() {
        const auto batches = static_cast<std::int64_t>(batch);
        const auto queries = static_cast<std::int64_t>(rows);
        const auto columns = static_cast<std::int64_t>(keys);
        return {data(),
                transposed ? std::vector<std::int64_t>{batches, 1, columns, queries}
                           : std::vector<std::int64_t>{batches, 1, queries, columns},
                dtype};
    }
};

AttentionMask make_attention_mask(std::size_t batch, std::size_t rows, std::size_t keys,
                                  DType dtype, float allowed, bool transposed = false);
void fill_attention_mask(AttentionMask& mask, std::size_t sample, const Sequence& sequence,
                         const RuntimeConfig& config, std::size_t first_query = 0);
bool prepared_attention(const ITrtModule& engine);
const char* attention_input_name(const ITrtModule& engine);
std::size_t attention_key_width(const ITrtModule& engine, std::size_t logical_length);

const EmbeddingTable* find_role(const RuntimeConfig& config, const char* role);
std::int32_t lookup(const EmbeddingTable& table, std::int64_t id);
std::size_t sequence_length(const RecommendationSequence& request, bool include_candidates);
Sequence assemble(const RecommendationSequence& request, const RuntimeConfig& config);
bool attention_allowed(std::int32_t row, std::int32_t column, const Sequence& sequence,
                       const RuntimeConfig& config);
void fill_positions(const Sequence& sequence, const RuntimeConfig& config, std::int32_t* target);
void fill_times(const RecommendationSequence& request, const Sequence& sequence,
                const RuntimeConfig& config, std::int32_t* target);

} // namespace trtmc::hstu
