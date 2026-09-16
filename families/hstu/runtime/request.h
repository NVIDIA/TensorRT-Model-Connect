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
