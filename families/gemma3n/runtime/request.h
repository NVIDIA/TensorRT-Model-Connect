/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "trtmc/internal/config.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace trtmc::gemma3n {

// Generation settings of one request. Temperature zero selects the first
// maximum logit (greedy); otherwise tokens are sampled from the softmax of
// logits / temperature restricted to the top_k (zero: all) most likely tokens
// and the top_p nucleus.
struct Generation {
    int32_t max_new_tokens{128};
    double temperature{0.0};
    int32_t top_k{0};
    double top_p{1.0};
    int64_t seed{-1};
};

inline const std::vector<internal::ConfigField>& config_fields() {
    using internal::ConfigKind;
    static const std::vector<internal::ConfigField> fields{
        {"max_new_tokens", ConfigKind::I64, int64_t{128}, "Maximum generated tokens"},
        {"temperature", ConfigKind::F64, 0.0, "Sampling temperature; zero is greedy decoding"},
        {"top_k", ConfigKind::I64, int64_t{0},
         "Sample among the k most likely tokens; zero is all"},
        {"top_p", ConfigKind::F64, 1.0, "Nucleus probability mass; one disables it"},
        {"seed", ConfigKind::I64, int64_t{-1}, "Sampling seed; -1 draws a nondeterministic seed"},
    };
    return fields;
}

inline Span<const internal::ConfigField> config_view() {
    return {config_fields().data(), config_fields().size()};
}

inline Generation read_generation(internal::ConfigView supplied) {
    internal::validate_config(config_view(), supplied);
    auto integer = [&](const char* name) {
        return *internal::config_get<int64_t>(supplied, config_view(), name);
    };
    auto real = [&](const char* name) {
        return *internal::config_get<double>(supplied, config_view(), name);
    };
    Generation generation;
    const auto max_new_tokens = integer("max_new_tokens");
    const auto top_k = integer("top_k");
    generation.temperature = real("temperature");
    generation.top_p = real("top_p");
    generation.seed = integer("seed");
    if (max_new_tokens < 1 || max_new_tokens > std::numeric_limits<int32_t>::max())
        throw internal::ConfigError("max_new_tokens must be a positive int32");
    if (top_k < 0 || top_k > std::numeric_limits<int32_t>::max())
        throw internal::ConfigError("top_k must be a non-negative int32");
    if (!std::isfinite(generation.temperature) || generation.temperature < 0.0)
        throw internal::ConfigError("temperature must be finite and non-negative");
    if (!std::isfinite(generation.top_p) || generation.top_p <= 0.0 || generation.top_p > 1.0)
        throw internal::ConfigError("top_p must lie in (0, 1]");
    if (generation.seed < -1)
        throw internal::ConfigError("seed must be -1 or non-negative");
    generation.max_new_tokens = static_cast<int32_t>(max_new_tokens);
    generation.top_k = static_cast<int32_t>(top_k);
    return generation;
}

} // namespace trtmc::gemma3n
