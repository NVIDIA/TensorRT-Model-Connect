/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "sampler.h"
#include "trtmc/internal/config.h"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace trtmc::smolvlm {

struct GenerationConfig {
    int32_t max_new_tokens{128};
    SamplingParams sampling;
};

inline const std::vector<internal::ConfigField>& config_fields() {
    using internal::ConfigKind;
    static const std::vector<internal::ConfigField> fields{
        {"max_new_tokens", ConfigKind::I64, int64_t{128}, "Maximum generated tokens"},
        {"temperature", ConfigKind::F64, 1.0, "Sampling temperature; zero selects greedy decoding"},
        {"top_k", ConfigKind::I64, int64_t{1},
         "Most likely tokens kept when sampling; 1 is greedy, 0 keeps all"},
        {"top_p", ConfigKind::F64, 1.0, "Nucleus probability mass kept when sampling"},
        {"seed", ConfigKind::I64, int64_t{-1}, "Sampling seed; -1 uses a fixed default"},
    };
    return fields;
}

inline Span<const internal::ConfigField> config_view() {
    const auto& fields = config_fields();
    return {fields.data(), fields.size()};
}

inline GenerationConfig read_config(internal::ConfigView supplied) {
    using internal::config_get;
    internal::validate_config(config_view(), supplied);
    GenerationConfig config;
    const auto max_new_tokens = *config_get<int64_t>(supplied, config_view(), "max_new_tokens");
    if (max_new_tokens < 1 || max_new_tokens > std::numeric_limits<int32_t>::max())
        throw internal::ConfigError("max_new_tokens must be a positive int32");
    config.max_new_tokens = static_cast<int32_t>(max_new_tokens);
    config.sampling.temperature = *config_get<double>(supplied, config_view(), "temperature");
    config.sampling.top_k = *config_get<int64_t>(supplied, config_view(), "top_k");
    config.sampling.top_p = *config_get<double>(supplied, config_view(), "top_p");
    config.sampling.seed = *config_get<int64_t>(supplied, config_view(), "seed");
    return config;
}

} // namespace trtmc::smolvlm
