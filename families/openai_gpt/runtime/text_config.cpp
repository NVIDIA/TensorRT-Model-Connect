/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/openai_gpt/runtime/text_config.h"

#include <cmath>
#include <limits>
#include <string>

namespace trtmc::openai_gpt {
namespace {

using internal::ConfigField;
using internal::ConfigKind;

const ConfigField kTextFields[] = {
    {"max_new_tokens", ConfigKind::I64, std::int64_t{128}, "Maximum number of generated tokens"},
    {"temperature", ConfigKind::F64, 1.0, "Sampling temperature; 0 selects greedy decoding"},
    {"top_k", ConfigKind::I64, std::int64_t{1}, "Top-k candidates; 1 is greedy, 0 is unlimited"},
    {"top_p", ConfigKind::F64, 1.0, "Nucleus probability mass; 1 disables nucleus filtering"},
    {"min_p", ConfigKind::F64, 0.0, "Minimum probability relative to the most likely token"},
    {"seed", ConfigKind::I64, std::int64_t{-1}, "Sampling seed; negative selects a fixed default"},
};

std::int64_t integer(internal::ConfigView supplied, std::string_view name) {
    return *internal::config_get<std::int64_t>(supplied, kTextFields, name);
}

// Kept as double: a positive value never rounds to zero on the way to the sampler.
double number(internal::ConfigView supplied, std::string_view name, double low, double high) {
    const auto value = *internal::config_get<double>(supplied, kTextFields, name);
    if (!std::isfinite(value) || value < low || value > high)
        throw internal::ConfigError("OpenAI GPT config '" + std::string(name) +
                                    "' is out of range");
    return value;
}

} // namespace

Span<const internal::ConfigField> text_config_fields() {
    return kTextFields;
}

TextConfig parse_text_config(internal::ConfigView supplied, std::int32_t capacity) {
    internal::validate_config(kTextFields, supplied);
    TextConfig config;
    config.max_new_tokens = integer(supplied, "max_new_tokens");
    // The last generated token is never fed back, so a prefix of n tokens and m new tokens
    // need n + m - 1 cache rows: m is at most the capacity.
    if (config.max_new_tokens <= 0 || config.max_new_tokens > capacity)
        throw internal::ConfigError("OpenAI GPT max_new_tokens must be in [1, " +
                                    std::to_string(capacity) + "]");
    const auto top_k = integer(supplied, "top_k");
    if (top_k < 0 || top_k > std::numeric_limits<std::int32_t>::max())
        throw internal::ConfigError("OpenAI GPT top_k is out of range");
    config.sampling.temperature =
        number(supplied, "temperature", 0.0, std::numeric_limits<double>::max());
    config.sampling.top_k = static_cast<std::int32_t>(top_k);
    config.sampling.top_p = number(supplied, "top_p", 0.0, 1.0);
    config.sampling.min_p = number(supplied, "min_p", 0.0, 1.0);
    config.sampling.seed = integer(supplied, "seed");
    return config;
}

} // namespace trtmc::openai_gpt
