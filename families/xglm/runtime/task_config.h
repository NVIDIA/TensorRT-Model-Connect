/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/internal/config.h"
#include "trtmc/task.h"

#include <cmath>
#include <limits>

namespace trtmc::xglm {

inline Span<const internal::ConfigField> text_config_fields() {
    using namespace internal;
    static const ConfigField fields[] = {
        {"max_new_tokens", ConfigKind::I64, ConfigValue{std::int64_t{128}},
         "Maximum generated tokens; zero returns an empty continuation"},
        {"temperature", ConfigKind::F64, ConfigValue{1.0}, "Sampling temperature"},
        {"top_k", ConfigKind::I64, ConfigValue{std::int64_t{1}}, "Top-k sampling cutoff"},
        {"top_p", ConfigKind::F64, ConfigValue{1.0}, "Nucleus sampling cutoff"},
        {"min_p", ConfigKind::F64, ConfigValue{0.0}, "Minimum relative token probability"},
        {"seed", ConfigKind::I64, ConfigValue{std::int64_t{-1}},
         "Sampling seed; negative uses the family default"},
        {"eos_token_id", ConfigKind::I64, ConfigValue{std::int64_t{-1}},
         "EOS override; negative uses the checkpoint value"},
        {"generation_mode", ConfigKind::String, ConfigValue{std::string_view{"auto"}},
         "Autoregressive generation mode"},
        {"use_chat_template", ConfigKind::Bool, ConfigValue{false},
         "Apply the checkpoint chat template to text inputs"},
        {"enable_thinking", ConfigKind::Bool, ConfigValue{true},
         "Enable thinking when the checkpoint template supports it"},
        {"stop_on_boxed_answer", ConfigKind::Bool, ConfigValue{false},
         "Stop at a completed boxed or final answer"},
        {"stop_check_interval", ConfigKind::I64, ConfigValue{std::int64_t{16}},
         "Generated-token interval between answer checks"},
        {"repetition_penalty", ConfigKind::F64, ConfigValue{1.0},
         "Only the neutral value 1 is supported"},
    };
    return fields;
}

inline TextGenerationConfig parse_text_config(internal::ConfigView supplied) {
    using namespace internal;
    const auto fields = text_config_fields();
    validate_config(fields, supplied);
    auto integer = [&](std::string_view name) {
        const auto value = config_get<std::int64_t>(supplied, fields, name).value();
        if (value < std::numeric_limits<std::int32_t>::min() ||
            value > std::numeric_limits<std::int32_t>::max())
            throw ConfigError(std::string(name) + " is outside the supported integer range");
        return static_cast<std::int32_t>(value);
    };
    auto real = [&](std::string_view name) {
        const auto value = config_get<double>(supplied, fields, name).value();
        if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max())
            throw ConfigError(std::string(name) + " must be a finite float32 value");
        return static_cast<float>(value);
    };
    TextGenerationConfig config;
    config.max_new_tokens = integer("max_new_tokens");
    if (config.max_new_tokens < 0)
        throw ConfigError("max_new_tokens must be nonnegative");
    config.temperature = real("temperature");
    config.top_k = integer("top_k");
    config.top_p = real("top_p");
    config.min_p = real("min_p");
    if (config.temperature < 0.0F || config.top_k < 0 || config.top_p < 0.0F ||
        config.top_p > 1.0F || config.min_p < 0.0F || config.min_p > 1.0F)
        throw ConfigError(
            "temperature and top_k must be nonnegative; top_p and min_p must be in [0, 1]");
    config.seed = integer("seed");
    config.eos_token_id = integer("eos_token_id");
    config.text_generation_mode =
        config_get<std::string_view>(supplied, fields, "generation_mode").value();
    config.use_chat_template = config_get<bool>(supplied, fields, "use_chat_template").value();
    config.enable_thinking = config_get<bool>(supplied, fields, "enable_thinking").value();
    config.stop_on_boxed_answer =
        config_get<bool>(supplied, fields, "stop_on_boxed_answer").value();
    config.stop_check_interval = integer("stop_check_interval");
    config.repetition_penalty = real("repetition_penalty");
    if (config.repetition_penalty != 1.0F)
        throw ConfigError("repetition_penalty supports only the neutral value 1");
    return config;
}

} // namespace trtmc::xglm
