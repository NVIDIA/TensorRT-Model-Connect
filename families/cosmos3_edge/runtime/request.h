/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "trtmc/internal/language.h"
#include "trtmc/internal/model.h"
#include "trtmc/internal/text.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace trtmc::cosmos3_edge {
using namespace internal;
inline const std::vector<ConfigField>& fields() {
    static const std::vector<ConfigField> result{
        {"max_new_tokens", ConfigKind::I64, int64_t{128}, "Maximum generated tokens"},
        {"temperature", ConfigKind::F64, 1.0, "Sampling temperature; zero is greedy"},
        {"top_k", ConfigKind::I64, int64_t{1}, "Top-k selection"},
        {"top_p", ConfigKind::F64, 1.0, "Nucleus sampling threshold"},
        {"min_p", ConfigKind::F64, 0.0, "Relative minimum probability"},
        {"seed", ConfigKind::I64, int64_t{-1}, "Sampling seed; -1 is unspecified"},
        {"use_chat_template", ConfigKind::Bool, true, "Format checkpoint chat messages"},
        {"enable_thinking", ConfigKind::Bool, true, "Start the reasoning section"},
        {"system_prompt", ConfigKind::String, std::string_view{}, "Optional system message"},
        {"repetition_penalty", ConfigKind::F64, 1.0, "Penalty applied to previously seen tokens"},
    };
    return result;
}
inline Span<const ConfigField> field_view(bool media = true) {
    static const auto text_fields = [] {
        auto copy = fields();
        for (auto& field : copy)
            if (field.name == "use_chat_template")
                field.default_value = false;
        return copy;
    }();
    const auto& selected = media ? fields() : text_fields;
    return {selected.data(), selected.size()};
}
inline TextGenerationConfig read_config(ConfigView supplied, bool media = true) {
    validate_config(field_view(media), supplied);
    TextGenerationConfig c;
    auto integer = [&](const char* name) {
        const auto n = *config_get<int64_t>(supplied, field_view(media), name);
        if (n < std::numeric_limits<int32_t>::min() || n > std::numeric_limits<int32_t>::max())
            throw ConfigError(std::string(name) + " is outside int32 range");
        return static_cast<int32_t>(n);
    };
    c.max_new_tokens = integer("max_new_tokens");
    c.top_k = integer("top_k");
    c.seed = integer("seed");
    c.temperature = *config_get<double>(supplied, field_view(media), "temperature");
    c.top_p = *config_get<double>(supplied, field_view(media), "top_p");
    c.min_p = *config_get<double>(supplied, field_view(media), "min_p");
    c.repetition_penalty = *config_get<double>(supplied, field_view(media), "repetition_penalty");
    c.use_chat_template = *config_get<bool>(supplied, field_view(media), "use_chat_template");
    c.enable_thinking = *config_get<bool>(supplied, field_view(media), "enable_thinking");
    c.system_prompt = *config_get<std::string_view>(supplied, field_view(media), "system_prompt");
    return c;
}
inline void validate_request(const TextGenerationConfig& c) {
    if (c.max_new_tokens <= 0 || !std::isfinite(c.temperature) || c.temperature < 0 ||
        !std::isfinite(c.top_p) || c.top_p < 0 || c.top_p > 1 || !std::isfinite(c.min_p) ||
        c.min_p < 0 || c.min_p > 1 || c.seed < -1 || !std::isfinite(c.repetition_penalty) ||
        c.repetition_penalty <= 0)
        throw std::invalid_argument("Invalid Cosmos3-Edge sampling configuration");
    if (c.eos_token_id != -1 || c.source_language_token_id != -1 || c.forced_bos_token_id != -1 ||
        c.guidance_scale != -1 || c.cfg_scale != -1 || c.num_steps != -1 || c.sde_gamma != -1 ||
        !c.initial_latents.empty() || !c.condition_latents.empty() || !c.condition_mask.empty() ||
        !c.sampling_steps.empty() || !c.sde_noises.empty() || c.block_length != 0 ||
        c.confidence_threshold != -1 || c.stop_on_boxed_answer || !c.lora_adapter_id.empty() ||
        (c.text_generation_mode != "auto" && c.text_generation_mode != "autoregressive"))
        throw std::invalid_argument(
            "Cosmos3-Edge reasoning does not implement the requested generation control");
}
inline std::string role_name(MessageRole role) {
    switch (role) {
    case MessageRole::System:
        return "system";
    case MessageRole::Developer:
        return "developer";
    case MessageRole::User:
        return "user";
    case MessageRole::Assistant:
        return "assistant";
    default:
        throw std::invalid_argument("Cosmos3-Edge reasoning task does not expose tool messages");
    }
}
} // namespace trtmc::cosmos3_edge
