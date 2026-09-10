/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "trtmc/task.h"

#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace trtmc::qwen3_8::edge_llm {

inline constexpr const char* kRevision = "e8b29522938901f6df19ebeedd4b69bc8edbcd97";

/// Return whether an artifact is a normalized file below one of the two Edge roots.
inline bool safe_artifact_path(const std::string& name) {
    if (name.find('\\') != std::string::npos || name.find('\0') != std::string::npos)
        return false;
    const std::filesystem::path path(name);
    if (path.is_absolute() || path.filename().empty())
        return false;
    for (const auto& part : path)
        if (part == "." || part == "..")
            return false;
    return path.generic_string() == name &&
           (name.rfind("edge_llm/engine/", 0) == 0 || name.rfind("edge_llm/checkpoint/", 0) == 0);
}

/// Reject invalid sampling settings and controls with no equivalent Edge request API.
inline void validate_generation(const TextGenerationConfig& c) {
    if (!std::isfinite(c.temperature) || c.temperature < 0 || !std::isfinite(c.top_p) ||
        c.top_p <= 0 || c.top_p > 1 || c.top_k < 0)
        throw std::invalid_argument("Invalid Qwen3.8 Edge sampling parameters");
    if (c.min_p != 0 || c.seed != -1 || c.eos_token_id != -1 || c.repetition_penalty != 1 ||
        !c.lora_adapter_id.empty() || c.stop_on_boxed_answer ||
        (c.text_generation_mode != "auto" && c.text_generation_mode != "autoregressive") ||
        c.source_language_token_id != -1 || c.forced_bos_token_id != -1 || c.guidance_scale != -1 ||
        c.cfg_scale != -1 || c.num_steps != -1 || c.sde_gamma != -1 || !c.initial_latents.empty() ||
        !c.condition_latents.empty() || !c.condition_mask.empty() || !c.sampling_steps.empty() ||
        !c.sde_noises.empty() || c.block_length != 0 || c.confidence_threshold != -1)
        throw std::invalid_argument(
            "Requested generation controls are unsupported by Qwen3.8 Edge");
}

/// Enforce prompt and total capacity without allowing Edge to silently clip generation.
inline void validate_capacity(int prompt_tokens, int input_limit, int capacity,
                              std::int64_t generated_tokens) {
    if (prompt_tokens <= 0 || prompt_tokens > input_limit || generated_tokens <= 0 ||
        generated_tokens > static_cast<std::int64_t>(capacity) - prompt_tokens)
        throw std::invalid_argument("Qwen3.8 Edge prompt and generation exceed bundle capacity");
}

} // namespace trtmc::qwen3_8::edge_llm
