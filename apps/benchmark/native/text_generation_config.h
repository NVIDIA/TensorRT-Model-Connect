/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/task.h"

#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>

namespace trtmc::benchmark_worker {

template <typename T>
T request_value(const nlohmann::json& request, const char* name, T default_value) {
    return request.contains(name) ? request.at(name).get<T>() : default_value;
}

// The ITextGeneration config of a ``generate`` request. A field the request omits keeps the default
// the command line uses. The language token ids are what translation families (M2M-100 / NLLB)
// frame the source and force the first target token with, so they are part of the request like the
// sampling fields.
inline trtmc::TextGenerationConfig text_generation_config(const nlohmann::json& request) {
    trtmc::TextGenerationConfig config;
    config.max_new_tokens = request_value<std::int32_t>(request, "max_new_tokens", 128);
    config.source_language_token_id =
        request_value<std::int32_t>(request, "source_language_token_id", -1);
    config.forced_bos_token_id = request_value<std::int32_t>(request, "forced_bos_token_id", -1);
    config.temperature = request_value<float>(request, "temperature", 1.0F);
    config.top_k = request_value<std::int32_t>(request, "top_k", 1);
    config.top_p = request_value<float>(request, "top_p", 1.0F);
    config.min_p = request_value<float>(request, "min_p", 0.0F);
    config.seed = request_value<std::int32_t>(request, "seed", -1);
    config.guidance_scale = request_value<float>(request, "guidance_scale", -1.0F);
    config.cfg_scale = request_value<float>(request, "cfg_scale", -1.0F);
    config.num_steps = request_value<std::int32_t>(request, "num_steps", -1);
    config.text_generation_mode =
        request_value<std::string>(request, "text_generation_mode", "auto");
    config.block_length = request_value<std::int32_t>(request, "block_length", 0);
    config.confidence_threshold = request_value<float>(request, "confidence_threshold", -1.0F);
    config.use_chat_template = request_value<bool>(request, "use_chat_template", false);
    config.enable_thinking = request_value<bool>(request, "enable_thinking", true);
    config.repetition_penalty = request_value<float>(request, "repetition_penalty", 1.0F);
    return config;
}

} // namespace trtmc::benchmark_worker
