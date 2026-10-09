/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "families/gemma/runtime/edge_llm/contract.h"

#include <edgellm/cpp/runtime/llmRuntimeUtils.h>

namespace trtmc::gemma::edge_llm {

/// Forward provider-template options to Edge 0.11 rather than rendering a second template.
inline trt_edgellm::rt::LLMGenerationRequest make_request(const std::string& prompt,
                                                          const TextGenerationConfig& config,
                                                          int default_length,
                                                          bool allow_sampling = false) {
    validate_generation(config, allow_sampling);
    trt_edgellm::rt::LLMGenerationRequest request{};
    request.requests.resize(1);
    auto& messages = request.requests.front().messages;
    if (!config.system_prompt.empty()) {
        if (!config.use_chat_template)
            throw std::invalid_argument("Gemma4 system messages require the chat template");
        trt_edgellm::rt::Message system{};
        system.role = "system";
        system.contents.push_back({"text", config.system_prompt});
        messages.push_back(std::move(system));
    }
    trt_edgellm::rt::Message user{};
    user.role = "user";
    user.contents.push_back({"text", prompt});
    messages.push_back(std::move(user));
    request.applyChatTemplate = config.use_chat_template;
    request.enableThinking = config.enable_thinking;
    request.temperature = config.temperature;
    request.topK = config.top_k;
    request.topP = config.top_p;
    request.maxGenerateLength = config.max_new_tokens > 0 ? config.max_new_tokens : default_length;
    if (config.seed >= 0)
        request.samplingSeed = static_cast<std::uint64_t>(config.seed);
    return request;
}

} // namespace trtmc::gemma::edge_llm
