/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "families/qwen3_5/runtime/edge_llm/contract.h"

#include <edgellm/cpp/runtime/llmRuntimeUtils.h>

namespace trtmc::qwen3_5::edge_llm {

/// Map Model Connect text arguments to the pinned Edge API; rejects unmapped controls.
inline trt_edgellm::rt::LLMGenerationRequest
make_request(const std::string& prompt, const TextGenerationConfig& config, int default_length) {
    validate_generation(config);
    trt_edgellm::rt::LLMGenerationRequest request{};
    request.requests.resize(1);
    request.requests.front().messages.push_back({"user", {{"text", prompt}}});
    request.applyChatTemplate = config.use_chat_template;
    request.enableThinking = config.enable_thinking;
    request.temperature = config.temperature;
    request.topK = config.top_k;
    request.topP = config.top_p;
    request.maxGenerateLength = config.max_new_tokens > 0 ? config.max_new_tokens : default_length;
    return request;
}

} // namespace trtmc::qwen3_5::edge_llm
