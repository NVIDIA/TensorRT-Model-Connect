/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "families/nemotron_bolt/runtime/edge_llm/contract.h"

#include <nlohmann/json.hpp>
#include <tensorrt_edgellm/runtime/llmRuntimeUtils.h>
#include <tensorrt_edgellm/runtime/streaming.h>

namespace trtmc::nemotron_bolt::edge_llm {

/// Validate complete responses before exposing any partial output, even early EOS.
inline void validate_response(const trt_edgellm::rt::LLMGenerationResponse& response,
                              int input_limit, int capacity, std::int64_t requested_budget) {
    if (response.outputIds.size() != 1 || response.outputTexts.size() != 1 ||
        response.inputTokenCounts.size() != 1 || response.finishReasons.size() != 1 ||
        response.outputIds.front().empty() || requested_budget <= 0 ||
        response.outputIds.front().size() > static_cast<std::uint64_t>(requested_budget))
        throw std::runtime_error("Nemotron Bolt Edge returned an invalid generation response");
    validate_capacity(response.inputTokenCounts.front(), input_limit, capacity, requested_budget);
    const auto reason = response.finishReasons.front();
    if (reason != trt_edgellm::rt::FinishReason::kEndId &&
        reason != trt_edgellm::rt::FinishReason::kLength)
        throw std::runtime_error("Nemotron Bolt Edge generation did not complete successfully");
    if (reason == trt_edgellm::rt::FinishReason::kLength &&
        response.outputIds.front().size() != static_cast<std::uint64_t>(requested_budget))
        throw std::runtime_error("Nemotron Bolt Edge clipped the original generation budget");
}

/// Map public raw/chat/thinking controls without imposing family-specific hidden defaults.
inline trt_edgellm::rt::LLMGenerationRequest
make_request(const std::string& prompt, const TextGenerationConfig& config, bool has_image) {
    validate_generation(config);
    trt_edgellm::rt::LLMGenerationRequest request{};
    request.requests.resize(1);
    auto& messages = request.requests.front().messages;
    if (has_image) {
        messages.push_back({"user", {{"image", "<image>"}, {"text", prompt}}});
    } else {
        messages.push_back({"user", {{"text", prompt}}});
    }
    if (!config.system_prompt.empty()) {
        if (!config.use_chat_template)
            throw std::invalid_argument("Bolt system prompts need the provider chat template");
        messages.insert(messages.begin(), {"system", {{"text", config.system_prompt}}});
    }
    if (config.seed >= 0)
        request.samplingSeed = static_cast<std::uint64_t>(config.seed);
    request.applyChatTemplate = config.use_chat_template;
    request.enableThinking = config.enable_thinking;
    request.temperature = config.temperature;
    request.topK = config.top_k;
    request.topP = config.top_p;
    // Match the public generation default for legacy callers.
    request.maxGenerateLength = config.max_new_tokens > 0 ? config.max_new_tokens : 128;
    return request;
}

} // namespace trtmc::nemotron_bolt::edge_llm
