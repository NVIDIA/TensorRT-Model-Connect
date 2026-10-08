/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "families/nemotron_h/runtime/chat_templates.h"
#include "families/nemotron_h/runtime/edge_llm/contract.h"
#include "families/nemotron_h/runtime/tokenizer.h"

#include <edgellm/cpp/runtime/llmRuntimeUtils.h>
#include <edgellm/cpp/runtime/streaming.h>

namespace trtmc::nemotron_h::edge_llm {

/// Apply the existing native family renderer, then pass already framed raw text.
inline trt_edgellm::rt::LLMGenerationRequest make_request(const std::string& prompt,
                                                          const TextGenerationConfig& config,
                                                          const std::string& chat_format,
                                                          const ITokenizer& tokenizer, int bos_id) {
    validate_generation(config);
    const auto formatted =
        config.use_chat_template && !chat_format.empty()
            ? nemotron_h_apply_chat_template(chat_format, prompt, config.enable_thinking)
            : prompt;
    trt_edgellm::rt::LLMGenerationRequest request{};
    request.requests.resize(1);
    request.requests.front().messages.push_back({"user", {{"text", formatted}}});
    auto ids = tokenizer.encode(formatted);
    if (config.use_chat_template && !chat_format.empty() && ids.size() >= 2 && bos_id >= 0 &&
        ids[0] == bos_id && ids[1] == bos_id)
        ids.erase(ids.begin());
    request.preTokenizedInputIds.push_back(std::move(ids));
    request.applyChatTemplate = false;
    request.addGenerationPrompt = false;
    request.enableThinking = false;
    const bool greedy = config.temperature < 1e-6F || config.top_p <= 0 ||
                        (config.top_k <= 1 && config.top_p >= 1.0F - 1e-6F);
    request.temperature = greedy ? 0.0F : config.temperature;
    request.topK = config.top_k;
    request.topP = config.top_p <= 0 ? 1.0F : config.top_p;
    request.maxGenerateLength =
        config.max_new_tokens > 0 ? config.max_new_tokens : kDefaultMaxNewTokens;
    return request;
}

/// Validate the ORIGINAL requested budget even after early EOS or upstream clipping.
inline void validate_response(const trt_edgellm::rt::LLMGenerationResponse& response,
                              std::int32_t input_limit, std::int32_t capacity,
                              std::int64_t requested_budget, int submitted_count) {
    namespace rt = trt_edgellm::rt;
    if (response.outputIds.size() != 1 || response.outputTexts.size() != 1 ||
        response.outputIds.front().empty() || requested_budget <= 0 ||
        response.outputIds.front().size() > static_cast<std::size_t>(requested_budget) ||
        response.finishReasons.size() != 1 ||
        (response.finishReasons.front() != rt::FinishReason::kEndId &&
         response.finishReasons.front() != rt::FinishReason::kLength))
        throw std::runtime_error("Nemotron-H Edge generation did not complete successfully");
    if (response.inputTokenCounts.size() != 1 ||
        response.inputTokenCounts.front() != submitted_count)
        throw std::runtime_error("Nemotron-H Edge returned invalid input counts");
    validate_capacity(response.inputTokenCounts.front(), input_limit, capacity, requested_budget);
    if (response.finishReasons.front() == rt::FinishReason::kLength &&
        response.outputIds.front().size() != static_cast<std::size_t>(requested_budget))
        throw std::runtime_error(
            "Nemotron-H Edge silently shortened the requested generation budget");
}

} // namespace trtmc::nemotron_h::edge_llm
