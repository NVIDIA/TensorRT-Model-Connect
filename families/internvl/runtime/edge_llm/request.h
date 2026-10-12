/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "families/internvl/runtime/edge_llm/contract.h"

#include <edgellm/cpp/runtime/llmRuntimeUtils.h>
#include <edgellm/cpp/runtime/streaming.h>
#include <nlohmann/json.hpp>

namespace trtmc::internvl::edge_llm {

/// ByteLevel postprocessing only changes offsets; reject unqualified token-ID framing.
inline void validate_raw_tokenizer(const nlohmann::json& tokenizer,
                                   const nlohmann::json& config = nlohmann::json::object()) {
    if (config.value("add_bos_token", false) || config.value("add_eos_token", false))
        throw std::invalid_argument("Unsupported InternVL Edge raw BOS/EOS tokenizer flags");
    if (!tokenizer.contains("post_processor") || !tokenizer.at("post_processor").is_object() ||
        tokenizer.at("post_processor").value("type", "") != "ByteLevel")
        throw std::invalid_argument("Unsupported InternVL Edge raw tokenizer postprocessor");
}

/// Validate complete responses before exposing any partial output, even early EOS.
inline void validate_response(const trt_edgellm::rt::LLMGenerationResponse& response,
                              int input_limit, int capacity, std::int64_t requested_budget) {
    if (response.outputIds.size() != 1 || response.outputTexts.size() != 1 ||
        response.inputTokenCounts.size() != 1 || response.finishReasons.size() != 1 ||
        response.outputIds.front().empty() || requested_budget <= 0 ||
        response.outputIds.front().size() > static_cast<std::uint64_t>(requested_budget))
        throw std::runtime_error("InternVL Edge returned an invalid generation response");
    validate_capacity(response.inputTokenCounts.front(), input_limit, capacity, requested_budget);
    const auto reason = response.finishReasons.front();
    if (reason != trt_edgellm::rt::FinishReason::kEndId &&
        reason != trt_edgellm::rt::FinishReason::kLength)
        throw std::runtime_error("InternVL Edge generation did not complete successfully");
    if (reason == trt_edgellm::rt::FinishReason::kLength &&
        response.outputIds.front().size() != static_cast<std::uint64_t>(requested_budget))
        throw std::runtime_error("InternVL Edge clipped the original generation budget");
}

/// Preserve native raw-no-image and fixed-system image mode, independent of ignored chat flags.
inline trt_edgellm::rt::LLMGenerationRequest
make_request(const std::string& prompt, const TextGenerationConfig& config, bool has_image) {
    validate_generation(config);
    trt_edgellm::rt::LLMGenerationRequest request{};
    request.requests.resize(1);
    auto& messages = request.requests.front().messages;
    if (has_image) {
        messages.push_back({"system", {{"text", "You are a helpful assistant."}}});
        messages.push_back({"user", {{"image", ""}, {"text", prompt}}});
    } else {
        messages.push_back({"user", {{"text", prompt}}});
    }
    request.applyChatTemplate = has_image;
    request.enableThinking = false;
    request.temperature = config.temperature;
    request.topK = config.top_k;
    request.topP = config.top_p;
    // Native accessor returns context, but explicit nonpositive request is the sentinel128.
    request.maxGenerateLength = config.max_new_tokens > 0 ? config.max_new_tokens : 128;
    return request;
}

} // namespace trtmc::internvl::edge_llm
