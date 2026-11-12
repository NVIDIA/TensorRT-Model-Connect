/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "families/qwen3_5_moe/runtime/edge_llm/contract.h"

#include <edgellm/cpp/runtime/llmRuntimeUtils.h>
#include <edgellm/cpp/runtime/streaming.h>
#include <nlohmann/json.hpp>

namespace trtmc::qwen3_5_moe::edge_llm {

/// Require source raw ByteLevel framing and preserve tokenizer/config EOS union.
inline void validate_raw_tokenizer(const nlohmann::json& tokenizer, const nlohmann::json& config,
                                   const nlohmann::json& engine) {
    const auto& post = tokenizer.at("post_processor");
    if (!post.is_object() || post.value("type", "") != "ByteLevel" ||
        config.value("add_bos_token", false) || config.value("add_eos_token", false) ||
        post.value("add_prefix_space", false) || post.value("trim_offsets", false) ||
        post.value("use_regex", false))
        throw std::invalid_argument("Unsupported Qwen3.5-MoE raw tokenizer framing");
    const auto& eos = config.at("eos_token");
    const std::string primary =
        eos.is_string() ? eos.get<std::string>() : eos.at("content").get<std::string>();
    bool found = false;
    for (const auto& token : tokenizer.at("added_tokens")) {
        if (token.at("content") == primary) {
            if (token.at("id") != 248046)
                throw std::invalid_argument("Unexpected Qwen3.5-MoE tokenizer primary EOS");
            found = true;
        }
    }
    const auto& ids = engine.at("eos_token_id");
    if (!found || !ids.is_array() || ids.empty())
        throw std::invalid_argument("Qwen3.5-MoE requires source EOS IDs");
    bool model_eos = false;
    for (const auto& id : ids) {
        if (id == 248044)
            model_eos = true;
        else if (id != 248046)
            throw std::invalid_argument("Unexpected exported source EOS");
    }
    if (!model_eos)
        throw std::invalid_argument("Missing model EOS from source union");
}

/// Validate complete responses before exposing any partial output, even early EOS.
inline void validate_response(const trt_edgellm::rt::LLMGenerationResponse& response,
                              int input_limit, int capacity, std::int64_t requested_budget) {
    if (response.outputIds.size() != 1 || response.outputTexts.size() != 1 ||
        response.inputTokenCounts.size() != 1 || response.finishReasons.size() != 1 ||
        response.outputIds.front().empty() || requested_budget <= 0 ||
        response.outputIds.front().size() > static_cast<std::uint64_t>(requested_budget))
        throw std::runtime_error("Qwen3.5-MoE Edge returned an invalid generation response");
    validate_capacity(response.inputTokenCounts.front(), input_limit, capacity, requested_budget);
    const auto reason = response.finishReasons.front();
    if (reason != trt_edgellm::rt::FinishReason::kEndId &&
        reason != trt_edgellm::rt::FinishReason::kLength)
        throw std::runtime_error("Qwen3.5-MoE Edge generation did not complete successfully");
    if (reason == trt_edgellm::rt::FinishReason::kLength &&
        response.outputIds.front().size() != static_cast<std::uint64_t>(requested_budget))
        throw std::runtime_error("Qwen3.5-MoE Edge clipped the original generation budget");
}

/// Map public raw/chat/thinking controls without imposing family-specific hidden defaults.
inline trt_edgellm::rt::LLMGenerationRequest
make_request(const std::string& prompt, const TextGenerationConfig& config, bool has_image) {
    validate_generation(config);
    trt_edgellm::rt::LLMGenerationRequest request{};
    request.requests.resize(1);
    auto& messages = request.requests.front().messages;
    if (has_image) {
        // Edge 0.11 raw mode requires preformatted media content. Its provider
        // template still owns chat-mode formatting of the typed image part.
        messages.push_back(
            {"user", {{"image", "<|vision_start|><|image_pad|><|vision_end|>"}, {"text", prompt}}});
    } else {
        messages.push_back({"user", {{"text", prompt}}});
    }
    request.applyChatTemplate = config.use_chat_template;
    request.enableThinking = config.enable_thinking;
    request.temperature = config.temperature;
    request.topK = config.top_k;
    request.topP = config.top_p;
    // The new family uses the public default128 and nonpositive sentinel128.
    request.maxGenerateLength = config.max_new_tokens > 0 ? config.max_new_tokens : 128;
    return request;
}

} // namespace trtmc::qwen3_5_moe::edge_llm
