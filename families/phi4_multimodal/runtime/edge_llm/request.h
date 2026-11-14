/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "families/phi4_multimodal/runtime/edge_llm/contract.h"

#include <edgellm/cpp/runtime/llmRuntimeUtils.h>
#include <edgellm/cpp/runtime/streaming.h>
#include <nlohmann/json.hpp>

namespace trtmc::phi4_multimodal::edge_llm {

/// Preserve raw no-framing and both official checkpoint generation stop tokens.
inline void validate_raw_tokenizer(const nlohmann::json& tokenizer, const nlohmann::json& config,
                                   const nlohmann::json& engine) {
    const auto identity = nlohmann::json::parse(R"({
        "type": "TemplateProcessing",
        "single": [{"Sequence": {"id": "A", "type_id": 0}}],
        "pair": [{"Sequence": {"id": "A", "type_id": 0}},
                 {"Sequence": {"id": "B", "type_id": 1}}],
        "special_tokens": {}
    })");
    const auto post = tokenizer.value("post_processor", nlohmann::json(nullptr));
    if ((!post.is_null() && post != identity) || config.value("add_bos_token", false) ||
        config.value("add_eos_token", false))
        throw std::invalid_argument("Unsupported Phi4 raw tokenizer framing");
    const auto& eos = config.at("eos_token");
    const std::string primary =
        eos.is_string() ? eos.get<std::string>() : eos.at("content").get<std::string>();
    bool found = false;
    for (const auto& token : tokenizer.at("added_tokens")) {
        if (token.at("content") == primary) {
            if (token.at("id") != 199999)
                throw std::invalid_argument("Phi4 tokenizer primary EOS differs from native");
            found = true;
        }
    }
    const auto& ids = engine.at("eos_token_id");
    if (!found || !ids.is_array() || ids.size() != 2)
        throw std::invalid_argument("Phi4 requires the checkpoint generation EOS union");
    const bool valid_order = ids.at(0) == 199999 && ids.at(1) == 200020;
    const bool reverse_order = ids.at(0) == 200020 && ids.at(1) == 199999;
    if (!valid_order && !reverse_order)
        throw std::invalid_argument("Phi4 exported EOS differs from checkpoint generation config");
}

/// Validate complete responses before exposing any partial output, even early EOS.
inline void validate_response(const trt_edgellm::rt::LLMGenerationResponse& response,
                              int input_limit, int capacity, std::int64_t requested_budget) {
    if (response.outputIds.size() != 1 || response.outputTexts.size() != 1 ||
        response.inputTokenCounts.size() != 1 || response.finishReasons.size() != 1 ||
        response.outputIds.front().empty() || requested_budget <= 0 ||
        response.outputIds.front().size() > static_cast<std::uint64_t>(requested_budget))
        throw std::runtime_error("Phi4 Edge returned an invalid generation response");
    validate_capacity(response.inputTokenCounts.front(), input_limit, capacity, requested_budget);
    const auto reason = response.finishReasons.front();
    if (reason != trt_edgellm::rt::FinishReason::kEndId &&
        reason != trt_edgellm::rt::FinishReason::kLength)
        throw std::runtime_error("Phi4 Edge generation did not complete successfully");
    if (reason == trt_edgellm::rt::FinishReason::kLength &&
        response.outputIds.front().size() != static_cast<std::uint64_t>(requested_budget))
        throw std::runtime_error("Phi4 Edge clipped the original generation budget");
}

/// Preserve native raw-no-image and no-system image mode, independent of ignored chat flags.
inline trt_edgellm::rt::LLMGenerationRequest
make_request(const std::string& prompt, const TextGenerationConfig& config, bool has_image) {
    validate_generation(config);
    trt_edgellm::rt::LLMGenerationRequest request{};
    request.requests.resize(1);
    auto& messages = request.requests.front().messages;
    trt_edgellm::rt::Message message{};
    message.role = "user";
    if (has_image) {
        message.contents = {{"image", ""}, {"text", prompt}};
    } else {
        message.contents = {{"text", prompt}};
    }
    messages.push_back(std::move(message));
    request.applyChatTemplate = has_image;
    request.enableThinking = false;
    request.temperature = config.temperature;
    request.topK = config.top_k;
    request.topP = config.top_p;
    // Native accessor returns context, but explicit nonpositive request is the sentinel128.
    request.maxGenerateLength = config.max_new_tokens > 0 ? config.max_new_tokens : 128;
    return request;
}

} // namespace trtmc::phi4_multimodal::edge_llm
