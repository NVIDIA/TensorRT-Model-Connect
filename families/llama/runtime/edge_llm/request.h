/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "families/llama/runtime/edge_llm/contract.h"

#include <edgellm/cpp/runtime/llmRuntimeUtils.h>
#include <nlohmann/json.hpp>

namespace trtmc::llama::edge_llm {

/// Preserve the documented fast-tokenizer single-sequence BOS template, not a guessed default.
inline std::string raw_prompt_prefix(const nlohmann::json& config,
                                     const nlohmann::json& tokenizer) {
    const auto& root = tokenizer.at("post_processor");
    const auto processors =
        root.at("type") == "Sequence" ? root.at("processors") : nlohmann::json::array({root});
    const nlohmann::json* templ = nullptr;
    for (const auto& processor : processors) {
        if (processor.at("type") == "ByteLevel")
            continue; // Post-encoding offset trimming does not change token IDs.
        if (processor.at("type") != "TemplateProcessing" || templ != nullptr)
            throw std::invalid_argument("Unsupported Llama Edge tokenizer postprocessor");
        templ = &processor;
    }
    if (!templ)
        throw std::invalid_argument("Llama Edge requires the documented BOS tokenizer template");
    const auto& single = templ->at("single");
    const auto& bos = config.at("bos_token");
    const std::string token =
        bos.is_string() ? bos.get<std::string>() : bos.at("content").get<std::string>();
    if (token.empty() || single.size() != 2 || single.at(0).at("SpecialToken").at("id") != token ||
        single.at(1).at("Sequence").at("id") != "A")
        throw std::invalid_argument("Unsupported Llama Edge raw tokenizer template");
    return token;
}

/// Map Model Connect text arguments to the pinned Edge API; rejects unmapped controls.
inline trt_edgellm::rt::LLMGenerationRequest make_request(const std::string& prompt,
                                                          const TextGenerationConfig& config,
                                                          int default_length,
                                                          const std::string& raw_prefix = {}) {
    validate_generation(config);
    trt_edgellm::rt::LLMGenerationRequest request{};
    request.requests.resize(1);
    request.requests.front().messages.push_back(
        {"user", {{"text", config.use_chat_template ? prompt : raw_prefix + prompt}}});
    request.applyChatTemplate = config.use_chat_template;
    request.enableThinking = config.enable_thinking;
    request.temperature = config.temperature;
    request.topK = config.top_k;
    request.topP = config.top_p;
    request.maxGenerateLength = config.max_new_tokens > 0 ? config.max_new_tokens : default_length;
    return request;
}

} // namespace trtmc::llama::edge_llm
