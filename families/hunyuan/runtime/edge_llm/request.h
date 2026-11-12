/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "families/hunyuan/runtime/edge_llm/contract.h"

#include <edgellm/cpp/runtime/llmRuntimeUtils.h>
#include <nlohmann/json.hpp>

namespace trtmc::hunyuan::edge_llm {

/// Hunyuan fast tokenizers do not insert BOS/EOS through a postprocessor.
inline std::string raw_prompt_prefix(const nlohmann::json&, const nlohmann::json& tokenizer) {
    const auto& processor = tokenizer.at("post_processor");
    if (!processor.is_null() && processor.value("type", "") != "ByteLevel")
        throw std::invalid_argument("Unsupported Hunyuan Edge tokenizer postprocessor");
    return {};
}

/// Map Model Connect text arguments to the pinned Edge API; rejects unmapped controls.
inline trt_edgellm::rt::LLMGenerationRequest make_request(const std::string& prompt,
                                                          const TextGenerationConfig& config,
                                                          int default_length,
                                                          const std::string& raw_prefix = {}) {
    validate_generation(config);
    trt_edgellm::rt::LLMGenerationRequest request{};
    request.requests.resize(1);
    auto& message = request.requests.front().messages.emplace_back();
    message.role = "user";
    message.contents.push_back({"text", config.use_chat_template ? prompt : raw_prefix + prompt});
    request.applyChatTemplate = config.use_chat_template;
    request.enableThinking = config.enable_thinking;
    request.temperature = config.temperature;
    request.topK = config.top_k;
    request.topP = config.top_p;
    request.maxGenerateLength = config.max_new_tokens > 0 ? config.max_new_tokens : default_length;
    return request;
}

} // namespace trtmc::hunyuan::edge_llm
