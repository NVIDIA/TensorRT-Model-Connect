/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/qwen3_5/runtime/edge_llm/request.h"

#include <iostream>

int main() {
    trtmc::TextGenerationConfig config;
    config.max_new_tokens = 27;
    config.temperature = 0.7F;
    config.top_k = 12;
    config.top_p = 0.8F;
    config.use_chat_template = true;
    config.enable_thinking = false;
    const auto request = trtmc::qwen3_5::edge_llm::make_request("Hello", config, 128);
    if (request.requests.size() != 1 || request.requests[0].messages.size() != 1 ||
        request.requests[0].messages[0].role != "user" ||
        request.requests[0].messages[0].contents[0].content != "Hello" ||
        request.temperature != config.temperature || request.topK != config.top_k ||
        request.topP != config.top_p || request.maxGenerateLength != config.max_new_tokens ||
        !request.applyChatTemplate || request.enableThinking || !request.addGenerationPrompt ||
        request.saveSystemPromptKVCache || !request.loraWeightsName.empty()) {
        std::cerr << "Model Connect arguments did not map to the pinned Edge request" << std::endl;
        return 1;
    }
    config.max_new_tokens = 0;
    config.temperature = 0;
    config.use_chat_template = false;
    const auto raw = trtmc::qwen3_5::edge_llm::make_request("raw", config, 32);
    if (raw.maxGenerateLength != 32 || raw.temperature != 0 || raw.applyChatTemplate)
        return 1;
    config.seed = 42;
    try {
        trtmc::qwen3_5::edge_llm::make_request("unsupported", config, 32);
        return 1;
    } catch (const std::invalid_argument&) {
    }
    return 0;
}
