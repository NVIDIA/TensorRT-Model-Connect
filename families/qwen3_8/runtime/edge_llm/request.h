/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "families/qwen3_8/runtime/edge_llm/contract.h"

#include <edgellm/cpp/runtime/llmRuntimeUtils.h>

namespace trtmc::qwen3_8::edge_llm {

/// Match the source Jinja trim filter without locale-dependent ASCII-only trimming.
inline std::string source_user_content(std::string text) {
    // Python str.strip whitespace, including C0 separators and Unicode spaces.
    constexpr const char* whitespace[] = {
        "\t",           "\n",           "\v",           "\f",           "\r",
        "\x1c",         "\x1d",         "\x1e",         "\x1f",         " ",
        "\xc2\x85",     "\xc2\xa0",     "\xe1\x9a\x80", "\xe2\x80\x80", "\xe2\x80\x81",
        "\xe2\x80\x82", "\xe2\x80\x83", "\xe2\x80\x84", "\xe2\x80\x85", "\xe2\x80\x86",
        "\xe2\x80\x87", "\xe2\x80\x88", "\xe2\x80\x89", "\xe2\x80\x8a", "\xe2\x80\xa8",
        "\xe2\x80\xa9", "\xe2\x80\xaf", "\xe2\x81\x9f", "\xe3\x80\x80"};
    bool changed = true;
    while (changed && !text.empty()) {
        changed = false;
        for (const std::string space : whitespace) {
            if (text.compare(0, space.size(), space) == 0) {
                text.erase(0, space.size());
                changed = true;
            }
            if (text.size() >= space.size() &&
                text.compare(text.size() - space.size(), space.size(), space) == 0) {
                text.resize(text.size() - space.size());
                changed = true;
            }
        }
    }
    if (text.rfind("<tool_response>", 0) == 0 && text.size() >= 16 &&
        text.compare(text.size() - 16, 16, "</tool_response>") == 0)
        throw std::invalid_argument("Qwen3.8 single-user prompt contains no user query");
    return text;
}

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

} // namespace trtmc::qwen3_8::edge_llm
