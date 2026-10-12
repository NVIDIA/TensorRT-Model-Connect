/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "families/gemma/runtime/edge_llm/contract.h"

#include <edgellm/cpp/runtime/llmRuntimeUtils.h>
#include <string_view>

namespace trtmc::gemma::edge_llm {

/// Match the checkpoint Jinja trim filter (Python Unicode whitespace), not the locale.
inline std::string trim_user_text(std::string_view text) {
    constexpr std::string_view whitespace[]{
        "\t",           "\n",           "\v",           "\f",           "\r",
        "\x1c",         "\x1d",         "\x1e",         "\x1f",         " ",
        "\xc2\x85",     "\xc2\xa0",     "\xe1\x9a\x80", "\xe2\x80\x80", "\xe2\x80\x81",
        "\xe2\x80\x82", "\xe2\x80\x83", "\xe2\x80\x84", "\xe2\x80\x85", "\xe2\x80\x86",
        "\xe2\x80\x87", "\xe2\x80\x88", "\xe2\x80\x89", "\xe2\x80\x8a", "\xe2\x80\xa8",
        "\xe2\x80\xa9", "\xe2\x80\xaf", "\xe2\x81\x9f", "\xe3\x80\x80"};
    for (;;) {
        const auto previous = text.size();
        for (const auto space : whitespace) {
            if (text.size() >= space.size() && text.substr(0, space.size()) == space)
                text.remove_prefix(space.size());
            if (text.size() >= space.size() && text.substr(text.size() - space.size()) == space)
                text.remove_suffix(space.size());
        }
        if (text.size() == previous)
            return std::string(text);
    }
}

/// Render the admitted Gemma4 checkpoint's single-user text template before Edge tokenization.
inline std::string render_chat(const std::string& prompt, bool thinking) {
    std::string result = "<bos>";
    if (thinking)
        result += "<|turn>system\n<|think|>\n<turn|>\n";
    result += "<|turn>user\n" + trim_user_text(prompt) + "<turn|>\n<|turn>model\n";
    if (!thinking)
        result += "<|channel>thought\n<channel|>";
    return result;
}

/// Map Model Connect text arguments to the pinned Edge API; rejects unmapped controls.
inline trt_edgellm::rt::LLMGenerationRequest make_request(const std::string& prompt,
                                                          const TextGenerationConfig& config,
                                                          int default_length, bool dspark = false) {
    validate_generation(config, dspark);
    trt_edgellm::rt::LLMGenerationRequest request{};
    request.requests.resize(1);
    const auto text =
        config.use_chat_template ? render_chat(prompt, config.enable_thinking) : prompt;
    request.requests.front().messages.push_back({"user", {{"text", text}}});
    // Edge 0.10.1's static Gemma template omits the disabled-thinking closure.
    // Pass the faithful family-rendered prompt as raw text; never strip model output.
    request.applyChatTemplate = false;
    request.enableThinking = config.enable_thinking;
    request.temperature = config.temperature;
    request.topK = config.top_k;
    request.topP = config.top_p;
    request.maxGenerateLength = config.max_new_tokens > 0 ? config.max_new_tokens : default_length;
    return request;
}

} // namespace trtmc::gemma::edge_llm
