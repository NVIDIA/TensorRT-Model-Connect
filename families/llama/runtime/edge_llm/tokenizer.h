/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

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

} // namespace trtmc::llama::edge_llm
