/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minicpmv4_6/runtime/prompt.h"

#include <cstdint>
#include <stdexcept>

namespace trtmc::minicpmv4_6 {
namespace {

std::string value(const nlohmann::json& object, const char* name) {
    return object.at(name).get<std::string>();
}

bool starts_with(const std::string& text, const std::string& prefix) {
    return text.compare(0, prefix.size(), prefix) == 0;
}

bool ends_with(const std::string& text, const std::string& suffix) {
    return text.size() >= suffix.size() &&
           text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

} // namespace

std::string strip(const std::string& text) {
    // Unicode White_Space plus the ASCII information separators, as Python's str.isspace().
    static const std::vector<std::string> spaces{" ",
                                                 "\t",
                                                 "\n",
                                                 "\r",
                                                 "\v",
                                                 "\f",
                                                 "\x1c",
                                                 "\x1d",
                                                 "\x1e",
                                                 "\x1f",
                                                 "\xc2\x85",
                                                 "\xc2\xa0",
                                                 "\xe1\x9a\x80",
                                                 "\xe2\x80\x80",
                                                 "\xe2\x80\x81",
                                                 "\xe2\x80\x82",
                                                 "\xe2\x80\x83",
                                                 "\xe2\x80\x84",
                                                 "\xe2\x80\x85",
                                                 "\xe2\x80\x86",
                                                 "\xe2\x80\x87",
                                                 "\xe2\x80\x88",
                                                 "\xe2\x80\x89",
                                                 "\xe2\x80\x8a",
                                                 "\xe2\x80\xa8",
                                                 "\xe2\x80\xa9",
                                                 "\xe2\x80\xaf",
                                                 "\xe2\x81\x9f",
                                                 "\xe3\x80\x80"};
    std::size_t begin = 0, end = text.size();
    for (bool found = true; found && begin < end;) {
        found = false;
        for (const auto& space : spaces)
            if (space.size() <= end - begin && text.compare(begin, space.size(), space) == 0) {
                begin += space.size();
                found = true;
                break;
            }
    }
    for (bool found = true; found && begin < end;) {
        found = false;
        for (const auto& space : spaces)
            if (space.size() <= end - begin &&
                text.compare(end - space.size(), space.size(), space) == 0) {
                end -= space.size();
                found = true;
                break;
            }
    }
    return text.substr(begin, end - begin);
}

void validate_recipe(const nlohmann::json& recipe) {
    for (const char* field : {"default_prefix", "empty_system_prefix", "generation"})
        (void)value(recipe, field);
    (void)recipe.at("parse_assistant_thinking").get<bool>();
    for (const char* kind : {"user", "system", "assistant_recent", "assistant_history"}) {
        (void)value(recipe.at(kind), "prefix");
        (void)value(recipe.at(kind), "suffix");
        if (std::string(kind).rfind("assistant", 0) == 0) {
            (void)recipe.at(kind).at("include_reasoning").get<bool>();
            (void)value(recipe.at(kind), "reasoning_suffix");
        }
    }
}

std::string format_conversation(const nlohmann::json& recipe,
                                const std::vector<PromptMessage>& messages) {
    if (messages.empty())
        throw std::invalid_argument("MiniCPM-V 4.6 requires at least one message");
    std::vector<std::string> content;
    std::int64_t last_user = -1;
    for (std::size_t i = 0; i < messages.size(); ++i) {
        content.push_back(strip(messages[i].content));
        const auto& text = content.back();
        const bool tool_response =
            starts_with(text, "<tool_response>") && ends_with(text, "</tool_response>");
        if (messages[i].role == Role::User && !tool_response)
            last_user = static_cast<std::int64_t>(i);
    }
    if (last_user < 0)
        throw std::invalid_argument("MiniCPM-V 4.6 requires a user query");
    std::string output;
    if (messages[0].role != Role::System)
        output = value(recipe, "default_prefix");
    else if (content[0].empty())
        output = value(recipe, "empty_system_prefix");
    else
        output = value(recipe.at("system"), "prefix") + content[0] +
                 value(recipe.at("system"), "suffix");
    for (std::size_t i = 0; i < messages.size(); ++i) {
        if (messages[i].role == Role::System) {
            if (i != 0)
                throw std::invalid_argument("MiniCPM-V 4.6 system message must be first");
            continue;
        }
        if (messages[i].role == Role::User) {
            output += value(recipe.at("user"), "prefix") + content[i] +
                      value(recipe.at("user"), "suffix");
            continue;
        }
        std::string body = content[i], reasoning;
        const auto close = body.find("</think>");
        if (recipe.at("parse_assistant_thinking").get<bool>() && close != std::string::npos) {
            reasoning = body.substr(0, close);
            while (!reasoning.empty() && reasoning.back() == '\n')
                reasoning.pop_back();
            const auto open = reasoning.rfind("<think>");
            if (open != std::string::npos)
                reasoning.erase(0, open + 7);
            reasoning = strip(reasoning);
            body.erase(0, body.rfind("</think>") + 8);
            const auto first = body.find_first_not_of('\n');
            body.erase(0, first == std::string::npos ? body.size() : first);
        }
        const auto& form = recipe.at(
            static_cast<std::int64_t>(i) > last_user ? "assistant_recent" : "assistant_history");
        output += value(form, "prefix");
        if (form.at("include_reasoning").get<bool>())
            output += reasoning + value(form, "reasoning_suffix");
        output += body + value(form, "suffix");
    }
    return output + value(recipe, "generation");
}

} // namespace trtmc::minicpmv4_6
