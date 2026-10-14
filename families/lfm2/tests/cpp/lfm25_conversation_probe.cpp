/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/lfm2/runtime/conversation.h"

#include <deque>
#include <iostream>
#include <nlohmann/json.hpp>

using Json = nlohmann::ordered_json;
using namespace trtmc;
using namespace trtmc::internal;

MessageRole role(const std::string& value) {
    if (value == "system")
        return MessageRole::System;
    if (value == "developer")
        return MessageRole::Developer;
    if (value == "user")
        return MessageRole::User;
    if (value == "assistant")
        return MessageRole::Assistant;
    if (value == "tool")
        return MessageRole::Tool;
    throw std::invalid_argument("invalid fixture role");
}

int main() {
    std::string line;
    while (std::getline(std::cin, line)) {
        try {
            const auto input = Json::parse(line);
            Json output;
            if (input.contains("decoded")) {
                auto result = lfm2_parse_conversation(input.at("decoded"),
                                                      input.value("starts_in_reasoning", false),
                                                      input.value("allow_json_tools", false));
                std::vector<ConversationPartView> call_history, result_history;
                if (input.contains("used_call_ids"))
                    for (const auto& id : input.at("used_call_ids"))
                        call_history.push_back(ToolCallView{id.get_ref<const std::string&>(),
                                                            "prior", "{}",
                                                            ToolCallState::Complete});
                if (input.contains("used_result_ids"))
                    for (const auto& id : input.at("used_result_ids"))
                        result_history.push_back(
                            ToolResultView{id.get_ref<const std::string&>(), "done", false});
                const ConversationMessageView history[] = {
                    {MessageRole::Assistant, {call_history.data(), call_history.size()}},
                    {MessageRole::Tool, {result_history.data(), result_history.size()}}};
                lfm2_assign_tool_call_ids(result, {history, {}});
                output["parts"] = Json::array();
                for (const auto& part : result.parts) {
                    if (const auto* text = std::get_if<TextPart>(&part))
                        output["parts"].push_back({{"type", "text"}, {"text", text->text}});
                    else if (const auto* reasoning = std::get_if<ReasoningPart>(&part))
                        output["parts"].push_back(
                            {{"type", "reasoning"}, {"text", reasoning->text}});
                    else {
                        const auto& call = std::get<ToolCall>(part);
                        output["parts"].push_back({{"type", "tool_call"},
                                                   {"call_id", call.call_id},
                                                   {"name", call.name},
                                                   {"arguments_json", call.arguments_json},
                                                   {"state", static_cast<int>(call.state)}});
                    }
                }
            } else {
                std::deque<std::string> strings;
                const auto save = [&](std::string text) -> std::string_view {
                    strings.push_back(std::move(text));
                    return strings.back();
                };
                std::vector<std::vector<ConversationPartView>> parts;
                std::vector<ConversationMessageView> messages;
                for (const auto& item : input.at("messages")) {
                    parts.emplace_back();
                    auto& value = parts.back();
                    if (item.contains("content") && !item.at("content").is_null()) {
                        if (item.at("role") == "tool")
                            value.push_back(ToolResultView{
                                {}, save(item.at("content").get<std::string>()), false});
                        else
                            value.push_back(
                                TextPartView{save(item.at("content").get<std::string>())});
                    }
                    if (item.contains("reasoning"))
                        value.push_back(
                            ReasoningPartView{save(item.at("reasoning").get<std::string>())});
                    if (item.contains("tool_calls"))
                        for (const auto& call : item.at("tool_calls")) {
                            const auto& function = call.at("function");
                            value.push_back(
                                ToolCallView{{},
                                             save(function.at("name").get<std::string>()),
                                             save(function.at("arguments").dump()),
                                             ToolCallState::Complete});
                        }
                    messages.push_back({role(item.at("role")), {value.data(), value.size()}});
                }
                std::vector<ToolDefinitionView> tools;
                for (const auto& item : input.value("tools", Json::array()))
                    tools.push_back({save(item.at("name").get<std::string>()),
                                     save(item.at("description").get<std::string>()),
                                     save(item.at("parameters").dump())});
                output["rendered"] = lfm2_render_conversation(
                    {{messages.data(), messages.size()}, {tools.data(), tools.size()}},
                    input.value("preserve_thinking", false));
            }
            std::cout << output.dump() << '\n';
        } catch (const std::exception& error) {
            std::cout << Json{{"error", error.what()}}.dump() << '\n';
        }
    }
}
