/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/lfm2_vl/runtime/conversation.h"
#include "families/lfm2_vl/runtime/pipeline.h"

#include <deque>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>

int main() {
    using namespace trtmc::internal;
    try {
        nlohmann::ordered_json fixture;
        std::ifstream(TRTMC_LFM2_VL_FIXTURE_DIR "/chat-rendering-reference.json") >> fixture;
        for (const auto& test : fixture.at("cases")) {
            std::deque<std::string> text;
            auto keep = [&](std::string value) -> std::string_view {
                text.push_back(std::move(value));
                return text.back();
            };
            std::vector<std::vector<ImagesTextPart>> parts;
            std::vector<ImagesTextMessage> messages;
            for (const auto& item : test.at("messages")) {
                parts.emplace_back();
                if (item.contains("reasoning"))
                    parts.back().emplace_back(ReasoningPartView{keep(item.at("reasoning"))});
                parts.back().emplace_back(TextPartView{keep(item.at("content"))});
            }
            for (std::size_t i = 0; i < parts.size(); ++i) {
                const auto role = test.at("messages").at(i).at("role").get<std::string>();
                const auto value = role == "system"      ? MessageRole::System
                                   : role == "assistant" ? MessageRole::Assistant
                                                         : MessageRole::User;
                messages.push_back({value, {parts[i].data(), parts[i].size()}});
            }
            std::vector<ToolDefinitionView> tools;
            for (const auto& tool : test.value("tools", nlohmann::ordered_json::array()))
                tools.push_back({keep(tool.at("name")), keep(tool.at("description")),
                                 keep(tool.at("parameters").dump())});
            std::vector<ConfigEntry> config;
            if (test.contains("keep_past_thinking"))
                config.push_back({"keep_past_thinking", test.at("keep_past_thinking").get<bool>()});
            const auto prepared = trtmc::lfm2_vl_prepare_request(
                {{messages.data(), messages.size()}, {tools.data(), tools.size()}},
                {config.data(), config.size()});
            if (prepared.prompt != test.at("expected").get<std::string>())
                throw std::runtime_error("original template mismatch: " +
                                         test.at("name").get<std::string>());
        }
        auto result = trtmc::lfm2_vl_parse_conversation(
            "<|tool_call_start|>[get_weather(location=\"Paris\", options={'units': 'C', "
            "'days': [1, 2], 'alert': True, 'extra': None})]<|tool_call_end|>Checking now.",
            false);
        if (result.parts.size() != 2)
            throw std::runtime_error("tool and text must remain distinct parts");
        const auto& call = std::get<ToolCall>(result.parts[0]);
        const auto args = nlohmann::json::parse(call.arguments_json);
        if (call.state != ToolCallState::Complete || call.name != "get_weather" ||
            args.at("location") != "Paris" ||
            args.at("options").at("days") != nlohmann::json({1, 2}) ||
            args.at("options").at("alert") != true || !args.at("options").at("extra").is_null() ||
            std::get<TextPart>(result.parts[1]).text != "Checking now.")
            throw std::runtime_error("literal tool arguments changed during parsing");
        for (const auto& input : {std::string("[run(path=__import__('os').system('command'))]"),
                                  std::string("[run(a=1, a=2)]")}) {
            const auto malformed = trtmc::lfm2_vl_parse_conversation(
                "<|tool_call_start|>" + input + "<|tool_call_end|>", false);
            if (std::get<ToolCall>(malformed.parts.at(0)).state != ToolCallState::Malformed)
                throw std::runtime_error("nonliteral or duplicate arguments must fail closed");
        }
        const auto incomplete =
            trtmc::lfm2_vl_parse_conversation("<|tool_call_start|>[get_weather(location=", false);
        if (std::get<ToolCall>(incomplete.parts.at(0)).state != ToolCallState::Incomplete)
            throw std::runtime_error("truncated calls must retain incomplete state");
        ToolCallView prior{"call_0", "get_weather", "{}", ToolCallState::Complete};
        ConversationPartView part = prior;
        ConversationMessageView message{MessageRole::Assistant, {&part, 1}};
        trtmc::lfm2_vl_assign_tool_call_ids(result, {{&message, 1}, {}});
        if (std::get<ToolCall>(result.parts[0]).call_id != "call_1")
            throw std::runtime_error("generated call IDs must not collide with history");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
