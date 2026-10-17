/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Reproducible model-card validation client using the public typed SDK.
// Inputs are read from a file so full-context prompts never depend on argv limits.
#include "trtmc/control.hpp"
#include "trtmc/language.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <set>

using Json = nlohmann::ordered_json;

trtmc::MessageRole role(const std::string& value) {
    if (value == "system")
        return trtmc::MessageRole::System;
    if (value == "developer")
        return trtmc::MessageRole::Developer;
    if (value == "user")
        return trtmc::MessageRole::User;
    if (value == "assistant")
        return trtmc::MessageRole::Assistant;
    if (value == "tool")
        return trtmc::MessageRole::Tool;
    throw std::invalid_argument("unknown message role");
}

std::string transport_id(const Json& value, const char* field) {
    if (!value.contains(field) || value.at(field).is_null())
        return {};
    if (!value.at(field).is_string())
        throw std::invalid_argument("tool transport IDs must be strings");
    return value.at(field).get<std::string>();
}

Json prepare_transport_request(Json input) {
    if (!input.contains("messages"))
        return input;
    std::set<std::string> reserved, explicit_calls, returned;
    for (const auto& message : input.at("messages")) {
        for (const auto& call : message.value("tool_calls", Json::array())) {
            const auto id = transport_id(call, "id");
            if (!id.empty()) {
                if (!explicit_calls.insert(id).second)
                    throw std::invalid_argument("duplicate explicit tool-call ID");
                reserved.insert(id);
            }
        }
        if (message.at("role") == "tool") {
            const auto id = transport_id(message, "tool_call_id");
            if (!id.empty())
                reserved.insert(id);
        }
    }
    std::vector<std::string> pending;
    std::size_t next = 0;
    for (auto& message : input.at("messages")) {
        if (message.contains("tool_calls")) {
            for (auto& call : message.at("tool_calls")) {
                auto id = transport_id(call, "id");
                if (id.empty()) {
                    do {
                        id = "call_" + std::to_string(next++);
                    } while (reserved.count(id));
                    reserved.insert(id);
                    call["id"] = id;
                }
                pending.push_back(id);
            }
        }
        if (message.at("role") == "tool") {
            auto id = transport_id(message, "tool_call_id");
            if (id.empty()) {
                if (pending.empty())
                    throw std::invalid_argument(
                        "ID-free tool result has no preceding unmatched call");
                id = pending.front();
                message["tool_call_id"] = id;
            }
            if (!returned.insert(id).second)
                throw std::invalid_argument("duplicate tool-result ID");
            const auto found = std::find(pending.begin(), pending.end(), id);
            if (found != pending.end())
                pending.erase(found);
        }
    }
    return input;
}

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--inspect-bundle") {
            const auto bundle = trtmc::Bundle::open(argv[2]);
            const auto metadata = bundle.info();
            const auto section = bundle.read_section("runtime.json");
            const auto bytes = section.bytes();
            const auto runtime = Json::parse(bytes.begin(), bytes.end());
            std::cout << Json{{"family", metadata.family},
                              {"task", metadata.task},
                              {"backend", metadata.backend},
                              {"precision", runtime.at("precision")},
                              {"max_cache_length", runtime.at("max_cache_length")}}
                             .dump()
                      << '\n';
            return 0;
        }
        const bool normalize_only = argc == 3 && std::string(argv[1]) == "--normalize-request";
        if (argc != 4 && !normalize_only)
            throw std::invalid_argument("usage: lfm25_sdk_client BUNDLE RUNTIME_ROOT REQUEST.json");
        std::ifstream file(argv[normalize_only ? 2 : 3]);
        if (!file)
            throw std::invalid_argument("cannot open request JSON");
        const auto input = prepare_transport_request(Json::parse(file));
        if (normalize_only) {
            std::cout << input.dump() << '\n';
            return 0;
        }
        trtmc::Config config;
        if (input.contains("config")) {
            for (auto it = input.at("config").begin(); it != input.at("config").end(); ++it) {
                if (it->is_boolean())
                    config.add(it.key(), it->get<bool>());
                else if (it->is_number_integer())
                    config.add(it.key(), it->get<std::int64_t>());
                else if (it->is_number_float())
                    config.add(it.key(), it->get<double>());
                else if (it->is_string())
                    config.add(it.key(), it->get<std::string>());
                else
                    throw std::invalid_argument("config requires scalar values");
            }
        }
        trtmc::LoadOptions options;
        options.runtime_root = argv[2];
        const auto model = trtmc::Model::load(argv[1], options);
        Json output;
        if (input.contains("messages")) {
            trtmc::TextConversationRequest request;
            for (const auto& source : input.at("messages")) {
                trtmc::ConversationMessage message{role(source.at("role")), {}};
                if (source.contains("content") && !source.at("content").is_null()) {
                    if (message.role == trtmc::MessageRole::Tool)
                        message.parts.emplace_back(
                            trtmc::ToolResult{source.value("tool_call_id", ""),
                                              source.at("content").get<std::string>(),
                                              source.value("is_error", false)});
                    else
                        message.parts.emplace_back(
                            trtmc::TextPart{source.at("content").get<std::string>()});
                }
                if (source.contains("reasoning"))
                    message.parts.emplace_back(
                        trtmc::ReasoningPart{source.at("reasoning").get<std::string>()});
                if (source.contains("tool_calls")) {
                    for (const auto& call : source.at("tool_calls")) {
                        const auto& function = call.at("function");
                        message.parts.emplace_back(trtmc::ToolCall{
                            call.value("id", ""), function.at("name").get<std::string>(),
                            function.at("arguments").dump(), trtmc::ToolCallState::Complete});
                    }
                }
                request.messages.push_back(std::move(message));
            }
            for (const auto& tool : input.value("tools", Json::array()))
                request.tools.push_back(
                    {tool.at("name"), tool.at("description"), tool.at("parameters").dump()});
            const auto result = model.task<trtmc::TextConversation>().run(request, config);
            output["parts"] = Json::array();
            for (const auto& part : result.message().parts) {
                if (const auto* text = std::get_if<trtmc::TextPart>(&part))
                    output["parts"].push_back({{"type", "text"}, {"text", text->text}});
                else if (const auto* reasoning = std::get_if<trtmc::ReasoningPart>(&part))
                    output["parts"].push_back({{"type", "reasoning"}, {"text", reasoning->text}});
                else {
                    const auto& call = std::get<trtmc::ToolCall>(part);
                    output["parts"].push_back({{"type", "tool_call"},
                                               {"call_id", call.call_id},
                                               {"name", call.name},
                                               {"arguments_json", call.arguments_json},
                                               {"state", static_cast<int>(call.state)}});
                }
            }
            const auto ids = result.result().token_ids();
            output["token_ids"] = std::vector<std::int32_t>(ids.begin(), ids.end());
            output["finish_reason"] = static_cast<int>(result.finish_reason());
            const auto usage = result.usage();
            if (usage.input_tokens)
                output["input_tokens"] = *usage.input_tokens;
            if (usage.output_tokens)
                output["output_tokens"] = *usage.output_tokens;
        } else {
            trtmc::TextContinuationRequest request;
            if (input.contains("token_ids"))
                request.prefix = input.at("token_ids").get<std::vector<std::int32_t>>();
            else
                request.prefix = input.at("prompt").get<std::string>();
            const auto result = model.task<trtmc::TextContinuation>().run(request, config);
            output["text"] = result.text();
            const auto ids = result.token_ids();
            output["token_ids"] = std::vector<std::int32_t>(ids.begin(), ids.end());
        }
        std::cout << output.dump() << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
