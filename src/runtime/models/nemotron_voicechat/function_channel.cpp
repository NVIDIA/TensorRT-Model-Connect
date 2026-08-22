/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/nemotron_voicechat/function_channel.h"

#include <algorithm>
#include <cctype>
#include <map>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace trtmc::nemotron_voicechat {

namespace {

using json = nlohmann::json;

bool is_ascii_text(std::string_view text) {
    return std::all_of(text.begin(), text.end(), [](unsigned char value) {
        return value == '\n' || value == '\r' || value == '\t' ||
               (value >= 0x20U && value <= 0x7eU);
    });
}

void require_ascii_text(std::string_view text, const char* label) {
    if (!is_ascii_text(text))
        throw std::invalid_argument(std::string(label) + " must contain printable ASCII only");
}

json parse_json(std::string_view text, const char* label) {
    if (text.empty())
        throw std::invalid_argument(std::string(label) + " must not be empty");
    try {
        return json::parse(text.begin(), text.end());
    } catch (const json::exception&) {
        throw std::invalid_argument(std::string(label) + " is not valid JSON");
    }
}

bool valid_tool_name(std::string_view name) {
    if (name.empty() || name.size() > 64)
        return false;
    return std::all_of(name.begin(), name.end(), [](unsigned char value) {
        return (value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z') ||
               (value >= '0' && value <= '9') || value == '_' || value == '-';
    });
}

bool json_strings_are_ascii(const json& value) {
    if (value.is_string())
        return is_ascii_text(value.get_ref<const std::string&>());
    if (value.is_array()) {
        return std::all_of(value.begin(), value.end(), json_strings_are_ascii);
    }
    if (value.is_object()) {
        for (const auto& item : value.items()) {
            if (!is_ascii_text(item.key()) || !json_strings_are_ascii(item.value()))
                return false;
        }
    }
    return true;
}

const json& tool_body(const json& entry) {
    if (!entry.is_object())
        throw std::invalid_argument("each function tool must be an object");
    if (!entry.contains("function")) {
        if (entry.contains("type") &&
            (!entry.at("type").is_string() || entry.at("type") != "function"))
            throw std::invalid_argument("function tool type must be function");
        return entry;
    }
    if (!entry.contains("type") || !entry.at("type").is_string() ||
        entry.at("type") != "function" || !entry.at("function").is_object())
        throw std::invalid_argument(
            "nested function tools require type=function and an object body");
    return entry.at("function");
}

std::string optional_ascii_string(const json& object, const char* field) {
    if (!object.contains(field))
        return {};
    if (!object.at(field).is_string())
        throw std::invalid_argument(std::string("tool ") + field + " must be a string");
    auto value = object.at(field).get<std::string>();
    require_ascii_text(value, field);
    return value;
}

std::vector<std::string> parse_ascii_messages(const json& value, const char* label,
                                              bool allow_single) {
    if (allow_single && value.is_string()) {
        auto message = value.get<std::string>();
        require_ascii_text(message, label);
        return {std::move(message)};
    }
    if (!value.is_array() || value.empty())
        throw std::invalid_argument(std::string(label) + " must be a non-empty string array");
    std::vector<std::string> messages;
    for (const auto& item : value) {
        if (!item.is_string() || item.get_ref<const std::string&>().empty())
            throw std::invalid_argument(std::string(label) + " entries must be non-empty strings");
        messages.push_back(item.get<std::string>());
        require_ascii_text(messages.back(), label);
    }
    return messages;
}

std::string parse_tool_name(const json& body) {
    if (!body.contains("name") || !body.at("name").is_string())
        throw std::invalid_argument("function tool name must be a string");
    auto name = body.at("name").get<std::string>();
    if (!valid_tool_name(name))
        throw std::invalid_argument(
            "function tool name must use 1-64 ASCII letters, digits, _ or -");
    return name;
}

std::string parse_parameters_json(const json& body) {
    const json parameters = body.value("parameters", json::object());
    if (!parameters.is_object())
        throw std::invalid_argument("function tool parameters must be an object schema");
    if (parameters.contains("type") && parameters.at("type") != "object")
        throw std::invalid_argument("function tool parameters must be an object schema");
    if (!json_strings_are_ascii(parameters))
        throw std::invalid_argument("function tool parameters must contain printable ASCII only");
    return parameters.dump();
}

const json* find_ack_messages(const json& entry, const json& body) {
    if (body.contains("ack_messages"))
        return &body.at("ack_messages");
    if (&body != &entry && entry.contains("ack_messages"))
        return &entry.at("ack_messages");
    return nullptr;
}

FunctionTool parse_tool(const json& entry) {
    const auto& body = tool_body(entry);
    FunctionTool tool;
    tool.name = parse_tool_name(body);
    tool.description = optional_ascii_string(body, "description");
    tool.parameters_json = parse_parameters_json(body);
    if (const auto* acknowledgements = find_ack_messages(entry, body)) {
        auto messages = parse_ascii_messages(*acknowledgements, "ack_messages", false);
        tool.ack_message = std::move(messages.front());
    }
    return tool;
}

using OnHoldMessages = std::map<std::string, std::string, std::less<>>;

OnHoldMessages parse_on_hold_messages(std::string_view messages_json) {
    OnHoldMessages messages;
    if (messages_json.empty())
        return messages;
    const auto root = parse_json(messages_json, "on-hold messages");
    if (!root.is_object())
        throw std::invalid_argument("on-hold messages must be a JSON object");
    for (const auto& item : root.items()) {
        if (item.key() != "default" && !valid_tool_name(item.key()))
            throw std::invalid_argument("on-hold message keys must be tool names or default");
        auto values = parse_ascii_messages(item.value(), "on-hold message", true);
        messages.emplace(item.key(), std::move(values.front()));
    }
    return messages;
}

void apply_on_hold_message(FunctionTool& tool, const OnHoldMessages& messages) {
    auto match = messages.find(tool.name);
    if (match == messages.end())
        match = messages.find("default");
    if (match != messages.end())
        tool.ack_message = match->second;
}

json parse_arguments(const json& call) {
    if (!call.contains("arguments"))
        return json::object();
    const auto& arguments = call.at("arguments");
    if (arguments.is_object())
        return arguments;
    if (arguments.is_string()) {
        const auto parsed = parse_json(arguments.get_ref<const std::string&>(), "tool arguments");
        if (parsed.is_object())
            return parsed;
    }
    throw std::invalid_argument("tool call arguments must be an object or encoded object");
}

std::string make_call_id(std::uint64_t epoch, std::uint64_t serial, std::size_t index) {
    std::ostringstream value;
    value << "call_" << epoch << '_' << serial << '_' << index;
    return value.str();
}

json normalize_result(std::string_view result_text) {
    require_ascii_text(result_text, "tool result");
    try {
        return json::parse(result_text.begin(), result_text.end());
    } catch (const json::exception&) {
        return std::string(result_text);
    }
}

std::string jinja_json(const json& value) {
    const std::string compact = value.dump();
    std::string rendered;
    rendered.reserve(compact.size() + compact.size() / 8U);
    bool in_string = false;
    bool escaped = false;
    for (const char character : compact) {
        rendered.push_back(character);
        if (in_string) {
            if (escaped)
                escaped = false;
            else if (character == '\\')
                escaped = true;
            else if (character == '"')
                in_string = false;
            continue;
        }
        if (character == '"')
            in_string = true;
        else if (character == ':' || character == ',')
            rendered.push_back(' ');
    }
    return rendered;
}

std::string_view strip_tool_call_wrapper(std::string_view text) {
    constexpr std::string_view kOpen = "<TOOLCALL>";
    constexpr std::string_view kClose = "</TOOLCALL>";
    const auto first = text.find_first_not_of(" \t\r\n");
    const auto last = text.find_last_not_of(" \t\r\n");
    if (first == std::string_view::npos)
        return {};
    text = text.substr(first, last - first + 1U);
    const bool has_open = text.substr(0, kOpen.size()) == kOpen;
    const bool has_close =
        text.size() >= kClose.size() && text.substr(text.size() - kClose.size()) == kClose;
    if (has_open != has_close)
        throw std::invalid_argument("tool call has an incomplete TOOLCALL wrapper");
    if (!has_open)
        return text;
    return text.substr(kOpen.size(), text.size() - kOpen.size() - kClose.size());
}

} // namespace

FunctionToolCatalog FunctionToolCatalog::from_json(std::string_view tools_json,
                                                   std::string_view on_hold_messages_json) {
    const auto root = parse_json(tools_json, "tools");
    if (!root.is_array() || root.empty())
        throw std::invalid_argument("tools must be a non-empty JSON array");

    FunctionToolCatalog catalog;
    const auto on_hold_messages = parse_on_hold_messages(on_hold_messages_json);
    for (const auto& entry : root) {
        auto tool = parse_tool(entry);
        if (catalog.find(tool.name) != nullptr)
            throw std::invalid_argument("function tool names must be unique");
        apply_on_hold_message(tool, on_hold_messages);
        catalog.tools_.push_back(std::move(tool));
    }
    return catalog;
}

const FunctionTool* FunctionToolCatalog::find(std::string_view name) const {
    const auto match = std::find_if(tools_.begin(), tools_.end(),
                                    [&](const FunctionTool& tool) { return tool.name == name; });
    return match == tools_.end() ? nullptr : &*match;
}

std::string FunctionToolCatalog::protocol_json() const {
    json root = json::array();
    for (const auto& tool : tools_) {
        json item = {{"name", tool.name}, {"parameters", json::parse(tool.parameters_json)}};
        if (!tool.description.empty())
            item["description"] = tool.description;
        root.push_back(std::move(item));
    }
    return jinja_json(root);
}

std::string render_function_system_prompt(std::string_view base_prompt,
                                          const FunctionToolCatalog& tools) {
    // Byte-compatible with public Speech commit 097dfe9e2f55,
    // examples/speechlm2/function_calling/template.jinja. The checkpoint is
    // prompt-sensitive, so keep the protocol text and JSON spacing stable.
    require_ascii_text(base_prompt, "system prompt");
    std::string result(base_prompt);
    if (!result.empty() && result.back() != '\n')
        result += "\n\n";
    result += "You can use the following tools to assist the user if required:\n";
    result += "<AVAILABLE_TOOLS>" + tools.protocol_json() + "</AVAILABLE_TOOLS>\n";
    result += "\nIf you decide to call any tool(s), use the following format:\n";
    result += "<TOOLCALL>[{\"name\": \"tool_name1\", \"arguments\": \"tool_args1\"}, "
              "{\"name\": \"tool_name2\", \"arguments\": \"tool_args2\"}]</TOOLCALL>\n\n";
    result += "The user will execute tool-calls and return responses from tool(s) in this "
              "format:\n";
    result += "<TOOL_RESPONSE>[{\"tool_response1\"}, {\"tool_response2\"}]"
              "</TOOL_RESPONSE>\n\n";
    result += "Based on the tool responses, you can call additional tools if needed, correct "
              "tool calls if any errors are found, or just respond to the user.";
    return result;
}

std::string_view default_function_system_message() {
    // Public VoiceChat function-calling default from Speech commit
    // 097dfe9e2f55. Keep this ASCII prompt stable for checkpoint parity.
    static constexpr std::string_view message =
        "You are an AI voice assistant developed by NVIDIA. Your name is NVIDIA Voice Chat. "
        "Your job is to be helpful and harmless and have engaging conversations in English. "
        "Maintain a warm and friendly tone. Keep the dialogue open and ongoing. Be clear and "
        "direct, especially when answering yes or no questions and multiple-choice questions. "
        "Avoid long answers unless the user asks you to provide details or context. You must "
        "provide diverse responses and rephrase answers if the user asks the same question. DO "
        "NOT interrupt the user when they are speaking, let them finish their turn before "
        "answering.\n\nWhen you receive a request, follow this decision process:\n1. Does the "
        "request match one of your available tools below? If yes, you MUST call that tool - never "
        "answer it directly from your own knowledge, even if you think you know the answer.\n2. "
        "Is it a general knowledge question (history, science, geography, math, facts, etc.)? If "
        "yes, answer directly from your own knowledge - do not call any tool.\n3. Does it require "
        "an external action or live data that none of your tools cover (e.g. ordering food, "
        "sending email)? If yes, politely say you don't have that capability.\n\nNEVER say \"I "
        "don't have a tool for that\" for general knowledge questions you can answer yourself.\n\n"
        "DO NOT use any tools when not needed to answer the user's requests, under no "
        "circumstance.\n\nYou are an expert across history, geography, science, math, literature, "
        "biographies, languages, recipes, programming, current affairs, and general knowledge. "
        "When the user asks about any of these, answer directly and conversationally from your "
        "own knowledge - no <TOOLCALL>.\n\nCall a tool ONLY when the user's request matches one "
        "of the tools listed in <AVAILABLE_TOOLS> below. For every other request, do not call any "
        "tool - just answer from your knowledge. Never invent or call a tool name that is not "
        "literally in <AVAILABLE_TOOLS>.\n\nTool-call arguments must be values the user spoke. If "
        "a required argument is missing, ask the user; never guess.\n\nIf a tool call fails or "
        "returns an error, do not retry the tool call for the same request. Tell the user that the "
        "API has an issue.";
    return message;
}

std::vector<FunctionCall> parse_function_calls(std::string_view calls_json,
                                               const FunctionToolCatalog& tools,
                                               std::uint64_t epoch, std::uint64_t serial) {
    const auto root = parse_json(strip_tool_call_wrapper(calls_json), "tool calls");
    if (!root.is_array() || root.empty())
        throw std::invalid_argument("tool calls must be a non-empty JSON array");

    std::vector<FunctionCall> calls;
    calls.reserve(root.size());
    for (std::size_t index = 0; index < root.size(); ++index) {
        const auto& item = root.at(index);
        if (!item.is_object() || !item.contains("name") || !item.at("name").is_string())
            throw std::invalid_argument("each tool call must contain a string name");
        const auto name = item.at("name").get<std::string>();
        if (tools.find(name) == nullptr)
            throw std::invalid_argument("tool call references an unknown tool: " + name);
        calls.push_back({make_call_id(epoch, serial, index), name, parse_arguments(item).dump()});
    }
    return calls;
}

FunctionChannelState::FunctionChannelState(std::size_t max_call_tokens)
    : max_call_tokens_(max_call_tokens) {
    if (max_call_tokens_ == 0)
        throw std::invalid_argument("function call token limit must be positive");
    call_tokens_.reserve(max_call_tokens_);
}

FunctionChannelObservation
FunctionChannelState::observe(int32_t token_id, std::uint64_t epoch, std::uint64_t serial,
                              const FunctionToolCatalog& tools,
                              const DecodeFunctionTokens& decode_tokens) {
    switch (phase_) {
    case Phase::kIdle:
        return observe_idle(token_id);
    case Phase::kCapturingCall:
        return observe_call(token_id, epoch, serial, tools, decode_tokens);
    case Phase::kAwaitingResponseEnd:
        return observe_response(token_id);
    }
    return fail("function channel entered an invalid state");
}

FunctionChannelObservation FunctionChannelState::observe_idle(int32_t token_id) {
    if (token_id == kFunctionSotcTokenId) {
        phase_ = Phase::kCapturingCall;
        call_tokens_.clear();
        return {FunctionChannelObservationKind::kCallStarted, {}, {}};
    }
    if (token_id == kFunctionEotcTokenId || token_id == kFunctionEotrTokenId)
        return fail("function channel received an end marker without SOTC");
    return {};
}

FunctionChannelObservation
FunctionChannelState::observe_call(int32_t token_id, std::uint64_t epoch, std::uint64_t serial,
                                   const FunctionToolCatalog& tools,
                                   const DecodeFunctionTokens& decode_tokens) {
    if (token_id == kFunctionSotcTokenId)
        return fail("function channel received nested SOTC");
    if (token_id == kFunctionEotrTokenId)
        return fail("function channel received EOTR before EOTC");
    if (token_id == kFunctionEotcTokenId)
        return complete_call(epoch, serial, tools, decode_tokens);
    if (token_id == kFunctionPadTokenId)
        return {};
    if (call_tokens_.size() >= max_call_tokens_)
        return fail("function call exceeded its token limit");
    call_tokens_.push_back(token_id);
    return {};
}

FunctionChannelObservation FunctionChannelState::observe_response(int32_t token_id) {
    if (token_id == kFunctionEotrTokenId) {
        reset();
        return {FunctionChannelObservationKind::kResponseFinished, {}, {}};
    }
    if (token_id == kFunctionSotcTokenId || token_id == kFunctionEotcTokenId)
        return fail("function channel received a call marker before EOTR");
    return {};
}

FunctionChannelObservation
FunctionChannelState::complete_call(std::uint64_t epoch, std::uint64_t serial,
                                    const FunctionToolCatalog& tools,
                                    const DecodeFunctionTokens& decode_tokens) {
    if (!decode_tokens)
        return fail("function channel requires a token decoder");
    try {
        auto calls = parse_function_calls(decode_tokens(call_tokens_), tools, epoch, serial);
        call_tokens_.clear();
        phase_ = Phase::kAwaitingResponseEnd;
        return {FunctionChannelObservationKind::kCallsReady, std::move(calls), {}};
    } catch (const std::exception& error) {
        return fail(error.what());
    }
}

FunctionChannelObservation FunctionChannelState::fail(std::string message) {
    reset();
    return {FunctionChannelObservationKind::kError, {}, std::move(message)};
}

void FunctionChannelState::reset() {
    phase_ = Phase::kIdle;
    call_tokens_.clear();
}

bool FunctionChannelState::active() const {
    return phase_ != Phase::kIdle;
}

bool FunctionChannelState::capturing_call() const {
    return phase_ == Phase::kCapturingCall;
}

bool FunctionChannelState::awaiting_response_end() const {
    return phase_ == Phase::kAwaitingResponseEnd;
}

std::vector<int32_t> build_tool_response_tokens(std::string_view result_json,
                                                const EncodeFunctionText& encode_text) {
    if (!encode_text)
        throw std::invalid_argument("tool response requires a tokenizer callback");
    auto result = normalize_result(result_json);
    if (!json_strings_are_ascii(result))
        throw std::invalid_argument("tool result must contain printable ASCII only");
    const json payload = result.is_array() ? std::move(result) : json::array({std::move(result)});
    const auto text = "<TOOL_RESPONSE>" + payload.dump() + "</TOOL_RESPONSE>";
    auto tokens = encode_text(text);
    if (tokens.empty())
        throw std::invalid_argument("tool response tokenizer returned no tokens");
    return tokens;
}

} // namespace trtmc::nemotron_voicechat
