/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/nemotron_voicechat/function_channel.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace voicechat = trtmc::nemotron_voicechat;

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

template <typename Callable>
void expect_invalid(Callable&& callable, const char* message) {
    bool rejected = false;
    try {
        callable();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, message);
}

voicechat::FunctionToolCatalog make_tools(std::string_view on_hold_messages = {}) {
    return voicechat::FunctionToolCatalog::from_json(R"json(
[
  {
    "name": "get_weather",
    "description": "Get current weather for a city",
    "ack_messages": ["Let me check the weather."],
    "parameters": {
      "type": "object",
      "properties": {"city": {"type": "string"}},
      "required": ["city"]
    }
  },
  {
    "type": "function",
    "function": {
      "name": "add",
      "description": "Add two integers",
      "parameters": {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}
      }
    },
    "ack_messages": ["I will add those numbers."]
  }
]
)json",
                                                     on_hold_messages);
}

voicechat::FunctionChannelObservation observe_call(voicechat::FunctionChannelState& state,
                                                   const voicechat::FunctionToolCatalog& tools,
                                                   std::string decoded, std::uint64_t epoch = 7,
                                                   std::uint64_t serial = 9) {
    const auto started = state.observe(voicechat::kFunctionSotcTokenId, epoch, serial, tools, {});
    check(started.kind == voicechat::FunctionChannelObservationKind::kCallStarted,
          "SOTC starts incremental function capture");
    (void)state.observe(101, epoch, serial, tools, {});
    (void)state.observe(voicechat::kFunctionPadTokenId, epoch, serial, tools, {});
    (void)state.observe(102, epoch, serial, tools, {});
    return state.observe(voicechat::kFunctionEotcTokenId, epoch, serial, tools,
                         [decoded = std::move(decoded)](const std::vector<int32_t>& tokens) {
                             check(tokens == std::vector<int32_t>({101, 102}),
                                   "function capture omits PAD and preserves token order");
                             return decoded;
                         });
}

void test_exact_markers_and_tool_catalog() {
    static_assert(voicechat::kFunctionSotcTokenId == 20);
    static_assert(voicechat::kFunctionEotcTokenId == 21);
    static_assert(voicechat::kFunctionEotrTokenId == 22);

    const auto tools = make_tools();
    check(tools.find("get_weather") != nullptr && tools.find("add") != nullptr &&
              tools.find("missing") == nullptr,
          "flat and nested OpenAI-style tools are indexed by name");
    check(tools.find("get_weather")->ack_message == "Let me check the weather." &&
              tools.find("add")->ack_message == "I will add those numbers.",
          "tool acknowledgement messages accept flat and nested definitions");

    const auto protocol = tools.protocol_json();
    check(protocol.find("get_weather") != std::string::npos &&
              protocol.find("ack_messages") == std::string::npos,
          "model protocol contains tool schemas but not host-only acknowledgements");
    const auto prompt = voicechat::render_function_system_prompt("Be helpful.", tools);
    const std::string expected_prompt =
        "Be helpful.\n\nYou can use the following tools to assist the user if required:\n"
        "<AVAILABLE_TOOLS>[{\"description\": \"Get current weather for a city\", \"name\": "
        "\"get_weather\", \"parameters\": {\"properties\": {\"city\": {\"type\": "
        "\"string\"}}, \"required\": [\"city\"], \"type\": \"object\"}}, {\"description\": "
        "\"Add two integers\", \"name\": \"add\", \"parameters\": {\"properties\": {\"a\": "
        "{\"type\": \"integer\"}, \"b\": {\"type\": \"integer\"}}, \"type\": "
        "\"object\"}}]</AVAILABLE_TOOLS>\n\nIf you decide to call any tool(s), use the following "
        "format:\n<TOOLCALL>[{\"name\": \"tool_name1\", \"arguments\": \"tool_args1\"}, "
        "{\"name\": \"tool_name2\", \"arguments\": \"tool_args2\"}]</TOOLCALL>\n\nThe user "
        "will execute tool-calls and return responses from tool(s) in this format:\n"
        "<TOOL_RESPONSE>[{\"tool_response1\"}, {\"tool_response2\"}]</TOOL_RESPONSE>\n\n"
        "Based on the tool responses, you can call additional tools if needed, correct tool "
        "calls if any errors are found, or just respond to the user.";
    check(prompt == expected_prompt,
          "function system prompt matches the pinned public Jinja rendering byte for byte");
    const auto default_prompt = voicechat::default_function_system_message();
    check(default_prompt.find("DO NOT interrupt the user") != std::string_view::npos &&
              default_prompt.find("Never invent or call a tool name") != std::string_view::npos &&
              default_prompt.find("API has an issue") != std::string_view::npos,
          "function default preserves the pinned selection and failure instructions");
}

void test_tool_validation() {
    expect_invalid(
        [] { (void)voicechat::FunctionToolCatalog::from_json(R"({"name":"not-array"})"); },
        "tool catalog rejects a non-array root");
    expect_invalid(
        [] { (void)voicechat::FunctionToolCatalog::from_json(R"([{"name":"bad name"}])"); },
        "tool catalog rejects invalid names");
    expect_invalid(
        [] {
            (void)voicechat::FunctionToolCatalog::from_json(R"([{"name":"dup"},{"name":"dup"}])");
        },
        "tool catalog rejects duplicate names");
    expect_invalid(
        [] {
            (void)voicechat::FunctionToolCatalog::from_json(
                R"([{"name":"bad_schema","parameters":{"type":"array"}}])");
        },
        "tool catalog requires object parameter schemas");
    expect_invalid(
        [] {
            (void)voicechat::FunctionToolCatalog::from_json(
                R"([{"type":"retrieval","name":"wrong_type"}])");
        },
        "tool catalog rejects non-function tool types");
    expect_invalid(
        [] {
            (void)voicechat::FunctionToolCatalog::from_json(
                R"([{"name":"unicode","description":"\u00e9"}])");
        },
        "tool catalog rejects non-ASCII descriptions");

    const auto tools = make_tools();
    expect_invalid(
        [&] {
            (void)voicechat::render_function_system_prompt(
                std::string("bad ") + static_cast<char>(0xc3) + static_cast<char>(0xa9), tools);
        },
        "system prompt rejects non-ASCII input");
}

void test_natural_and_multi_call_observation() {
    const auto tools = make_tools();
    voicechat::FunctionChannelState state;
    const auto ready =
        observe_call(state, tools, R"([{"name":"get_weather","arguments":{"city":"Paris"}}])");
    check(ready.kind == voicechat::FunctionChannelObservationKind::kCallsReady &&
              ready.calls.size() == 1 && ready.calls[0].call_id == "call_7_9_0" &&
              ready.calls[0].name == "get_weather" &&
              ready.calls[0].arguments_json == R"({"city":"Paris"})",
          "natural function call is decoded with deterministic identity");
    check(state.active() && state.awaiting_response_end(), "completed call waits for EOTR");
    check(state.observe(400, 7, 9, tools, {}).kind ==
              voicechat::FunctionChannelObservationKind::kNone,
          "forced response tokens do not disturb the marker state");
    const auto finished = state.observe(voicechat::kFunctionEotrTokenId, 7, 9, tools, {});
    check(finished.kind == voicechat::FunctionChannelObservationKind::kResponseFinished &&
              !state.active() && !state.awaiting_response_end(),
          "EOTR closes the function response cycle");

    voicechat::FunctionChannelState wrapped_state;
    const auto wrapped =
        observe_call(wrapped_state, tools,
                     "  <TOOLCALL>[{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Tokyo\"}}]"
                     "</TOOLCALL>\n  ",
                     8, 10);
    check(wrapped.kind == voicechat::FunctionChannelObservationKind::kCallsReady &&
              wrapped.calls.size() == 1 && wrapped.calls[0].name == "get_weather" &&
              wrapped.calls[0].arguments_json == R"({"city":"Tokyo"})",
          "real model TOOLCALL wrappers are removed before JSON parsing");

    voicechat::FunctionChannelState multi_state;
    const auto multi = observe_call(
        multi_state, tools,
        R"([{"name":"get_weather","arguments":{"city":"Oslo"}},{"name":"add","arguments":"{\"a\":2,\"b\":3}"}])",
        11, 42);
    check(multi.kind == voicechat::FunctionChannelObservationKind::kCallsReady &&
              multi.calls.size() == 2 && multi.calls[0].call_id == "call_11_42_0" &&
              multi.calls[1].call_id == "call_11_42_1" &&
              multi.calls[1].arguments_json == R"({"a":2,"b":3})",
          "parallel tool calls preserve order and accept encoded argument objects");
}

void test_invalid_calls_and_markers() {
    const auto tools = make_tools();

    voicechat::FunctionChannelState malformed;
    const auto malformed_event = observe_call(malformed, tools, "not-json");
    check(malformed_event.kind == voicechat::FunctionChannelObservationKind::kError &&
              !malformed.active() && !malformed.capturing_call(),
          "malformed call JSON fails and resets capture");

    voicechat::FunctionChannelState unknown;
    const auto unknown_event =
        observe_call(unknown, tools, R"([{"name":"invented","arguments":{}}])");
    check(unknown_event.kind == voicechat::FunctionChannelObservationKind::kError &&
              unknown_event.error.find("unknown tool") != std::string::npos,
          "unknown model-selected tools are rejected");

    voicechat::FunctionChannelState wrong;
    check(wrong.observe(voicechat::kFunctionEotcTokenId, 1, 1, tools, {}).kind ==
              voicechat::FunctionChannelObservationKind::kError,
          "EOTC without SOTC is rejected");
    (void)wrong.observe(voicechat::kFunctionSotcTokenId, 1, 2, tools, {});
    check(wrong.observe(voicechat::kFunctionSotcTokenId, 1, 2, tools, {}).kind ==
              voicechat::FunctionChannelObservationKind::kError,
          "nested SOTC is rejected");
    (void)wrong.observe(voicechat::kFunctionSotcTokenId, 1, 3, tools, {});
    check(wrong.observe(voicechat::kFunctionEotrTokenId, 1, 3, tools, {}).kind ==
              voicechat::FunctionChannelObservationKind::kError,
          "EOTR before EOTC is rejected");

    voicechat::FunctionChannelState awaiting;
    (void)observe_call(awaiting, tools, R"([{"name":"add","arguments":{}}])");
    check(awaiting.observe(voicechat::kFunctionSotcTokenId, 7, 10, tools, {}).kind ==
              voicechat::FunctionChannelObservationKind::kError,
          "new call markers are rejected until EOTR");
}

void test_bounds_and_reset() {
    const auto tools = make_tools();
    voicechat::FunctionChannelState bounded(2);
    (void)bounded.observe(voicechat::kFunctionSotcTokenId, 1, 1, tools, {});
    (void)bounded.observe(101, 1, 1, tools, {});
    (void)bounded.observe(102, 1, 1, tools, {});
    check(bounded.buffered_tokens() == 2, "function capture admits its exact token bound");
    check(bounded.observe(103, 1, 1, tools, {}).kind ==
                  voicechat::FunctionChannelObservationKind::kError &&
              bounded.buffered_tokens() == 0,
          "function capture rejects overflow and clears buffered tokens");

    (void)bounded.observe(voicechat::kFunctionSotcTokenId, 2, 1, tools, {});
    (void)bounded.observe(104, 2, 1, tools, {});
    bounded.reset();
    check(!bounded.active() && !bounded.capturing_call() && !bounded.awaiting_response_end() &&
              bounded.buffered_tokens() == 0,
          "explicit reset returns the parser to idle");
    expect_invalid([] { voicechat::FunctionChannelState invalid(0); },
                   "function capture requires a positive token bound");
}

void test_tool_response_tokens() {
    std::string encoded_text;
    const auto forced =
        voicechat::build_tool_response_tokens(R"({"temperature":72})", [&](std::string_view text) {
            encoded_text = std::string(text);
            return std::vector<int32_t>{501, 502, 503};
        });
    check(encoded_text == R"(<TOOL_RESPONSE>[{"temperature":72}]</TOOL_RESPONSE>)" &&
              forced == std::vector<int32_t>({501, 502, 503}),
          "tool result is wrapped exactly once and encoded through the injected tokenizer");

    std::string array_text;
    const auto already_array = voicechat::build_tool_response_tokens(
        R"([{"first":1},{"second":2}])", [&](std::string_view text) {
            array_text = text;
            return std::vector<int32_t>{7};
        });
    check(already_array == std::vector<int32_t>({7}) &&
              array_text == R"(<TOOL_RESPONSE>[{"first":1},{"second":2}]</TOOL_RESPONSE>)",
          "tool result arrays are not nested a second time");

    std::string plain_text;
    const auto plain = voicechat::build_tool_response_tokens("ready", [&](std::string_view text) {
        plain_text = text;
        return std::vector<int32_t>{8};
    });
    check(plain == std::vector<int32_t>({8}) &&
              plain_text == R"(<TOOL_RESPONSE>["ready"]</TOOL_RESPONSE>)",
          "plain ASCII tool results become one JSON string result");

    expect_invalid(
        [] {
            (void)voicechat::build_tool_response_tokens(
                R"({"value":"\u00e9"})", [](std::string_view) { return std::vector<int32_t>{1}; });
        },
        "tool response rejects non-ASCII values");
    expect_invalid(
        [] {
            (void)voicechat::build_tool_response_tokens(
                "{}", [](std::string_view) { return std::vector<int32_t>{}; });
        },
        "tool response rejects an empty tokenizer result");
}

void test_on_hold_lookup() {
    const auto tools =
        make_tools(R"({"get_weather":["Let me check.","One moment."],"default":"Please wait."})");
    check(tools.find("get_weather")->ack_message == "Let me check.",
          "on-hold override takes precedence over the tool acknowledgement");
    check(tools.find("add")->ack_message == "Please wait.",
          "on-hold default applies when a tool has no override");

    const auto no_default = make_tools(R"({"get_weather":"Checking."})");
    check(no_default.find("add")->ack_message == "I will add those numbers.",
          "tool acknowledgement remains when no override or default exists");
    expect_invalid([] { (void)make_tools(R"({"default":[]})"); },
                   "on-hold overrides reject empty message lists");
}

} // namespace

int main() {
    test_exact_markers_and_tool_catalog();
    test_tool_validation();
    test_natural_and_multi_call_observation();
    test_invalid_calls_and_markers();
    test_bounds_and_reset();
    test_tool_response_tokens();
    test_on_hold_lookup();
    return failures;
}
