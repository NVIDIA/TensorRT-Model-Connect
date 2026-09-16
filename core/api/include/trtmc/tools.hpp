/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/core.hpp"
#include "trtmc/tools.h"

namespace trtmc {

enum class ToolCallState : std::uint32_t {
    Unknown = TRTMC_TOOL_CALL_UNKNOWN,
    Complete = TRTMC_TOOL_CALL_COMPLETE,
    Incomplete = TRTMC_TOOL_CALL_INCOMPLETE,
    Malformed = TRTMC_TOOL_CALL_MALFORMED
};

struct ToolDefinition {
    std::string name;
    std::string description;
    std::string parameters_schema_json;
};
struct ToolCall {
    std::string call_id;
    std::string name;
    std::string arguments_json;
    ToolCallState state{ToolCallState::Unknown};
};
struct ToolResult {
    std::string call_id;
    std::string content_text;
    bool is_error{false};
};

namespace detail {
inline trtmc_tool_definition_v1 c_tool_definition(const ToolDefinition& value) noexcept {
    return {c_string(value.name), c_string(value.description),
            c_string(value.parameters_schema_json)};
}
inline trtmc_tool_call_v1 c_tool_call(const ToolCall& value) noexcept {
    return {c_string(value.call_id), c_string(value.name), c_string(value.arguments_json),
            static_cast<std::uint32_t>(value.state)};
}
inline trtmc_tool_result_v1 c_tool_result(const ToolResult& value) noexcept {
    return {c_string(value.call_id), c_string(value.content_text), value.is_error ? 1U : 0U};
}
inline ToolCall copy_tool_call(const trtmc_tool_call_v1& value) {
    return {std::string(string_view(value.call_id)), std::string(string_view(value.name)),
            std::string(string_view(value.arguments_json)),
            static_cast<ToolCallState>(value.state)};
}
inline ToolResult copy_tool_result(const trtmc_tool_result_v1& value) {
    return {std::string(string_view(value.call_id)), std::string(string_view(value.content_text)),
            value.is_error != 0};
}
} // namespace detail
} // namespace trtmc
