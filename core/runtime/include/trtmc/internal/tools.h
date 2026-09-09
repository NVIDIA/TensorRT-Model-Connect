/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace trtmc::internal {

enum class ToolCallState : std::uint32_t {
    Unknown = 0,
    Complete = 1,
    Incomplete = 2,
    Malformed = 3
};

// JSON is confined to the explicitly named tool schema/argument payloads.
// Family/application code interprets these payloads; the SDK executes no tools.
struct ToolDefinitionView {
    std::string_view name;
    std::string_view description;
    std::string_view parameters_schema_json;
};
struct ToolCallView {
    std::string_view call_id;
    std::string_view name;
    std::string_view arguments_json;
    ToolCallState state{ToolCallState::Unknown};
};
struct ToolResultView {
    std::string_view call_id;
    std::string_view content_text;
    bool is_error{false};
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

} // namespace trtmc::internal
