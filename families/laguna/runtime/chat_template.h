/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <string_view>

namespace trtmc::laguna {

// The system message Laguna's chat template inserts when none is supplied.
inline constexpr std::string_view kDefaultSystemMessage =
    "You are a helpful, conversationally-fluent assistant made by Poolside. You are here to "
    "be helpful to users through natural language conversations.";

// Renders Laguna's chat_template.jinja for one user turn without tools, with
// the generation prompt: `<assistant>\n` followed by `<think>` (thinking) or
// `</think>` (no thinking). An empty system prompt selects the default message.
std::string render_chat(std::string_view user, std::string_view system_prompt,
                        bool enable_thinking);

} // namespace trtmc::laguna
