/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "trtmc/internal/language.h"

#include <string>

namespace trtmc {
std::string lfm2_moe_render_conversation(const internal::TextConversationRequest& request,
                                         bool preserve_thinking);
internal::ConversationResult lfm2_moe_parse_conversation(const std::string& decoded,
                                                         bool starts_in_reasoning,
                                                         bool allow_json_tools = false);
void lfm2_moe_assign_tool_call_ids(internal::ConversationResult& result,
                                   const internal::TextConversationRequest& request);
} // namespace trtmc
