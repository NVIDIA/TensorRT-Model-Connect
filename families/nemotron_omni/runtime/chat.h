/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <string>
#include <vector>
namespace trtmc::nemotron_omni {
struct ChatMessage {
    std::string role, content;
};
// Family-owned plain-content projection of the checkpoint template. Media
// placeholder expansion is performed separately, preserving tokenizer order.
std::string render_chat(const std::vector<ChatMessage>& messages, bool enable_thinking);
} // namespace trtmc::nemotron_omni
