/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace trtmc::minicpmv4_6 {

enum class Role { System, User, Assistant };

struct PromptMessage {
    Role role{Role::User};
    std::string content;
};

// Python str.strip() whitespace (the template's trim filter).
std::string strip(const std::string& text);

// Render a conversation with a provider recipe extracted at build time
// (vision.json "prompts"), including the generation prompt.
std::string format_conversation(const nlohmann::json& recipe,
                                const std::vector<PromptMessage>& messages);
void validate_recipe(const nlohmann::json& recipe);

} // namespace trtmc::minicpmv4_6
