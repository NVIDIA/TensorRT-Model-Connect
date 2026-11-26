/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc::minicpmv4_6 {

enum class Role { System, User, Assistant };

struct PromptMessage {
    Role role{Role::User};
    std::string content;
};

// Python str.strip() whitespace (the template's trim filter).
std::string strip(const std::string& text);

// Whether strip() removes this code point.
bool is_strip_space(std::uint32_t code_point);

// One message's content as strip() leaves it, assembled part by part: leading white space is
// dropped, and white space is copied (and counted) only once text follows it, so trailing white
// space is never copied. The added texts must stay alive until the content is taken.
class StrippedContent {
  public:
    // Appends `text`; false once the kept code points exceed `limit` (nothing more is copied).
    bool add(std::string_view text, std::size_t limit);
    std::size_t kept() const { return kept_; }
    std::string take() { return std::move(content_); }
    // White-space pieces held until text follows; none beyond the limit (text after them is
    // refused).
    std::size_t pending_pieces() const { return pending_.size(); }

  private:
    std::string content_;
    std::vector<std::string_view> pending_; // white space not yet followed by text
    std::size_t kept_{0}, pending_points_{0};
    bool started_{false};
};

// Render a conversation with a provider recipe extracted at build time
// (vision.json "prompts"), including the generation prompt.
std::string format_conversation(const nlohmann::json& recipe,
                                const std::vector<PromptMessage>& messages);
void validate_recipe(const nlohmann::json& recipe);

} // namespace trtmc::minicpmv4_6
